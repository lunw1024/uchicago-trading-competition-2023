#!/usr/bin/env python

from utc_bot import UTCBot, start_bot
import proto.utc_bot as pb
import betterproto
import asyncio
import json
from dataclasses import dataclass
from py_vollib.black_scholes_merton import black_scholes_merton
from py_vollib.black_scholes_merton.greeks.analytical import delta, gamma, vega, theta, rho
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility

PARAM_FILE = "params.json"
SYMBOLS = ['SPY', 'SPY65C', 'SPY65P', 'SPY70C', 'SPY70P', 'SPY75C', 'SPY75P', 'SPY80C', 'SPY80P', 'SPY85C', 'SPY85P', 'SPY90C', 'SPY90P', 'SPY95C', 'SPY95P', 'SPY100C', 'SPY100P', 'SPY105C', 'SPY105P', 'SPY110C', 'SPY110P', 'SPY115C', 'SPY115P', 'SPY120C', 'SPY120P', 'SPY125C', 'SPY125P', 'SPY130C', 'SPY130P', 'SPY135C', 'SPY135P']
CALLS = ['SPY65C', 'SPY70C', 'SPY75C', 'SPY80C', 'SPY85C', 'SPY90C', 'SPY95C', 'SPY100C', 'SPY105C', 'SPY110C', 'SPY115C', 'SPY120C', 'SPY125C', 'SPY130C', 'SPY135C']
PUTS = ['SPY65P', 'SPY70P', 'SPY75P', 'SPY80P', 'SPY85P', 'SPY90P', 'SPY95P', 'SPY100P', 'SPY105P', 'SPY110P', 'SPY115P', 'SPY120P', 'SPY125P', 'SPY130P', 'SPY135P']
UNDERLYING = 'SPY'
Type = pb.OrderSpecType
Side = pb.OrderSpecSide

@dataclass
class Order:
    id: int
    symbol: str
    price: int
    qty: int
    side: Side

class OptionBot(UTCBot):
    """
    An example bot that reads from a file to set internal parameters during the round
    """

    async def handle_round_started(self):
        self.orders: dict[int, Order] = {}
        self.t = 0
        await asyncio.sleep(0.1)
        asyncio.create_task(self.handle_read_params())

    async def handle_read_params(self):
        """read the config every 1s"""
        while True:
            try:
                self.params = json.load(open(PARAM_FILE, "r"))
            except:
                print("Unable to read file " + PARAM_FILE)
            await asyncio.sleep(1)

    async def handle_exchange_update(self, update: pb.FeedMessage):
        kind, _ = betterproto.which_one_of(update, "msg")
        # Competition event messages
        if kind == "generic_msg":
            msg = update.generic_msg.message
            print(msg)
        elif kind == "market_snapshot_msg":
            msg = update.market_snapshot_msg
            self.t += 1
            ts = msg.timestamp
            books = msg.books
            self.make_market(books)
        elif kind == "fill_msg":
            msg = update.fill_msg
            if msg.remaining_qty == 0:
                print(f"Order {msg.order_id} done.")
                self.orders.pop(msg.order_id, None)
        elif kind == "order_cancelled_msg":
            msg = update.order_cancelled_msg
            if msg.intentional: # TODO: deal with unintential cancels, if it breaks
                for id in msg.order_ids:
                    self.orders.pop(id)

    async def quote(self, symbol, side, qty, price):
        response = await self.place_order(symbol, Type.LIMIT, side, qty, price)
        if not response.ok:
            print("Can't place order:", response.message)
            return False
        self.orders[response.order_id] = Order(response.order_id, symbol, price, qty, side)
        return True

    def make_market(self, books):
        for symbol, book in books.items():
            if symbol == "SPY":
                continue
            best_bid = float(book.bids[0].px) if book.bids else 0.1
            best_ask = float(book.asks[0].px) if book.asks else 100
            # join bb ba
            asyncio.create_task(self.quote(symbol, Side.BID, 2, best_bid))
            asyncio.create_task(self.quote(symbol, Side.ASK, 2, best_ask))
            print(f"{symbol}: Quoting {best_bid}@{best_ask}")
            
            
            



if __name__ == "__main__":
    start_bot(OptionBot)

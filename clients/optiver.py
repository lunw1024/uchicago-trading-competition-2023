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
import numpy as np
from datetime import datetime, timedelta
from math import floor, ceil

PARAM_FILE = "params.json"
SYMBOLS = ['SPY', 'SPY65C', 'SPY65P', 'SPY70C', 'SPY70P', 'SPY75C', 'SPY75P', 'SPY80C', 'SPY80P', 'SPY85C', 'SPY85P', 'SPY90C', 'SPY90P', 'SPY95C', 'SPY95P', 'SPY100C', 'SPY100P', 'SPY105C', 'SPY105P', 'SPY110C', 'SPY110P', 'SPY115C', 'SPY115P', 'SPY120C', 'SPY120P', 'SPY125C', 'SPY125P', 'SPY130C', 'SPY130P', 'SPY135C', 'SPY135P']
DONT_TRADE = ['SPY65C', 'SPY65P', 'SPY70C', 'SPY70P', 'SPY75C', 'SPY75P','SPY125C', 'SPY125P', 'SPY130C', 'SPY130P', 'SPY135C', 'SPY135P']
CALLS = ['SPY65C', 'SPY70C', 'SPY75C', 'SPY80C', 'SPY85C', 'SPY90C', 'SPY95C', 'SPY100C', 'SPY105C', 'SPY110C', 'SPY115C', 'SPY120C', 'SPY125C', 'SPY130C', 'SPY135C']
PUTS = ['SPY65P', 'SPY70P', 'SPY75P', 'SPY80P', 'SPY85P', 'SPY90P', 'SPY95P', 'SPY100P', 'SPY105P', 'SPY110P', 'SPY115P', 'SPY120P', 'SPY125P', 'SPY130P', 'SPY135P']
UNDERLYING = 'SPY'
RISK_LIMITS = {
    "delta": 2000,
    "gamma": 5000,
    "vega": 1000000,
    "theta": 5000
}
Type = pb.OrderSpecType
Side = pb.OrderSpecSide
lock = asyncio.Lock()

@dataclass
class Order:
    id: int
    symbol: str
    price: int
    qty: int
    side: Side
    cancelling: bool = False

def parse_sym(s):
    if s == "SPY":
        return None
    flag = s[-1].lower()
    strike = int(s[3:-1])
    return flag, strike

class OptionBot(UTCBot):
    """
    An example bot that reads from a file to set internal parameters during the round
    """

    async def handle_round_started(self):
        self.orders: dict[str, list] = {}
        self.positions: dict[str, int] = {}
        self.risks: dict[str, int] = {"delta": 0, "gamma": 0, "vega": 0, "theta": 0}
        self.last_price = 100
        self.anchor = 3.5
        self.returns = []
        self.start_t = datetime.now()
        self.t = 0
        await asyncio.sleep(0.1)
        asyncio.create_task(self.handle_read_params())
        # asyncio.create_task(self.update_position())

    async def handle_read_params(self):
        """read the config every 1s"""
        while True:
            try:
                self.params = json.load(open(PARAM_FILE, "r"))
            except:
                print("Unable to read file " + PARAM_FILE)
            await asyncio.sleep(1)

    def get_t(self):
        now = datetime.now()
        d = now - self.start_t
        secs = d // timedelta(minutes=1) * 30 + min(timedelta(seconds=30), d % timedelta(minutes=1)) / timedelta(seconds=1)
        return secs
    
    def get_expiry(self):
        return 0.25-self.get_t()/3600

    async def update_position(self):
        """Update position every 0.1s"""
        # while True:
        resp = await self.get_positions()
        self.positions = resp.positions
        self.risks = {"delta": 0, "gamma": 0, "vega": 0, "theta": 0}
        under = self.last_price
        expiry = self.get_expiry()
        for sym, v in self.positions.items():
            if sym == "SPY":
                self.risks["delta"] += v
                continue
            flag, strike = parse_sym(sym)
            vol = self.vol(strike)
            # print(flag, under, strike, expiration, vol)
            self.risks["delta"] += v * delta(flag, under, strike, expiry, 0, vol, 0)
            self.risks["gamma"] += v * gamma(flag, under, strike, expiry, 0, vol, 0)
            self.risks["vega"] += v * vega(flag, under, strike, expiry, 0, vol, 0)
            self.risks["theta"] += v * theta(flag, under, strike, expiry, 0, vol, 0)
            # await asyncio.sleep(0.5)

    async def handle_exchange_update(self, update: pb.FeedMessage):
        kind, _ = betterproto.which_one_of(update, "msg")
        # Competition event messages
        if kind == "generic_msg":
            msg = update.generic_msg
            print(msg.event_type, msg.message)
        elif kind == "pnl_msg":
            msg = update.pnl_msg
            print(f"pnl: {msg.m2m_pnl}")
        elif kind == "market_snapshot_msg":
            msg = update.market_snapshot_msg
            ts = msg.timestamp
            books = msg.books
            # print("SPY:", len(books["SPY"].bids), len(books["SPY"].asks))
            if self.t == floor(self.get_t()) or not books["SPY"].bids or not books["SPY"].asks:
                return
            self.t = floor(self.get_t())
            print(ts, self.get_t())
            new_price = (float(books["SPY"].bids[0].px) + float(books["SPY"].asks[0].px)) / 2
            self.last_price = new_price

            # update anchor
            if books["SPY100C"].bids and books["SPY100C"].asks:
                self.anchor = (float(books["SPY100C"].bids[0].px) + float(books["SPY100C"].asks[0].px)) / 2

            await self.update_position()
            if not self.params['halt'] and not self.params['unwind']:
                await self.make_market(books)
                await self.hedge()
            if self.params['unwind']:
                await self.unwind()

            # debug
            print(sorted(self.positions.items()))
            print(self.risks)
            print(f"Anchor: {self.anchor}")
            # print(self.last_price, [self.vol(i) for i in range(65, 140, 5)])

        elif kind == "fill_msg":
            msg = update.fill_msg

        elif kind == "order_cancelled_msg":
            msg = update.order_cancelled_msg
            if msg.intentional: # TODO: deal with unintential cancels, if it breaks
                pass
            else:
                print("WARNING: unintentional cancels")

    async def quote(self, symbol, side, qty, price):
        if price <= 0:
            return False
        async with lock:
            handle = symbol+("b" if side == Side.BID else "a")
            id = self.orders.get(handle, "")
            # print(f"{handle}: Modifying order {id}")
            response = await self.modify_order(id, symbol, Type.LIMIT, side, qty, price)
            if not response.ok:
                print("Can't place order:", response.message)
                return False
            else:
                self.orders[handle] = response.order_id
                # print(f"{handle}: Updated order id to {response.order_id}")
                return True

    async def make_market(self, books):
        for symbol, book in books.items():
            if symbol == "SPY" or symbol in DONT_TRADE:
                continue
            flag, strike = parse_sym(symbol)
            best_bid = float(book.bids[0].px) if book.bids else 0.1
            best_ask = float(book.asks[0].px) if book.asks else 100
            fair_price = black_scholes_merton(flag, self.last_price, strike, self.get_expiry(), 0, self.vol(strike), 0)

            # quote a constant spread
            size = self.params["quote_size"]
            hs = self.params["half_spread"]
            bid = round(fair_price - hs, 1)
            ask = round(fair_price + hs, 1)
            if bid >= 0:
                asyncio.gather(
                    self.quote(symbol, Side.BID, size, bid),
                    self.quote(symbol, Side.ASK, size, ask),
                )
            # print(f"{symbol}: Quoting {bid}@{ask}")
    
    async def hedge(self):
        tasks = []
        if self.risks["delta"] > 3:
            tasks.append(self.place_order("SPY", Type.MARKET, Side.ASK, ceil(self.risks["delta"] - 3), None))
        elif self.risks["delta"] < -3:
            tasks.append(self.place_order("SPY", Type.MARKET, Side.BID, ceil(-self.risks["delta"] - 3), None))
        asyncio.gather(*tasks)

    async def unwind(self):
        tasks = []
        for sym, v in self.positions.items():
            if v > 0:
                tasks.append(self.place_order(sym, Type.MARKET, Side.ASK, min(v, 10)))
            elif v < 0:
                tasks.append(self.place_order(sym, Type.MARKET, Side.BID, min(-v, 10)))
        asyncio.gather(*tasks)

    def realized_vol(self, win: int) -> float:
        win = min(len(self.returns), win)
        if win == 0:
            return 0
        # print(np.square(self.returns[-win:]).sum())
        return np.sqrt(np.square(self.returns[-win:]).sum() * (3600/win))

    def vol(self, strike) -> float:
        # horizontal shift
        x = self.last_price - 100 + strike
        # vertical shift
        y = implied_volatility(self.anchor, self.last_price, 100, self.get_expiry(), 0, 0, 'c') - self.params["vol_curve"]["100"]
        # l, r = max(65, floor(x / 5) * 5), min(135, ceil(x / 5) * 5)
        # a = self.params["vol_curve"][str(l)] + self.params["vol_curve_offset"][str(r)]
        # b = self.params["vol_curve"][str(l)] + self.params["vol_curve_offset"][str(r)]
        # iv = a + (b - a) * (x - l) + y
        iv = self.params["vol_curve"][str(round(x))] + y + self.params["vol_curve_offset"][self.params["offset"]][str(strike)] + self.params["overall_offset"]
        return iv

if __name__ == "__main__":
    start_bot(OptionBot)

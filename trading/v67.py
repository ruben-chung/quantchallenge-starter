# Strategy: VW-spread maker (no import-time dependency on Side/Ticker)
from __future__ import annotations
from collections import deque

class Strategy:
    def __init__(self) -> None:
        self.reset_state()

    def reset_state(self) -> None:
        # discovered at runtime
        self.ticker = None         # engine's Ticker object
        self.SideBUY = None        # engine's Side.BUY value
        self.SideSELL = None       # engine's Side.SELL value

        # our resting orders (one bid + one ask)
        self.bid_oid = None
        self.ask_oid = None

        # params
        self.depth = 5                 # top-N levels for VW prices
        self.inside_frac = 0.25        # how far inside the spread we quote
        self.min_half_spread = 0.10    # never quote tighter than this (points)
        self.base_qty = 4.0            # resting quote size
        self.max_qty = 16.0
        self.tick_size = 0.0           # set if venue has ticks, e.g. 0.05

        # one-time “prove pipeline” orders on first snapshot
        self.sanity_done = False
        self.sanity_qty = 3.0

        # optional bookkeeping
        self._mid_hist = deque(maxlen=64)

        # cache of best prices (for IOC fallbacks)
        self._best_bid = None
        self._best_ask = None

    # ---------- tiny utils ----------
    @staticmethod
    def _clamp(x, lo, hi): return lo if x < lo else hi if x > hi else x

    def _round_tick(self, px):
        t = float(self.tick_size or 0.0)
        return round(float(px) / t) * t if t > 0 else float(px)

    @staticmethod
    def _vw(levels, n):
        """volume-weighted price and total qty from top-n levels"""
        tot_q = 0.0; tot_pq = 0.0; used = 0
        for i in range(min(n, len(levels))):
            try:
                p, q = levels[i]
                p = float(p); q = float(q)
                if q <= 0: continue
                tot_q += q; tot_pq += p * q; used += 1
            except Exception:
                continue
        if tot_q <= 0 or used == 0:
            return None, 0.0
        return tot_pq / tot_q, tot_q

    # ---------- enum discovery / helpers ----------
    def _ensure_side_enum(self):
        """Populate SideBUY/SELL if engine exposed a Side enum."""
        if self.SideBUY is not None and self.SideSELL is not None:
            return
        Side = globals().get("Side")  # engine often defines this in globals
        if Side is not None:
            try:
                self.SideBUY  = getattr(Side, "BUY")
                self.SideSELL = getattr(Side, "SELL")
            except Exception:
                pass

    def _side_val(self, want_buy: bool):
        self._ensure_side_enum()
        if want_buy and self.SideBUY is not None:   return self.SideBUY
        if (not want_buy) and self.SideSELL is not None: return self.SideSELL
        # last-ditch: some engines accept strings
        return "BUY" if want_buy else "SELL"

    # ---------- robust order wrappers (use YOUR template signatures) ----------
    def _place_market(self, want_buy: bool, qty: float):
        if self.ticker is None:
            # try to default if engine didn't call us yet
            self.ticker = globals().get("Ticker", None)
            if hasattr(self.ticker, "TEAM_A"):
                self.ticker = self.ticker.TEAM_A
        if self.ticker is None:
            return None
        side = self._side_val(want_buy)
        try:
            return place_market_order(side, self.ticker, float(qty))
        except Exception:
            # fallback: emulate with IOC at top-of-book
            if want_buy and self._best_ask is not None:
                return place_limit_order(self._side_val(True),  self.ticker, float(qty), self._best_ask, True)
            if (not want_buy) and self._best_bid is not None:
                return place_limit_order(self._side_val(False), self.ticker, float(qty), self._best_bid, True)
            return None

    def _place_limit(self, want_buy: bool, qty: float, price: float, ioc: bool = False):
        if self.ticker is None:
            T = globals().get("Ticker", None)
            if hasattr(T, "TEAM_A"): self.ticker = T.TEAM_A
        if self.ticker is None:
            return None
        return place_limit_order(self._side_val(want_buy), self.ticker, float(qty), float(price), bool(ioc))

    def _cancel(self, oid):
        if oid is None or self.ticker is None: return
        try:
            cancel_order(self.ticker, oid)
        except Exception:
            pass

    # ---------- exchange callbacks ----------
    def on_trade_update(self, ticker, side, quantity, price) -> None:
        # just keep ticker fresh and remember last execution price if you want
        self.ticker = ticker

    def on_orderbook_update(self, ticker, side, quantity, price) -> None:
        # capture enum from the side object if possible
        try:
            cls = getattr(side, "__class__", None)
            buy  = getattr(cls, "BUY", None)
            sell = getattr(cls, "SELL", None)
            if buy is not None and sell is not None:
                self.SideBUY, self.SideSELL = buy, sell
        except Exception:
            pass
        self.ticker = ticker

    def on_account_update(self, ticker, side, price, quantity, capital_remaining) -> None:
        # not used for logic here, but you could track PnL/pos
        self.ticker = ticker

    def on_game_event_update(self, *args, **kwargs) -> None:
        # strategy is book-driven; no-op here
        pass

    # >>> core driver: full book snapshot <<<
    def on_orderbook_snapshot(self, ticker, bids, asks) -> None:
        self.ticker = ticker
        self._best_bid = float(bids[0][0]) if bids else None
        self._best_ask = float(asks[0][0]) if asks else None

        # one-time sanity: BUY then SELL to prove pipeline; cross if market fails
        if not self.sanity_done:
            self._place_market(True,  self.sanity_qty)
            self._place_market(False, self.sanity_qty)
            self.sanity_done = True

        # need both sides to quote
        if not bids or not asks:
            self._cancel(self.bid_oid); self.bid_oid = None
            self._cancel(self.ask_oid); self.ask_oid = None
            return

        vwbid, bid_vol = self._vw(bids, self.depth)
        vwask, ask_vol = self._vw(asks, self.depth)
        if vwbid is None or vwask is None or vwask <= vwbid:
            self._cancel(self.bid_oid); self.bid_oid = None
            self._cancel(self.ask_oid); self.ask_oid = None
            return

        spread = vwask - vwbid
        mid = 0.5 * (vwbid + vwask)
        self._mid_hist.append(mid)

        half = max(self.min_half_spread, self.inside_frac * spread)
        bid_px = self._round_tick(self._clamp(mid - half, 0.0, vwask))
        ask_px = self._round_tick(self._clamp(mid + half, vwbid, 100.0))

        # simple imbalance sizing
        tot = bid_vol + ask_vol
        imb = ((bid_vol - ask_vol) / tot) if tot > 0 else 0.0
        bid_qty = self._clamp(self.base_qty * (1.0 + max(0.0, -imb)), 1.0, self.max_qty)
        ask_qty = self._clamp(self.base_qty * (1.0 + max(0.0,  imb)), 1.0, self.max_qty)

        # refresh our quotes every snapshot
        if self.bid_oid: self._cancel(self.bid_oid); self.bid_oid = None
        if self.ask_oid: self._cancel(self.ask_oid); self.ask_oid = None

        self.bid_oid = self._place_limit(True,  bid_qty, bid_px, ioc=False)
        self.ask_oid = self._place_limit(False, ask_qty, ask_px, ioc=False)

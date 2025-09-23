# Strategy: simple volume-weighted spread maker
# Uses the orderbook snapshot only. Posts one bid + one ask each snapshot.

from collections import deque

class Strategy:
    def __init__(self):
        self.reset_state()

    def reset_state(self):
        # engine objects / state
        self.ticker = None
        self.bid_oid = None
        self.ask_oid = None

        # params (tweak)
        self.depth = 5                # how many levels to use for VW bests
        self.inside_frac = 0.25       # 25% of current spread toward mid
        self.min_half_spread = 0.10   # don’t quote tighter than this
        self.base_qty = 4             # integer size for each quote
        self.max_qty = 12
        self._mid_hist = deque(maxlen=50)

    # ---- util ----
    @staticmethod
    def _clamp(x, lo, hi):
        return lo if x < lo else hi if x > hi else x

    @staticmethod
    def _vw(levels, n):
        """Volume-weighted price from top n levels. levels: [(price, qty), ...]"""
        tot_q = 0.0
        tot_pq = 0.0
        used = 0
        for i in range(min(n, len(levels))):
            try:
                p, q = levels[i]
                p = float(p); q = float(q)
                if q <= 0: 
                    continue
                tot_q += q
                tot_pq += p * q
                used += 1
            except Exception:
                continue
        if tot_q <= 0 or used == 0:
            return None, 0.0
        return tot_pq / tot_q, tot_q

    def _cancel(self, oid):
        if oid is None or self.ticker is None:
            return
        try:
            cancel_order(self.ticker, oid)
        except Exception:
            try:
                cancel_order(ticker=self.ticker, order_id=oid)
            except Exception:
                pass

    # ---- core: snapshot-driven quoting ----
    def on_orderbook_snapshot(self, ticker, bids, asks) -> None:
        """
        Called periodically with full book. We compute VW best bid/ask and quote
        inside the spread with fixed sizes, refreshing each snapshot.
        """
        self.ticker = ticker

        if not bids or not asks:
            # No two-sided market → pull any resting quotes
            self._cancel(self.bid_oid); self.bid_oid = None
            self._cancel(self.ask_oid); self.ask_oid = None
            return

        vwbid, bid_vol = self._vw(bids, self.depth)
        vwask, ask_vol = self._vw(asks, self.depth)
        if vwbid is None or vwask is None or vwask <= vwbid:
            # Degenerate book; pull quotes
            self._cancel(self.bid_oid); self.bid_oid = None
            self._cancel(self.ask_oid); self.ask_oid = None
            return

        spread = vwask - vwbid
        mid = 0.5 * (vwbid + vwask)
        self._mid_hist.append(mid)

        half = max(self.min_half_spread, self.inside_frac * spread)
        bid_px = self._clamp(mid - half, 0.0, vwask)          # keep inside spread
        ask_px = self._clamp(mid + half, vwbid, 100.0)

        # simple imbalance sizing: lean toward the heavy side
        tot = bid_vol + ask_vol
        imb = ((bid_vol - ask_vol) / tot) if tot > 0 else 0.0   # [-1, 1]
        bid_qty = int(self._clamp(self.base_qty * (1.0 + max(0.0, -imb)), 1, self.max_qty))
        ask_qty = int(self._clamp(self.base_qty * (1.0 + max(0.0,  imb)), 1, self.max_qty))

        # Refresh quotes every snapshot (cancel then repost)
        if self.bid_oid:
            self._cancel(self.bid_oid); self.bid_oid = None
        if self.ask_oid:
            self._cancel(self.ask_oid); self.ask_oid = None

        # Use the canonical signatures from your doc:
        # place_limit_order(ticker, side, quantity, price, IOC=False)
        try:
            self.bid_oid = place_limit_order(self.ticker, "BUY",  bid_qty, bid_px, False)
        except Exception:
            self.bid_oid = place_limit_order(ticker=self.ticker, side="BUY",  quantity=bid_qty, price=bid_px, IOC=False)

        try:
            self.ask_oid = place_limit_order(self.ticker, "SELL", ask_qty, ask_px, False)
        except Exception:
            self.ask_oid = place_limit_order(ticker=self.ticker, side="SELL", quantity=ask_qty, price=ask_px, IOC=False)

    # ---- other callbacks (unused here) ----
    def on_orderbook_update(self, *_, **__): 
        pass

    def on_trade_update(self, *_, **__): 
        pass

    def on_account_update(self, *_, **__): 
        pass

    def on_game_event_update(self, *_, **__): 
        pass

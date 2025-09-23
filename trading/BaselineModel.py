"""
Quant Challenge 2025

Algorithmic strategy – VW spread maker with full local orderbook and 30/min rate limit
"""

from __future__ import annotations
import time
from enum import Enum
from typing import Optional, Dict, List, Tuple
from collections import deque

# ─────────────────────────────────────────────────────────────────────────────
# Provided API surface (keep these signatures exactly)
# ─────────────────────────────────────────────────────────────────────────────

class Side(Enum):
    BUY = 0
    SELL = 1

class Ticker(Enum):
    # TEAM_A (home team)
    TEAM_A = 0

def place_market_order(side: Side, ticker: Ticker, quantity: float) -> None:
    """Place a market order."""
    return

def place_limit_order(side: Side, ticker: Ticker, quantity: float, price: float, ioc: bool = False) -> int:
    """Place a limit order. Returns order_id."""
    return 0

def cancel_order(ticker: Ticker, order_id: int) -> bool:
    """Cancel an order. Returns success flag."""
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Strategy
# ─────────────────────────────────────────────────────────────────────────────

class Strategy:
    """Simple volume-weighted spread maker.
    
    • Maintains a complete **local orderbook** from snapshots + incremental updates  
    • Quotes **one bid + one ask** inside the spread (VW top-N levels)  
    • One-time **sanity cross** on first two-sided book to prove trading path  
    • Respects **rate limit: 30 orders/min** (place + cancel counted)  
    • Updates position/cash on fills (account updates)  
    """

    # ───────── Lifecycle ─────────
    def __init__(self) -> None:
        self.reset_state()

    def reset_state(self) -> None:
        # Engine enums/functions resolved at runtime (robust to import-time stubs)
        self._Side = None
        self._Ticker = None
        self._fn_place_market = None
        self._fn_place_limit = None
        self._fn_cancel = None

        # Identity / state
        self._ticker: Optional[Ticker] = None

        # Full local orderbook
        self._bids: Dict[float, float] = {}      # price -> qty
        self._asks: Dict[float, float] = {}
        self._bid_prices: List[float] = []       # sorted desc
        self._ask_prices: List[float] = []       # sorted asc
        self._best_bid: Optional[Tuple[float, float]] = None
        self._best_ask: Optional[Tuple[float, float]] = None

        # Our resting orders (one bid + one ask) and last posted prices
        self._bid_oid: Optional[int] = None
        self._ask_oid: Optional[int] = None
        self._last_bid_px: Optional[float] = None
        self._last_ask_px: Optional[float] = None

        # Position / PnL (tracked on our fills)
        self.position: float = 0.0
        self.cash: float = 0.0
        self.capital_remaining: Optional[float] = None

        # Quoting params (tune safely)
        self.depth: int = 5                 # VW over top N levels
        self.inside_frac: float = 0.25      # fraction of spread toward mid
        self.min_half_spread: float = 0.10  # don’t quote tighter than this
        self.base_qty: float = 4.0          # resting quote size
        self.max_qty: float = 16.0          # cap size
        self.tick_size: float = 0.0         # set if venue enforces tick (e.g., 0.05)
        self.price_epsilon: float = 1e-6    # avoid churn on tiny float diffs

        # One-time “prove we can trade” burst
        self._did_sanity: bool = False
        self._sanity_qty: float = 3.0

        # History (optional)
        self._mid_hist: deque[float] = deque(maxlen=64)

        # Rate limit: 30 orders/min (place + cancel)
        self._rl_capacity: float = 30.0
        self._rl_tokens: float = 30.0     # start full
        self._rl_refill_per_sec: float = 30.0 / 60.0  # = 0.5 tokens/sec
        self._rl_last_t: float = time.monotonic()

    # ───────── Runtime API resolution ─────────
    def _ensure_api(self) -> None:
        g = globals()
        if self._Side is None:
            self._Side = g.get("Side", None)
        if self._Ticker is None:
            self._Ticker = g.get("Ticker", None)
        if self._fn_place_market is None:
            self._fn_place_market = g.get("place_market_order", None)
        if self._fn_place_limit is None:
            self._fn_place_limit = g.get("place_limit_order", None)
        if self._fn_cancel is None:
            self._fn_cancel = g.get("cancel_order", None)

    def _ensure_ticker(self) -> None:
        if self._ticker is None:
            self._ensure_api()
            if self._Ticker is not None and hasattr(self._Ticker, "TEAM_A"):
                self._ticker = getattr(self._Ticker, "TEAM_A")

    # ───────── Rate limit: 30/min (leaky bucket) ─────────
    def _rl_allow(self, cost: float = 1.0) -> bool:
        now = time.monotonic()
        dt = max(0.0, now - self._rl_last_t)
        self._rl_last_t = now
        self._rl_tokens = min(self._rl_capacity, self._rl_tokens + dt * self._rl_refill_per_sec)
        if self._rl_tokens >= cost:
            self._rl_tokens -= cost
            return True
        return False

    # ───────── Math helpers ─────────
    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return lo if x < lo else hi if x > hi else x

    def _round_tick(self, px: float) -> float:
        t = float(self.tick_size or 0.0)
        return round(px / t) * t if t > 0 else px

    @staticmethod
    def _vw(levels: List[Tuple[float, float]], n: int) -> Tuple[Optional[float], float]:
        tq = 0.0; tpq = 0.0; used = 0
        m = min(n, len(levels))
        for i in range(m):
            try:
                p, q = levels[i]
                p = float(p); q = float(q)
                if q <= 0: 
                    continue
                tq += q; tpq += p * q; used += 1
            except Exception:
                continue
        if tq <= 0 or used == 0:
            return None, 0.0
        return tpq / tq, tq

    # ───────── Local orderbook maintenance ─────────
    def _rebuild_books(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> None:
        self._bids.clear(); self._asks.clear()
        self._bid_prices = []; self._ask_prices = []

        for p, q in bids:
            p = float(p); q = float(q)
            if q > 0:
                self._bids[p] = q; self._bid_prices.append(p)
        for p, q in asks:
            p = float(p); q = float(q)
            if q > 0:
                self._asks[p] = q; self._ask_prices.append(p)

        self._bid_prices.sort(reverse=True)
        self._ask_prices.sort()

        self._best_bid = (self._bid_prices[0], self._bids[self._bid_prices[0]]) if self._bid_prices else None
        self._best_ask = (self._ask_prices[0], self._asks[self._ask_prices[0]]) if self._ask_prices else None

    def _apply_level_update(self, side: Side, quantity: float, price: float) -> None:
        """Apply incremental (side, quantity, price) to our local book."""
        p = float(price); q = float(quantity)
        is_bid = (side == Side.BUY) if isinstance(side, Side) else (getattr(side, "name", str(side)).upper() in ("BUY","BID"))
        book = self._bids if is_bid else self._asks
        plist = self._bid_prices if is_bid else self._ask_prices

        existed = p in book
        if q <= 0:
            if existed:
                del book[p]
                try: plist.remove(p)
                except ValueError: pass
        else:
            book[p] = q
            if not existed:
                plist.append(p)
                plist.sort(reverse=is_bid)

        # refresh bests
        self._best_bid = (self._bid_prices[0], self._bids[self._bid_prices[0]]) if self._bid_prices else None
        self._best_ask = (self._ask_prices[0], self._asks[self._ask_prices[0]]) if self._ask_prices else None

    # ───────── Robust order wrappers (rate-limited) ─────────
    def _place_market(self, want_buy: bool, qty: float):
        self._ensure_api(); self._ensure_ticker()
        if not callable(self._fn_place_market) or self._ticker is None:
            return None
        if not self._rl_allow(1.0):
            return None
        side = Side.BUY if want_buy else Side.SELL if self._Side is Side else ("BUY" if want_buy else "SELL")
        try:
            return self._fn_place_market(side, self._ticker, float(qty))
        except Exception:
            # Fallback: cross with IOC at top-of-book
            px = (self._best_ask[0] if want_buy and self._best_ask else
                  self._best_bid[0] if (not want_buy) and self._best_bid else None)
            if px is not None:
                return self._place_limit(want_buy, qty, px, ioc=True)
            return None

    def _place_limit(self, want_buy: bool, qty: float, price: float, ioc: bool = False):
        self._ensure_api(); self._ensure_ticker()
        if not callable(self._fn_place_limit) or self._ticker is None:
            return None
        if not self._rl_allow(1.0):
            return None
        side = Side.BUY if want_buy else Side.SELL if self._Side is Side else ("BUY" if want_buy else "SELL")
        try:
            return self._fn_place_limit(side, self._ticker, float(qty), float(price), bool(ioc))
        except Exception:
            return None

    def _cancel(self, oid: Optional[int]) -> None:
        if oid is None:
            return
        self._ensure_api(); self._ensure_ticker()
        if not callable(self._fn_cancel) or self._ticker is None:
            return
        if not self._rl_allow(1.0):
            return
        try:
            self._fn_cancel(self._ticker, oid)
        except Exception:
            pass

    # ───────── Exchange callbacks ─────────
    def on_trade_update(
        self, ticker: Ticker, side: Side, quantity: float, price: float
    ) -> None:
        # Keep ticker fresh; optional: observe last trade price
        self._ticker = ticker
        # print(f"[trade] {side.name} {quantity} @ {price}")

    def on_orderbook_update(
        self, ticker: Ticker, side: Side, quantity: float, price: float
    ) -> None:
        # Maintain full local book incrementally
        self._ticker = ticker
        self._apply_level_update(side, quantity, price)

    def on_account_update(
        self,
        ticker: Ticker,
        side: Side,
        price: float,
        quantity: float,
        capital_remaining: float,
    ) -> None:
        # Track simple position/cash on OUR fills
        self._ticker = ticker
        q = float(quantity); px = float(price)
        if side == Side.BUY:
            self.position += q
            self.cash -= q * px
        else:
            self.position -= q
            self.cash += q * px
        self.capital_remaining = float(capital_remaining)
        # print(f"[fill] {side.name} {q} @ {px} | pos={self.position:.1f} cash={self.cash:.2f}")

    def on_game_event_update(self,
                           event_type: str,
                           home_away: str,
                           home_score: int,
                           away_score: int,
                           player_name: Optional[str],
                           substituted_player_name: Optional[str],
                           shot_type: Optional[str],
                           assist_player: Optional[str],
                           rebound_type: Optional[str],
                           coordinate_x: Optional[float],
                           coordinate_y: Optional[float],
                           time_seconds: Optional[float]
        ) -> None:
        # Strategy is primarily book-driven; leave events for future alpha.
        # Still honor END_GAME to clean up & reset.
        if event_type == "END_GAME":
            self._cancel(self._bid_oid); self._bid_oid = None
            self._cancel(self._ask_oid); self._ask_oid = None
            self.reset_state()
            return
        # print(f"{event_type} {home_score} - {away_score}")

    def on_orderbook_snapshot(self, ticker: Ticker, bids: list, asks: list) -> None:
        """Periodic full snapshot: rebuild book, compute VW bests, quote inside spread."""
        self._ticker = ticker

        # 1) Rebuild complete local book from snapshot
        self._rebuild_books(bids, asks)

        # 2) One-time sanity: if two-sided, try to guarantee a print (cross or market)
        if not self._did_sanity and self._best_bid and self._best_ask:
            # Try market both ways (rate-limited). If market fails, IOC cross falls back inside _place_market.
            self._place_market(True,  self._sanity_qty)
            self._place_market(False, self._sanity_qty)
            self._did_sanity = True

        # 3) If not two-sided, pull our quotes and bail
        if not self._best_bid or not self._best_ask:
            self._cancel(self._bid_oid); self._bid_oid = None
            self._cancel(self._ask_oid); self._ask_oid = None
            return

        # 4) Compute VW bests (top-N)
        vwbid, bid_vol = self._vw(bids, self.depth)
        vwask, ask_vol = self._vw(asks, self.depth)
        if vwbid is None or vwask is None or vwask <= vwbid:
            self._cancel(self._bid_oid); self._bid_oid = None
            self._cancel(self._ask_oid); self._ask_oid = None
            return

        mid = 0.5 * (vwbid + vwask)
        spread = vwask - vwbid
        self._mid_hist.append(mid)

        # 5) Choose inside-spread prices (respect minimum half-spread; clamp safely)
        half = max(self.min_half_spread, self.inside_frac * spread)
        target_bid_px = self._round_tick(self._clamp(mid - half, 0.0, vwask - 1e-9))
        target_ask_px = self._round_tick(self._clamp(mid + half, vwbid + 1e-9, 100.0))

        # 6) Size leaning by displayed imbalance
        tot = bid_vol + ask_vol
        imb = ((bid_vol - ask_vol) / tot) if tot > 0 else 0.0      # [-1, 1]
        bid_qty = self._clamp(self.base_qty * (1.0 + max(0.0, -imb)), 1.0, self.max_qty)
        ask_qty = self._clamp(self.base_qty * (1.0 + max(0.0,  imb)), 1.0, self.max_qty)

        # 7) Refresh quotes only when needed (conserves rate budget)
        def needs_replace(prev_px: Optional[float], new_px: float) -> bool:
            if prev_px is None: return True
            eps = max(self.price_epsilon, self.tick_size * 0.5)
            return abs(prev_px - new_px) > eps

        # Bid
        if needs_replace(self._last_bid_px, target_bid_px):
            if self._bid_oid is not None:
                self._cancel(self._bid_oid); self._bid_oid = None
            oid = self._place_limit(True, bid_qty, target_bid_px, ioc=False)
            if oid is not None:
                self._bid_oid = oid
                self._last_bid_px = target_bid_px

        # Ask
        if needs_replace(self._last_ask_px, target_ask_px):
            if self._ask_oid is not None:
                self._cancel(self._ask_oid); self._ask_oid = None
            oid = self._place_limit(False, ask_qty, target_ask_px, ioc=False)
            if oid is not None:
                self._ask_oid = oid
                self._last_ask_px = target_ask_px

#14K p&L

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
    BUY = 1
    SELL = 2

class Ticker(Enum):
    TEAM_A = 1
    TEAM_B = 2

# These are stubs so the module imports cleanly;
# the actual engine provides the real implementations at runtime.
def place_market_order(side: Side, ticker: Ticker, quantity: float) -> bool:
    """Place a market order. Returns success flag."""
    return False

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

    • Maintains a complete local orderbook from snapshots + incremental updates
    • Quotes one bid + one ask inside the spread (VW top-N levels)
    • One-time sanity cross on first two-sided book to prove trading path
    • Respects rate limit: 30 orders/min (place + cancel counted)
    • Updates position/cash on fills (account updates)
    • Handles game rollover in continuous feeds (cooldown + optional flatten)
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

        # ───────── Rollover detection state ─────────
        self._last_score_sum: Optional[int] = None
        self._delta_window: deque = deque(maxlen=60)  # rolling deltas
        self._cooldown_until: float = 0.0
        self._last_event_ts: Optional[float] = None

        # Rollover knobs (tune)
        self.rollover_cooldown_sec: float = 6.0    # seconds to pause quoting
        self.min_drop: int = 20                    # absolute score drop to flag rollover
        self.frac_drop: float = 0.40               # fractional drop threshold
        self.z_sigma: float = 4.0                  # z-score threshold for unusual negative jump
        self.flatten_on_rollover: bool = True      # flatten inventory on rollover

        # Quoting params (tune safely)
        self.depth: int = 5                 # VW over top N levels
        self.inside_frac: float = 0.25      # fraction of spread toward mid
        self.base_qty: float = 5.0          # base order size
        self.max_qty: float = 20.0          # cap by imbalance scaling
        self.tick_size: float = 0.5         # price tick granularity
        self.price_epsilon: float = 0.01    # min change to replace
        self._did_sanity: bool = False
        self._sanity_qty: float = 1.0

        # Rate limit
        self._rl_capacity = 30.0
        self._rl_tokens = self._rl_capacity
        self._rl_refill_per_sec = 30.0 / 60.0
        self._rl_last_t = time.monotonic()

        # Ensure engine funcs are bound
        self._ensure_api()
        self._ensure_ticker()

    # ───────── Engine resolution helpers ─────────
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
        # refill
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
        t = self.tick_size
        return round(px / t) * t

    # ───────── Book builders ─────────
    def _rebuild_books(self, bids: list, asks: list) -> None:
        """Rebuild the full local book from snapshot arrays of [price, qty]."""
        self._bids.clear(); self._asks.clear()
        self._bid_prices.clear(); self._ask_prices.clear()

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

        # keep bests fresh
        self._best_bid = (self._bid_prices[0], self._bids[self._bid_prices[0]]) if self._bid_prices else None
        self._best_ask = (self._ask_prices[0], self._asks[self._ask_prices[0]]) if self._ask_prices else None

    # ───────── Order helpers ─────────
    def _place_market(self, want_buy: bool, qty: float) -> bool:
        self._ensure_api(); self._ensure_ticker()
        if not callable(self._fn_place_market) or self._ticker is None:
            return False
        if not self._rl_allow(1.0):
            return False
        # tolerate either actual Enum or string
        side = Side.BUY if want_buy else Side.SELL if self._Side is Side else ("BUY" if want_buy else "SELL")
        try:
            return bool(self._fn_place_market(side, self._ticker, float(qty)))
        except Exception:
            return False

    def _place_limit(self, want_buy: bool, qty: float, price: float, ioc: bool = False) -> Optional[int]:
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

    # ───────── Rollover helpers ─────────
    def _handle_rollover(self, reason: str) -> None:
        """Trigger a new-game rollover: cancel quotes, optionally flatten, clear local book, and start cooldown."""
        # Cancel resting quotes
        if getattr(self, "_bid_oid", None) is not None:
            try: self._cancel(self._bid_oid)
            except Exception: pass
            self._bid_oid = None
        if getattr(self, "_ask_oid", None) is not None:
            try: self._cancel(self._ask_oid)
            except Exception: pass
            self._ask_oid = None

        # Optional: flatten inventory via market
        try:
            if self.flatten_on_rollover and abs(self.position) > 0.0:
                qty = abs(self.position)
                if self.position > 0:
                    self._place_market(False, qty)
                else:
                    self._place_market(True, qty)
        except Exception:
            pass

        # Reset local book memory
        try:
            self._bids.clear(); self._asks.clear()
            self._bid_prices.clear(); self._ask_prices.clear()
        except Exception:
            pass
        self._best_bid = None; self._best_ask = None
        self._last_bid_px = None; self._last_ask_px = None

        # Start cooldown and clear stats
        now = time.time()
        self._cooldown_until = now + float(self.rollover_cooldown_sec)
        self._last_score_sum = None
        try: self._delta_window.clear()
        except Exception: self._delta_window = deque(maxlen=60)
        self._last_event_ts = None
        # print(f"[rollover] {reason} | cooldown={self.rollover_cooldown_sec}s")

    # ───────── Exchange callbacks ─────────
    def on_trade_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        # Keep ticker fresh
        self._ticker = ticker
        # Cooldown: ignore trading logic during rollover
        if time.time() < getattr(self, "_cooldown_until", 0.0):
            return
        # Strategy currently doesn't use last trade directly

    def on_orderbook_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        self._ticker = ticker
        self._apply_level_update(side, quantity, price)

        # Pause quoting during rollover cooldown, but keep rebuilding the book
        if time.time() < getattr(self, "_cooldown_until", 0.0):
            return

        # Compute VW prices at depth N
        def vw_best(prices: List[float], book: Dict[float, float], take_n: int, reverse: bool) -> Tuple[float, float]:
            if not prices:
                return (0.0, 0.0)
            pn = prices[:take_n] if not reverse else prices[:take_n]
            vol = 0.0; notional = 0.0
            for p in pn:
                q = book.get(p, 0.0)
                vol += q
                notional += p * q
            if vol <= 0:
                return (0.0, 0.0)
            return (notional / vol, vol)

        # Compute mid using VW bests at depth
        if self._bid_prices and self._ask_prices:
            vwbid, bid_vol = vw_best(self._bid_prices, self._bids, self.depth, reverse=True)
            vwask, ask_vol = vw_best(self._ask_prices, self._asks, self.depth, reverse=False)
            if vwbid > 0 and vwask > 0:
                mid = (vwbid + vwask) / 2.0
                spread = max(self._round_tick(vwask - vwbid), self.tick_size)
                half = self.inside_frac * spread
                target_bid_px = self._round_tick(max(vwbid, mid - half))
                target_ask_px = self._round_tick(min(vwask, mid + half))

                # Size leaning by displayed imbalance
                tot = bid_vol + ask_vol
                imb = ((bid_vol - ask_vol) / tot) if tot > 0 else 0.0      # [-1, 1]
                bid_qty = self._clamp(self.base_qty * (1.0 + max(0.0, -imb)), 1.0, self.max_qty)
                ask_qty = self._clamp(self.base_qty * (1.0 + max(0.0,  imb)), 1.0, self.max_qty)

                # Replace only when needed
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

    def on_account_update(self, *args, **kwargs) -> None:
        """
        Accepts:
          (position, cash, capital_remaining)
          (ticker, position, cash, capital_remaining)
          (ticker, position, cash, capital_remaining, realized_pnl, unrealized_pnl)
          or kwargs: position=, cash=, capital_remaining=, realized_pnl=, unrealized_pnl=
        """
        ticker = None
        position = cash = capital_remaining = None
        realized_pnl = kwargs.get("realized_pnl")
        unrealized_pnl = kwargs.get("unrealized_pnl")

        if len(args) >= 6:
            ticker, position, cash, capital_remaining, realized_pnl, unrealized_pnl = args[:6]
        elif len(args) == 5:
            ticker, position, cash, capital_remaining, realized_pnl = args[:5]
        elif len(args) == 4:
            ticker, position, cash, capital_remaining = args[:4]
        elif len(args) == 3:
            position, cash, capital_remaining = args[:3]
        else:
            position = kwargs.get("position")
            cash = kwargs.get("cash")
            capital_remaining = kwargs.get("capital_remaining")

        if ticker is not None:
            self._ticker = ticker
        if position is not None:
            self.position = float(position)
        if cash is not None:
            self.cash = float(cash)
        if capital_remaining is not None:
            self.capital_remaining = float(capital_remaining)
        if realized_pnl is not None:
            self.realized_pnl = float(realized_pnl)
        if unrealized_pnl is not None:
            self.unrealized_pnl = float(unrealized_pnl)

    def on_fill_update(self, *args, **kwargs) -> None:
        """
        Accepts:
          (order_id, side, quantity, price)
          (ticker, order_id, side, quantity, price[, ...])
          or kwargs: order_id=, side=, quantity=, price=
        """
        if len(args) >= 5:
            _, order_id, side, quantity, price = args[:5]
        elif len(args) == 4:
            order_id, side, quantity, price = args[:4]
        else:
            order_id = kwargs.get("order_id")
            side = kwargs.get("side")
            quantity = kwargs.get("quantity")
            price = kwargs.get("price")

        if order_id is None or side is None or quantity is None or price is None:
            return

        q = float(quantity); p = float(price)
        try:
            is_buy = (side == self._Side.BUY) if self._Side else (getattr(side, "name", "").upper() == "BUY")
        except Exception:
            is_buy = str(side).upper() in ("BUY", "BID")

        if is_buy:
            self.position += q
            self.cash -= q * p
        else:
            self.position -= q
            self.cash += q * p

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
                             time_seconds: Optional[float]) -> None:
        # Explicit end marker
        if event_type == "END_GAME":
            self._handle_rollover("END_GAME event")
            return

        now = time.time()

        # During cooldown: just keep baselines fresh
        if now < getattr(self, "_cooldown_until", 0.0):
            self._last_score_sum = int(home_score) + int(away_score)
            self._last_event_ts = now
            return

        curr_sum = int(home_score) + int(away_score)

        # Score-drop rollover detector (robust to nonzero resets)
        if self._last_score_sum is not None:
            delta = curr_sum - self._last_score_sum  # negative on drop
            neg_jump = delta < 0

            # rolling stats
            if len(self._delta_window) >= 10:
                mean_d = sum(self._delta_window) / len(self._delta_window)
                var_d = sum((d - mean_d) ** 2 for d in self._delta_window) / max(1, len(self._delta_window) - 1)
                std_d = max(1e-6, var_d ** 0.5)
            else:
                mean_d, std_d = 0.0, 1.0

            drop_abs = -delta  # positive when it’s a drop
            drop_frac = drop_abs / max(1.0, float(self._last_score_sum))
            z = (delta - mean_d) / std_d  # negative when unusually large drop

            if neg_jump and (drop_abs >= self.min_drop or drop_frac >= self.frac_drop or z <= -self.z_sigma):
                self._handle_rollover(f"score negative jump Δ={delta:.0f}, frac={drop_frac:.2f}, z={z:.1f}")
                # seed new baseline from this first event of the next game
                self._last_score_sum = curr_sum
                self._last_event_ts = now
                return

            # no trigger → update rolling window
            self._delta_window.append(delta)
        else:
            # first observation: seed baseline, neutral delta
            self._delta_window.append(0.0)

        self._last_score_sum = curr_sum
        self._last_event_ts = now

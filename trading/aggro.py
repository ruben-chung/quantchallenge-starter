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
        self.tick_size: float = 0.5         # price tick granularity
        self.price_epsilon: float = 0.01    # min change to replace
        self._did_sanity: bool = False
        self._sanity_qty: float = 1.0

        # Rate limit
        self._rl_capacity = 30.0
        self._rl_tokens = self._rl_capacity
        self._rl_refill_per_sec = 30.0 / 60.0
        self._rl_last_t = time.monotonic()

        # ───────── Enhancements: risk/vol/cooldown/health ─────────
        # inventory skew & churn guard
        self.kappa: float = 0.02              # inventory skew per unit position (price units)
        self.min_order_life: float = 0.80     # seconds before we allow a replace on the same side
        self._bid_ts: Optional[float] = None
        self._ask_ts: Optional[float] = None

        # volatility-aware quoting
        self.c_vol: float = 0.50              # widen factor versus vol
        self.vol_ref: float = 0.002           # reference vol (fractional mid move)
        self.vol_window: deque = deque(maxlen=60)
        self._last_mid: Optional[float] = None

        # dynamic cooldown exit based on book health
        self.dynamic_cooldown: bool = True
        self.health_spread_max: float = 3.0    # don’t quote if spread wider than this
        self.health_min_depth: float = 20.0    # min combined top-N depth to consider healthy
        self._good_book_streak: int = 0

        # risk sizing knobs
        self.contract_value: float = 1.0     # $ P&L per 1 price-unit per 1 qty
        self.fill_risk_bp: float = 2.0       # per-fill risk budget (bp of capital)
        self.max_inv_frac: float = 0.02      # max net inventory notional as fraction of capital
        self.depth_frac_cap: float = 0.25    # cap size as fraction of displayed depth

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

    # ───────── Book helpers ─────────
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
        if self._Side:
            side = self._Side.BUY if want_buy else self._Side.SELL
        else:
            side = Side.BUY if want_buy else Side.SELL
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
        if self._Side:
            side = self._Side.BUY if want_buy else self._Side.SELL
        else:
            side = Side.BUY if want_buy else Side.SELL
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

    # ───────── Health/vol helpers ─────────
    def _book_health(self) -> bool:
        """Simple health check: spread tight enough and top-N depth sufficient."""
        if not (self._best_bid and self._best_ask):
            return False
        best_spread = self._best_ask[0] - self._best_bid[0]
        if best_spread > self.health_spread_max:
            return False
        take = max(1, int(self.depth))
        bid_depth = sum(self._bids.get(p, 0.0) for p in self._bid_prices[:take])
        ask_depth = sum(self._asks.get(p, 0.0) for p in self._ask_prices[:take])
        return (bid_depth + ask_depth) >= self.health_min_depth

    def _update_vol(self, mid: float) -> float:
        """Update and return rolling absolute return volatility (fractional)."""
        if self._last_mid is not None and self._last_mid > 0:
            ret = abs(mid / self._last_mid - 1.0)
            self.vol_window.append(ret)
        self._last_mid = mid
        if not self.vol_window:
            return 0.0
        sorted_rets = sorted(self.vol_window)
        m = sorted_rets[len(sorted_rets)//2]
        return m * 1.4826  # ~robust std from MAD for Laplace-ish tails

    # ───────── Risk sizing helper ─────────
    def _safe_qty(self, mid: float, spread: float, vol_frac: float,
                  side_is_bid: bool, displayed_depth: float) -> float:
        """
        Return a per-order quantity bounded by:
          • per-fill risk budget (bp of capital)
          • max inventory vs capital
          • fraction of displayed depth
        """
        # estimate capital if not provided by engine yet
        est_cap = self.capital_remaining
        if est_cap is None:
            est_cap = self.cash + abs(self.position) * max(mid, 1.0) * self.contract_value
        cap = max(float(est_cap), 1.0)

        # worst-case unit loss for a taker hit shortly after you quote
        worst_move = 0.5 * spread + max(self.tick_size, 0.0) + 2.0 * vol_frac * max(mid, 1.0)
        risk_per_unit = self.contract_value * max(worst_move, 1e-6)

        # per-fill risk budget in $
        risk_budget = (self.fill_risk_bp * 1e-4) * cap
        q_risk = risk_budget / risk_per_unit

        # inventory cap in qty: max notional = max_inv_frac * cap
        max_inv_notional = self.max_inv_frac * cap
        q_inv_cap = max_inv_notional / max(self.contract_value * max(mid, 1.0), 1e-6)

        # depth cap
        q_depth_cap = self.depth_frac_cap * max(displayed_depth, 0.0)

        q = min(q_risk, q_inv_cap, q_depth_cap, float('inf'))
        q = min(q, self._clamp(q, 0.0, self.max_qty))
        return max(1.0, q)  # ensure at least 1 unit if allowed

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
        self._bid_ts = None; self._ask_ts = None

        # Start cooldown and clear stats
        now = time.time()
        self._cooldown_until = now + float(self.rollover_cooldown_sec)
        self._last_score_sum = None
        try: self._delta_window.clear()
        except Exception: self._delta_window = deque(maxlen=60)
        self._last_event_ts = None
        self._good_book_streak = 0
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

        # Pause quoting during cooldown, but allow early exit if book looks healthy
        if time.time() < getattr(self, "_cooldown_until", 0.0):
            if self.dynamic_cooldown and self._book_health():
                self._good_book_streak += 1
                if self._good_book_streak >= 3:   # 3 consecutive healthy snapshots
                    self._cooldown_until = 0.0
                    self._good_book_streak = 0
            else:
                self._good_book_streak = 0
            return

        # Compute VW prices at depth N
        def vw_best(prices: List[float], book: Dict[float, float], take_n: int, reverse: bool) -> Tuple[float, float]:
            if not prices:
                return (0.0, 0.0)
            pn = prices[:take_n]
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
                raw_mid = (vwbid + vwask) / 2.0
                vol = self._update_vol(raw_mid)

                # widen in high vol, tighten in low vol
                widen = self.c_vol * min(1.0, vol / max(1e-9, self.vol_ref))
                inside_frac_eff = max(0.10, min(0.45, self.inside_frac + widen))

                spread = max(self._round_tick(vwask - vwbid), self.tick_size)
                half = inside_frac_eff * spread

                # inventory skew: push prices away from current position
                mid = raw_mid - self.kappa * self.position

                target_bid_px = self._round_tick(max(vwbid, mid - half))
                target_ask_px = self._round_tick(min(vwask, mid + half))

                # don’t quote if unhealthy book
                if not self._book_health():
                    return

                # displayed depth for sizing
                take = max(1, int(self.depth))
                displayed_bid_depth = sum(self._bids.get(p, 0.0) for p in self._bid_prices[:take])
                displayed_ask_depth = sum(self._asks.get(p, 0.0) for p in self._ask_prices[:take])

                # baseline size from risk/depth/capital
                base_bid = self._safe_qty(raw_mid, spread, vol, True,  displayed_bid_depth)
                base_ask = self._safe_qty(raw_mid, spread, vol, False, displayed_ask_depth)

                # lean size by imbalance, but never exceed safe cap computed above
                tot = bid_vol + ask_vol
                imb = ((bid_vol - ask_vol) / tot) if tot > 0 else 0.0
                tilt = 0.50
                bid_qty = self._clamp(base_bid * (1.0 + tilt * max(0.0, -imb)), 1.0, base_bid)
                ask_qty = self._clamp(base_ask * (1.0 + tilt * max(0.0,  imb)), 1.0, base_ask)

                # Replace only when needed and after min order life
                now = time.time()
                def needs_replace(prev_px: Optional[float], prev_ts: Optional[float], new_px: float) -> bool:
                    if prev_px is None:
                        return True
                    if prev_ts is not None and (now - prev_ts) < self.min_order_life:
                        return False
                    eps = max(self.price_epsilon, self.tick_size * 0.5)
                    return abs(prev_px - new_px) > eps

                # Bid
                if needs_replace(self._last_bid_px, self._bid_ts, target_bid_px):
                    if self._bid_oid is not None:
                        self._cancel(self._bid_oid); self._bid_oid = None
                    oid = self._place_limit(True, bid_qty, target_bid_px, ioc=False)
                    if oid is not None:
                        self._bid_oid = oid
                        self._last_bid_px = target_bid_px
                        self._bid_ts = now

                # Ask
                if needs_replace(self._last_ask_px, self._ask_ts, target_ask_px):
                    if self._ask_oid is not None:
                        self._cancel(self._ask_oid); self._ask_oid = None
                    oid = self._place_limit(False, ask_qty, target_ask_px, ioc=False)
                    if oid is not None:
                        self._ask_oid = oid
                        self._last_ask_px = target_ask_px
                        self._ask_ts = now

    def on_orderbook_snapshot(self, ticker: Ticker, bids: list, asks: list) -> None:
        """Optional full snapshot handler: rebuild then reuse the same quoting logic via updates."""
        self._ticker = ticker
        self._rebuild_books(bids, asks)

        # during cooldown, let dynamic exit logic happen in incremental updates
        if time.time() < getattr(self, "_cooldown_until", 0.0):
            return

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

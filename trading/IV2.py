"""
Quant Challenge 2025

Algorithmic strategy – VW spread maker with full local orderbook and 30/min rate limit
Volatility-harvesting + mean-revert exit + IV-based regime switching + adaptive MOMO skew
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

# Stubs so the module imports cleanly; engine provides real implementations.
def place_market_order(side: Side, ticker: Ticker, quantity: float) -> bool:
    return False

def place_limit_order(side: Side, ticker: Ticker, quantity: float, price: float, ioc: bool = False) -> int:
    return 0

def cancel_order(ticker: Ticker, order_id: int) -> bool:
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Strategy
# ─────────────────────────────────────────────────────────────────────────────

class Strategy:
    """
    Simple volume-weighted spread maker.

    • Maintains a complete local orderbook from snapshots + incremental updates
    • Quotes one bid + one ask inside the spread (VW top-N levels)
    • Enforces 30 orders/min with a leaky bucket rate limiter
    • Handles game rollovers (score resets) with cooldown + optional flatten
    • Tracks fills to maintain position and cash

    Volatility harvesting
    • Adaptive half-spread = inside_frac*spread + k_vol*std(diff(mid))
    • Mean-reversion lean (z-score of mid vs rolling mean)
    • Inventory skew to keep book neutral
    • Volatility-based size dampening

    Mean-revert exit
    • When price snaps back toward entry by a configurable edge, send IOC to reduce/flatten.

    IV-based regime switching
    • Use std(diff(mid)) as a practical intraday IV proxy. Switch between:
      - VOL: the volatility-harvesting mode above
      - MOMO: tighter quoting with adaptive momentum skew in low IV
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

        # Track average entry price for current net position (abs)
        self.pos_avg_price: float = 0.0

        # Rollover detection state
        self._last_score_sum: Optional[int] = None
        self._delta_window: deque = deque(maxlen=60)  # rolling deltas
        self._cooldown_until: float = 0.0
        self._last_event_ts: Optional[float] = None

        # Rollover knobs (tune)
        self.rollover_cooldown_sec: float = 4.0    # seconds to pause quoting
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

        # Volatility-harvesting controls
        self.vol_window: int = 60            # number of mid observations for rolling stats
        self.k_vol: float = 1.25             # half-spread add-on = k_vol * std(diff(mid))
        self.min_inside_ticks: float = 1.0   # floor for half-spread in ticks
        self.max_inside_ticks: float = 8.0   # cap for half-spread in ticks
        self.z_lean: float = 0.35            # lean against deviations (z-score of mid vs mean)
        self.inv_skew_per_unit: float = 0.05 # additional skew per 1 unit inventory (in ticks)
        self.qty_vol_dampen: float = 0.5     # damp sizes when vol is high (0=no damp, 1=strong)

        # Mean-revert exit controls
        self.mr_sigma_take: float = 0.5      # trigger when move vs entry >= 0.5 * std(diff(mid))
        self.mr_min_edge_ticks: float = 0.5  # at least this many ticks of edge
        self.mr_clip: float = 10.0           # max qty to reduce per check
        self.mr_use_ioc: bool = True         # IOC so we don't rest during exit

        # IV-based regime controls
        self.iv_high_thresh_ticks: float = 2.0   # high IV if _diff_std/tick_size >= 2
        self.iv_low_thresh_ticks: float  = 1.0   # low IV if <= 1 (hysteresis band 1..2)
        self.regime: str = "VOL"                 # "VOL" or "MOMO"
        self.min_regime_hold_sec: float = 20.0
        self._last_regime_switch: float = 0.0

        # MOMO mode controls (for low IV)
        self.momo_half_ticks: float = 1.5        # tighter quoting in low IV
        self.momo_skew: float = 0.20             # (legacy; not used with adaptive, retained for compat)
        self.momo_pos_cap: float = 25.0          # stricter inventory cap in low IV

        # MOMO adaptive momentum (new)
        self.momo_mom_window: int = 8            # lookback (# mids → #diffs = window-1)
        self.momo_mom_decay: float = 0.7         # EWMA decay for recent diffs
        self.momo_skew_gain: float = 0.35        # ticks of skew per 1.0 unit signal
        self.momo_skew_max_ticks: float = 2.0    # cap skew magnitude (ticks)

        # Rolling mid stats
        self._mid_hist = deque(maxlen=self.vol_window)
        self._mid_diff_hist = deque(maxlen=self.vol_window)
        self._mid_mean: Optional[float] = None
        self._mid_std: float = 0.0
        self._diff_std: float = 0.0

        self._did_sanity: bool = False
        self._sanity_qty: float = 1.0

        # Rate limit (30 orders/min)
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
        self._rl_tokens = min(self._rl_capacity, self._rl_tokens + dt * self._rl_refill_per_sec)
        if self._rl_tokens >= cost:
            self._rl_tokens -= cost
            return True
        return False

    # ───────── Round helpers ─────────
    def _round_tick(self, px: float) -> float:
        if self.tick_size <= 0:
            return float(px)
        k = round(px / self.tick_size)
        return float(k * self.tick_size)

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

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
        """Apply incremental level update to the local book."""
        try:
            p = float(price); q = float(quantity)
        except Exception:
            return
        try:
            is_bid = (side == Side.BUY) if isinstance(side, Side) else (getattr(side, "name", "").upper() == "BUY")
        except Exception:
            is_bid = str(side).upper() in ("BUY", "BID")

        book = self._bids if is_bid else self._asks
        prices = self._bid_prices if is_bid else self._ask_prices

        if q <= 0:
            if p in book:
                del book[p]
                try:
                    prices.remove(p)
                except ValueError:
                    pass
        else:
            new = p not in book
            book[p] = q
            if new:
                prices.append(p)
                prices.sort(reverse=is_bid is True)

        self._best_bid = (self._bid_prices[0], self._bids[self._bid_prices[0]]) if self._bid_prices else None
        self._best_ask = (self._ask_prices[0], self._asks[self._ask_prices[0]]) if self._ask_prices else None

    # ───────── Order helpers ─────────
    def _place_limit(self, want_buy: bool, qty: float, price: float, ioc: bool = False) -> Optional[int]:
        self._ensure_api(); self._ensure_ticker()
        if not callable(self._fn_place_limit) or self._ticker is None:
            return None
        if not self._rl_allow(1.0):
            return None
        if self._Side and hasattr(self._Side, "BUY"):
            side = self._Side.BUY if want_buy else self._Side.SELL
        else:
            side = "BUY" if want_buy else "SELL"
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
        """Trigger a new-game rollover: cancel quotes, optionally flatten, clear local book, start cooldown."""
        # Cancel resting quotes
        if self._bid_oid is not None:
            try: self._cancel(self._bid_oid)
            except Exception: pass
            self._bid_oid = None; self._last_bid_px = None
        if self._ask_oid is not None:
            try: self._cancel(self._ask_oid)
            except Exception: pass
            self._ask_oid = None; self._last_ask_px = None

        # Flatten inventory if configured
        if self.flatten_on_rollover and abs(self.position) > 0 and callable(self._fn_place_market):
            want_buy = self.position < 0
            try:
                if self._Side and self._ticker:
                    self._fn_place_market(self._Side.BUY if want_buy else self._Side.SELL,
                                          self._ticker, abs(self.position))
                    self.position = 0.0
                    self.pos_avg_price = 0.0
            except Exception:
                pass

        # Clear local book and set cooldown
        self._bids.clear(); self._asks.clear()
        self._bid_prices.clear(); self._ask_prices.clear()
        self._best_bid = None; self._best_ask = None
        self._cooldown_until = time.time() + self.rollover_cooldown_sec

    # ───────── Volatility stats ─────────
    def _update_mid_stats(self, mid: float) -> None:
        if mid is None:
            return
        if self._mid_hist and self._mid_hist[-1] is not None:
            last_mid = self._mid_hist[-1]
            self._mid_diff_hist.append(float(mid) - float(last_mid))
        self._mid_hist.append(float(mid))

        n = len(self._mid_hist)
        if n >= 2:
            m = sum(self._mid_hist) / n
            var = sum((x - m)**2 for x in self._mid_hist) / max(1, n - 1)
            self._mid_mean = m
            self._mid_std = var ** 0.5
        else:
            self._mid_mean = mid
            self._mid_std = 0.0

        nd = len(self._mid_diff_hist)
        if nd >= 2:
            md = sum(self._mid_diff_hist) / nd
            var_d = sum((x - md)**2 for x in self._mid_diff_hist) / max(1, nd - 1)
            self._diff_std = var_d ** 0.5
        elif nd == 1:
            self._diff_std = abs(self._mid_diff_hist[0])
        else:
            self._diff_std = 0.0

    # ───────── IV regime helper ─────────
    def _update_regime(self) -> None:
        now = time.time()
        if now - self._last_regime_switch < self.min_regime_hold_sec:
            return
        vol_in_ticks = (self._diff_std or 0.0) / max(self.tick_size, 1e-9)
        if self.regime != "VOL" and vol_in_ticks >= self.iv_high_thresh_ticks:
            self.regime = "VOL"; self._last_regime_switch = now
        elif self.regime != "MOMO" and vol_in_ticks <= self.iv_low_thresh_ticks:
            self.regime = "MOMO"; self._last_regime_switch = now

    # ───────── Adaptive MOMO momentum signal ─────────
    def _momo_trend_signal(self) -> float:
        """
        Adaptive momentum signal for low-IV regime.
        • EWMA of recent mid diffs (window, decay)
        • Normalized by a scale (use _diff_std or mean |diff|) → unitless
        • Returns a clipped signal in roughly [-3, +3]
        """
        n_hist = len(self._mid_hist)
        if n_hist < 2:
            return 0.0

        # Build recent diffs (up to window)
        k = min(self.momo_mom_window + 1, n_hist)
        diffs: List[float] = []
        for i in range(1, k):
            diffs.append(self._mid_hist[-i] - self._mid_hist[-i-1])
        if not diffs:
            return 0.0

        # EWMA (recent moves weigh more)
        decay = float(self.momo_mom_decay)
        w = 1.0
        num = 0.0
        den = 0.0
        for d in diffs:
            num += w * d
            den += w
            w *= decay
        ewma = (num / den) if den > 0 else 0.0

        # Scale to unitless
        if self._diff_std and self._diff_std > 0:
            scale = self._diff_std
        else:
            avg_abs = sum(abs(d) for d in diffs) / max(1, len(diffs))
            scale = max(self.tick_size, avg_abs)
        sig = ewma / max(scale, 1e-9)

        # Clip to avoid extreme skews
        return self._clamp(sig, -3.0, 3.0)

    # ───────── Mean-revert exit helper ─────────
    def _maybe_mean_revert_exit(self, mid: float, vwbid: float, vwask: float) -> None:
        if self.position == 0:
            return
        sigma_diff = float(self._diff_std or 0.0)
        edge_abs = max(self.mr_sigma_take * sigma_diff, self.mr_min_edge_ticks * self.tick_size)

        qty = min(abs(self.position), self.mr_clip)
        if qty <= 0:
            return

        # Long → if mid moved up from avg by edge, sell IOC near bid
        if self.position > 0 and (mid - self.pos_avg_price) >= edge_abs:
            px = self._round_tick(max(vwbid, mid - self.tick_size))
            self._place_limit(False, qty, px, ioc=self.mr_use_ioc)
        # Short → if mid moved down from avg by edge, buy IOC near ask
        elif self.position < 0 and (self.pos_avg_price - mid) >= edge_abs:
            px = self._round_tick(min(vwask, mid + self.tick_size))
            self._place_limit(True, qty, px, ioc=self.mr_use_ioc)

    # ───────── Public engine hooks ─────────
    def on_orderbook_snapshot(self, ticker: Ticker, bids: list, asks: list) -> None:
        self._ticker = ticker
        self._rebuild_books(bids, asks)

    def on_orderbook_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        self._ticker = ticker
        self._apply_level_update(side, quantity, price)

        # Pause quoting during rollover cooldown, but keep rebuilding the book
        if time.time() < self._cooldown_until:
            return

        # Compute VW prices at depth N
        def vw_best(prices: List[float], book: Dict[float, float], take_n: int) -> Tuple[float, float]:
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

        if self._bid_prices and self._ask_prices:
            vwbid, bid_vol = vw_best(self._bid_prices, self._bids, self.depth)
            vwask, ask_vol = vw_best(self._ask_prices, self._asks, self.depth)
            if vwbid > 0 and vwask > 0:
                mid = (vwbid + vwask) / 2.0
                spread = max(self._round_tick(vwask - vwbid), self.tick_size)

                # Volatility stats and regime
                self._update_mid_stats(mid)
                self._update_regime()

                # Compute half-spread, center skew, size scale per regime
                if self.regime == "VOL":
                    sigma_diff = float(self._diff_std or 0.0)
                    base_half = self.inside_frac * spread
                    vol_add = self.k_vol * sigma_diff
                    half = self._round_tick(
                        self._clamp(base_half + vol_add,
                                    self.min_inside_ticks * self.tick_size,
                                    self.max_inside_ticks * self.tick_size)
                    )

                    z = 0.0
                    if (self._mid_mean is not None) and (self._mid_std and self._mid_std > 1e-9):
                        z = (mid - float(self._mid_mean)) / float(self._mid_std)
                    center_skew_ticks = (- self.z_lean * z) + (- self.inv_skew_per_unit * self.position)

                    sigma = max(0.0, sigma_diff)
                    size_scale = 1.0 / (1.0 + self.qty_vol_dampen * sigma / max(self.tick_size, 1e-6))

                else:  # MOMO (low IV)
                    half = self._round_tick(self.momo_half_ticks * self.tick_size)

                    # Adaptive momentum skew (unitless signal ~[-3,+3] → ticks)
                    sig = self._momo_trend_signal()
                    mom_skew_ticks = self._clamp(self.momo_skew_gain * sig,
                                                 -self.momo_skew_max_ticks,
                                                 self.momo_skew_max_ticks)

                    # Inventory anti-skew stays as is
                    center_skew_ticks = mom_skew_ticks + (- self.inv_skew_per_unit * self.position)

                    # Size policy in low IV (steady; shrink if over cap)
                    size_scale = 1.0
                    if abs(self.position) > self.momo_pos_cap:
                        size_scale = 0.5  # gently bias toward flattening when over cap

                center_skew = center_skew_ticks * self.tick_size

                # Targets with skew (ensure we don't cross best levels)
                target_bid_px = self._round_tick(max(vwbid, (mid + center_skew) - half))
                target_ask_px = self._round_tick(min(vwask, (mid + center_skew) + half))

                # Size leaning by displayed imbalance + regime scale
                tot = bid_vol + ask_vol
                imb = ((bid_vol - ask_vol) / tot) if tot > 0 else 0.0      # [-1, 1]
                bid_qty = self._clamp(self.base_qty * (1.0 + max(0.0, -imb)), 1.0, self.max_qty)
                ask_qty = self._clamp(self.base_qty * (1.0 + max(0.0,  imb)), 1.0, self.max_qty)
                bid_qty = max(self._sanity_qty, bid_qty * size_scale)
                ask_qty = max(self._sanity_qty, ask_qty * size_scale)

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

                # Opportunistic mean-revert exit (IOC, does not disturb resting quotes)
                self._maybe_mean_revert_exit(mid, vwbid, vwask)

    def on_rejected(self, *args, **kwargs) -> None:
        # Could adaptively widen on rejects
        pass

    # Flexible callbacks to match engines that pass many positional args
    def on_game_event_update(self, *args, **kwargs) -> None:
        """
        Compatibility handler: accepts any arg list or dict payload.
        We only need a timestamp for cooldown book-keeping; score-based rollover
        is handled here only if scores are present.
        """
        try:
            now = kwargs.get("timestamp") or kwargs.get("ts") or kwargs.get("time")
            if now is None:
                nums = [a for a in args if isinstance(a, (int, float))]
                if nums:
                    now = float(nums[-1])
            self._last_event_ts = float(now) if now is not None else time.time()
        except Exception:
            self._last_event_ts = time.time()

        home_score = kwargs.get("home_score")
        away_score = kwargs.get("away_score")
        try:
            if home_score is not None and away_score is not None:
                curr_sum = int(home_score) + int(away_score)
                if self._last_score_sum is not None:
                    delta = curr_sum - int(self._last_score_sum)
                    if len(self._delta_window) >= 20:
                        mu = sum(self._delta_window) / len(self._delta_window)
                        var = sum((x - mu)**2 for x in self._delta_window) / max(1, len(self._delta_window) - 1)
                        sd = var ** 0.5
                        z = (delta - mu) / max(sd, 1e-6)
                    else:
                        z = 0.0

                    abs_drop = (delta <= -self.min_drop)
                    frac_drop = (delta <= -self.frac_drop * max(1, self._last_score_sum)) if self._last_score_sum > 0 else False
                    z_trigger = (z <= -self.z_sigma)

                    if abs_drop or frac_drop or z_trigger:
                        self._handle_rollover("score_reset")
                        self._last_score_sum = curr_sum
                        self._delta_window.clear()
                        return

                    self._delta_window.append(delta)
                else:
                    self._delta_window.append(0.0)
                self._last_score_sum = curr_sum
        except Exception:
            pass

    def on_account_update(self, *args, **kwargs) -> None:
        """
        Engine callback: updates cash/capital/position safely whether payload arrives
        as a dict (args[0]) or kwargs. Ignores missing fields.
        """
        payload = args[0] if args and isinstance(args[0], dict) else (kwargs or {})
        try:
            if payload.get("cash") is not None:
                self.cash = float(payload["cash"])
        except Exception:
            pass
        try:
            if payload.get("capital_remaining") is not None:
                self.capital_remaining = float(payload["capital_remaining"])
        except Exception:
            pass
        try:
            if payload.get("position") is not None:
                self.position = float(payload["position"])
        except Exception:
            pass

    def on_trade_update(self, *args, **kwargs) -> None:
        return

    def on_order_update(self, *args, **kwargs) -> None:
        return

    def on_order_filled(self, *args, **kwargs) -> None:
        # Extract fill fields (support dict or kwargs)
        if len(args) >= 1 and isinstance(args[0], dict):
            d = args[0]
            side = d.get("side"); quantity = d.get("quantity"); price = d.get("price")
        else:
            side = kwargs.get("side")
            quantity = kwargs.get("quantity")
            price = kwargs.get("price")

        if side is None or quantity is None or price is None:
            return

        try:
            is_buy = (side == self._Side.BUY) if self._Side else (getattr(side, "name", "").upper() == "BUY")
        except Exception:
            is_buy = str(side).upper() in ("BUY", "BID")

        q = float(quantity); p = float(price)

        prev_pos = self.position
        if is_buy:
            new_pos = prev_pos + q
        else:
            new_pos = prev_pos - q

        # pos_avg_price maintenance
        if prev_pos == 0.0:
            self.pos_avg_price = p
        elif (prev_pos > 0 and new_pos > prev_pos) or (prev_pos < 0 and new_pos < prev_pos):
            abs_prev = abs(prev_pos)
            abs_new = abs(new_pos)
            if abs_new > 1e-12:
                self.pos_avg_price = (self.pos_avg_price * abs_prev + p * abs(abs_new - abs_prev)) / abs_new
        elif (prev_pos > 0 and new_pos < 0) or (prev_pos < 0 and new_pos > 0):
            self.pos_avg_price = p
        # else: reducing without crossing zero → keep avg

        # Commit position/cash
        if is_buy:
            self.position = new_pos
            self.cash -= q * p
        else:
            self.position = new_pos
            self.cash += q * p

        if self.position == 0.0:
            self.pos_avg_price = 0.0

    def on_pnl_update(self, *args, **kwargs) -> None:
        # Optional: some runners send this instead of on_account_update
        payload = args[0] if args and isinstance(args[0], dict) else (kwargs or {})
        for key in ("position", "cash", "capital_remaining"):
            try:
                if payload.get(key) is not None:
                    setattr(self, key, float(payload[key]))
            except Exception:
                pass


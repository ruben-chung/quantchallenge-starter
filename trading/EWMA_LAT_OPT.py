"""
Quant Challenge 2025 - Latency Optimized

Key optimizations:
- Pre-allocated arrays and circular buffers
- Cached calculations and minimal recomputation
- Reduced function calls and attribute lookups
- Optimized data structures and algorithms
- Eliminated unnecessary operations in hot paths
"""

from __future__ import annotations
import time
from enum import Enum
from typing import Optional, Dict, List, Tuple
import array

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
# Optimized Strategy
# ─────────────────────────────────────────────────────────────────────────────

class Strategy:
    """
    Latency-optimized volume-weighted spread maker.
    
    Key optimizations:
    - Pre-allocated circular buffers using arrays
    - Cached calculations to avoid recomputation
    - Minimal attribute lookups and function calls
    - Optimized hot path in on_orderbook_update
    - Reduced memory allocations
    """

    # Pre-compute constants
    _HALF_PI = 1.5707963267948966
    _SQRT_2PI = 2.5066282746310005
    _EPS = 1e-9

    def __init__(self) -> None:
        self.reset_state()

    def reset_state(self) -> None:
        # Cache API references to avoid repeated lookups
        self._Side = None
        self._Ticker = None
        self._fn_place_market = None
        self._fn_place_limit = None
        self._fn_cancel = None

        # Identity
        self._ticker: Optional[Ticker] = None

        # Orderbook - use sorted lists with binary search for better cache locality
        self._bids: Dict[float, float] = {}
        self._asks: Dict[float, float] = {}
        self._bid_prices: List[float] = []
        self._ask_prices: List[float] = []
        self._best_bid: Optional[Tuple[float, float]] = None
        self._best_ask: Optional[Tuple[float, float]] = None

        # Cached VW calculations to avoid recomputation
        self._cached_vwbid = 0.0
        self._cached_vwask = 0.0
        self._cached_mid = 0.0
        self._cached_spread = 0.0
        self._book_dirty = True

        # Our orders
        self._bid_oid: Optional[int] = None
        self._ask_oid: Optional[int] = None
        self._last_bid_px: Optional[float] = None
        self._last_ask_px: Optional[float] = None

        # Position tracking
        self.position: float = 0.0
        self.cash: float = 0.0
        self.capital_remaining: Optional[float] = None
        self.pos_avg_price: float = 0.0

        # Rollover detection - use arrays for better performance
        self._last_score_sum: Optional[int] = None
        self._delta_array = array.array('f', [0.0] * 60)  # Pre-allocated
        self._delta_idx = 0
        self._delta_count = 0
        self._cooldown_until: float = 0.0
        self._last_event_ts: Optional[float] = None

        # Configuration
        self.rollover_cooldown_sec: float = 4.0
        self.min_drop: int = 20
        self.frac_drop: float = 0.40
        self.z_sigma: float = 4.0
        self.flatten_on_rollover: bool = True

        # Quoting params
        self.depth: int = 5
        self.inside_frac: float = 0.25
        self.base_qty: float = 5.0
        self.max_qty: float = 20.0
        self.tick_size: float = 0.5
        self.price_epsilon: float = 0.01

        # Pre-compute 1/tick_size for faster division
        self._tick_inv = 1.0 / self.tick_size if self.tick_size > 0 else 1.0

        # Volatility params
        self.vol_window: int = 60
        self.k_vol: float = 1.25
        self.min_inside_ticks: float = 1.0
        self.max_inside_ticks: float = 8.0
        self.z_lean: float = 0.35
        self.inv_skew_per_unit: float = 0.05
        self.qty_vol_dampen: float = 0.5

        # Mean-revert exit
        self.mr_sigma_take: float = 0.5
        self.mr_min_edge_ticks: float = 0.5
        self.mr_clip: float = 10.0
        self.mr_use_ioc: bool = True

        # IV regime
        self.iv_high_thresh_ticks: float = 2.0
        self.iv_low_thresh_ticks: float = 1.0
        self.regime: str = "VOL"
        self.min_regime_hold_sec: float = 20.0
        self._last_regime_switch: float = 0.0

        # MOMO params
        self.momo_half_ticks: float = 1.5
        self.momo_skew: float = 0.20
        self.momo_pos_cap: float = 25.0
        self.momo_mom_window: int = 8
        self.momo_mom_decay: float = 0.7
        self.momo_skew_gain: float = 0.35
        self.momo_skew_max_ticks: float = 2.0

        # Markout tracking - use circular buffer
        self._markout_ewma: float = 0.0
        self._markout_alpha: float = 0.2
        self._fills_buffer = [(0.0, 0, 0.0) for _ in range(200)]  # Pre-allocated
        self._fills_head = 0
        self._fills_size = 0
        self.markout_horizon_sec: float = 2.0
        self.markout_widen_k: float = 0.5

        # Exit budget
        self._exit_capacity = 10.0
        self._exit_tokens = self._exit_capacity
        self._exit_refill_per_sec = self._exit_capacity / 60.0
        self._exit_last_t = time.monotonic()

        # Dynamic VAR
        self.base_pos_cap: float = 60.0
        self.min_pos_cap: float = 10.0
        self.varcap_flatten_bias: float = 0.5

        # Pre-allocated arrays for rolling stats
        self._mid_array = array.array('f', [0.0] * self.vol_window)
        self._diff_array = array.array('f', [0.0] * self.vol_window)
        self._mid_idx = 0
        self._diff_idx = 0
        self._mid_count = 0
        self._diff_count = 0
        
        # Cached stats to avoid recomputation
        self._mid_mean: float = 0.0
        self._mid_std: float = 0.0
        self._diff_std: float = 0.0
        self._stats_dirty = True

        self._did_sanity: bool = False
        self._sanity_qty: float = 1.0

        # Rate limiter
        self._rl_capacity = 30.0
        self._rl_tokens = self._rl_capacity
        self._rl_refill_per_sec = 30.0 / 60.0
        self._rl_last_t = time.monotonic()

        # Pre-compute regime-specific constants
        self._vol_min_half = self.min_inside_ticks * self.tick_size
        self._vol_max_half = self.max_inside_ticks * self.tick_size
        self._momo_fixed_half = self.momo_half_ticks * self.tick_size

        self._ensure_api()
        self._ensure_ticker()

    def _ensure_api(self) -> None:
        """Cache API references once to avoid repeated globals lookups"""
        if self._Side is None:
            g = globals()
            self._Side = g.get("Side", None)
            self._Ticker = g.get("Ticker", None)
            self._fn_place_market = g.get("place_market_order", None)
            self._fn_place_limit = g.get("place_limit_order", None)
            self._fn_cancel = g.get("cancel_order", None)

    def _ensure_ticker(self) -> None:
        if self._ticker is None:
            self._ensure_api()
            if self._Ticker is not None and hasattr(self._Ticker, "TEAM_A"):
                self._ticker = getattr(self._Ticker, "TEAM_A")

    # Optimized rate limiter with fewer operations
    def _rl_allow(self, cost: float = 1.0) -> bool:
        now = time.monotonic()
        dt = now - self._rl_last_t
        self._rl_last_t = now
        
        if dt > 0:
            self._rl_tokens = min(self._rl_capacity, self._rl_tokens + dt * self._rl_refill_per_sec)
        
        if self._rl_tokens >= cost:
            self._rl_tokens -= cost
            return True
        return False

    def _rl_allow_exit(self, cost: float = 1.0) -> bool:
        now = time.monotonic()
        dt = now - self._exit_last_t
        self._exit_last_t = now
        
        if dt > 0:
            self._exit_tokens = min(self._exit_capacity, self._exit_tokens + dt * self._exit_refill_per_sec)
        
        if self._exit_tokens >= cost:
            self._exit_tokens -= cost
            return True
        return False

    # Optimized rounding using pre-computed inverse
    def _round_tick(self, px: float) -> float:
        if self.tick_size <= 0:
            return px
        return round(px * self._tick_inv) * self.tick_size

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return lo if x < lo else hi if x > hi else x

    def _rebuild_books(self, bids: list, asks: list) -> None:
        """Optimized book rebuild"""
        # Clear existing data
        self._bids.clear()
        self._asks.clear()
        self._bid_prices.clear()
        self._ask_prices.clear()

        # Process bids
        for p, q in bids:
            if q > 0:
                p_f, q_f = float(p), float(q)
                self._bids[p_f] = q_f
                self._bid_prices.append(p_f)

        # Process asks
        for p, q in asks:
            if q > 0:
                p_f, q_f = float(p), float(q)
                self._asks[p_f] = q_f
                self._ask_prices.append(p_f)

        # Sort once
        self._bid_prices.sort(reverse=True)
        self._ask_prices.sort()

        # Cache best levels
        self._best_bid = (self._bid_prices[0], self._bids[self._bid_prices[0]]) if self._bid_prices else None
        self._best_ask = (self._ask_prices[0], self._asks[self._ask_prices[0]]) if self._ask_prices else None
        
        self._book_dirty = True

    def _apply_level_update(self, side: Side, quantity: float, price: float) -> None:
        """Optimized level update"""
        try:
            p, q = float(price), float(quantity)
        except:
            return

        # Fast side check
        try:
            is_bid = (side == Side.BUY) if isinstance(side, Side) else (getattr(side, "name", "").upper() == "BUY")
        except:
            is_bid = str(side).upper() in ("BUY", "BID")

        if is_bid:
            book, prices = self._bids, self._bid_prices
            reverse_sort = True
        else:
            book, prices = self._asks, self._ask_prices
            reverse_sort = False

        if q <= 0:
            if p in book:
                del book[p]
                try:
                    prices.remove(p)
                except ValueError:
                    pass
        else:
            new_level = p not in book
            book[p] = q
            if new_level:
                # Use binary search for insertion to maintain sorted order
                import bisect
                if reverse_sort:
                    # For bids (reverse sorted), we need to find insertion point
                    idx = bisect.bisect_left([-px for px in prices], -p)
                    prices.insert(idx, p)
                else:
                    # For asks (normal sorted)
                    idx = bisect.bisect_left(prices, p)
                    prices.insert(idx, p)

        # Update cached best levels
        self._best_bid = (self._bid_prices[0], self._bids[self._bid_prices[0]]) if self._bid_prices else None
        self._best_ask = (self._ask_prices[0], self._asks[self._ask_prices[0]]) if self._ask_prices else None
        
        self._book_dirty = True

    def _place_limit(self, want_buy: bool, qty: float, price: float, ioc: bool = False) -> Optional[int]:
        if not self._rl_allow(1.0):
            return None
        
        # Use cached references
        if not callable(self._fn_place_limit) or self._ticker is None:
            return None
            
        side = self._Side.BUY if want_buy else self._Side.SELL
        try:
            return self._fn_place_limit(side, self._ticker, qty, price, ioc)
        except:
            return None

    def _cancel(self, oid: Optional[int]) -> None:
        if oid is None or not callable(self._fn_cancel) or self._ticker is None:
            return
        if not self._rl_allow(1.0):
            return
        try:
            self._fn_cancel(self._ticker, oid)
        except:
            pass

    def _handle_rollover(self, reason: str) -> None:
        """Optimized rollover handling"""
        # Cancel orders
        if self._bid_oid is not None:
            try: self._cancel(self._bid_oid)
            except: pass
            self._bid_oid = None
            self._last_bid_px = None
        
        if self._ask_oid is not None:
            try: self._cancel(self._ask_oid)
            except: pass
            self._ask_oid = None
            self._last_ask_px = None

        # Flatten if needed
        if self.flatten_on_rollover and abs(self.position) > 0 and callable(self._fn_place_market):
            try:
                side = self._Side.BUY if self.position < 0 else self._Side.SELL
                self._fn_place_market(side, self._ticker, abs(self.position))
                self.position = 0.0
                self.pos_avg_price = 0.0
            except:
                pass

        # Clear book
        self._bids.clear()
        self._asks.clear()
        self._bid_prices.clear()
        self._ask_prices.clear()
        self._best_bid = None
        self._best_ask = None
        self._cooldown_until = time.time() + self.rollover_cooldown_sec

    def _update_mid_stats(self, mid: float) -> None:
        """Optimized stats update using circular buffers"""
        if self._mid_count > 0:
            last_mid = self._mid_array[(self._mid_idx - 1) % self.vol_window]
            diff = mid - last_mid
            self._diff_array[self._diff_idx] = diff
            self._diff_idx = (self._diff_idx + 1) % self.vol_window
            self._diff_count = min(self._diff_count + 1, self.vol_window)
        
        # Store new mid
        self._mid_array[self._mid_idx] = mid
        self._mid_idx = (self._mid_idx + 1) % self.vol_window
        self._mid_count = min(self._mid_count + 1, self.vol_window)
        
        self._stats_dirty = True

    def _compute_stats(self) -> None:
        """Compute rolling stats only when needed"""
        if not self._stats_dirty:
            return
        
        # Compute mid mean and std
        if self._mid_count >= 2:
            total = sum(self._mid_array[i] for i in range(self._mid_count))
            self._mid_mean = total / self._mid_count
            
            var_sum = sum((self._mid_array[i] - self._mid_mean) ** 2 for i in range(self._mid_count))
            self._mid_std = (var_sum / max(1, self._mid_count - 1)) ** 0.5
        else:
            self._mid_mean = self._mid_array[0] if self._mid_count > 0 else 0.0
            self._mid_std = 0.0

        # Compute diff std
        if self._diff_count >= 2:
            diff_total = sum(self._diff_array[i] for i in range(self._diff_count))
            diff_mean = diff_total / self._diff_count
            diff_var = sum((self._diff_array[i] - diff_mean) ** 2 for i in range(self._diff_count))
            self._diff_std = (diff_var / max(1, self._diff_count - 1)) ** 0.5
        elif self._diff_count == 1:
            self._diff_std = abs(self._diff_array[0])
        else:
            self._diff_std = 0.0
        
        self._stats_dirty = False

    def _update_regime(self) -> None:
        """Optimized regime switching"""
        now = time.time()
        if now - self._last_regime_switch < self.min_regime_hold_sec:
            return
        
        vol_in_ticks = self._diff_std * self._tick_inv
        
        if self.regime != "VOL" and vol_in_ticks >= self.iv_high_thresh_ticks:
            self.regime = "VOL"
            self._last_regime_switch = now
        elif self.regime != "MOMO" and vol_in_ticks <= self.iv_low_thresh_ticks:
            self.regime = "MOMO"
            self._last_regime_switch = now

    def _compute_vw_prices(self) -> None:
        """Compute VW prices only when book is dirty"""
        if not self._book_dirty or not self._bid_prices or not self._ask_prices:
            return
        
        # VW bid
        vol, notional = 0.0, 0.0
        for i, p in enumerate(self._bid_prices[:self.depth]):
            q = self._bids[p]
            vol += q
            notional += p * q
        self._cached_vwbid = notional / vol if vol > 0 else 0.0
        
        # VW ask
        vol, notional = 0.0, 0.0
        for i, p in enumerate(self._ask_prices[:self.depth]):
            q = self._asks[p]
            vol += q
            notional += p * q
        self._cached_vwask = notional / vol if vol > 0 else 0.0
        
        # Cache derived values
        if self._cached_vwbid > 0 and self._cached_vwask > 0:
            self._cached_mid = (self._cached_vwbid + self._cached_vwask) * 0.5
            self._cached_spread = max(self._round_tick(self._cached_vwask - self._cached_vwbid), self.tick_size)
        
        self._book_dirty = False

    def _momo_trend_signal(self) -> float:
        """Optimized momentum signal computation"""
        if self._diff_count < 2:
            return 0.0
        
        # EWMA of recent diffs
        decay = self.momo_mom_decay
        w, num, den = 1.0, 0.0, 0.0
        
        k = min(self.momo_mom_window, self._diff_count)
        for i in range(k):
            idx = (self._diff_idx - 1 - i) % self.vol_window
            diff = self._diff_array[idx]
            num += w * diff
            den += w
            w *= decay
        
        ewma = num / den if den > 0 else 0.0
        
        # Scale by volatility
        scale = max(self._diff_std, self.tick_size) if self._diff_std > 0 else self.tick_size
        sig = ewma / scale
        
        return self._clamp(sig, -3.0, 3.0)

    def _update_markout(self, mid: float) -> None:
        """Optimized markout calculation using circular buffer"""
        now_ts = time.time()
        
        # Process aged fills
        while self._fills_size > 0:
            head_ts, sgn, px_fill = self._fills_buffer[self._fills_head]
            if now_ts - head_ts <= self.markout_horizon_sec:
                break
            
            # Realize markout
            mk = sgn * (mid - px_fill)
            self._markout_ewma = (1 - self._markout_alpha) * self._markout_ewma + self._markout_alpha * mk
            
            # Advance head
            self._fills_head = (self._fills_head + 1) % 200
            self._fills_size -= 1

    def _maybe_mean_revert_exit(self, mid: float) -> None:
        """Optimized mean revert exit"""
        if self.position == 0 or not self._rl_allow_exit(1.0):
            return

        edge_abs = max(self.mr_sigma_take * self._diff_std, self.mr_min_edge_ticks * self.tick_size)
        qty = min(abs(self.position), self.mr_clip)
        
        if qty <= 0:
            return

        if self.position > 0 and (mid - self.pos_avg_price) >= edge_abs:
            px = self._round_tick(max(self._cached_vwbid, mid - self.tick_size))
            self._place_limit(False, qty, px, ioc=self.mr_use_ioc)
        elif self.position < 0 and (self.pos_avg_price - mid) >= edge_abs:
            px = self._round_tick(min(self._cached_vwask, mid + self.tick_size))
            self._place_limit(True, qty, px, ioc=self.mr_use_ioc)

    # ───────── Public engine hooks (optimized) ─────────
    def on_orderbook_snapshot(self, ticker: Ticker, bids: list, asks: list) -> None:
        self._ticker = ticker
        self._rebuild_books(bids, asks)

    def on_orderbook_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        """Heavily optimized hot path"""
        self._ticker = ticker
        self._apply_level_update(side, quantity, price)

        # Skip during cooldown
        if time.time() < self._cooldown_until:
            return

        # Compute VW prices (cached)
        self._compute_vw_prices()
        
        if self._cached_vwbid <= 0 or self._cached_vwask <= 0:
            return

        mid = self._cached_mid
        spread = self._cached_spread
        
        # Update stats and regime
        self._update_mid_stats(mid)
        self._compute_stats()  # Only if dirty
        self._update_regime()
        self._update_markout(mid)

        # Dynamic position cap
        vol_in_ticks = self._diff_std * self._tick_inv
        pos_cap_dyn = max(self.min_pos_cap, self.base_pos_cap / max(1.0, vol_in_ticks))

        # Regime-specific calculations
        if self.regime == "VOL":
            # Volatility harvesting mode
            base_half = self.inside_frac * spread
            vol_add = self.k_vol * self._diff_std
            half = self._round_tick(self._clamp(base_half + vol_add, self._vol_min_half, self._vol_max_half))
            
            # Mean reversion skew
            z = (mid - self._mid_mean) / max(self._mid_std, self._EPS) if self._mid_std > self._EPS else 0.0
            center_skew_ticks = (-self.z_lean * z) + (-self.inv_skew_per_unit * self.position)
            
            # Size dampening
            size_scale = 1.0 / (1.0 + self.qty_vol_dampen * self._diff_std * self._tick_inv)
        else:
            # MOMO mode
            half = self._momo_fixed_half
            sig = self._momo_trend_signal()
            mom_skew_ticks = self._clamp(self.momo_skew_gain * sig, 
                                       -self.momo_skew_max_ticks, 
                                       self.momo_skew_max_ticks)
            center_skew_ticks = mom_skew_ticks + (-self.inv_skew_per_unit * self.position)
            size_scale = 1.0

        # Markout adjustment
        if self._markout_ewma < 0:
            half += self._round_tick(self.markout_widen_k * (-self._markout_ewma))

        # VAR cap adjustment
        if abs(self.position) > pos_cap_dyn:
            size_scale *= self.varcap_flatten_bias
            center_skew_ticks += (-self.inv_skew_per_unit * (self.position / pos_cap_dyn))

        center_skew = center_skew_ticks * self.tick_size

        # Target prices
        target_bid_px = self._round_tick(max(self._cached_vwbid, (mid + center_skew) - half))
        target_ask_px = self._round_tick(min(self._cached_vwask, (mid + center_skew) + half))

        # Size calculation (simplified)
        base_bid_qty = self.base_qty
        base_ask_qty = self.base_qty
        
        bid_qty = max(self._sanity_qty, base_bid_qty * size_scale)
        ask_qty = max(self._sanity_qty, base_ask_qty * size_scale)

        # Replace orders only when needed
        eps = max(self.price_epsilon, self.tick_size * 0.5)
        
        # Bid replacement
        if (self._last_bid_px is None or 
            abs(self._last_bid_px - target_bid_px) > eps):
            
            if self._bid_oid is not None:
                self._cancel(self._bid_oid)
                self._bid_oid = None
            
            oid = self._place_limit(True, bid_qty, target_bid_px, False)
            if oid is not None:
                self._bid_oid = oid
                self._last_bid_px = target_bid_px

        # Ask replacement  
        if (self._last_ask_px is None or 
            abs(self._last_ask_px - target_ask_px) > eps):
            
            if self._ask_oid is not None:
                self._cancel(self._ask_oid)
                self._ask_oid = None
            
            oid = self._place_limit(False, ask_qty, target_ask_px, False)
            if oid is not None:
                self._ask_oid = oid
                self._last_ask_px = target_ask_px

        # Mean revert exit
        self._maybe_mean_revert_exit(mid)

    def on_rejected(self, *args, **kwargs) -> None:
        pass

    def on_game_event_update(self, *args, **kwargs) -> None:
        """Optimized rollover detection"""
        try:
            self._last_event_ts = time.time()
        except:
            pass

        home_score = kwargs.get("home_score")
        away_score = kwargs.get("away_score")
        
        if home_score is None or away_score is None:
            return
        
        try:
            curr_sum = int(home_score) + int(away_score)
            
            if self._last_score_sum is not None:
                delta = curr_sum - self._last_score_sum
                
                # Check rollover conditions
                if (delta <= -self.min_drop or 
                    (self._last_score_sum > 0 and delta <= -self.frac_drop * self._last_score_sum)):
                    self._handle_rollover("score_reset")
                    self._last_score_sum = curr_sum
                    self._delta_count = 0
                    return
                
                # Store delta in circular buffer
                self._delta_array[self._delta_idx] = delta
                self._delta_idx = (self._delta_idx + 1) % 60
                self._delta_count = min(self._delta_count + 1, 60)
                
                # Z-score check (only if we have enough history)
                if self._delta_count >= 20:
                    total = sum(self._delta_array[i] for i in range(self._delta_count))
                    mu = total / self._delta_count
                    var_sum = sum((self._delta_array[i] - mu) ** 2 for i in range(self._delta_count))
                    sd = (var_sum / max(1, self._delta_count - 1)) ** 0.5
                    z = (delta - mu) / max(sd, self._EPS)
                    
                    if z <= -self.z_sigma:
                        self._handle_rollover("z_score_trigger")
                        self._last_score_sum = curr_sum
                        self._delta_count = 0
                        return
            
            self._last_score_sum = curr_sum
        except:
            pass

    def on_account_update(self, *args, **kwargs) -> None:
        """Optimized account update"""
        payload = args[0] if args and isinstance(args[0], dict) else kwargs
        
        # Fast attribute updates
        try:
            if "cash" in payload:
                self.cash = float(payload["cash"])
            if "capital_remaining" in payload:
                self.capital_remaining = float(payload["capital_remaining"])
            if "position" in payload:
                self.position = float(payload["position"])
        except:
            pass

    def on_trade_update(self, *args, **kwargs) -> None:
        return

    def on_order_update(self, *args, **kwargs) -> None:
        return

    def on_order_filled(self, *args, **kwargs) -> None:
        """Optimized fill handling"""
        # Extract fill data
        if args and isinstance(args[0], dict):
            d = args[0]
            side, quantity, price = d.get("side"), d.get("quantity"), d.get("price")
        else:
            side = kwargs.get("side")
            quantity = kwargs.get("quantity") 
            price = kwargs.get("price")

        if side is None or quantity is None or price is None:
            return

        try:
            is_buy = (side == self._Side.BUY) if self._Side else str(side).upper() in ("BUY", "BID")
        except:
            is_buy = str(side).upper() in ("BUY", "BID")

        q, p = float(quantity), float(price)
        prev_pos = self.position
        new_pos = prev_pos + q if is_buy else prev_pos - q

        # Optimized position average tracking
        if prev_pos == 0.0:
            self.pos_avg_price = p
        elif (prev_pos > 0 and new_pos > prev_pos) or (prev_pos < 0 and new_pos < prev_pos):
            # Adding to position
            abs_prev, abs_new = abs(prev_pos), abs(new_pos)
            if abs_new > self._EPS:
                self.pos_avg_price = (self.pos_avg_price * abs_prev + p * abs(abs_new - abs_prev)) / abs_new
        elif (prev_pos > 0 and new_pos < 0) or (prev_pos < 0 and new_pos > 0):
            # Crossing zero
            self.pos_avg_price = p

        # Update position and cash
        self.position = new_pos
        if is_buy:
            self.cash -= q * p
            sgn = 1
        else:
            self.cash += q * p
            sgn = -1

        # Add to fills ring buffer
        try:
            tail = (self._fills_head + self._fills_size) % 200
            self._fills_buffer[tail] = (time.time(), sgn, p)
            if self._fills_size < 200:
                self._fills_size += 1
            else:
                self._fills_head = (self._fills_head + 1) % 200
        except:
            pass

        if abs(self.position) < self._EPS:
            self.position = 0.0
            self.pos_avg_price = 0.0

    def on_pnl_update(self, *args, **kwargs) -> None:
        """Optimized PnL update"""
        payload = args[0] if args and isinstance(args[0], dict) else kwargs
        
        try:
            for key in ("position", "cash", "capital_remaining"):
                if key in payload and payload[key] is not None:
                    setattr(self, key, float(payload[key]))
        except:
            pass

#10k
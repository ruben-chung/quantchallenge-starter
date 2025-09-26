"""
Quant Challenge 2025

Algorithmic strategy – VW spread maker with full local orderbook and 30/min rate limit
Volatility-harvesting + mean-revert exit + IV-based regime switching + adaptive MOMO skew
+ Markout-aware quoting + Exit sub-budget + Dynamic inventory VAR cap + Time-decay risk management
"""

from __future__ import annotations
import time
import math
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

    Additions
    • Markout-aware quoting (short-horizon toxicity filter)
    • Exit sub-budget (prevents IOC exits from consuming the full 30/min)
    • Dynamic inventory VAR cap (cap shrinks as intraday IV rises)
    • Time-decay risk management (reduce position limits as event progresses)
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
        self.momo_pos_cap: float = 25.0          # legacy cap; kept for reference

        # MOMO adaptive momentum
        self.momo_mom_window: int = 8            # lookback (# mids → #diffs = window-1)
        self.momo_mom_decay: float = 0.7         # EWMA decay for recent diffs
        self.momo_skew_gain: float = 0.35        # ticks of skew per 1.0 unit signal
        self.momo_skew_max_ticks: float = 2.0    # cap skew magnitude (ticks)

        # Markout-aware quoting (toxicity filter)
        self._markout_ewma: float = 0.0          # >0 good, <0 toxic
        self._markout_alpha: float = 0.2         # EWMA rate
        self._fills_ring = deque(maxlen=200)     # ring of (ts, sign, fill_px)
        self.markout_horizon_sec: float = 2.0    # realized markout horizon
        self.markout_widen_k: float = 0.5        # widen factor per unit negative markout

        # Exit sub-budget (protects order budget from IOC exits)
        self._exit_capacity = 10.0               # exits per minute
        self._exit_tokens = self._exit_capacity
        self._exit_refill_per_sec = self._exit_capacity / 60.0
        self._exit_last_t = time.monotonic()

        # Dynamic inventory VAR cap
        self.base_pos_cap: float = 60.0          # position units at ~1 tick vol
        self.min_pos_cap: float = 10.0           # never go below this
        self.varcap_flatten_bias: float = 0.5    # size scale when beyond cap

        # Time-decay risk management attributes
        self.event_start_time: Optional[float] = None
        self.event_duration_minutes: float = 120.0  # Default 2 hours for most sports
        self.game_phase: str = "UNKNOWN"  # PREGAME, Q1, Q2, HALFTIME, Q3, Q4, OVERTIME, FINAL
        self.time_remaining_seconds: Optional[float] = None
        
        # Time-based risk scaling factors
        self.time_decay_curve: str = "EXPONENTIAL"  # or "LINEAR" or "STEP"
        self.late_game_threshold: float = 0.15  # Last 15% of game time
        self.critical_threshold: float = 0.05   # Last 5% of game time
        
        # Position limit decay parameters
        self.end_game_pos_mult: float = 0.3      # Reduce to 30% of normal at game end
        self.halftime_pos_mult: float = 0.7      # Reduce to 70% during transition periods
        self.overtime_pos_mult: float = 0.4      # Very conservative in overtime
        
        # Size scaling parameters
        self.late_game_size_mult: float = 0.6    # Reduce order sizes in late game
        self.critical_size_mult: float = 0.3     # Minimal sizes in final minutes
        
        # Spread widening parameters
        self.late_game_spread_mult: float = 1.4   # Widen spreads 40% in late game
        self.critical_spread_mult: float = 2.0    # Double spreads in critical periods
        self.timeout_spread_mult: float = 1.3     # Widen during timeouts/breaks
        
        # Period transition detection
        self.period_transition_window: float = 30.0  # 30 seconds around period changes
        self.last_period_change: Optional[float] = None
        self.in_transition: bool = False
        
        # Game situation tracking
        self.score_differential: float = 0.0
        self.is_close_game: bool = False
        self.close_game_threshold: int = 7  # Within 7 points = close game
        self.blowout_threshold: int = 20    # 20+ point differential = blowout
        self._last_risk_log: float = 0.0
        self._last_period: Optional[int] = None

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

    # Exit sub-budget limiter
    def _rl_allow_exit(self, cost: float = 1.0) -> bool:
        now = time.monotonic()
        dt = max(0.0, now - self._exit_last_t)
        self._exit_last_t = now
        self._exit_tokens = min(self._exit_capacity, self._exit_tokens + dt * self._exit_refill_per_sec)
        if self._exit_tokens >= cost:
            self._exit_tokens -= cost
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

    # ───────── Time-decay risk management methods ─────────
    def _initialize_event_timing(self) -> None:
        """Initialize event timing when first game event is received"""
        if self.event_start_time is None:
            self.event_start_time = time.time()

    def _update_game_context(self, **kwargs) -> None:
        """Update game context from event data"""
        # Extract timing information
        period = kwargs.get('period', kwargs.get('quarter', None))
        time_left = kwargs.get('time_remaining', kwargs.get('clock', None))
        
        # Update game phase
        if period is not None:
            if period == 0:
                self.game_phase = "PREGAME"
            elif period == 1:
                self.game_phase = "Q1"
            elif period == 2:
                self.game_phase = "Q2" 
            elif period == 3:
                self.game_phase = "Q3"
            elif period == 4:
                self.game_phase = "Q4"
            elif period > 4:
                self.game_phase = "OVERTIME"
            
            # Check for halftime or period transitions
            if hasattr(self, '_last_period') and self._last_period != period:
                self.last_period_change = time.time()
                self.in_transition = True
            self._last_period = period
        
        # Update time remaining
        if time_left is not None:
            try:
                if isinstance(time_left, str) and ':' in time_left:
                    # Parse "MM:SS" format
                    parts = time_left.split(':')
                    self.time_remaining_seconds = int(parts[0]) * 60 + int(parts[1])
                else:
                    self.time_remaining_seconds = float(time_left)
            except (ValueError, IndexError):
                pass
        
        # Update score differential
        home_score = kwargs.get('home_score', 0)
        away_score = kwargs.get('away_score', 0)
        try:
            self.score_differential = abs(int(home_score) - int(away_score))
            self.is_close_game = self.score_differential <= self.close_game_threshold
        except (ValueError, TypeError):
            pass

    def _calculate_time_progress(self) -> float:
        """Calculate how far through the event we are (0.0 to 1.0)"""
        if self.time_remaining_seconds is not None:
            # Use actual game clock if available
            total_regulation_time = 60 * 60  # 60 minutes for most sports
            if self.game_phase == "OVERTIME":
                # In overtime, consider we're at 100%+ progress
                return 1.0 + (600 - self.time_remaining_seconds) / 600.0  # 10min OT periods
            else:
                time_elapsed = total_regulation_time - self.time_remaining_seconds
                return min(1.0, max(0.0, time_elapsed / total_regulation_time))
        
        elif self.event_start_time is not None:
            # Fallback to wall clock time
            elapsed_minutes = (time.time() - self.event_start_time) / 60.0
            return min(1.0, max(0.0, elapsed_minutes / self.event_duration_minutes))
        
        return 0.0  # Unknown progress

    def _check_transition_periods(self) -> bool:
        """Check if we're in a period transition (high volatility time)"""
        if self.last_period_change is None:
            return False
        
        time_since_change = time.time() - self.last_period_change
        if time_since_change <= self.period_transition_window:
            return True
        else:
            self.in_transition = False
            return False

    def _calculate_time_decay_multipliers(self) -> Dict[str, float]:
        """Calculate risk adjustment multipliers based on time progress"""
        progress = self._calculate_time_progress()
        is_transition = self._check_transition_periods()
        
        # Base multipliers
        pos_mult = 1.0
        size_mult = 1.0
        spread_mult = 1.0
        
        # Handle different phases
        if self.game_phase == "OVERTIME":
            pos_mult = self.overtime_pos_mult
            size_mult = self.critical_size_mult
            spread_mult = self.critical_spread_mult
            
        elif is_transition or self.game_phase == "HALFTIME":
            pos_mult = self.halftime_pos_mult
            size_mult = 0.8
            spread_mult = self.timeout_spread_mult
            
        elif progress >= (1.0 - self.critical_threshold):
            # Final 5% of game - very conservative
            pos_mult = self.end_game_pos_mult
            size_mult = self.critical_size_mult
            spread_mult = self.critical_spread_mult
            
        elif progress >= (1.0 - self.late_game_threshold):
            # Late game (last 15%) - moderately conservative
            # Smooth interpolation from normal to end-game values
            late_progress = (progress - (1.0 - self.late_game_threshold)) / self.late_game_threshold
            
            if self.time_decay_curve == "EXPONENTIAL":
                # Exponential decay - more aggressive near the end
                decay_factor = math.exp(-3 * (1 - late_progress))
            elif self.time_decay_curve == "LINEAR":
                # Linear decay
                decay_factor = 1 - late_progress
            else:  # STEP
                # Step function at critical threshold
                decay_factor = 1.0 if late_progress < 0.67 else 0.5
            
            pos_mult = 1.0 - (1.0 - self.end_game_pos_mult) * (1 - decay_factor)
            size_mult = 1.0 - (1.0 - self.late_game_size_mult) * (1 - decay_factor)
            spread_mult = 1.0 + (self.late_game_spread_mult - 1.0) * (1 - decay_factor)
        
        # Additional adjustments for game situation
        if self.is_close_game and progress > 0.8:
            # Close games in final 20% - extra conservative
            pos_mult *= 0.8
            spread_mult *= 1.2
            
        elif self.score_differential >= self.blowout_threshold and progress < 0.8:
            # Blowout games early - can be slightly more aggressive
            pos_mult *= 1.1
            spread_mult *= 0.95
        
        return {
            "position_mult": pos_mult,
            "size_mult": size_mult, 
            "spread_mult": spread_mult,
            "time_progress": progress
        }

    def _get_time_adjusted_position_cap(self) -> float:
        """Get position cap adjusted for time decay"""
        multipliers = self._calculate_time_decay_multipliers()
        
        # Start with your existing dynamic VAR cap
        vol_in_ticks = (self._diff_std or 0.0) / max(self.tick_size, 1e-9)
        base_cap = max(self.min_pos_cap, self.base_pos_cap / max(1.0, vol_in_ticks))
        
        # Apply time decay
        time_adjusted_cap = base_cap * multipliers["position_mult"]
        
        return max(self.min_pos_cap * 0.5, time_adjusted_cap)  # Never go below 50% of min

    def _get_time_adjusted_sizing(self, base_bid_qty: float, base_ask_qty: float) -> Tuple[float, float]:
        """Get order sizes adjusted for time decay"""
        multipliers = self._calculate_time_decay_multipliers()
        size_mult = multipliers["size_mult"]
        
        adj_bid_qty = max(self._sanity_qty, base_bid_qty * size_mult)
        adj_ask_qty = max(self._sanity_qty, base_ask_qty * size_mult)
        
        return adj_bid_qty, adj_ask_qty

    def _get_time_adjusted_spreads(self, base_half_spread: float) -> float:
        """Get half-spread adjusted for time decay"""
        multipliers = self._calculate_time_decay_multipliers()
        spread_mult = multipliers["spread_mult"]
        
        adjusted_half = base_half_spread * spread_mult
        
        # Ensure we don't go below minimum or above maximum
        min_half = self.min_inside_ticks * self.tick_size
        max_half = self.max_inside_ticks * self.tick_size * 1.5  # Allow extra wide in late game
        
        return self._clamp(adjusted_half, min_half, max_half)

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
        if self._bid_oid is not None:
            try: self._cancel(self._bid_oid)
            except Exception: pass
            self._bid_oid = None; self._last_bid_px = None
        if self._ask_oid is not None:
            try: self._cancel(self._ask_oid)
            except Exception: pass
            self._ask_oid = None; self._last_ask_px = None

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
        n_hist = len(self._mid_hist)
        if n_hist < 2:
            return 0.0
        k = min(self.momo_mom_window + 1, n_hist)
        diffs: List[float] = []
        for i in range(1, k):
            diffs.append(self._mid_hist[-i] - self._mid_hist[-i-1])
        if not diffs:
            return 0.0
        decay = float(self.momo_mom_decay)
        w = 1.0; num = 0.0; den = 0.0
        for d in diffs:
            num += w * d; den += w; w *= decay
        ewma = (num / den) if den > 0 else 0.0
        if self._diff_std and self._diff_std > 0:
            scale = self._diff_std
        else:
            avg_abs = sum(abs(d) for d in diffs) / max(1, len(diffs))
            scale = max(self.tick_size, avg_abs)
        sig = ewma / max(scale, 1e-9)
        return self._clamp(sig, -3.0, 3.0)

    # ───────── Mean-revert exit helper ─────────
    def _maybe_mean_revert_exit(self, mid: float, vwbid: float, vwask: float) -> None:
        if self.position == 0:
            return
        if not self._rl_allow_exit(1.0):
            return  # preserve order budget; quotes will lean to flatten anyway

        sigma_diff = float(self._diff_std or 0.0)
        edge_abs = max(self.mr_sigma_take * sigma_diff, self.mr_min_edge_ticks * self.tick_size)
        qty = min(abs(self.position), self.mr_clip)
        if qty <= 0:
            return

        if self.position > 0 and (mid - self.pos_avg_price) >= edge_abs:
            px = self._round_tick(max(vwbid, mid - self.tick_size))
            self._place_limit(False, qty, px, ioc=self.mr_use_ioc)
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
                vol += q; notional += p * q
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

                # Update short-horizon markout (realize aged fills vs current mid)
                now_ts = time.time()
                while self._fills_ring and now_ts - self._fills_ring[0][0] > self.markout_horizon_sec:
                    _, sgn, px_fill = self._fills_ring.popleft()
                    mk = sgn * (mid - px_fill)  # >0 good, <0 toxic
                    self._markout_ewma = (1 - self._markout_alpha) * self._markout_ewma + self._markout_alpha * mk

                # Dynamic VAR cap (shrinks as vol rises) - NOW WITH TIME DECAY
                pos_cap_dyn = self._get_time_adjusted_position_cap()

                # Compute half/center/size per regime
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
                    sig = self._momo_trend_signal()
                    mom_skew_ticks = self._clamp(self.momo_skew_gain * sig,
                                                 -self.momo_skew_max_ticks,
                                                 self.momo_skew_max_ticks)
                    center_skew_ticks = mom_skew_ticks + (- self.inv_skew_per_unit * self.position)
                    size_scale = 1.0

                # Apply time decay adjustments to spreads
                half = self._get_time_adjusted_spreads(half)

                # Markout-aware widening: if toxicity (negative EWMA), widen by fraction of its magnitude
                markout_penalty = max(0.0, -self._markout_ewma)
                if markout_penalty > 0:
                    half += self._round_tick(self.markout_widen_k * markout_penalty)

                # Extra flatten bias when beyond dynamic cap (reduce sizes too)
                if abs(self.position) > pos_cap_dyn:
                    size_scale *= self.varcap_flatten_bias
                    # push center opposite inventory a bit more
                    center_skew_ticks += (- self.inv_skew_per_unit * (self.position / max(1.0, pos_cap_dyn)))

                center_skew = center_skew_ticks * self.tick_size

                # Targets with skew (ensure we don't cross best levels)
                target_bid_px = self._round_tick(max(vwbid, (mid + center_skew) - half))
                target_ask_px = self._round_tick(min(vwask, (mid + center_skew) + half))

                # Size leaning by displayed imbalance + regime/dynamic scaling
                tot = bid_vol + ask_vol
                imb = ((bid_vol - ask_vol) / tot) if tot > 0 else 0.0      # [-1, 1]
                bid_qty = self._clamp(self.base_qty * (1.0 + max(0.0, -imb)), 1.0, self.max_qty)
                ask_qty = self._clamp(self.base_qty * (1.0 + max(0.0,  imb)), 1.0, self.max_qty)
                bid_qty = max(self._sanity_qty, bid_qty * size_scale)
                ask_qty = max(self._sanity_qty, ask_qty * size_scale)

                # Apply time decay adjustments to sizes
                bid_qty, ask_qty = self._get_time_adjusted_sizing(bid_qty, ask_qty)

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

                # Opportunistic mean-revert exit (IOC, respects exit budget)
                self._maybe_mean_revert_exit(mid, vwbid, vwask)

    def on_rejected(self, *args, **kwargs) -> None:
        # Could adaptively widen on rejects
        pass

    # Flexible callbacks to match engines that pass many positional args
    def on_game_event_update(self, *args, **kwargs) -> None:
        try:
            now = kwargs.get("timestamp") or kwargs.get("ts") or kwargs.get("time")
            if now is None:
                nums = [a for a in args if isinstance(a, (int, float))]
                if nums:
                    now = float(nums[-1])
            self._last_event_ts = float(now) if now is not None else time.time()
        except Exception:
            self._last_event_ts = time.time()

        # Initialize timing on first event
        self._initialize_event_timing()
        
        # Update game context with timing information
        self._update_game_context(**kwargs)

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

        # Log current risk state periodically
        if time.time() - self._last_risk_log > 60:  # Every minute
            self._last_risk_log = time.time()
            multipliers = self._calculate_time_decay_multipliers()
            progress = multipliers["time_progress"]
            pos_cap = self._get_time_adjusted_position_cap()
            print(f"Time-decay Risk: Progress={progress:.2%}, Phase={self.game_phase}, "
                  f"PosCap={pos_cap:.1f}, Pos={self.position:.1f}, "
                  f"Mults: pos={multipliers['position_mult']:.2f}, "
                  f"size={multipliers['size_mult']:.2f}, spread={multipliers['spread_mult']:.2f}")

    def on_account_update(self, *args, **kwargs) -> None:
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
        new_pos = prev_pos + q if is_buy else prev_pos - q

        # pos_avg_price maintenance
        if prev_pos == 0.0:
            self.pos_avg_price = p
        elif (prev_pos > 0 and new_pos > prev_pos) or (prev_pos < 0 and new_pos < prev_pos):
            abs_prev = abs(prev_pos); abs_new = abs(new_pos)
            if abs_new > 1e-12:
                self.pos_avg_price = (self.pos_avg_price * abs_prev + p * abs(abs_new - abs_prev)) / abs_new
        elif (prev_pos > 0 and new_pos < 0) or (prev_pos < 0 and new_pos > 0):
            self.pos_avg_price = p

        # Commit position/cash
        self.position = new_pos
        if is_buy:
            self.cash -= q * p
            sgn = +1
        else:
            self.cash += q * p
            sgn = -1

        # Queue the fill for later markout measurement
        try:
            self._fills_ring.append((time.time(), sgn, p))
        except Exception:
            pass

        if self.position == 0.0:
            self.pos_avg_price = 0.0

    def on_pnl_update(self, *args, **kwargs) -> None:
        payload = args[0] if args and isinstance(args[0], dict) else (kwargs or {})
        for key in ("position", "cash", "capital_remaining"):
            try:
                if payload.get(key) is not None:
                    setattr(self, key, float(payload[key]))
            except Exception:
                pass


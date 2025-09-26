"""
Ultra Low-Latency Optimized Trading Strategy (EWMA-only improvements)

Key optimizations:
1. Incremental orderbook updates (no full rebuilds)
2. Pre-computed lookup tables and cached calculations
3. Minimal branching in hot paths
4. Batch operations and lazy evaluation
5. Memory pools and object reuse
6. Fast numerical operations with numpy
7. Streamlined risk checks
8. Optimized data structures

Live robustness add-ons (fast):
- EWMA markout (toxicity) with persistent asymmetric widening
- Microprice lean (top-of-book imbalance)
- Queue-aware shading (thin/heavy best size)
- Replace throttling (per side)
- Inventory-aware widen floor
"""

from __future__ import annotations
import time
import numpy as np
from enum import Enum
from typing import Optional, Dict, List, Tuple
from collections import deque
from dataclasses import dataclass

# ─────────────────────────────────────────────────────────────────────────────
# Provided API surface
# ─────────────────────────────────────────────────────────────────────────────

class Side(Enum):
    BUY = 1
    SELL = 2

class Ticker(Enum):
    TEAM_A = 1
    TEAM_B = 2

def place_market_order(side: Side, ticker: Ticker, quantity: float) -> bool:
    return False

def place_limit_order(side: Side, ticker: Ticker, quantity: float, price: float, ioc: bool = False) -> int:
    return 0

def cancel_order(ticker: Ticker, order_id: int) -> bool:
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Fast Data Structures and Caching
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FastLevel:
    """Memory-efficient orderbook level"""
    __slots__ = ('price', 'qty')
    price: float
    qty: float

class FastOrderbook:
    """Ultra-fast orderbook with incremental updates and cached calculations"""
    
    def __init__(self, max_levels: int = 50):
        self.bid_prices = np.zeros(max_levels, dtype=np.float64)
        self.bid_qtys   = np.zeros(max_levels, dtype=np.float64)
        self.ask_prices = np.zeros(max_levels, dtype=np.float64)
        self.ask_qtys   = np.zeros(max_levels, dtype=np.float64)
        self.bid_count = 0
        self.ask_count = 0
        self.max_levels = max_levels
        # Cached values
        self._best_bid = 0.0
        self._best_ask = 0.0
        self._mid = 0.0
        self._spread = 0.0
        self._cache_valid = False
        # VW cache
        self.vw_cache = np.zeros((2, 10, 2), dtype=np.float64)  # [side][depth][price,vol]
        self.vw_cache_valid = False
    
    def update_level(self, is_buy: bool, price: float, qty: float) -> bool:
        prices = self.bid_prices if is_buy else self.ask_prices
        qtys   = self.bid_qtys  if is_buy else self.ask_qtys
        count  = self.bid_count if is_buy else self.ask_count
        
        if qty <= 0.0:  # Remove level
            idx = self._find_price_idx(prices, count, price, is_buy)
            if idx >= 0:
                if idx < count - 1:
                    prices[idx:count-1] = prices[idx+1:count]
                    qtys[idx:count-1]   = qtys[idx+1:count]
                if is_buy: self.bid_count -= 1
                else:      self.ask_count -= 1
                self._invalidate_cache()
                return True
        else:  # Add/update
            idx = self._find_insert_idx(prices, count, price, is_buy)
            if idx < count and abs(prices[idx] - price) < 1e-9:
                qtys[idx] = qty
            else:
                if count < self.max_levels:
                    if idx < count:
                        prices[idx+1:count+1] = prices[idx:count]
                        qtys[idx+1:count+1]   = qtys[idx:count]
                    prices[idx] = price
                    qtys[idx]   = qty
                    if is_buy: self.bid_count += 1
                    else:      self.ask_count += 1
            self._invalidate_cache()
            return True
        return False
    
    def _find_price_idx(self, prices: np.ndarray, count: int, price: float, is_buy: bool) -> int:
        if count == 0: return -1
        # Linear scan (small count); avoids reversing for desc bids
        for i in range(count):
            if abs(prices[i] - price) < 1e-9:
                return i
        return -1
    
    def _find_insert_idx(self, prices: np.ndarray, count: int, price: float, is_buy: bool) -> int:
        if count == 0: return 0
        if is_buy:  # Descending
            for i in range(count):
                if price > prices[i]: return i
            return count
        else:      # Ascending
            for i in range(count):
                if price < prices[i]: return i
            return count
    
    def _invalidate_cache(self) -> None:
        self._cache_valid = False
        self.vw_cache_valid = False
    
    def get_bbo(self) -> Tuple[float, float, float, float]:
        if not self._cache_valid: self._update_cache()
        return (self._best_bid, self.bid_qtys[0] if self.bid_count > 0 else 0.0,
                self._best_ask, self.ask_qtys[0] if self.ask_count > 0 else 0.0)
    
    def get_mid_spread(self) -> Tuple[float, float]:
        if not self._cache_valid: self._update_cache()
        return self._mid, self._spread
    
    def _update_cache(self) -> None:
        self._best_bid = self.bid_prices[0] if self.bid_count > 0 else 0.0
        self._best_ask = self.ask_prices[0] if self.ask_count > 0 else 0.0
        if self._best_bid > 0 and self._best_ask > 0:
            self._mid = 0.5 * (self._best_bid + self._best_ask)
            self._spread = self._best_ask - self._best_bid
        else:
            self._mid = self._spread = 0.0
        self._cache_valid = True
    
    def get_vw_price(self, is_buy: bool, depth: int) -> Tuple[float, float]:
        if not self.vw_cache_valid: self._update_vw_cache()
        side_idx = 0 if is_buy else 1
        depth_idx = min(depth - 1, 9)
        return self.vw_cache[side_idx, depth_idx, 0], self.vw_cache[side_idx, depth_idx, 1]
    
    def _update_vw_cache(self) -> None:
        # bids
        for depth in range(1, min(11, self.bid_count + 1)):
            vol = 0.0; notional = 0.0
            for i in range(depth):
                q = self.bid_qtys[i]; p = self.bid_prices[i]
                vol += q; notional += p * q
            if vol > 0:
                self.vw_cache[0, depth-1, 0] = notional / vol
                self.vw_cache[0, depth-1, 1] = vol
        # asks
        for depth in range(1, min(11, self.ask_count + 1)):
            vol = 0.0; notional = 0.0
            for i in range(depth):
                q = self.ask_qtys[i]; p = self.ask_prices[i]
                vol += q; notional += p * q
            if vol > 0:
                self.vw_cache[1, depth-1, 0] = notional / vol
                self.vw_cache[1, depth-1, 1] = vol
        self.vw_cache_valid = True


class FastRollingStats:
    """Ultra-fast rolling statistics with incremental updates"""
    def __init__(self, window: int = 60):
        self.window = window
        self.values = np.zeros(window, dtype=np.float64)
        self.diffs  = np.zeros(window-1, dtype=np.float64)
        self.count = 0
        self.idx = 0
        self.mean = 0.0
        self.std = 0.0
        self.diff_std = 0.0
        self.last_value = 0.0
        self.sum_values = 0.0
        self.sum_sq_values = 0.0
        self.sum_diffs = 0.0
        self.sum_sq_diffs = 0.0
    
    def add_value(self, value: float) -> None:
        old_value = self.values[self.idx] if self.count >= self.window else 0.0
        self.values[self.idx] = value
        if self.count > 0:
            diff = value - self.last_value
            diff_idx = (self.idx - 1) % (self.window - 1)
            old_diff = self.diffs[diff_idx] if self.count >= self.window else 0.0
            self.diffs[diff_idx] = diff
            self.sum_diffs += diff - old_diff
            self.sum_sq_diffs += diff*diff - old_diff*old_diff
        self.sum_values += value - old_value
        self.sum_sq_values += value*value - old_value*old_value
        self.idx = (self.idx + 1) % self.window
        self.count = min(self.count + 1, self.window)
        self.last_value = value
        self._update_stats()
    
    def _update_stats(self) -> None:
        if self.count <= 1:
            self.mean = self.last_value; self.std = 0.0; self.diff_std = 0.0
            return
        n = self.count
        self.mean = self.sum_values / n
        var = max(0.0, (self.sum_sq_values - self.sum_values * self.mean) / (n - 1))
        self.std = np.sqrt(var)
        if self.count >= 2:
            n_diff = min(self.count - 1, self.window - 1)
            diff_mean = self.sum_diffs / n_diff
            diff_var  = max(0.0, (self.sum_sq_diffs - self.sum_diffs * diff_mean) / max(1, n_diff - 1))
            self.diff_std = np.sqrt(diff_var)


class FastRiskManager:
    """Streamlined risk manager for hot path"""
    def __init__(self, max_position_pct: float = 0.25, daily_loss_pct: float = 0.03):
        self.max_position_pct = max_position_pct
        self.daily_loss_pct = daily_loss_pct
        self.current_capital = 100000.0
        self.daily_start_capital = 100000.0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 8
        self.risk_level = 0
        self.position_scale_factors = np.array([1.0, 0.7, 0.4, 0.1], dtype=np.float64)
        self.circuit_breaker_active = False
        self.circuit_breaker_until = 0.0
        
    def quick_risk_check(self, position: float, mid_price: float) -> Tuple[int, float]:
        if self.circuit_breaker_active and time.time() < self.circuit_breaker_until:
            return 3, 0.0
        position_value = abs(position * mid_price)
        position_pct = position_value / max(self.current_capital, 1.0)
        if position_pct >= self.max_position_pct:
            return 3, 0.1
        daily_pnl_pct = (self.current_capital - self.daily_start_capital) / max(self.daily_start_capital, 1.0)
        if daily_pnl_pct <= -self.daily_loss_pct:
            self.circuit_breaker_active = True
            self.circuit_breaker_until = time.time() + 1800
            return 3, 0.0
        if self.consecutive_losses >= self.max_consecutive_losses:
            risk_level = 3
        elif daily_pnl_pct <= -self.daily_loss_pct * 0.7 or position_pct >= self.max_position_pct * 0.8:
            risk_level = 2
        elif daily_pnl_pct <= -self.daily_loss_pct * 0.4 or position_pct >= self.max_position_pct * 0.6:
            risk_level = 1
        else:
            risk_level = 0
        return risk_level, self.position_scale_factors[risk_level]


# ─────────────────────────────────────────────────────────────────────────────
# Ultra Fast Strategy Implementation (with EWMA improvements)
# ─────────────────────────────────────────────────────────────────────────────

class FastStrategy:
    """
    Ultra low-latency optimized trading strategy with EWMA-only robustness.
    """
    def __init__(self, initial_capital: float = 100000.0):
        # Fast data structures
        self.orderbook = FastOrderbook(max_levels=20)
        self.mid_stats = FastRollingStats(window=60)
        self.risk_manager = FastRiskManager()
        
        # API resolution
        self._Side = None
        self._Ticker = None
        self._fn_place_market = None
        self._fn_place_limit = None
        self._fn_cancel = None
        self._ticker = None
        
        # Position tracking
        self.position = 0.0
        self.cash = initial_capital
        self.pos_avg_price = 0.0
        
        # Current orders (one bid/ask)
        self.bid_oid = None
        self.ask_oid = None
        self.last_bid_px = 0.0
        self.last_ask_px = 0.0
        
        # Params
        self.tick_size = 0.5
        self.base_qty = 5.0
        self.max_qty = 20.0
        self.depth = 5
        
        # Regimes: 0=VOL, 1=MOMO
        self.regime = 0
        self.regime_params = np.array([
            [0.25, 1.25, 1.0, 8.0],  # VOL: inside_frac, k_vol, min_ticks, max_ticks
            [1.5, 0.0, 0.0, 0.0]     # MOMO: half_ticks, unused, unused, unused
        ], dtype=np.float64)
        
        # Cached metrics
        self.last_mid = 0.0
        self.last_spread = 0.0
        self.last_vol_ticks = 0.0
        
        # Rate limiting
        self.rl_tokens = 30.0
        self.rl_last_refill = time.monotonic()
        
        # Inventory skew
        self.inv_skew_per_unit = 0.05
        
        # === EWMA markout + widening (NEW) ===
        self._fills_ring = deque(maxlen=200)        # (ts, sign, fill_px)
        self._markout_ewma = 0.0                    # >0 good, <0 toxic
        self._markout_alpha = 0.2
        self.markout_horizon_sec = 2.0
        # persistent asymmetric widening
        self._spread_penalty_px = 0.0
        self._spread_penalty_decay = 0.985          # ~gentle decay per sec
        self.markout_widen_bad_k  = 0.70            # widen fast on toxic
        self.markout_widen_good_k = 0.25            # unwind slowly on benign
        
        # Queue-aware shading (NEW)
        self.queue_thin_qty  = 5.0
        self.queue_heavy_qty = 50.0
        
        # Microprice lean (NEW)
        self.microprice_lean = 0.35                 # fraction of (ask-bid) by imbalance
        
        # Replace throttling (NEW)
        self.min_replace_ms = 200.0
        self._last_replace_ms = {"bid": 0.0, "ask": 0.0}
        
        # Inventory-aware widen floor (NEW)
        self.inv_widen_ticks_per_unit = 0.02
        self.inv_widen_cap_ticks      = 3.0
        
        self._ensure_api()
    
    # API and rate limit
    def _ensure_api(self) -> None:
        g = globals()
        self._Side = g.get("Side", None)
        self._Ticker = g.get("Ticker", None)
        self._fn_place_market = g.get("place_market_order", None)
        self._fn_place_limit = g.get("place_limit_order", None)
        self._fn_cancel = g.get("cancel_order", None)
        if self._Ticker and hasattr(self._Ticker, "TEAM_A"):
            self._ticker = self._Ticker.TEAM_A
    
    def _refill_tokens(self) -> None:
        now = time.monotonic()
        dt = now - self.rl_last_refill
        self.rl_last_refill = now
        self.rl_tokens = min(30.0, self.rl_tokens + dt * 0.5)  # 30/min
    
    def _can_trade(self) -> bool:
        return self.rl_tokens >= 1.0 and (self._fn_place_limit and self._ticker and self._Side)
    
    def _place_limit_fast(self, is_buy: bool, qty: float, price: float) -> Optional[int]:
        if not self._can_trade(): return None
        self.rl_tokens -= 1.0
        side = self._Side.BUY if is_buy else self._Side.SELL
        try:
            return self._fn_place_limit(side, self._ticker, float(qty), float(price), False)
        except Exception:
            return None
    
    def _cancel_fast(self, oid: Optional[int]) -> None:
        if oid is None or not self._can_trade(): return
        self.rl_tokens -= 1.0
        try:
            self._fn_cancel(self._ticker, oid)
        except Exception:
            pass
    
    # Regimes
    def _update_regime_fast(self) -> None:
        vol_ticks = (self.mid_stats.diff_std / max(self.tick_size, 1e-9)) if self.tick_size > 0 else 0.0
        self.last_vol_ticks = vol_ticks
        self.regime = 1 if vol_ticks <= 1.0 else 0  # MOMO if low-vol
    
    # Quote params (now returns center and half for adjustments)
    def _calculate_quote_params_fast(self, mid: float, spread: float, vw_bid: float, vw_ask: float, 
                                     bid_vol: float, ask_vol: float) -> Tuple[float, float, float, float, float, float]:
        params = self.regime_params[self.regime]
        if self.regime == 0:  # VOL
            inside_frac, k_vol, min_ticks, max_ticks = params
            base_half = inside_frac * spread
            vol_add = k_vol * self.mid_stats.diff_std
            half_spread = np.clip(base_half + vol_add,
                                  min_ticks * self.tick_size,
                                  max_ticks * self.tick_size)
            z = (mid - self.mid_stats.mean) / max(self.mid_stats.std, 1e-9) if self.mid_stats.std > 0 else 0.0
            center_skew = -0.35 * z * self.tick_size
        else:  # MOMO
            half_ticks = params[0]
            half_spread = half_ticks * self.tick_size
            momentum = 0.0
            if self.mid_stats.count >= 3:
                # take last 3 diffs from the circular buffer cheaply
                idxs = [(self.mid_stats.idx - 1 - i) % (self.mid_stats.window - 1) for i in range(3)]
                recent = np.array([self.mid_stats.diffs[j] for j in idxs], dtype=np.float64)
                momentum = float(np.mean(recent)) / max(self.mid_stats.diff_std, 1e-9)
            center_skew = np.clip(0.35 * momentum * self.tick_size,
                                  -2.0 * self.tick_size, 2.0 * self.tick_size)
        # Inventory skew
        inv_skew = -self.inv_skew_per_unit * self.position * self.tick_size
        total_skew = center_skew + inv_skew
        center = mid + total_skew
        
        # Initial targets
        bid_target = max(vw_bid, center - half_spread)
        ask_target = min(vw_ask, center + half_spread)
        
        # Sizes from imbalance
        tot = bid_vol + ask_vol
        if tot > 0:
            imb = (bid_vol - ask_vol) / tot
            bid_qty = self.base_qty * (1.0 + max(0.0, -imb))
            ask_qty = self.base_qty * (1.0 + max(0.0,  imb))
        else:
            bid_qty = ask_qty = self.base_qty
        bid_qty = min(bid_qty, self.max_qty)
        ask_qty = min(ask_qty, self.max_qty)
        return bid_target, ask_target, bid_qty, ask_qty, center, half_spread
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ULTRA-FAST HOT PATH
    # ═══════════════════════════════════════════════════════════════════════════
    def on_orderbook_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        is_buy = (side == Side.BUY) if hasattr(side, 'name') else str(side).upper() == 'BUY'
        self.orderbook.update_level(is_buy, float(price), float(quantity))
        
        self._refill_tokens()
        if self.rl_tokens < 2.0:  # need 2 tokens for bid/ask maintenance
            return
        
        mid, spread = self.orderbook.get_mid_spread()
        if mid <= 0 or spread <= 0:
            return
        
        vw_bid, bid_vol = self.orderbook.get_vw_price(True,  self.depth)
        vw_ask, ask_vol = self.orderbook.get_vw_price(False, self.depth)
        if vw_bid <= 0 or vw_ask <= 0:
            return
        
        # Risk gate
        risk_level, scale_factor = self.risk_manager.quick_risk_check(self.position, mid)
        if risk_level >= 3:
            if self.bid_oid: self._cancel_fast(self.bid_oid); self.bid_oid = None
            if self.ask_oid: self._cancel_fast(self.ask_oid); self.ask_oid = None
            return
        
        # Rolling stats + regime
        if abs(mid - self.last_mid) > 1e-9:
            self.mid_stats.add_value(mid)
            self.last_mid = mid
            self._update_regime_fast()
        
        # Realize markout for aged fills vs current mid (EWMA toxicity)
        now_ts = time.time()
        while self._fills_ring and (now_ts - self._fills_ring[0][0]) > self.markout_horizon_sec:
            _, sgn, px_fill = self._fills_ring.popleft()
            mk = sgn * (mid - px_fill)  # >0 good, <0 toxic
            self._markout_ewma = (1 - self._markout_alpha) * self._markout_ewma + self._markout_alpha * mk
        
        # Base quote params
        bid_target, ask_target, bid_qty, ask_qty, center, half_spread = self._calculate_quote_params_fast(
            mid, spread, vw_bid, vw_ask, bid_vol, ask_vol
        )
        
        # Inventory-aware widen floor
        inv_widen_ticks = min(self.inv_widen_cap_ticks, abs(self.position) * self.inv_widen_ticks_per_unit)
        inv_widen_px = inv_widen_ticks * self.tick_size
        
        # Persistent asymmetric widening (decay + update from markout)
        # approximate 1s between meaningful updates; for higher accuracy keep a timestamp delta
        self._spread_penalty_px *= self._spread_penalty_decay
        neg = max(0.0, -self._markout_ewma)
        pos = max(0.0,  self._markout_ewma)
        self._spread_penalty_px += self.markout_widen_bad_k  * neg
        self._spread_penalty_px -= self.markout_widen_good_k * pos
        # clamp to sane range
        self._spread_penalty_px = max(0.0, min(self._spread_penalty_px, 8.0 * self.tick_size))
        
        # Microprice lean (top-of-book imbalance)
        best_bid = self.orderbook.bid_prices[0] if self.orderbook.bid_count > 0 else 0.0
        best_ask = self.orderbook.ask_prices[0] if self.orderbook.ask_count > 0 else 0.0
        bbq = self.orderbook.bid_qtys[0] if self.orderbook.bid_count > 0 else 0.0
        baq = self.orderbook.ask_qtys[0] if self.orderbook.ask_count > 0 else 0.0
        den = (bbq + baq) if (bbq + baq) > 0 else 1.0
        micro_center_shift = 0.0
        if best_bid > 0 and best_ask > 0:
            micro = (bbq * best_ask + baq * best_bid) / den
            center0 = 0.5 * (best_bid + best_ask)
            micro_center_shift = self.microprice_lean * (micro - center0)
        
        # Rebuild targets with penalty & inventory widen & micro lean
        center_adj = center + micro_center_shift
        half_adj   = half_spread + self._spread_penalty_px + inv_widen_px
        bid_target = max(vw_bid, center_adj - half_adj)
        ask_target = min(vw_ask, center_adj + half_adj)
        
        # Queue-aware shading
        tick = self.tick_size
        if bbq > self.queue_heavy_qty:
            bid_target = max(vw_bid, bid_target - 0.5 * tick)
        if baq > self.queue_heavy_qty:
            ask_target = min(vw_ask, ask_target + 0.5 * tick)
        # thin = ok to join as is
        
        # Risk scaling on sizes
        bid_qty *= scale_factor
        ask_qty *= scale_factor
        if bid_qty < 1.0 or ask_qty < 1.0:
            return
        
        # Round to ticks
        bid_target = round(bid_target / tick) * tick
        ask_target = round(ask_target / tick) * tick
        
        # Replace throttling helper
        now_ms = time.time() * 1000.0
        def throttle(side_name: str, prev_px: float, new_px: float) -> bool:
            if prev_px == 0.0: return True
            if abs(new_px - prev_px) < 0.25 * tick:
                if (now_ms - self._last_replace_ms[side_name]) < self.min_replace_ms:
                    return False
            return True
        
        # Bid replace
        if (self.bid_oid is None) or (abs(bid_target - self.last_bid_px) > 0.5 * tick and throttle("bid", self.last_bid_px, bid_target)):
            if self.bid_oid:
                self._cancel_fast(self.bid_oid); self.bid_oid = None
            new_oid = self._place_limit_fast(True, bid_qty, bid_target)
            if new_oid:
                self.bid_oid = new_oid; self.last_bid_px = bid_target; self._last_replace_ms["bid"] = now_ms
            else:
                self.bid_oid = None
        
        # Ask replace
        if (self.ask_oid is None) or (abs(ask_target - self.last_ask_px) > 0.5 * tick and throttle("ask", self.last_ask_px, ask_target)):
            if self.ask_oid:
                self._cancel_fast(self.ask_oid); self.ask_oid = None
            new_oid = self._place_limit_fast(False, ask_qty, ask_target)
            if new_oid:
                self.ask_oid = new_oid; self.last_ask_px = ask_target; self._last_replace_ms["ask"] = now_ms
            else:
                self.ask_oid = None
        
        # NOTE: mean-revert IOC exit intentionally disabled (EWMA-only model)
        # self._maybe_mean_revert_exit(mid)  # ← removed by design
    
    # Snapshot/init
    def on_orderbook_snapshot(self, ticker: Ticker, bids: list, asks: list) -> None:
        self._ticker = ticker
        self.orderbook.bid_count = 0
        self.orderbook.ask_count = 0
        for price, qty in bids:
            if self.orderbook.bid_count < self.orderbook.max_levels:
                self.orderbook.update_level(True,  float(price), float(qty))
        for price, qty in asks:
            if self.orderbook.ask_count < self.orderbook.max_levels:
                self.orderbook.update_level(False, float(price), float(qty))
    
    # Fills
    def on_order_filled(self, *args, **kwargs) -> None:
        if args and isinstance(args[0], dict):
            d = args[0]; side = d.get("side"); quantity = d.get("quantity"); price = d.get("price")
        else:
            side = kwargs.get("side"); quantity = kwargs.get("quantity"); price = kwargs.get("price")
        if side is None or quantity is None or price is None:
            return
        is_buy = (side == Side.BUY) if hasattr(side, 'name') else str(side).upper() == 'BUY'
        q = float(quantity); p = float(price)
        prev_pos = self.position
        self.position += q if is_buy else -q
        # avg price
        if prev_pos == 0.0:
            self.pos_avg_price = p
        elif (prev_pos > 0 and self.position > prev_pos) or (prev_pos < 0 and self.position < prev_pos):
            ap = abs(prev_pos); an = abs(self.position)
            if an > 1e-12: self.pos_avg_price = (self.pos_avg_price * ap + p * abs(an - ap)) / an
        elif (prev_pos > 0 and self.position < 0) or (prev_pos < 0 and self.position > 0):
            self.pos_avg_price = p
        # cash
        if is_buy: self.cash -= q * p
        else:      self.cash += q * p
        # record fill for markout realization
        try:
            self._fills_ring.append((time.time(), +1 if is_buy else -1, p))
        except Exception:
            pass
        # update capital proxy
        self.risk_manager.current_capital = self.cash + self.position * self.last_mid
        # reset avg if flat
        if abs(self.position) < 1e-9:
            self.pos_avg_price = 0.0
    
    # Account updates
    def on_account_update(self, *args, **kwargs) -> None:
        payload = args[0] if args and isinstance(args[0], dict) else kwargs
        if "cash" in payload:     self.cash = float(payload["cash"])
        if "position" in payload: self.position = float(payload["position"])
        self.risk_manager.current_capital = self.cash + self.position * self.last_mid
    
    # Unused hooks kept for compatibility (no overhead)
    def on_rejected(self, *args, **kwargs): pass
    def on_game_event_update(self, *args, **kwargs): pass
    def on_trade_update(self, *args, **kwargs): pass
    def on_order_update(self, *args, **kwargs): pass
    def on_pnl_update(self, *args, **kwargs): pass

# Compatibility for environments expecting `Strategy`
class Strategy(FastStrategy):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# (Optional) perf harness for local testing
# ─────────────────────────────────────────────────────────────────────────────
def _profile_memory_usage():
    import sys
    orderbook = FastOrderbook(max_levels=50)
    stats = FastRollingStats(window=60)
    risk_mgr = FastRiskManager()
    strategy = FastStrategy()
    for i in range(100):
        orderbook.update_level(True, 100.0 - i*0.1, 10.0)
        orderbook.update_level(False, 100.1 + i*0.1, 10.0)
        stats.add_value(100.0 + i*0.01)
    for name, obj in [("FastOrderbook", orderbook), ("FastRollingStats", stats),
                      ("FastRiskManager", risk_mgr), ("FastStrategy", strategy)]:
        print(f"{name}: {sys.getsizeof(obj)} bytes")

def _run_benchmark():
    import random
    s = FastStrategy()
    bids = [[100.0 - i*0.5, np.random.uniform(5, 20)] for i in range(10)]
    asks = [[100.5 + i*0.5, np.random.uniform(5, 20)] for i in range(10)]
    s.on_orderbook_snapshot(Ticker.TEAM_A, bids, asks)
    # warm
    for _ in range(200):
        isb = random.choice([True, False])
        side = Side.BUY if isb else Side.SELL
        px = 100.0 - np.random.randint(0, 9) * 0.5 if isb else 100.5 + np.random.randint(0, 9) * 0.5
        qty = np.random.uniform(0, 25)
        s.on_orderbook_update(Ticker.TEAM_A, side, qty, px)

if __name__ == "__main__":
    # Local-only diagnostics (safe for linter imports)
    _profile_memory_usage()
    _run_benchmark()

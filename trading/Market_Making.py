"""
Quant Challenge 2025

HF scalp market-maker – fast exits, tiny edges, strict risk/time stops.
Designed to be lint-clean and engine-friendly:
- Provides on_account_update / on_fill_update / on_orderbook_update / on_trade_update / on_game_event_update.
- Flexible callback signatures for account/fill updates.
- Full local orderbook; quotes 1 bid + 1 ask inside the spread using VW depth.
- Immediate IOC take-profit on fill, adverse-move cut, and max-hold timeout flatten.
- Size leaning by imbalance + optional momentum nudge.
- Strict 30 orders/min rate limit (place + cancel count).
- Rollover handling with optional flatten + cooldown.
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


# Stubs so the module imports cleanly; engine binds real functions at runtime.
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
    HF scalp market-maker.

    Key ideas:
      • Build/maintain a local orderbook (price->qty + sorted price lists).
      • Quote narrow bid/ask inside the VW spread; lean sizes by imbalance.
      • After any fill: fire an IOC scalp exit at tiny take-profit ticks.
      • Cut losers fast if mid moves adversely; hard timeout on inventory.
      • Momentum gate can nudge quotes a tick with drift.
      • Respect 30 orders/min including cancels via a token bucket.
      • Handle game rollovers (cooldown + optional flatten).
    """

    # ───────── Lifecycle ─────────
    def __init__(self) -> None:
        # engine bindings (resolved lazily)
        self._Side: Optional[type[Side]] = None
        self._Ticker: Optional[type[Ticker]] = None
        self._fn_place_market = None
        self._fn_place_limit = None
        self._fn_cancel = None

        # identity / latest ticker
        self._ticker: Optional[Ticker] = None

        # local orderbook
        self._bids: Dict[float, float] = {}
        self._asks: Dict[float, float] = {}
        self._bid_prices: List[float] = []   # sorted desc
        self._ask_prices: List[float] = []   # sorted asc
        self._best_bid: Optional[Tuple[float, float]] = None
        self._best_ask: Optional[Tuple[float, float]] = None

        # our resting quotes
        self._bid_oid: Optional[int] = None
        self._ask_oid: Optional[int] = None
        self._last_bid_px: Optional[float] = None
        self._last_ask_px: Optional[float] = None

        # inventory / PnL
        self.position: float = 0.0
        self.cash: float = 0.0
        self.capital_remaining: Optional[float] = None
        self.realized_pnl: float = 0.0
        self.unrealized_pnl: float = 0.0
        self._inventory_timestamp: Optional[float] = None
        self._avg_entry_price: Optional[float] = None

        # momentum / mid history
        self._mid_hist: deque[float] = deque(maxlen=256)

        # rollover state
        self._cooldown_until: float = 0.0
        self._last_score_sum: Optional[int] = None
        self._delta_window: deque[float] = deque(maxlen=60)
        self._last_event_ts: Optional[float] = None

        # rate limit (token bucket)
        self._rl_capacity: float = 30.0
        self._rl_tokens: float = self._rl_capacity
        self._rl_refill_per_sec: float = 30.0 / 60.0
        self._rl_last_t: float = time.monotonic()

        # ── Tunables ─────────────────────────────────────────────────────────
        self.tick_size: float = 0.5                # adjust to venue
        self.depth: int = 3                        # VW over top-N levels
        self.inside_frac: float = 0.25             # quote inside % of spread
        self.price_epsilon: float = 0.01           # min delta to reprice

        self.base_qty: float = 5.0                 # baseline order size
        self.max_qty: float = 25.0                 # cap per quote
        self.inventory_limit: float = 100.0        # hard position cap

        # scalp exits
        self.take_profit_ticks: float = 1.0        # TP distance from fill
        self.adverse_exit_ticks: float = 2.0       # cut if mid moves this many ticks against
        self.max_hold_seconds: float = 3.0         # force-exit inventory older than this

        # momentum gating (optional)
        self.use_momentum: bool = True
        self.momo_window: int = 15                 # updates in window
        self.momo_threshold_ticks: float = 0.5     # require ≥ this many ticks drift

        # rollover handling
        self.flatten_on_rollover: bool = True
        self.rollover_cooldown_sec: float = 4.0

        # protective controls
        self.kill_switch_enabled: bool = True
        self.kill_switch_drawdown: float = -1000.0  # stop if cash PnL < this

    # ───────── Engine resolution helpers ─────────
    def _ensure_api(self) -> None:
        g = globals()
        if self._Side is None:
            self._Side = g.get("Side")
        if self._Ticker is None:
            self._Ticker = g.get("Ticker")
        if self._fn_place_market is None:
            self._fn_place_market = g.get("place_market_order")
        if self._fn_place_limit is None:
            self._fn_place_limit = g.get("place_limit_order")
        if self._fn_cancel is None:
            self._fn_cancel = g.get("cancel_order")

    def _ensure_ticker(self) -> None:
        if self._ticker is None:
            self._ensure_api()
            if self._Ticker is not None and hasattr(self._Ticker, "TEAM_A"):
                self._ticker = getattr(self._Ticker, "TEAM_A")

    # ───────── Rate limit: 30/min (token bucket) ─────────
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
    def _clamp(v: float, lo: float, hi: float) -> float:
        return lo if v < lo else hi if v > hi else v

    def _round_tick(self, px: float) -> float:
        t = self.tick_size
        # keep rounding stable and numeric
        return round(round(px / t) * t, 8)

    # ───────── Book builders ─────────
    def _rebuild_books(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> None:
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
        p = float(price); q = float(quantity)
        is_bid = (side == Side.BUY) if isinstance(side, Side) else (getattr(side, "name", str(side)).upper() in ("BUY", "BID"))
        book = self._bids if is_bid else self._asks
        plist = self._bid_prices if is_bid else self._ask_prices

        existed = p in book
        if q <= 0:
            if existed:
                del book[p]
                try:
                    plist.remove(p)
                except ValueError:
                    pass
        else:
            book[p] = q
            if not existed:
                plist.append(p)
                plist.sort(reverse=is_bid)

        self._best_bid = (self._bid_prices[0], self._bids[self._bid_prices[0]]) if self._bid_prices else None
        self._best_ask = (self._ask_prices[0], self._asks[self._ask_prices[0]]) if self._ask_prices else None

    # ───────── Order helpers (ratelimited, safe) ─────────
    def _place_market(self, is_buy: bool, qty: float) -> bool:
        self._ensure_api(); self._ensure_ticker()
        if not callable(self._fn_place_market) or self._ticker is None:
            return False
        if not self._rl_allow(1.0):
            return False
        side_obj = self._Side.BUY if is_buy else self._Side.SELL if self._Side is not None else ("BUY" if is_buy else "SELL")
        try:
            return bool(self._fn_place_market(side_obj, self._ticker, float(qty)))
        except Exception:
            return False

    def _place_limit(self, is_buy: bool, qty: float, px: float, ioc: bool = False) -> Optional[int]:
        self._ensure_api(); self._ensure_ticker()
        if not callable(self._fn_place_limit) or self._ticker is None:
            return None
        if not self._rl_allow(1.0):
            return None
        side_obj = self._Side.BUY if is_buy else self._Side.SELL if self._Side is not None else ("BUY" if is_buy else "SELL")
        try:
            return self._fn_place_limit(side_obj, self._ticker, float(qty), float(px), bool(ioc))
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
            self._fn_cancel(self._ticker, int(oid))
        except Exception:
            pass

    # ───────── Mid / spread helpers ─────────
    def _get_mid_spread(self) -> Tuple[Optional[float], Optional[float]]:
        if self._best_bid and self._best_ask:
            bb = self._best_bid[0]; ba = self._best_ask[0]
            mid = 0.5 * (bb + ba)
            spr = max(self._round_tick(ba - bb), self.tick_size)
            return mid, spr
        return None, None

    def _vw_best(self, prices: List[float], book: Dict[float, float], take_n: int) -> Tuple[float, float]:
        if not prices:
            return (0.0, 0.0)
        vol = 0.0
        notional = 0.0
        for p in prices[:take_n]:
            q = book.get(p, 0.0)
            vol += q
            notional += p * q
        if vol <= 0:
            return (0.0, 0.0)
        return (notional / vol, vol)

    # ───────── Momentum filter ─────────
    def _momo_gate(self, mid: float) -> Tuple[bool, float]:
        self._mid_hist.append(mid)
        if not self.use_momentum or len(self._mid_hist) < self.momo_window:
            return True, 0.0
        prev = list(self._mid_hist)[-self.momo_window]
        drift_ticks = (mid - prev) / self.tick_size
        if abs(drift_ticks) >= self.momo_threshold_ticks:
            return True, drift_ticks
        return True, 0.0  # soft gate; we still trade

    # ───────── Risk / exits ─────────
    def _mark_inventory_entry(self, entry_px: float) -> None:
        self._inventory_timestamp = time.time()
        self._avg_entry_price = entry_px

    def _inventory_age(self) -> float:
        if not self._inventory_timestamp:
            return 0.0
        return max(0.0, time.time() - self._inventory_timestamp)

    def _adverse_cut_needed(self, mid: float) -> bool:
        if self.position == 0 or self._avg_entry_price is None:
            return False
        dt = self.adverse_exit_ticks * self.tick_size
        if self.position > 0:
            return mid <= (self._avg_entry_price - dt)
        return mid >= (self._avg_entry_price + dt)

    def _force_flatten_needed(self) -> bool:
        return (self.position != 0) and (self._inventory_age() >= self.max_hold_seconds)

    def _try_flatten_fast(self, mid: float) -> None:
        if self.position == 0:
            return
        qty = abs(self.position)
        self._place_market(is_buy=(self.position < 0), qty=qty)
        # Approximate realized PnL using mid; engine's actual fill may differ.
        if self._avg_entry_price is not None:
            pnl = (mid - self._avg_entry_price) * (-self.position)
            self.cash += pnl
        self.position = 0.0
        self._inventory_timestamp = None
        self._avg_entry_price = None

    # ───────── Exchange callbacks ─────────
    def on_trade_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        # keep ticker fresh; trading logic is driven by book updates
        self._ticker = ticker
        if time.time() < self._cooldown_until:
            return

    def on_orderbook_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        self._ticker = ticker
        self._apply_level_update(side, quantity, price)

        # build book during cooldown; no quoting
        if time.time() < self._cooldown_until:
            return

        if not (self._bid_prices and self._ask_prices):
            return

        vwbid, bid_vol = self._vw_best(self._bid_prices, self._bids, self.depth)
        vwask, ask_vol = self._vw_best(self._ask_prices, self._asks, self.depth)
        if vwbid <= 0 or vwask <= 0:
            return

        mid = 0.5 * (vwbid + vwask)
        spread = max(self._round_tick(vwask - vwbid), self.tick_size)
        half = self.inside_frac * spread
        target_bid_px = self._round_tick(max(vwbid, mid - half))
        target_ask_px = self._round_tick(min(vwask, mid + half))

        # record mid for momentum & controls
        self._mid_hist.append(mid)
        _, drift_ticks = self._momo_gate(mid)

        # size leaning by displayed imbalance
        tot = bid_vol + ask_vol
        imb = ((bid_vol - ask_vol) / tot) if tot > 0 else 0.0  # [-1, 1]
        bid_qty = self._clamp(self.base_qty * (1.0 + max(0.0, -imb)), 1.0, self.max_qty)
        ask_qty = self._clamp(self.base_qty * (1.0 + max(0.0,  imb)), 1.0, self.max_qty)

        # momentum nudge (slight chase in drift direction)
        if self.use_momentum and drift_ticks != 0.0:
            if drift_ticks > 0:  # up drift
                target_bid_px = self._round_tick(min(target_bid_px + self.tick_size, mid))
                ask_qty *= 0.75
            else:               # down drift
                target_ask_px = self._round_tick(max(target_ask_px - self.tick_size, mid))
                bid_qty *= 0.75

        # inventory guards
        if abs(self.position) > 0.5 * self.inventory_limit:
            if self.position > 0:
                bid_qty *= 0.5
                target_bid_px = self._round_tick(target_bid_px - self.tick_size)
            else:
                ask_qty *= 0.5
                target_ask_px = self._round_tick(target_ask_px + self.tick_size)

        # exits first
        if self._adverse_cut_needed(mid) or self._force_flatten_needed():
            self._try_flatten_fast(mid)
            if self._bid_oid is not None:
                self._cancel(self._bid_oid); self._bid_oid = None
            if self._ask_oid is not None:
                self._cancel(self._ask_oid); self._ask_oid = None
            self._last_bid_px = None; self._last_ask_px = None

        # replace only when needed
        def needs_replace(prev_px: Optional[float], new_px: float) -> bool:
            if prev_px is None:
                return True
            eps = max(self.price_epsilon, 0.5 * self.tick_size)
            return abs(prev_px - new_px) > eps

        # manage bid
        if needs_replace(self._last_bid_px, target_bid_px):
            if self._bid_oid is not None:
                self._cancel(self._bid_oid); self._bid_oid = None
            if self.position < self.inventory_limit:
                oid = self._place_limit(is_buy=True, qty=bid_qty, px=target_bid_px, ioc=False)
                if oid is not None:
                    self._bid_oid = oid
                    self._last_bid_px = target_bid_px

        # manage ask
        if needs_replace(self._last_ask_px, target_ask_px):
            if self._ask_oid is not None:
                self._cancel(self._ask_oid); self._ask_oid = None
            if -self.position < self.inventory_limit:
                oid = self._place_limit(is_buy=False, qty=ask_qty, px=target_ask_px, ioc=False)
                if oid is not None:
                    self._ask_oid = oid
                    self._last_ask_px = target_ask_px

        # kill switch on deep drawdown
        if self.kill_switch_enabled and self.cash <= self.kill_switch_drawdown:
            if self._bid_oid is not None:
                self._cancel(self._bid_oid); self._bid_oid = None
            if self._ask_oid is not None:
                self._cancel(self._ask_oid); self._ask_oid = None
            self._cooldown_until = time.time() + 10.0

    # fills / account callbacks (flexible signatures for engine compatibility)
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

        prev_pos = self.position
        if is_buy:
            self.position += q
            self.cash -= q * p
        else:
            self.position -= q
            self.cash += q * p

        # inventory bookkeeping and scalp IOC exit
        if prev_pos == 0.0 and self.position != 0.0:
            self._mark_inventory_entry(entry_px=p)
        elif self.position == 0.0:
            if self._avg_entry_price is not None:
                closed_qty = abs(prev_pos)
                # realized profit approx: (exit - entry) * signed close
                pnl = (p - self._avg_entry_price) * (closed_qty if not is_buy else -closed_qty)
                self.cash += pnl
            self._avg_entry_price = None
            self._inventory_timestamp = None

        if self.position != 0:
            tp = self.take_profit_ticks * self.tick_size
            is_buy_exit = (self.position < 0)
            target_px = self._round_tick(p - tp if is_buy_exit else p + tp)
            exit_qty = min(abs(self.position), self.base_qty)
            self._place_limit(is_buy=is_buy_exit, qty=exit_qty, px=target_px, ioc=True)

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

    def on_game_event_update(
        self,
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
        time_seconds: Optional[float],
    ) -> None:
        # explicit end marker / hard rollover trigger if present
        if event_type == "END_GAME":
            self._handle_rollover("END_GAME event")
            return

        now = time.time()

        # During cooldown: just keep baselines fresh
        if now < self._cooldown_until:
            self._last_score_sum = int(home_score) + int(away_score)
            self._last_event_ts = now
            return

        curr_sum = int(home_score) + int(away_score)

        # score-drop rollover detector (robust to nonzero resets)
        if self._last_score_sum is not None:
            delta = curr_sum - self._last_score_sum  # negative on drop
            neg_jump = delta < 0

            # simple rolling stats for z-score
            if len(self._delta_window) >= 10:
                mean_d = sum(self._delta_window) / len(self._delta_window)
                var_d = sum((d - mean_d) ** 2 for d in self._delta_window) / max(1, len(self._delta_window) - 1)
                std_d = max(1e-6, var_d ** 0.5)
            else:
                mean_d, std_d = 0.0, 1.0

            drop_abs = -delta
            drop_frac = drop_abs / max(1.0, float(self._last_score_sum))
            z = (delta - mean_d) / std_d  # negative when unusually large drop

            if neg_jump and (drop_abs >= 20 or drop_frac >= 0.40 or z <= -4.0):
                self._handle_rollover(f"score negative jump Δ={delta:.0f}, frac={drop_frac:.2f}, z={z:.1f}")
                self._last_score_sum = curr_sum
                self._last_event_ts = now
                return

            self._delta_window.append(delta)
        else:
            self._delta_window.append(0.0)

        self._last_score_sum = curr_sum
        self._last_event_ts = now

    # ───────── Rollover helpers ─────────
    def _handle_rollover(self, reason: str) -> None:
        # cancel resting quotes
        if self._bid_oid is not None:
            self._cancel(self._bid_oid)
            self._bid_oid = None
        if self._ask_oid is not None:
            self._cancel(self._ask_oid)
            self._ask_oid = None

        # optional flatten
        if self.flatten_on_rollover and abs(self.position) > 0:
            # pick a sensible mid proxy if book unknown
            mid = 0.0
            if self._best_bid and self._best_ask:
                mid = 0.5 * (self._best_bid[0] + self._best_ask[0])
            self._try_flatten_fast(mid=mid)

        # reset local book & timers
        self._bids.clear(); self._asks.clear()
        self._bid_prices.clear(); self._ask_prices.clear()
        self._best_bid = None; self._best_ask = None
        self._last_bid_px = None; self._last_ask_px = None
        self._mid_hist.clear()

        self._cooldown_until = time.time() + float(self.rollover_cooldown_sec)
        self._last_event_ts = None

    # ───────── External hook ─────────
    def notify_game_rollover(self, reason: str = "score reset / new game") -> None:
        self._handle_rollover(reason)

"""
Quant Challenge 2025

Algorithmic strategy – VW spread maker with full local orderbook and 30/min rate limit
Game-theory + session control upgrades:
- queue-aware quoting
- microprice & toxicity gating
- inventory-aware skew
- regime ladder (stable/normal/volatile)
- anti-spoof persistence weighting
- jittered, hysteretic replaces to avoid being farmed
- rollover hardening + kill-switches
- pre-close flatten before game end (clock-based)
- optional opening probe trade on new game start
"""

from __future__ import annotations
import time
import random
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
    """Volume-weighted spread maker, hardened for adversarial bots, with session controls."""

    # ───────── Lifecycle ─────────
    def __init__(self) -> None:
        self.reset_state()

    def reset_state(self) -> None:
        # Engine enums/functions resolved at runtime
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

        # Per-price persistence (anti-spoof)
        # key: (side_str, price) -> dict(last_seen, present, score)
        self._level_meta: Dict[Tuple[str, float], Dict[str, float]] = {}

        # Our resting orders (one bid + one ask) and last posted prices
        self._bid_oid: Optional[int] = None
        self._ask_oid: Optional[int] = None
        self._last_bid_px: Optional[float] = None
        self._last_ask_px: Optional[float] = None
        self._last_replace_ts = {"bid": 0.0, "ask": 0.0}

        # Position / PnL (tracked on our fills)
        self.position: float = 0.0
        self.cash: float = 0.0
        self.capital_remaining: Optional[float] = None
        self.realized_pnl: float = 0.0
        self.unrealized_pnl: float = 0.0

        # Risk limits
        self.pos_max: float = 200.0              # hard inventory cap (tune to product)
        self.drawdown_halt: float = -700.0       # stop if realized pnl below this since start (currency units)

        # Rollover detection state
        self._last_score_sum: Optional[int] = None
        self._delta_window: deque = deque(maxlen=60)  # rolling deltas (score or synthetic flow)
        self._cooldown_until: float = 0.0
        self._last_event_ts: Optional[float] = None

        # Rollover knobs
        self.rollover_cooldown_sec: float = 6.0
        self.min_drop: int = 20
        self.frac_drop: float = 0.40
        self.z_sigma: float = 4.0
        self.flatten_on_rollover: bool = True

        # Quoting params
        self.depth: int = 5                 # VW over top N levels
        self.inside_frac: float = 0.25      # fraction of spread toward mid
        self.base_qty: float = 5.0          # base order size
        self.max_qty: float = 20.0          # cap by imbalance scaling
        self.tick_size: float = 0.5
        self.price_epsilon: float = 0.01
        self._did_sanity: bool = False
        self._sanity_qty: float = 1.0

        # Toxicity/queue gates
        self.toxicity_tau: float = 0.10     # post filters: tox > tau blocks bid, tox < -tau blocks ask
        self.queue_join_max_frac: float = 0.8  # if our size would be >80% of displayed at top, skip

        # Replace throttles
        self.min_replace_dt: float = 0.7     # per-side throttle
        self.cancel_then_place: bool = True  # batch cancels then places

        # IOC feeler probability
        self.feeler_prob: float = 0.05
        self.feeler_qty: float = 0.5

        # Rate limit (leaky bucket)
        self._rl_capacity = 30.0
        self._rl_tokens = self._rl_capacity
        self._rl_refill_per_sec = 30.0 / 60.0
        self._rl_last_t = time.monotonic()

        # ───────── Session control: pre-close & new-game probe ─────────
        # If the feed provides time_seconds as seconds remaining in the game,
        # we can arm a pre-close flatten a few seconds before 0.
        self.preclose_flatten_enabled: bool = True
        self.preclose_flatten_sec: float = 8.0         # flatten when time_left <= this (seconds)
        self.preclose_cooldown_sec: float = 5.0        # brief freeze after pre-close flatten
        self._preclose_done: bool = False              # per-game flag to avoid repeats

        # Optionally place a tiny probe at the start of a new game (after rollover cooldown).
        # Set side to "BUY", "SELL", or "NONE".
        self.open_probe_side: str = "BUY"
        self.open_probe_qty: float = 1.0
        self.open_probe_enabled: bool = False
        self._opened_this_game: bool = False           # reset at rollover

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

    def _mk_side(self, want_buy: bool):
        # Robust to stubbed or Enum side
        if self._Side:
            return self._Side.BUY if want_buy else self._Side.SELL
        return "BUY" if want_buy else "SELL"

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

    # ───────── Book builders ─────────
    def _rebuild_books(self, bids: list, asks: list) -> None:
        """Rebuild the full local book from snapshot arrays of [price, qty]."""
        self._bids.clear(); self._asks.clear()
        self._bid_prices.clear(); self._ask_prices.clear()

        for p, q in bids:
            p = float(p); q = float(q)
            if q > 0:
                self._bids[p] = q; self._bid_prices.append(p)
                self._touch_level("BUY", p, present=True)
        for p, q in asks:
            p = float(p); q = float(q)
            if q > 0:
                self._asks[p] = q; self._ask_prices.append(p)
                self._touch_level("SELL", p, present=True)

        self._bid_prices.sort(reverse=True)
        self._ask_prices.sort()

        self._best_bid = (self._bid_prices[0], self._bids[self._bid_prices[0]]) if self._bid_prices else None
        self._best_ask = (self._ask_prices[0], self._asks[self._ask_prices[0]]) if self._ask_prices else None

    def _apply_level_update(self, side: Side, quantity: float, price: float) -> None:
        """Apply incremental (side, quantity, price) to our local book."""
        p = float(price); q = float(quantity)
        is_bid = (side == Side.BUY) if isinstance(side, Side) else (getattr(side, "name", str(side)).upper() in ("BUY","BID"))
        side_str = "BUY" if is_bid else "SELL"
        book = self._bids if is_bid else self._asks
        plist = self._bid_prices if is_bid else self._ask_prices

        existed = p in book
        if q <= 0:
            if existed:
                del book[p]
                try: plist.remove(p)
                except ValueError: pass
            # mark absence to decay persistence
            self._touch_level(side_str, p, present=False)
        else:
            book[p] = q
            if not existed:
                plist.append(p)
                plist.sort(reverse=is_bid)
            self._touch_level(side_str, p, present=True)

        # keep bests fresh
        self._best_bid = (self._bid_prices[0], self._bids[self._bid_prices[0]]) if self._bid_prices else None
        self._best_ask = (self._ask_prices[0], self._asks[self._ask_prices[0]]) if self._ask_prices else None

    # ───────── Level persistence (anti-spoof) ─────────
    def _touch_level(self, side_str: str, price: float, present: bool) -> None:
        now = time.time()
        k = (side_str, float(price))
        meta = self._level_meta.get(k)
        if meta is None:
            meta = {"last_seen": now, "present": 1.0 if present else 0.0, "score": 0.5}
            self._level_meta[k] = meta
            return
        # EWMA update with time-aware decay
        dt = max(0.0, now - meta["last_seen"])
        alpha = 1.0 - pow(0.5, dt / 0.5)  # half-life 0.5s for presence memory
        target = 1.0 if present else 0.0
        meta["score"] = (1 - alpha) * meta["score"] + alpha * target
        meta["present"] = 1.0 if present else 0.0
        meta["last_seen"] = now

    def _persistence_score(self, side_str: str, price: float) -> float:
        meta = self._level_meta.get((side_str, float(price)))
        if not meta:
            return 0.5
        return self._clamp(meta["score"], 0.05, 1.0)

    # ───────── Microprice / toxicity / regime / queue helpers ─────────
    def _micro_price(self) -> Tuple[float, float, float]:
        if not (self._best_bid and self._best_ask):
            return (0.0, 0.0, 0.0)
        bb, qb = self._best_bid; ba, qa = self._best_ask
        mid = (bb + ba) / 2.0
        tot = max(1e-9, (qb + qa))
        micro = (ba * qb + bb * qa) / tot
        spread = max(self.tick_size, ba - bb)
        skew = (micro - mid) / spread  # [-0.5, 0.5] roughly
        return micro, mid, skew

    def _toxicity(self) -> float:
        micro, mid, skew = self._micro_price()
        d = list(self._delta_window)
        last_n = d[-10:] if len(d) >= 10 else d
        neg_share = sum(1 for x in last_n if x < 0) / max(1, len(last_n))
        wide = 1.0 if (self._best_ask and self._best_bid and (self._best_ask[0] - self._best_bid[0] >= 2 * self.tick_size)) else 0.0
        tox = 0.6 * skew + 0.3 * (neg_share - 0.5) + 0.1 * wide
        return self._clamp(tox, -1.0, 1.0)

    def _queue_rank(self, want_buy: bool) -> float:
        book = self._bids if want_buy else self._asks
        prices = self._bid_prices if want_buy else self._ask_prices
        if not prices:
            return 1.0
        top = prices[0]; top_sz = book.get(top, 0.0)
        frac = self.base_qty / max(1e-6, top_sz)
        return self._clamp(frac, 0.0, 1.0)

    def _risk_skew(self) -> float:
        k = 0.0075  # risk aversion per unit position (tune)
        return self._clamp(-k * self.position, -0.5, 0.5)

    def _regime(self) -> str:
        if not (self._best_bid and self._best_ask):
            return "idle"
        s = self._best_ask[0] - self._best_bid[0]
        churn = sum(1 for d in list(self._delta_window)[-20:] if d != 0)
        if s >= 2 * self.tick_size or churn > 10:
            return "volatile"
        elif s <= self.tick_size and churn < 5:
            return "stable"
        return "normal"

    # Depth-adaptive VW using persistence caps
    def _vw(self, book: Dict[float, float], prices: List[float], take_n: int, side_str: str, cap: float = 10.0) -> Tuple[float, float]:
        notional = 0.0; vol = 0.0
        for p in prices[:take_n]:
            q_raw = book.get(p, 0.0)
            persist = self._persistence_score(side_str, p)
            q = min(q_raw, cap) * persist
            if q <= 0:
                continue
            notional += p * q
            vol += q
        if vol <= 0:
            return (0.0, 0.0)
        return (notional / vol, vol)

    # ───────── Order helpers ─────────
    def _place_market(self, want_buy: bool, qty: float) -> bool:
        self._ensure_api(); self._ensure_ticker()
        if not callable(self._fn_place_market) or self._ticker is None:
            return False
        if not self._rl_allow(1.0):
            return False
        side = self._mk_side(want_buy)
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
        side = self._mk_side(want_buy)
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

    def _should_replace(self, prev_px: Optional[float], new_px: float, side_key: str, now: float) -> bool:
        if prev_px is None:
            return True
        eps = max(self.price_epsilon, self.tick_size)  # hysteresis
        moved = abs(prev_px - new_px) > eps
        ok_time = (now - self._last_replace_ts.get(side_key, 0.0)) > self.min_replace_dt
        jitter_ok = random.random() > 0.15  # break predictability
        if moved and ok_time and jitter_ok:
            self._last_replace_ts[side_key] = now
            return True
        return False

    # ───────── Kill switches ─────────
    def _risk_halts(self) -> bool:
        if abs(self.position) > self.pos_max:
            return True
        if self.realized_pnl <= self.drawdown_halt:
            return True
        return False

    # ───────── Pre-close flatten ─────────
    def _preclose_flatten(self, reason: str) -> None:
        # Pull resting quotes
        if self._bid_oid is not None:
            self._cancel(self._bid_oid); self._bid_oid = None
        if self._ask_oid is not None:
            self._cancel(self._ask_oid); self._ask_oid = None
        # Flatten inventory via market orders (best-effort)
        try:
            if abs(self.position) > 0.0:
                qty = abs(self.position)
                if self.position > 0:
                    self._place_market(False, qty)
                else:
                    self._place_market(True, qty)
        except Exception:
            pass
        # Short cooldown to avoid last-second toxicity
        self._cooldown_until = time.time() + float(self.preclose_cooldown_sec)
        self._preclose_done = True
        # print(f"[preclose] flattened due to {reason}")

    # ───────── Rollover helpers ─────────
    def _handle_rollover(self, reason: str) -> None:
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

        # Reset per-game flags
        self._preclose_done = False
        self._opened_this_game = False
        # print(f"[rollover] {reason} | cooldown={self.rollover_cooldown_sec}s")

    # ───────── Decision loop ─────────
    def _decide_and_quote(self) -> None:
        # Do nothing during cooldown or if risk halt engaged
        if time.time() < getattr(self, "_cooldown_until", 0.0):
            return
        if self._risk_halts():
            if self._bid_oid is not None:
                self._cancel(self._bid_oid); self._bid_oid = None
            if self._ask_oid is not None:
                self._cancel(self._ask_oid); self._ask_oid = None
            return

        if not (self._bid_prices and self._ask_prices):
            return

        # Compute persistence-weighted VW bid/ask and volumes
        vwbid, bid_vol = self._vw(self._bids, self._bid_prices, self.depth, "BUY")
        vwask, ask_vol = self._vw(self._asks, self._ask_prices, self.depth, "SELL")
        if not (vwbid > 0 and vwask > 0):
            return

        mid = (vwbid + vwask) / 2.0
        spread_raw = max(self.tick_size, vwask - vwbid)
        half = self.inside_frac * spread_raw

        target_bid_px = self._round_tick(max(vwbid, mid - half))
        target_ask_px = self._round_tick(min(vwask, mid + half))

        # Size leaning by displayed imbalance
        tot = bid_vol + ask_vol
        imb = ((bid_vol - ask_vol) / tot) if tot > 0 else 0.0      # [-1, 1]
        bid_qty = self._clamp(self.base_qty * (1.0 + max(0.0, -imb)), 1.0, self.max_qty)
        ask_qty = self._clamp(self.base_qty * (1.0 + max(0.0,  imb)), 1.0, self.max_qty)

        # Risk skew on price and size
        r = self._risk_skew()
        target_bid_px = self._round_tick(target_bid_px - r * self.tick_size)
        target_ask_px = self._round_tick(target_ask_px + r * self.tick_size)
        if r > 0:   # long; prefer to sell
            bid_qty *= (1.0 - min(0.5, r))
            ask_qty *= (1.0 + min(0.5, r))
        elif r < 0: # short; prefer to buy
            bid_qty *= (1.0 + min(0.5, -r))
            ask_qty *= (1.0 - min(0.5, -r))

        # Regime-dependent inside fraction and size adjustments
        reg = self._regime()
        if reg == "volatile":
            self.inside_frac = 0.45
            half = self.inside_frac * spread_raw
            target_bid_px = self._round_tick(max(vwbid, mid - half))
            target_ask_px = self._round_tick(min(vwask, mid + half))
            bid_qty *= 0.6; ask_qty *= 0.6
        elif reg == "stable":
            self.inside_frac = 0.20
            half = self.inside_frac * spread_raw
            target_bid_px = self._round_tick(max(vwbid, mid - half))
            target_ask_px = self._round_tick(min(vwask, mid + half))
        else:
            self.inside_frac = 0.25

        # Toxicity and queue rank filters
        tox = self._toxicity()
        post_bid = tox <= self.toxicity_tau
        post_ask = tox >= -self.toxicity_tau

        if self._queue_rank(True) > self.queue_join_max_frac:
            post_bid = False
        if self._queue_rank(False) > self.queue_join_max_frac:
            post_ask = False

        now = time.time()

        # Apply with hysteresis + rate limit budgeting (cancel-then-place batches)
        if self.cancel_then_place:
            # Bid side
            if post_bid and self._should_replace(self._last_bid_px, target_bid_px, "bid", now):
                if self._bid_oid is not None:
                    self._cancel(self._bid_oid); self._bid_oid = None
                oid = self._place_limit(True, bid_qty, target_bid_px, ioc=False)
                if oid is not None:
                    self._bid_oid = oid
                    self._last_bid_px = target_bid_px
            elif not post_bid and self._bid_oid is not None:
                self._cancel(self._bid_oid); self._bid_oid = None

            # Ask side
            if post_ask and self._should_replace(self._last_ask_px, target_ask_px, "ask", now):
                if self._ask_oid is not None:
                    self._cancel(self._ask_oid); self._ask_oid = None
                oid = self._place_limit(False, ask_qty, target_ask_px, ioc=False)
                if oid is not None:
                    self._ask_oid = oid
                    self._last_ask_px = target_ask_px
            elif not post_ask and self._ask_oid is not None:
                self._cancel(self._ask_oid); self._ask_oid = None
        else:
            # Simple replace path
            if post_bid and self._should_replace(self._last_bid_px, target_bid_px, "bid", now):
                oid = self._place_limit(True, bid_qty, target_bid_px, ioc=False)
                if oid is not None:
                    if self._bid_oid is not None:
                        self._cancel(self._bid_oid)
                    self._bid_oid = oid
                    self._last_bid_px = target_bid_px
            if post_ask and self._should_replace(self._last_ask_px, target_ask_px, "ask", now):
                oid = self._place_limit(False, ask_qty, target_ask_px, ioc=False)
                if oid is not None:
                    if self._ask_oid is not None:
                        self._cancel(self._ask_oid)
                    self._ask_oid = oid
                    self._last_ask_px = target_ask_px

        # Occasional IOC feeler on strong signal flips
        if random.random() < self.feeler_prob:
            if tox < -0.25 and post_bid is False and self._best_ask:
                self._place_limit(True, self.feeler_qty, self._best_ask[0], ioc=True)
            elif tox > 0.25 and post_ask is False and self._best_bid:
                self._place_limit(False, self.feeler_qty, self._best_bid[0], ioc=True)

    # ───────── Exchange callbacks ─────────
    def on_trade_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        self._ticker = ticker
        if time.time() < getattr(self, "_cooldown_until", 0.0):
            return
        # could incorporate trade direction; book deltas already captured

    def on_orderbook_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        self._ticker = ticker
        self._apply_level_update(side, quantity, price)

        # During cooldown, only maintain book
        if time.time() < getattr(self, "_cooldown_until", 0.0):
            return

        # One-time sanity cross on first healthy book
        if not self._did_sanity and self._best_bid and self._best_ask:
            if (self._best_ask[0] - self._best_bid[0]) <= 2 * self.tick_size:
                self._place_limit(True, self._sanity_qty, self._best_ask[0], ioc=True)
                self._place_limit(False, self._sanity_qty, self._best_bid[0], ioc=True)
                self._did_sanity = True

        self._decide_and_quote()

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
        """
        Assumptions:
          - time_seconds, when provided, is the seconds remaining in the game clock.
          - event_type may include "END_GAME" exactly at the end (engine dependent).
        """

        # 1) Pre-close flatten if the clock is low and we haven't done it for this game
        if self.preclose_flatten_enabled and not self._preclose_done and time_seconds is not None:
            try:
                # accept both float/int; guard against bad values
                tl = float(time_seconds)
                if 0.0 <= tl <= float(self.preclose_flatten_sec):
                    self._preclose_flatten("clock threshold")
            except Exception:
                pass

        # 2) Explicit end marker → full rollover
        if event_type == "END_GAME":
            self._handle_rollover("END_GAME event")
            return

        now = time.time()

        # During cooldown: keep baselines fresh, and prepare to open on first post-cooldown event
        if now < getattr(self, "_cooldown_until", 0.0):
            self._last_score_sum = int(home_score) + int(away_score)
            self._last_event_ts = now
            return

        curr_sum = int(home_score) + int(away_score)

        # Optional opening probe on the first event after a rollover cooldown
        # Detect "first event of new game": last_score_sum is None because rollover cleared it.
        if self.open_probe_enabled and not self._opened_this_game and self._last_score_sum is None:
            # place a tiny probe trade to establish inventory or test liquidity
            side_str = str(self.open_probe_side or "NONE").upper()
            if side_str in ("BUY", "SELL") and self.open_probe_qty > 0:
                try:
                    if side_str == "BUY":
                        # prefer IOC at best ask if we have a book; else market
                        if self._best_ask:
                            self._place_limit(True, self.open_probe_qty, self._best_ask[0], ioc=True)
                        else:
                            self._place_market(True, self.open_probe_qty)
                    else:
                        if self._best_bid:
                            self._place_limit(False, self.open_probe_qty, self._best_bid[0], ioc=True)
                        else:
                            self._place_market(False, self.open_probe_qty)
                except Exception:
                    pass
            self._opened_this_game = True  # only once per game

        # 3) Score-drop rollover detector (robust to nonzero resets)
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
            # first observation post-rollover: seed baseline, neutral delta
            self._delta_window.append(0.0)

        self._last_score_sum = curr_sum
        self._last_event_ts = now


#9k
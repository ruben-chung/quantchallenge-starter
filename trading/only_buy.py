# QuantChallenge 2025 – Strategy (BUY on home score only)
# Minimal, hard-coded behavior: whenever the home team scores, place a market BUY.
# No orderbook dependency. Robust to flexible callback signatures and engine enums.

import math
from collections import deque

class Strategy:
    def __init__(self):
        self.reset_state()

    # ---------------- init/reset ----------------
    def reset_state(self):
        # engine-provided objects (captured when seen)
        self.ticker = None      # engine's ticker object
        self.SideBUY = None     # engine's BUY enum/value (captured)
        self.SideSELL = None    # engine's SELL enum/value (captured)

        # game state (only what's needed to detect home scores + optional fair calc)
        self.time_seconds = None
        self.game_total_seconds = 2880  # will infer 2400 on first timestamp
        self.home_score = 0
        self.away_score = 0
        self.last_team = None
        self.last_event_type = None

        # small momentum/smoothing (not strictly needed, but harmless)
        self.diff_prev = 0
        self.momentum_ema = 0.0
        self._fair_snaps = deque(maxlen=8)

        # trading control
        self.tick = 0
        self.buy_size = 5.0      # <-- fixed clip size for each home score
        self.max_clip = 20.0     # clamp any accidental oversize
        self.cooldown_until = 0  # minimal throttle

        # sanity: fire once at start to prove path (optional: set False to disable)
        self.did_boot_probe = False
        self.boot_probe_size = 2.0

    # ---------------- small utils ----------------
    @staticmethod
    def _clamp(x, lo, hi):
        return lo if x < lo else hi if x > hi else x

    def _capture_side_enum(self, side_obj):
        if side_obj is None:
            return
        SideCls = getattr(side_obj, "__class__", None)
        if SideCls is None:
            return
        buy = getattr(SideCls, "BUY", None) or getattr(SideCls, "Bid", None) or getattr(SideCls, "BID", None)
        sell = getattr(SideCls, "SELL", None) or getattr(SideCls, "Ask", None) or getattr(SideCls, "ASK", None)
        if buy is not None:
            self.SideBUY = buy
        if sell is not None:
            self.SideSELL = sell

    def _engine_side(self, want_buy: bool):
        if want_buy and self.SideBUY is not None:
            return self.SideBUY
        if (not want_buy) and self.SideSELL is not None:
            return self.SideSELL
        return "BUY" if want_buy else "SELL"

    # ---------------- order helpers (try multiple signatures) ----------------
    def _place_market(self, want_buy: bool, qty: float):
        qty = float(self._clamp(qty, 0.0, self.max_clip))
        if qty <= 0.0:
            return None
        side = self._engine_side(want_buy)

        # try with ticker (common)
        if self.ticker is not None:
            try:
                return place_market_order(self.ticker, side, qty)
            except Exception:
                pass
            try:
                return place_market_order(ticker=self.ticker, side=side, quantity=qty)
            except Exception:
                pass

        # fallbacks without ticker
        try:
            return place_market_order(side, qty)
        except Exception:
            pass
        try:
            return place_market_order(side=side, quantity=qty)
        except Exception:
            return None

    # ---------------- optional fair/momentum (not required for behavior) ----------------
    def _update_momentum(self):
        diff = self.home_score - self.away_score
        delta = diff - self.diff_prev
        self.diff_prev = diff
        alpha = 0.3
        self.momentum_ema = (1 - alpha) * self.momentum_ema + alpha * (delta / 3.0)

    def _quick_fair_anchor(self):
        # purely cosmetic; helps if engine marks to mid somewhere
        diff = self.home_score - self.away_score
        t_frac = (float(self.time_seconds) if self.time_seconds else 0.0) / float(self.game_total_seconds or 2880)
        base = 50.0 + 2.0 * diff + 6.0 * t_frac
        return self._clamp(base, 0.0, 100.0)

    # ---------------- callbacks ----------------
    # If called, this often looks like: (ticker, side, price, total_quantity, ...)
    def on_orderbook_update(self, *args, **kwargs):
        if args:
            # capture ticker if first arg looks like a non-numeric object/string
            if self.ticker is None and not isinstance(args[0], (int, float)):
                self.ticker = args[0]
            # capture engine side enum from second arg if present
            if len(args) >= 2:
                self._capture_side_enum(args[1])
        if "ticker" in kwargs and self.ticker is None:
            self.ticker = kwargs["ticker"]
        if "side" in kwargs:
            self._capture_side_enum(kwargs["side"])

    # Some engines report trades as (ticker, aggressive_side, price, quantity)
    def on_trade_update(self, *args, **kwargs):
        if self.ticker is None and "ticker" in kwargs and not isinstance(kwargs["ticker"], (int, float)):
            self.ticker = kwargs["ticker"]
        if self.ticker is None and args and not isinstance(args[0], (int, float)):
            self.ticker = args[0]

    def on_account_update(self, *args, **kwargs):
        pass

    def on_game_event_update(self, *args, **kwargs):
        self.tick += 1
        evt = self._normalize_event(*args, **kwargs)

        # capture ticker if present in event kwargs
        if self.ticker is None and isinstance(evt.get("ticker"), str):
            self.ticker = evt["ticker"]

        # update time format (2400 vs 2880) and scores
        ts = evt.get("time_seconds")
        if ts is not None:
            try:
                tsf = float(ts)
                if self.time_seconds is None:
                    self.game_total_seconds = 2880 if tsf > 2400 else 2400
                self.time_seconds = tsf
            except Exception:
                pass
        try:
            self.home_score = int(evt.get("home_score", self.home_score))
        except Exception:
            pass
        try:
            self.away_score = int(evt.get("away_score", self.away_score))
        except Exception:
            pass

        # classify event + team
        et = evt.get("event_type", self.last_event_type)
        self.last_event_type = str(et) if et is not None else self.last_event_type
        ha = evt.get("home_away", self.last_team)
        self.last_team = str(ha) if ha is not None else self.last_team

        # END_GAME → reset
        if self.last_event_type == "END_GAME":
            self.reset_state()
            return

        # OPTIONAL: tiny probe once to prove order path
        if not self.did_boot_probe and self.ticker is not None:
            self._place_market(True, self.boot_probe_size)   # one buy
            self.did_boot_probe = True

        # BUY RULE: if the event is a "home score", buy.
        # We treat any event_type containing "SCORE" (case-insensitive) as a score,
        # and also match a few common shot-made labels just in case.
        etu = (self.last_event_type or "").upper()
        is_score = ("SCORE" in etu) or (etu in {"MADE_SHOT", "FG_MADE", "THREE_MADE", "FREE_THROW_MADE"})
        is_home = (str(self.last_team or "").lower() == "home")

        if is_score and is_home:
            self._place_market(True, self.buy_size)  # BUY clip

        # maintain tiny momentum/fair tracking (not necessary for behavior)
        self._update_momentum()
        self._fair_snaps.append(self._quick_fair_anchor())

    # ---------------- event normalizer ----------------
    def _normalize_event(self, *args, **kwargs):
        # kwargs: either fields or event=<str/dict>
        if kwargs:
            if "event" in kwargs and not isinstance(kwargs["event"], dict):
                return {"event_type": str(kwargs["event"])}
            return dict(kwargs)
        # single dict
        if len(args) == 1 and isinstance(args[0], dict):
            return dict(args[0])
        # single string like "SCORE"
        if len(args) == 1 and isinstance(args[0], str):
            return {"event_type": args[0]}
        # positional tuple shape:
        # (home_away, home_score, away_score, event_type, time_seconds, ...)
        pos = list(args)
        while len(pos) < 5:
            pos.append(None)
        (home_away, home_score, away_score, event_type, time_seconds) = pos[:5]
        return {
            "home_away": home_away,
            "home_score": home_score,
            "away_score": away_score,
            "event_type": event_type,
            "time_seconds": time_seconds,
        }

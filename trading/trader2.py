# QuantChallenge 2025 – Strategy (ultra-aggressive, engine-enum aware, no-OB dependency)
# Goal: FORCE activity to verify end-to-end trading path.

import math
from collections import deque

class Strategy:
    def __init__(self):
        self.reset_state()

    # ---------------- init/reset ----------------
    def reset_state(self):
        # engine-provided objects (captured from callbacks)
        self.ticker = None           # keep engine ticker object as-is
        self.SideBUY = None          # engine's BUY enum/value (captured)
        self.SideSELL = None         # engine's SELL enum/value (captured)

        # market anchor (no public order book)
        self.last_trade_price = None

        # active resting orders we own (to refresh cleanly each event)
        self._resting_oids = []

        # game state
        self.time_seconds = None
        self.game_total_seconds = 2880   # will infer to 2400 from first timestamp
        self.home_score = 0
        self.away_score = 0
        self.last_team = None
        self.last_event_type = None
        self.diff_prev = 0
        self.momentum_ema = 0.0
        self._fair_snaps = deque(maxlen=24)

        # trading / risk
        self.position = 0.0
        self.cash = 0.0
        self.tick = 0
        self.cooldown_until = 0  # kept tiny; essentially no cooldown

        # AGGRESSION KNOBS (turned way up)
        self.min_time_to_trade = 0        # start immediately
        self.min_clip = 3.0               # minimum order size (bigger to ensure prints)
        self.max_clip = 10.0
        self.k_pos = 0.8                  # large target pos per 1pt edge
        self.max_position_abs = 200.0     # very wide leash early
        self.base_half_spread = 0.15      # ± around fair for resting quotes (tight!)
        self.fire_every_tick = True       # send a market clip EVERY event
        self.fire_on_impact = True        # extra clip after SCORE/TO/STEAL/BLOCK
        self.impact_extra = 0.3           # fraction of max_clip for impact clip

        # one-time sanity burst to prove API path
        self.did_boot_sanity = False

    # ---------------- small utils ----------------
    @staticmethod
    def _clamp(x, lo, hi):
        return lo if x < lo else hi if x > hi else x

    def _engine_side(self, want_buy: bool):
        # Prefer engine enum/value if captured; otherwise fall back to strings.
        if want_buy and self.SideBUY is not None:
            return self.SideBUY
        if (not want_buy) and self.SideSELL is not None:
            return self.SideSELL
        return "BUY" if want_buy else "SELL"

    def _capture_side_enum(self, side_obj):
        # Try to capture engine BUY/SELL constants from a side object / enum instance.
        if side_obj is None:
            return
        SideCls = getattr(side_obj, "__class__", None)
        if SideCls is None:
            return
        # Common attribute names across engines
        buy = getattr(SideCls, "BUY", None) or getattr(SideCls, "Bid", None) or getattr(SideCls, "BID", None)
        sell = getattr(SideCls, "SELL", None) or getattr(SideCls, "Ask", None) or getattr(SideCls, "ASK", None)
        # Some enums hold values on the instance (name/value pattern)
        if buy is None and hasattr(side_obj, "name"):
            # Attempt to glean from class dir
            for attr in ("BUY","SELL","Bid","Ask","BID","ASK"):
                val = getattr(SideCls, attr, None)
                if val is not None:
                    if "BUY" in attr.upper() or "BID" in attr.upper():
                        buy = buy or val
                    if "SELL" in attr.upper() or "ASK" in attr.upper():
                        sell = sell or val
        if buy is not None:  self.SideBUY  = buy
        if sell is not None: self.SideSELL = sell

    # ---------------- order helpers (robust call patterns) ----------------
    def _place_market(self, want_buy: bool, qty: float):
        qty = float(qty)
        side = self._engine_side(want_buy)

        # Try with ticker first (common pattern)
        if self.ticker is not None:
            try:
                return place_market_order(self.ticker, side, qty)
            except Exception:
                pass
            try:
                return place_market_order(ticker=self.ticker, side=side, quantity=qty)
            except Exception:
                pass

        # Try without ticker (some engines don't require it)
        try:
            return place_market_order(side, qty)
        except Exception:
            pass
        try:
            return place_market_order(side=side, quantity=qty)
        except Exception:
            pass
        return None

    def _place_limit(self, want_buy: bool, qty: float, price: float, ioc: bool = False):
        qty = float(qty); price = float(price)
        side = self._engine_side(want_buy)

        # prefer with ticker
        if self.ticker is not None:
            try:
                return place_limit_order(self.ticker, side, qty, price, ioc)
            except Exception:
                pass
            try:
                return place_limit_order(ticker=self.ticker, side=side, quantity=qty, price=price, IOC=ioc)
            except Exception:
                pass
            try:
                return place_limit_order(ticker=self.ticker, side=side, quantity=qty, price=price, ioc=ioc)
            except Exception:
                pass

        # fallback without ticker
        try:
            return place_limit_order(side, qty, price, ioc)
        except Exception:
            pass
        try:
            return place_limit_order(side=side, quantity=qty, price=price, IOC=ioc)
        except Exception:
            pass
        try:
            return place_limit_order(side=side, quantity=qty, price=price, ioc=ioc)
        except Exception:
            return None

    def _cancel_order(self, oid):
        if oid is None:
            return
        # try with ticker
        if self.ticker is not None:
            try:
                cancel_order(self.ticker, oid)
                return
            except Exception:
                pass
            try:
                cancel_order(ticker=self.ticker, order_id=oid)
                return
            except Exception:
                pass
        # fallback without ticker
        try:
            cancel_order(oid)
        except Exception:
            try:
                cancel_order(order_id=oid)
            except Exception:
                pass

    def _cancel_all_resting(self):
        for oid in self._resting_oids:
            self._cancel_order(oid)
        self._resting_oids.clear()

    # ---------------- passive quoting band ----------------
    def _post_resting_band(self, fair):
        # refresh: cancel and repost 3 tight levels either side of fair
        self._cancel_all_resting()
        levels = (
            self.base_half_spread,
            self.base_half_spread + 0.25,
            self.base_half_spread + 0.5,
        )
        for half in levels:
            bid_px = self._clamp(fair - half, 0.0, 100.0)
            ask_px = self._clamp(fair + half, 0.0, 100.0)

            bid_oid = self._place_limit(True,  self.min_clip, bid_px, ioc=False)
            if bid_oid is not None: self._resting_oids.append(bid_oid)

            ask_oid = self._place_limit(False, self.min_clip, ask_px, ioc=False)
            if ask_oid is not None: self._resting_oids.append(ask_oid)

    # ---------------- callbacks ----------------
    # If the engine calls this, capture ticker and Side enum; we ignore price/qty.
    def on_orderbook_update(self, *args, **kwargs):
        # common level-update: (ticker, side, price, total_qty, ...)
        if args:
            if self.ticker is None and isinstance(args[0], str) or (args and not isinstance(args[0], (int,float))):
                self.ticker = self.ticker or args[0]
            if len(args) >= 2:
                self._capture_side_enum(args[1])
        if "ticker" in kwargs and self.ticker is None:
            self.ticker = kwargs["ticker"]
        if "side" in kwargs:
            self._capture_side_enum(kwargs["side"])

    # Trades reported by engine; capture ticker and last trade price if present
    def on_trade_update(self, *args, **kwargs):
        if "ticker" in kwargs and self.ticker is None and isinstance(kwargs["ticker"], str):
            self.ticker = kwargs["ticker"]
        if args and self.ticker is None and not isinstance(args[0], (int,float)):
            self.ticker = args[0]

        px = None
        if "price" in kwargs:
            try: px = float(kwargs["price"])
            except Exception: px = None
        if px is None:
            for v in args:
                try:
                    fv = float(v)
                    px = fv; break
                except Exception:
                    continue
        if px is not None:
            self.last_trade_price = self._clamp(px, 0.0, 100.0)

    def on_account_update(self, *args, **kwargs):
        # Optionally track balances/capital if provided
        pass

    def on_game_event_update(self, *args, **kwargs):
        self.tick += 1
        evt = self._normalize_event(*args, **kwargs)

        # capture ticker if provided via events
        if self.ticker is None and isinstance(evt.get("ticker"), str):
            self.ticker = evt["ticker"]

        # capture Side enum if passed oddly via kwargs
        if "side" in kwargs:
            self._capture_side_enum(kwargs["side"])

        # time/format
        ts = evt.get("time_seconds")
        if ts is not None:
            try:
                tsf = float(ts)
                if self.time_seconds is None:
                    self.game_total_seconds = 2880 if tsf > 2400 else 2400
                self.time_seconds = tsf
            except Exception:
                pass

        # scores & labels
        try: self.home_score = int(evt.get("home_score", self.home_score))
        except Exception: pass
        try: self.away_score = int(evt.get("away_score", self.away_score))
        except Exception: pass

        et = evt.get("event_type", self.last_event_type)
        self.last_event_type = str(et) if et is not None else self.last_event_type
        ha = evt.get("home_away", self.last_team)
        self.last_team = str(ha) if ha is not None else self.last_team

        if self.last_event_type == "END_GAME":
            self._cancel_all_resting()
            self.reset_state()
            return

        # momentum + smoothing for fair calc
        self._update_momentum()
        self._fair_snaps.append(self._quick_fair_anchor())

        # start immediately
        if (self.time_seconds is None) or (self.tick < self.cooldown_until):
            return

        # compute fair; use it as anchor if no trades yet
        fair = self.fair_price()
        mid = self.last_trade_price if self.last_trade_price is not None else fair

        # generous risk leash
        time_left = max(float(self.time_seconds), 1.0)
        tightener = max(0.2, min(1.0, time_left / 300.0))  # only really tight in final ~5 min
        max_pos = self.max_position_abs * tightener

        edge = fair - mid
        target_pos = self._clamp(self.k_pos * edge, -max_pos, +max_pos)
        desired_delta = target_pos - self.position
        clip = self._clamp(max(self.min_clip, abs(desired_delta)), self.min_clip, self.max_clip)

        # one-time sanity: FIRE both sides regardless of state to prove API path
        if not self.did_boot_sanity:
            # try with/without ticker internally (helpers handle both)
            self._place_market(True,  self.min_clip)
            self._place_market(False, self.min_clip)
            self.did_boot_sanity = True

        # always post a tight resting band so others can hit you
        self._post_resting_band(fair)

        # fire a market clip EVERY event (direction by edge; if edge ~0, buy)
        want_buy = True if edge >= 0.0 else False
        self._place_market(want_buy, clip)

        # extra market nudge on impactful events toward the advantaged team
        if self.fire_on_impact and self.last_event_type in ("SCORE","TURNOVER","STEAL","BLOCK"):
            adv_buy = (self.last_team == "home")
            self._place_market(adv_buy, max(self.min_clip, self.impact_extra * self.max_clip))

        # essentially no cooldown; keep it to zero or one tick
        self.cooldown_until = self.tick

    # ---------------- event normalizer ----------------
    def _normalize_event(self, *args, **kwargs):
        if kwargs:
            if "event" in kwargs and not isinstance(kwargs["event"], dict):
                return {"event_type": str(kwargs["event"])}
            return dict(kwargs)
        if len(args) == 1 and isinstance(args[0], dict):
            return dict(args[0])
        if len(args) == 1 and isinstance(args[0], str):
            return {"event_type": args[0]}
        pos = list(args)
        while len(pos) < 11:
            pos.append(None)
        (home_away, home_score, away_score, event_type, time_seconds,
         player_name, shot_type, assist_player, rebound_type, coordinate_x, coordinate_y) = pos[:11]
        return {
            "home_away": home_away,
            "home_score": home_score,
            "away_score": away_score,
            "event_type": event_type,
            "time_seconds": time_seconds,
            "player_name": player_name,
            "shot_type": shot_type,
            "assist_player": assist_player,
            "rebound_type": rebound_type,
            "coordinate_x": coordinate_x,
            "coordinate_y": coordinate_y,
        }

    # ---------------- pricing ----------------
    def win_prob(self):
        diff = float(self.home_score - self.away_score)
        t = max(float(self.time_seconds) if self.time_seconds is not None else 0.0, 1.0)
        logt_scaled = math.log1p(t) / math.log1p(float(self.game_total_seconds or 2880))
        clutch = 1.0 if t <= 60.0 else 0.0
        x = (-3.2 + 0.085 * diff + 0.55 * logt_scaled + 0.45 * self.momentum_ema + 0.65 * clutch)
        if self.last_event_type in ("SCORE", "TURNOVER", "BLOCK", "STEAL"):
            x += 0.08 if self.last_team == "home" else -0.08
        return 1.0 / (1.0 + math.exp(-x))

    def fair_price(self):
        p = 100.0 * self.win_prob()
        if self._fair_snaps:
            p = 0.7 * p + 0.3 * (sum(self._fair_snaps) / len(self._fair_snaps))
        return self._clamp(p, 0.0, 100.0)

    def _update_momentum(self):
        diff = self.home_score - self.away_score
        delta = diff - self.diff_prev
        self.diff_prev = diff
        alpha = 0.25
        self.momentum_ema = (1 - alpha) * self.momentum_ema + alpha * (delta / 3.0)

    def _quick_fair_anchor(self):
        diff = self.home_score - self.away_score
        t_frac = (float(self.time_seconds) if self.time_seconds else 0.0) / float(self.game_total_seconds or 2880)
        base = 50.0 + 2.3 * diff + 8.0 * t_frac
        return self._clamp(base, 0.0, 100.0)

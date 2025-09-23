# QuantChallenge 2025 – Strategy (super aggressive event-driven trader)
# Trades very frequently to validate end-to-end wiring.
# Pure stdlib; robust to flexible callback arg shapes; no orderbook dependency.

import math
from collections import deque

class Strategy:
    def __init__(self):
        self.reset_state()

    # --------------- init/reset ---------------
    def reset_state(self):
        # we ignore order book; keep last trade as market anchor
        self.ticker = None
        self.last_trade_price = None

        # active quotes (so we can refresh/cancel)
        self.bid_order_id = None
        self.ask_order_id = None

        # game state
        self.time_seconds = None
        self.home_score = 0
        self.away_score = 0
        self.last_team = None
        self.last_event_type = None
        self.diff_prev = 0
        self.momentum_ema = 0.0
        self.events_window = deque(maxlen=24)
        self._fair_snaps = deque(maxlen=24)
        self.game_total_seconds = 2880  # flip to 2400 when we see first ts

        # trading / risk
        self.position = 0.0
        self.cash = 0.0
        self.tick = 0
        self.cooldown_until = 0

        # aggression knobs (turned way up)
        self.min_time_to_trade = 1      # start almost immediately
        self.base_half_spread = 0.3     # ± around fair for passive quotes
        self.min_clip = 2.0             # minimum order size
        self.max_clip = 8.0
        self.k_pos = 0.75               # target position per 1pt edge (big)
        self.max_position_abs = 120.0   # wide leash
        self.impact_bump_take = 0.2     # extra clip on scores/turnovers
        self.every_tick_take = True     # fire a small market clip each tick

    # --------------- utils ---------------
    @staticmethod
    def _clamp(x, lo, hi):
        return lo if x < lo else hi if x > hi else x

    def _cancel_if(self, oid):
        if not oid or not self.ticker:
            return
        try:
            cancel_order(self.ticker, oid)
        except Exception:
            try:
                cancel_order(ticker=self.ticker, order_id=oid)
            except Exception:
                pass

    def _place_limit(self, side, qty, price, ioc=False):
        if not self.ticker:
            return None
        qty = float(qty); price = float(price)
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
            return None

    def _place_market(self, side, qty):
        if not self.ticker:
            return None
        qty = float(qty)
        try:
            return place_market_order(self.ticker, side, qty)
        except Exception:
            try:
                return place_market_order(ticker=self.ticker, side=side, quantity=qty)
            except Exception:
                return None

    # --------------- callbacks ---------------
    # If the engine calls this, we just capture ticker and ignore the rest.
    def on_orderbook_update(self, ticker, *_, **__):
        self.ticker = self.ticker or ticker

    # trades from the exchange; store last trade price as rough market anchor
    def on_trade_update(self, *args, **kwargs):
        # common forms: (ticker, aggressive_side, price, quantity) or a dict
        px = None
        if kwargs:
            # dict-like
            if "price" in kwargs:
                try: px = float(kwargs["price"])
                except Exception: px = None
        elif len(args) == 1 and isinstance(args[0], dict):
            d = args[0]
            try: px = float(d.get("price"))
            except Exception: px = None
        else:
            # positional: try to find a numeric 'price' among args
            for val in args:
                try:
                    v = float(val)
                    px = v
                    break
                except Exception:
                    continue
        if px is not None:
            self.last_trade_price = self._clamp(px, 0.0, 100.0)
        # also try to learn ticker if present
        if len(args) and isinstance(args[0], str) and self.ticker is None:
            self.ticker = args[0]
        if "ticker" in kwargs and isinstance(kwargs["ticker"], str) and self.ticker is None:
            self.ticker = kwargs["ticker"]

    def on_account_update(self, *args, **kwargs):
        # optional: could track capital_remaining if provided
        pass

    def on_game_event_update(self, *args, **kwargs):
        self.tick += 1
        evt = self._normalize_event(*args, **kwargs)

        # time / format
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
            self._cancel_if(self.bid_order_id); self.bid_order_id = None
            self._cancel_if(self.ask_order_id); self.ask_order_id = None
            self.reset_state()
            return

        # ticker may be included in evt; capture if so
        if self.ticker is None and isinstance(evt.get("ticker"), str):
            self.ticker = evt["ticker"]

        # momentum + smoothing
        self._update_momentum()
        self._fair_snaps.append(self._quick_fair_anchor())
        self.events_window.append(self.last_event_type)

        # start trading almost immediately
        if (self.time_seconds is None) or (self.time_seconds <= self.min_time_to_trade):
            return
        if self.tick < self.cooldown_until:
            return

        # compute fair and desired position
        fair = self.fair_price()
        mid = self.last_trade_price if self.last_trade_price is not None else fair
        edge = fair - mid

        # risk tightening as time decays (still very generous)
        time_left = max(float(self.time_seconds), 1.0)
        tightener = max(0.2, min(1.0, time_left / 300.0))  # tighten only in last ~5 min
        max_pos = self.max_position_abs * tightener

        target_pos = self._clamp(self.k_pos * edge, -max_pos, +max_pos)
        desired_delta = target_pos - self.position

        # always refresh quotes: tiny passive band around fair
        self._refresh_quotes(fair, desired_delta)

        # fire a market clip every tick to prove trading path (can disable later)
        if self.every_tick_take and self.ticker:
            side = "BUY" if edge >= 0 else "SELL"
            clip = self._clamp(max(self.min_clip, abs(desired_delta)), self.min_clip, self.max_clip)
            self._place_market(side, clip)

        # after high-impact events, add an extra taker bump in direction of advantage
        if self.last_event_type in ("SCORE", "TURNOVER", "STEAL", "BLOCK") and self.ticker:
            side = "BUY" if (self.last_team == "home") else "SELL"
            self._place_market(side, max(self.min_clip, self.impact_bump_take * self.max_clip))

        # tiny cooldown to avoid overwhelming the engine (still very active)
        self.cooldown_until = self.tick + 1

    # --------------- quoting ---------------
    def _refresh_quotes(self, fair, desired_delta):
        # cancel prior quotes
        if self.bid_order_id:
            self._cancel_if(self.bid_order_id); self.bid_order_id = None
        if self.ask_order_id:
            self._cancel_if(self.ask_order_id); self.ask_order_id = None
        if not self.ticker:
            return

        # inventory skew: quote away from risk
        skew = 0.10 * self.position
        bid_px = self._clamp(fair - self.base_half_spread - skew, 0.0, 100.0)
        ask_px = self._clamp(fair + self.base_half_spread - skew, 0.0, 100.0)

        # size to move toward target, but always at least a clip
        buy_qty  = self._clamp(max(self.min_clip,  desired_delta), 0.0, self.max_clip)
        sell_qty = self._clamp(max(self.min_clip, -desired_delta), 0.0, self.max_clip)

        # post both sides to maximize fills while flat-ish
        self.bid_order_id = self._place_limit("BUY",  buy_qty,  bid_px)
        self.ask_order_id = self._place_limit("SELL", sell_qty, ask_px)

    # --------------- event normalizer ---------------
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

    # --------------- pricing ---------------
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

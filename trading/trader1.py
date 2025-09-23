# QuantChallenge 2025 – Strategy (event-driven quoting; no orderbook dependence)

import math
from collections import deque

class Strategy:
    def __init__(self):
        self.reset_state()

    # ---------------- init/reset ----------------
    def reset_state(self):
        # we ignore order book; keep last trade as market anchor
        self.ticker = None
        self.last_trade_price = None

        # active quotes we manage (ids) so we can cancel/refresh
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
        self.events_window = deque(maxlen=16)
        self._fair_snaps = deque(maxlen=16)
        self.game_total_seconds = 2880  # will infer 2400 on first timestamp

        # trading state
        self.position = 0.0
        self.cash = 0.0
        self.tick = 0
        self.cooldown_until = 0

        # controls (tune)
        self.base_spread = 1.6          # total spread around fair (we'll split in half)
        self.edge_take = 1.0            # take a small clip after big events
        self.k_pos = 0.25               # target position per 1pt edge
        self.max_position_abs = 60.0
        self.min_time_to_trade = 2
        self.min_clip = 1.0
        self.max_clip = 6.0
        self.replace_every_ticks = 1    # refresh quotes every event
        self.impact_bump = 0.4          # tighten/take after SCORE/TO/STEAL/BLOCK

    # ---------------- small utils ----------------
    @staticmethod
    def _clamp(x, lo, hi):
        return lo if x < lo else hi if x > hi else x

    def _cancel_if(self, oid):
        if not oid or not self.ticker:
            return
        # try both signatures
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
        # try positional
        try:
            return place_limit_order(self.ticker, side, qty, price, ioc)
        except Exception:
            pass
        # try named
        try:
            return place_limit_order(ticker=self.ticker, side=side, quantity=qty, price=price, IOC=ioc)
        except Exception:
            pass
        # try named lowercase ioc
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

    # ---------------- callbacks ----------------
    # If the engine calls this, we just capture ticker and ignore the rest.
    def on_orderbook_update(self, ticker, *_, **__):
        self.ticker = self.ticker or ticker

    # trades from the exchange; store last trade price as rough market
    def on_trade_update(self, ticker_or_side=None, aggressive_side=None, price=None, quantity=None, *args, **kwargs):
        # try to recover ticker and price regardless of signature
        if self.ticker is None and isinstance(ticker_or_side, str):
            self.ticker = ticker_or_side
        px = None
        # sometimes price is second/third argument; grab first numerical value seen
        for val in (price, quantity) + args:
            try:
                v = float(val)
                # the first float we see is likely price (quantity tends to be integerish but either works)
                if px is None:
                    px = v
            except Exception:
                continue
        if px is not None:
            # sanity clamp
            self.last_trade_price = self._clamp(px, 0.0, 100.0)

    def on_account_update(self, *args, **kwargs):
        # optional: could track capital_remaining if provided
        pass

    def on_game_event_update(self, *args, **kwargs):
        self.tick += 1
        evt = self._normalize_event(*args, **kwargs)

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
            # flatten: cancel quotes and reset
            self._cancel_if(self.bid_order_id); self.bid_order_id = None
            self._cancel_if(self.ask_order_id); self.ask_order_id = None
            self.reset_state()
            return

        # update momentum and smoothing
        self._update_momentum()
        self._fair_snaps.append(self._quick_fair_anchor())
        self.events_window.append(self.last_event_type)

        # throttle early start
        if (self.time_seconds is None) or (self.time_seconds <= self.min_time_to_trade):
            return
        if self.tick < self.cooldown_until:
            return

        # compute fair and desired position
        fair = self.fair_price()
        # if no ticker known yet, try to infer from event kwargs (some runners pass it)
        if self.ticker is None:
            self.ticker = evt.get("ticker") if isinstance(evt.get("ticker"), str) else self.ticker

        # risk tightening as time decays
        time_left = max(float(self.time_seconds), 1.0)
        tightener = max(0.15, min(1.0, time_left / 600.0))
        max_pos = self.max_position_abs * tightener

        # edge vs last trade (if any). If none, edge=0 and we’ll still quote passively.
        mid = self.last_trade_price if self.last_trade_price is not None else fair
        edge = fair - mid
        target_pos = self._clamp(self.k_pos * edge, -max_pos, +max_pos)
        desired_delta = target_pos - self.position

        # size we’ll try to move (ensure at least min_clip if meaningfully off target)
        if abs(desired_delta) < self.min_clip and abs(edge) >= 0.25:
            desired_delta = self.min_clip if edge > 0 else -self.min_clip

        # cancel/refresh quotes every event (keeps book clean)
        if self.bid_order_id:
            self._cancel_if(self.bid_order_id); self.bid_order_id = None
        if self.ask_order_id:
            self._cancel_if(self.ask_order_id); self.ask_order_id = None

        # set quoting spread (narrower after impactful events)
        impact = self.last_event_type in ("SCORE","TURNOVER","STEAL","BLOCK")
        half_spread = 0.5 * (self.base_spread - (self.impact_bump if impact else 0.0))
        half_spread = max(0.3, half_spread)

        # inventory skew: quote away from the side that increases risk
        skew = 0.12 * self.position
        bid_px = self._clamp(fair - half_spread - skew, 0.0, 100.0)
        ask_px = self._clamp(fair + half_spread - skew, 0.0, 100.0)

        # passive quotes
        if self.ticker:
            # bias which side we post if we need to move inventory
            post_bid = (self.position <= target_pos)  # want to buy or stay flat
            post_ask = (self.position >= target_pos)  # want to sell or stay flat

            if post_bid:
                qty = self._clamp(max(self.min_clip, desired_delta), 0.0, self.max_clip)
                if qty >= self.min_clip:
                    self.bid_order_id = self._place_limit("BUY", qty, bid_px)
            if post_ask:
                qty = self._clamp(max(self.min_clip, -desired_delta), 0.0, self.max_clip)
                if qty >= self.min_clip:
                    self.ask_order_id = self._place_limit("SELL", qty, ask_px)

        # opportunistic taker after impact if edge vs last trade is big
        if impact and abs(edge) >= self.edge_take and self.ticker:
            clip = self._clamp(abs(desired_delta), self.min_clip, self.max_clip)
            side = "BUY" if edge > 0 else "SELL"
            self._place_market(side, clip)
            self.cooldown_until = self.tick + 1

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

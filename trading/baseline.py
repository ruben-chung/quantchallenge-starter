# QuantChallenge 2025 – Strategy (robust to arg shapes)
# Pure stdlib; defines class Strategy; handles level-updates + flexible game-event args.

import math
from collections import deque

class Strategy:
    def __init__(self):
        self.reset_state()

    # ---------------- init/reset ----------------
    def reset_state(self):
        # order book (price -> size)
        self.bids = {}
        self.asks = {}
        self.best_bid = None   # (price, size)
        self.best_ask = None   # (price, size)
        self.mid = None
        self.ticker = None

        # game state
        self.time_seconds = None
        self.home_score = 0
        self.away_score = 0
        self.last_team = None
        self.last_event_type = None
        self.diff_prev = 0
        self.momentum_ema = 0.0
        self.events_window = deque(maxlen=12)
        self._fair_snaps = deque(maxlen=12)
        self.game_total_seconds = 2880   # will switch to 2400 when inferred

        # trading state
        self.position = 0.0
        self.cash = 0.0
        self.open_orders = {}
        self.tick = 0
        self.cooldown_until = 0

        # controls
        self.edge_threshold = 1.0
        self.take_edge = 2.0
        self.quote_half_spread = 0.9
        self.k_pos = 0.12
        self.max_position_abs = 60.0
        self.min_time_to_trade = 10

    # ---------------- small utils ----------------
    @staticmethod
    def _side_to_str(side):
        name = getattr(side, "name", None)
        if isinstance(name, str):
            return name
        s = str(side).upper()
        return "BUY" if ("BUY" in s or "BID" in s) else "SELL"

    @staticmethod
    def _clamp(x, lo, hi):
        return lo if x < lo else hi if x > hi else x

    def _recompute_best(self):
        self.best_bid = (max(self.bids), self.bids[max(self.bids)]) if self.bids else None
        self.best_ask = (min(self.asks), self.asks[min(self.asks)]) if self.asks else None
        if self.best_bid and self.best_ask:
            self.mid = 0.5 * (self.best_bid[0] + self.best_ask[0])
        elif self.best_bid:
            self.mid = self.best_bid[0]
        elif self.best_ask:
            self.mid = self.best_ask[0]

    # ---------------- callbacks ----------------
    def on_orderbook_update(self, ticker, side, price, total_quantity, *args, **kwargs):
        """Level update: total quantity at (side, price) changed."""
        self.ticker = self.ticker or ticker
        s = self._side_to_str(side)
        try:
            px = float(price)
            qty = float(total_quantity)
        except Exception:
            return
        if s == "BUY":
            if qty <= 0:
                self.bids.pop(px, None)
            else:
                self.bids[px] = qty
        else:
            if qty <= 0:
                self.asks.pop(px, None)
            else:
                self.asks[px] = qty
        self._recompute_best()

    def on_trade_update(self, ticker, aggressive_side, price, quantity, *args, **kwargs):
        """Two orders matched; side is the aggressive side. (We don’t need to do anything here.)"""
        pass

    def on_account_update(self, capital_remaining=None, *args, **kwargs):
        """Your order matched; includes new capital_remaining (if provided)."""
        # Optionally track capital_remaining to tighten risk.
        pass

    def on_game_event_update(self, *args, **kwargs):
        """
        Robust handler for multiple call styles:
        1) dict: on_game_event_update(event_dict)
        2) kwargs: on_game_event_update(home_away='home', home_score=..., ...)
        3) positional tuple (common in these comps):
           (home_away, home_score, away_score, event_type, time_seconds,
            player_name, shot_type, assist_player, rebound_type, coordinate_x, coordinate_y)
        Only the first 5 are required; the rest may be None.
        """
        self.tick += 1

        # --- normalize to an 'event' dict ---
        if kwargs:
            event = dict(kwargs)
        elif args and isinstance(args[0], dict):
            event = args[0]
        else:
            # positional form
            pos = list(args)
            # pad to length 11
            while len(pos) < 11:
                pos.append(None)
            (home_away, home_score, away_score, event_type, time_seconds,
             player_name, shot_type, assist_player, rebound_type, coordinate_x, coordinate_y) = pos[:11]
            event = {
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

        # --- update internal state safely ---
        ts = event.get("time_seconds", None)
        if ts is not None:
            try:
                ts = float(ts)
                if self.time_seconds is None:
                    self.game_total_seconds = 2880 if ts > 2400 else 2400
                self.time_seconds = ts
            except Exception:
                pass

        try:
            self.home_score = int(event.get("home_score", self.home_score))
        except Exception:
            pass
        try:
            self.away_score = int(event.get("away_score", self.away_score))
        except Exception:
            pass

        et = event.get("event_type", self.last_event_type)
        if isinstance(et, (bytes, bytearray)):
            et = et.decode("utf-8", errors="ignore")
        self.last_event_type = str(et) if et is not None else self.last_event_type

        ha = event.get("home_away", self.last_team)
        self.last_team = str(ha) if ha is not None else self.last_team

        # END_GAME → full reset
        if self.last_event_type == "END_GAME":
            self.reset_state()
            return

        # momentum & smoothing
        self._update_momentum()
        self._fair_snaps.append(self._quick_fair_anchor())
        self.events_window.append(self.last_event_type)

        # throttles
        if (self.time_seconds is None) or (self.time_seconds >= (self.game_total_seconds - self.min_time_to_trade)):
            return
        if self.tick < self.cooldown_until:
            return

        # risk tightening
        time_left = max(float(self.time_seconds), 1.0)
        tightener = max(0.15, min(1.0, time_left / 600.0))
        max_pos = self.max_position_abs * tightener

        # pricing & target
        fair = self.fair_price()
        mid = self.mid if self.mid is not None else fair
        edge = fair - mid
        target_pos = self._clamp(self.k_pos * edge, -max_pos, +max_pos)
        delta = target_pos - self.position

        # execution
        if abs(edge) >= self.take_edge and abs(delta) >= 1.0:
            qty = min(5.0, abs(delta))
            side = "BUY" if edge > 0 else "SELL"
            self._place_market(side, qty)
            return

        if abs(edge) >= self.edge_threshold and abs(delta) >= 1.0:
            skew = 0.15 * self.position
            buy_px = self._clamp(fair - self.quote_half_spread - skew, 0.0, 100.0)
            sell_px = self._clamp(fair + self.quote_half_spread - skew, 0.0, 100.0)
            buy_qty = max(0.0, min(5.0, target_pos - self.position))
            sell_qty = max(0.0, min(5.0, self.position - target_pos))
            if buy_qty >= 1.0:
                oid = self._place_limit("BUY", buy_qty, buy_px)
                if oid is not None:
                    self.open_orders[oid] = ("BUY", buy_px, buy_qty)
            if sell_qty >= 1.0:
                oid = self._place_limit("SELL", sell_qty, sell_px)
                if oid is not None:
                    self.open_orders[oid] = ("SELL", sell_px, sell_qty)

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

    # ---------------- order helpers ----------------
    def _place_market(self, side, qty):
        if not self.ticker:
            return None
        try:
            return place_market_order(self.ticker, side, float(qty))
        except Exception:
            try:
                return place_market_order(ticker=self.ticker, side=side, quantity=float(qty))
            except Exception:
                pass
        # emulate with limit if needed
        if side == "BUY" and self.best_ask:
            return self._place_limit("BUY", qty, float(self.best_ask[0]))
        if side == "SELL" and self.best_bid:
            return self._place_limit("SELL", qty, float(self.best_bid[0]))
        return self._place_limit(side, qty, self.fair_price())

    def _place_limit(self, side, qty, price):
        if not self.ticker:
            return None
        price = float(price); qty = float(qty)
        try:
            return place_limit_order(self.ticker, side, qty, price)
        except Exception:
            try:
                return place_limit_order(ticker=self.ticker, side=side, quantity=qty, price=price)
            except Exception:
                return None

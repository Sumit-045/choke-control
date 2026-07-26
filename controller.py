"""
controller.py

Brute-force MPC controller.

    measure -> bias update -> enumerate candidates -> predict over horizon
            -> reject infeasible -> score survivors -> clamp -> apply

Two structural changes from the previous version:

1. compute() now RECEIVES THE MEASUREMENTS. The brief states that at each
   control interval the controller receives Oil Rate, WHP, FLP, BHP and the
   current choke position. The old signature was compute(current_choke, target),
   which is open-loop: the plant's actual state never entered the decision.

2. A bias term is formed from those measurements:

       bias = measured - what the model predicted for this instant

   Without it the controller has no integral action, so any model gain error
   becomes a permanent steady-state offset. It looks like a tuning problem and
   is not - no tracking weight can fix it, because the controller believes it
   has already arrived.
"""

import numpy as np

import config
from predictor import Predictor, OUTPUTS


CONSTRAINED = ("WHP", "FLP", "BHP")


class MPCController:

    def __init__(self, predictor=None, backoff=None):
        if predictor is None:
            predictor = Predictor()
            predictor.load()
        self.predictor = predictor

        self.backoff = backoff or {k: config.BACKOFF_FLOOR[k] for k in CONSTRAINED}

        self._model_state = None
        self._pred_for_now = None
        self.bias = {k: 0.0 for k in OUTPUTS}

        self.log = {"n_feasible": [], "fallback": [], "bias_BHP": [],
                    "blocked": [], "limiting": []}

    # ------------------------------------------------------------------

    @classmethod
    def with_noise_backoff(cls, sigma, predictor=None):
        """Build with back-off sized from measured noise sigma."""
        bo = {k: max(config.BACKOFF_SIGMA * float(sigma.get(k, 0.0)),
                     config.BACKOFF_FLOOR[k]) for k in CONSTRAINED}
        return cls(predictor=predictor, backoff=bo)

    def initialise(self, measurement):
        self._model_state = {k: float(measurement[k]) for k in OUTPUTS}
        self._pred_for_now = None
        self.bias = {k: 0.0 for k in OUTPUTS}

    # ------------------------------------------------------------------
    # Limits
    # ------------------------------------------------------------------

    def limits(self):
        return {"WHP": config.WHP_LIMIT, "FLP": config.FLP_LIMIT,
                "BHP": config.BHP_LIMIT}

    def effective_limits(self):
        """What the controller steers to: limit + back-off.

        Each limit is defined AS the settled pressure at a chosen choke opening,
        so that opening sits exactly on the boundary by construction. Aiming at
        it directly is a coin flip on every noisy sample.
        """
        lo = self.limits()
        return {k: lo[k] + self.backoff[k] for k in CONSTRAINED}

    @staticmethod
    def _sign():
        """+1 when the limits are lower bounds, -1 when upper.

        Detected by characterize() on the PROVIDED simulator rather than
        assumed. Getting this backwards judges every candidate unsafe and
        freezes the choke - which is exactly how an earlier version failed."""
        return 1.0 if config.CONSTRAINT_DIRECTION == "lower" else -1.0

    def effective_limits_signed(self):
        lo = self.limits()
        sgn = self._sign()
        return {k: lo[k] + sgn * self.backoff[k] for k in CONSTRAINED}

    def safe(self, whp, flp, bhp):
        """Kept for backward compatibility. Direction-aware, with back-off."""
        eff = self.effective_limits_signed()
        sgn = self._sign()
        vals = {"WHP": whp, "FLP": flp, "BHP": bhp}
        return all(sgn * (vals[k] - eff[k]) >= 0 for k in CONSTRAINED)

    # ------------------------------------------------------------------
    # Candidates
    # ------------------------------------------------------------------

    def get_candidates(self, current_choke):
        """Candidate chokes within the ramp limit.

        np.linspace, not range() - the old version required an integer choke and
        gave only 1 % resolution, and would raise TypeError the moment a float
        reached it.
        """
        lo = max(config.MIN_CHOKE, current_choke - config.MAX_CHOKE_MOVE)
        hi = min(config.MAX_CHOKE, current_choke + config.MAX_CHOKE_MOVE)
        return np.linspace(lo, hi, config.N_CANDIDATES)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def compute(self, current_choke, target,
                oil=None, whp=None, flp=None, bhp=None):
        """Return the choke position to apply for the next interval.

        Measurements are optional so old call sites do not break, but supplying
        them is what enables the bias correction.
        """
        current_choke = float(current_choke)

        measurement = None
        if None not in (oil, whp, flp, bhp):
            measurement = {"Oil": float(oil), "WHP": float(whp),
                           "FLP": float(flp), "BHP": float(bhp)}

        # --- 1. bias update ------------------------------------------------
        if measurement is not None:
            if self._model_state is None:
                self.initialise(measurement)
            if self._pred_for_now is None:
                self.bias = {k: 0.0 for k in OUTPUTS}
            else:
                self.bias = {k: measurement[k] - self._pred_for_now[k]
                             for k in OUTPUTS}
            state = {k: measurement[k] - self.bias[k] for k in OUTPUTS}
        else:
            state = self._model_state or {
                k: float(self.predictor.static(k, current_choke)) for k in OUTPUTS}

        # --- 2. candidates and prediction ----------------------------------
        cand = self.get_candidates(current_choke)
        N = config.PREDICTION_HORIZON
        traj = self.predictor.predict_batch(state, cand, N, bias=self.bias)

        # --- 3. feasibility over the WHOLE horizon -------------------------
        # Not just the endpoint: a move that dips under a floor at hour 12 and
        # recovers by hour 60 is unsafe, and with a 21 h BHP time constant that
        # is a live failure mode rather than a theoretical one.
        eff = self.effective_limits_signed()
        sgn = self._sign()
        violation = np.zeros(cand.size)
        for k in CONSTRAINED:
            short = np.clip(-sgn * (traj[k] - eff[k]), 0.0, None)
            violation += short.sum(axis=1) / max(self.backoff[k], 1e-6)
        feasible = violation <= 0.0

        # --- 4. cost -------------------------------------------------------
        err = traj["Oil"] - float(target)
        j_track = config.WEIGHT_TRACKING * np.sum(err ** 2, axis=1) / N
        j_move = config.WEIGHT_CHOKE_MOVEMENT * (cand - current_choke) ** 2
        cost = j_track + j_move

        # --- 5. select -----------------------------------------------------
        fallback = False
        if feasible.any():
            idx = int(np.argmin(np.where(feasible, cost, np.inf)))
        else:
            # Nothing is safe across the horizon. Take the least-unsafe move.
            # For a pressure-floor breach the winner is always a closing move,
            # which is the correct physical response. The old version silently
            # returned current_choke here, which meant the controller froze
            # exactly when it most needed to act.
            fallback = True
            idx = int(np.argmin(config.WEIGHT_VIOLATION * violation + cost))

        u_next = float(cand[idx])

        # --- 6. hard clamps ------------------------------------------------
        u_next = float(np.clip(u_next,
                               max(config.MIN_CHOKE, current_choke - config.MAX_CHOKE_MOVE),
                               min(config.MAX_CHOKE, current_choke + config.MAX_CHOKE_MOVE)))

        # --- 7. remember this prediction for the next bias update ----------
        self._pred_for_now = self.predictor.advance(state, u_next)
        self._model_state = self._pred_for_now

        # --- 8. diagnostics ------------------------------------------------
        wanted_more = bool(feasible.any() and int(np.argmin(cost)) > idx)
        margins = {k: float(np.min(sgn * (traj[k][idx] - eff[k])))
                   for k in CONSTRAINED}
        gains = {k: abs(self.predictor.gain(k, u_next)) or 1e-9 for k in CONSTRAINED}
        # ranked in choke-percent of headroom, not psi: BHP moves ~6 psi per %
        # of choke and FLP under 1, so equal psi margins are not equal safety
        limiting = min(CONSTRAINED, key=lambda k: margins[k] / gains[k])

        self.log["n_feasible"].append(int(feasible.sum()))
        self.log["fallback"].append(fallback)
        self.log["bias_BHP"].append(self.bias["BHP"])
        self.log["blocked"].append(wanted_more)
        self.log["limiting"].append(limiting)

        return u_next


def max_safe_rate(predictor, backoff=None, n=2001):
    """Highest oil rate that satisfies every limit at steady state.

    Computed independently of the controller by sweeping the identified static
    maps, so Scenario C can be scored against a number rather than eyeballed.
    Also used to resolve the fractional scenario targets.
    """
    sgn = 1.0 if config.CONSTRAINT_DIRECTION == "lower" else -1.0
    lo = {"WHP": config.WHP_LIMIT, "FLP": config.FLP_LIMIT, "BHP": config.BHP_LIMIT}
    backoff = backoff or {k: 0.0 for k in CONSTRAINED}

    u = np.linspace(config.MIN_CHOKE, config.MAX_CHOKE, n)
    ok = np.ones_like(u, dtype=bool)
    for k in CONSTRAINED:
        y = np.asarray(predictor.static(k, u), dtype=float)
        ok &= (sgn * (y - (lo[k] + sgn * backoff[k])) >= 0)

    q = np.asarray(predictor.static("Oil", u), dtype=float)
    if not ok.any():
        return {"feasible": False, "Q": float("nan"), "u": float("nan"),
                "limiting": None}

    i = int(np.argmax(np.where(ok, q, -np.inf)))
    gains = {k: abs(predictor.gain(k, float(u[i]))) or 1e-9 for k in CONSTRAINED}
    head = {k: sgn * (float(predictor.static(k, u[i])) - lo[k]) / gains[k]
            for k in CONSTRAINED}
    return {"feasible": True, "Q": float(q[i]), "u": float(u[i]),
            "limiting": min(CONSTRAINED, key=lambda k: head[k])}

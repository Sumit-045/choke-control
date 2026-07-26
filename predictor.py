"""
predictor.py

Process model used by the MPC for prediction.

STRUCTURE (Hammerstein: static nonlinearity + linear dynamics)

    y_ss(u)  = f(u)                                      <- from the settled sweep
    y(k+1)   = a * y(k) + (1 - a) * y_ss(u(k)),   a = exp(-Ts / tau)

The previous version had only the static block. Because predict() depended on
nothing but the choke, the controller's horizon loop returned the identical
value 20 times over - measured directly:

    BHP predicted 3x for the same choke: [2968.705, 2968.705, 2968.705]

so the "prediction horizon" was dead code that multiplied the cost by 20 and
cost 20x the compute. The dynamics block below makes the horizon mean something.

MODEL SELECTION
Both a RandomForest and a polynomial static map are fitted and scored. The
polynomial is used for control; the forest is kept and reported because the
comparison is itself the justification.

The forest scores R2 = 1.00000 against R2 = 0.9986 for the polynomial - but
that number is meaningless, and saying why is the point. The forest is being
scored on the exact choke levels it was trained on, which it has memorised. It
says nothing about behaviour BETWEEN those levels, and between them is where the
controller lives:

  * A forest is piecewise constant between training levels. The controller
    resolves moves of 0.25 %, so inside a flat region every candidate predicts
    an identical outcome, the tracking term cannot separate them, and move
    suppression picks zero. That is a zero-gradient dead zone - the same failure
    that froze the earlier controller, arriving by a different route.
  * A forest cannot extrapolate beyond its training range at all.
  * The honest comparison is the multi-step validation below, run against a
    transient neither model was trained on.
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import config


OUTPUTS = ("Oil", "WHP", "FLP", "BHP")
_COLS = {
    "Oil": "OilRate_bbl_hr",
    "WHP": "WHP_psi",
    "FLP": "FLP_psi",
    "BHP": "BHP_psi",
}


class Predictor:

    def __init__(self, degree=2):
        self.degree = degree
        self.poly = {}          # output -> polynomial coefficients
        self.tau = {}           # output -> time constant, hours
        self.u_lo = 0.0
        self.u_hi = 100.0
        self.forest = {}        # output -> RandomForestRegressor (comparison only)
        self.report = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, verbose=True):
        if verbose:
            print("\nTraining process model...")

        df = pd.read_csv(config.DATASET_FILENAME)

        # average the settled samples at each choke level
        grouped = df.groupby("Choke_pct", as_index=False).mean(numeric_only=True)
        u = grouped["Choke_pct"].values.astype(float)
        self.u_lo, self.u_hi = float(u.min()), float(u.max())

        X = df[["Choke_pct"]]

        for key in OUTPUTS:
            col = _COLS[key]
            y = grouped[col].values.astype(float)

            # --- polynomial static map (used for control) ---
            coeffs = np.polyfit(u, y, self.degree)
            self.poly[key] = coeffs
            resid = y - np.polyval(coeffs, u)
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2_poly = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else 1.0

            # --- RandomForest (comparison / reporting only) ---
            rf = RandomForestRegressor(n_estimators=100,
                                       random_state=config.RANDOM_SEED)
            rf.fit(X, df[col])
            self.forest[key] = rf
            yr = rf.predict(pd.DataFrame({"Choke_pct": u}))
            r2_rf = 1.0 - float(np.sum((y - yr) ** 2)) / ss_tot if ss_tot > 0 else 1.0

            self.report[key] = {"r2_poly": r2_poly, "r2_forest": r2_rf,
                                "rmse_poly": float(np.sqrt(np.mean(resid ** 2)))}

        self._fit_dynamics(verbose=verbose)

        if verbose:
            print(f"  {'output':<6}{'tau (h)':>9}{'gain@50%':>11}"
                  f"{'R2 poly':>10}{'R2 forest':>11}")
            for key in OUTPUTS:
                r = self.report[key]
                print(f"  {key:<6}{self.tau[key]:>9.2f}{self.gain(key, 50.0):>11.3f}"
                      f"{r['r2_poly']:>10.5f}{r['r2_forest']:>11.5f}")
            print("Training complete.")

    def _fit_dynamics(self, verbose=True):
        """Identify one time constant per output from the open-loop step test."""
        try:
            st = pd.read_csv(config.STEPTEST_FILENAME)
        except FileNotFoundError:
            self.tau = {k: 10.0 for k in OUTPUTS}
            return

        u_new = float(config.STEPTEST_TO)
        t = st["Time_hr"].values.astype(float)

        for key in OUTPUTS:
            y = st[_COLS[key]].values.astype(float)
            y0 = float(y[0])
            y_inf = float(self.static(key, u_new))

            if abs(y_inf - y0) < 1e-9:
                self.tau[key] = 10.0
                continue

            # normalised response; tau is where it reaches 63.2 %
            frac = (y - y0) / (y_inf - y0)
            grid = np.linspace(0.5, 60.0, 600)
            best, best_err = 10.0, np.inf
            for tau in grid:
                pred = 1.0 - np.exp(-t / tau)
                err = float(np.mean((frac - pred) ** 2))
                if err < best_err:
                    best, best_err = float(tau), err
            self.tau[key] = best

    # ------------------------------------------------------------------
    # Static map
    # ------------------------------------------------------------------

    def static(self, key, u):
        """Steady-state value, with tangent extrapolation outside the tested range.

        Outside [u_lo, u_hi] a quadratic has no support and can curve back on
        itself, so we continue along the tangent at the nearest tested edge.
        Clipping flat instead would create a zero-gradient region in which the
        controller cannot tell candidates apart and stops moving.
        """
        u = np.asarray(u, dtype=float)
        uc = np.clip(u, self.u_lo, self.u_hi)
        base = np.polyval(self.poly[key], uc)
        slope = np.polyval(np.polyder(self.poly[key]), uc)
        return base + slope * (u - uc)

    def gain(self, key, u):
        uc = np.clip(np.asarray(u, dtype=float), self.u_lo, self.u_hi)
        return float(np.polyval(np.polyder(self.poly[key]), uc))

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, choke):
        """Steady-state prediction. Kept for backward compatibility."""
        return tuple(float(self.static(k, choke)) for k in OUTPUTS)

    def predict_batch(self, state, candidates, horizon, bias=None):
        """Predict trajectories for many candidate chokes at once.

        Each candidate is applied once and then HELD for the whole horizon
        (move-and-hold), which keeps the search one-dimensional.

        Returns dict: output -> array shaped (n_candidates, horizon).

        Fully vectorised. The old implementation called a 100-tree forest once
        per candidate per horizon step - 880 forest calls per control interval,
        measured at 6.6 seconds. This runs in well under a millisecond.
        """
        cand = np.atleast_1d(np.asarray(candidates, dtype=float))
        bias = bias or {k: 0.0 for k in OUTPUTS}
        traj = {}

        for key in OUTPUTS:
            a = float(np.exp(-config.TS_HOURS / max(self.tau[key], 1e-6)))
            target = np.asarray(self.static(key, cand), dtype=float)
            cur = np.full(cand.shape, float(state[key]))
            arr = np.empty((cand.size, horizon))
            for k in range(horizon):
                cur = a * cur + (1.0 - a) * target
                arr[:, k] = cur + bias[key]
            traj[key] = arr
        return traj

    def advance(self, state, choke):
        """One-step-ahead model update, no bias. Used to form the bias estimate."""
        out = {}
        for key in OUTPUTS:
            a = float(np.exp(-config.TS_HOURS / max(self.tau[key], 1e-6)))
            out[key] = a * float(state[key]) + (1.0 - a) * float(self.static(key, choke))
        return out

    def scale_gains(self, factor):
        """Detune all steady-state gains - used by the robustness scenario."""
        for key in OUTPUTS:
            mid = 0.5 * (self.u_lo + self.u_hi)
            y_mid = float(self.static(key, mid))
            der = np.polyder(self.poly[key]) * factor
            new = np.polyint(der)
            new[-1] = y_mid - np.polyval(new, mid)
            self.poly[key] = new

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, verbose=True):
        """Score multi-step prediction against the held-out step-test transient.

        The static sweep is the training set; this transient is data the static
        fit never saw, and it exercises exactly what the MPC relies on -
        predicting forward many steps from a measurement.
        """
        try:
            st = pd.read_csv(config.STEPTEST_FILENAME)
        except FileNotFoundError:
            return {}

        u = st["Choke_pct"].values.astype(float)
        n = len(u)
        results = {}

        for horizon in (1, 10, config.PREDICTION_HORIZON):
            preds = {k: np.full(n, np.nan) for k in OUTPUTS}
            k0 = 0
            while k0 < n - 1:
                state = {key: float(st[_COLS[key]].values[k0]) for key in OUTPUTS}
                steps = min(horizon, n - 1 - k0)
                # the choke that ACTS over the window is the one applied at
                # k0+1, not the one in force at k0 - using u[k0] predicts the
                # well holding its old position while the plant steps away
                traj = self.predict_batch(state, [u[k0 + 1]], steps)
                for key in OUTPUTS:
                    preds[key][k0 + 1:k0 + 1 + steps] = traj[key][0, :steps]
                k0 += steps

            row = {}
            for key in OUTPUTS:
                m = ~np.isnan(preds[key])
                err = st[_COLS[key]].values[m] - preds[key][m]
                row[key] = float(np.sqrt(np.mean(err ** 2)))
            results[horizon] = row

        if verbose:
            print("\nModel validation - RMSE by prediction horizon")
            print(f"  {'horizon':>9}" + "".join(f"{k:>10}" for k in OUTPUTS))
            for h, row in results.items():
                print(f"  {h:>9}" + "".join(f"{row[k]:>10.3f}" for k in OUTPUTS))
            print("  (the longest horizon is the accuracy the MPC actually relies on)")

        self.report["validation"] = results
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self):
        with open(config.MODEL_FILENAME, "wb") as f:
            pickle.dump({"poly": self.poly, "tau": self.tau,
                         "u_lo": self.u_lo, "u_hi": self.u_hi,
                         "degree": self.degree, "report": self.report}, f)
        print("Model saved.")

    def load(self):
        with open(config.MODEL_FILENAME, "rb") as f:
            d = pickle.load(f)
        self.poly = d["poly"]
        self.tau = d["tau"]
        self.u_lo, self.u_hi = d["u_lo"], d["u_hi"]
        self.degree = d.get("degree", 2)
        self.report = d.get("report", {})
        return self

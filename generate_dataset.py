"""
generate_dataset.py

Generates training data from the Honeywell simulator.

Two datasets are produced:

  train_dataset.csv    steady-state map:  choke -> Oil, WHP, FLP, BHP
                       Recorded ONLY after the well has settled.

  steptest_dataset.csv one open-loop step change, full transient recorded.
                       Used to identify the time constants.

Also computes the safety limits and the measurement noise sigma, both of which
are needed on every startup.
"""

import csv

import numpy as np

from well import Simulator
import config


class DatasetGenerator:

    def __init__(self):
        self.sim = Simulator()

    # ------------------------------------------------------------------
    # Characterisation  -- RUNS FIRST, ON THE PROVIDED SIMULATOR
    # ------------------------------------------------------------------

    def characterize(self, verbose=True):
        """Measure the plant before assuming anything about it.

        The brief states four separate times that the simulator is PROVIDED and
        is the source of process behaviour. Everything this pipeline needs to
        know about timescales and constraint direction is therefore measured
        here, on whatever simulator is actually loaded, and written back into
        config. Nothing downstream carries a number tuned to a stand-in.

        Determines:
          1. time constants        -> SETTLE_HOURS and PREDICTION_HORIZON
          2. constraint direction  -> lower bounds or upper bounds
        """
        PROBE = 400          # generous: long enough for any plausible tau

        # --- 1. time constants from one open-loop step -------------------
        self.sim.reset()
        for _ in range(PROBE):
            self.sim.step(config.STEPTEST_FROM)
        y0 = np.asarray(self.sim.step(config.STEPTEST_FROM), dtype=float)

        traj = []
        for _ in range(PROBE):
            traj.append(self.sim.step(config.STEPTEST_TO))
        traj = np.asarray(traj, dtype=float)
        y_inf = traj[-config.RECORD_HOURS:].mean(axis=0)

        t = np.arange(1, len(traj) + 1, dtype=float)
        taus = {}
        for i, key in enumerate(("Oil", "WHP", "FLP", "BHP")):
            span = y_inf[i] - y0[i]
            if abs(span) < 1e-9:
                taus[key] = 1.0
                continue
            frac = (traj[:, i] - y0[i]) / span
            # first crossing of 63.2 % is the time constant
            idx = np.flatnonzero(frac >= 0.632)
            taus[key] = float(t[idx[0]]) if idx.size else float(t[-1])

        tau_max = max(taus.values())
        config.SETTLE_HOURS = int(np.ceil(4.0 * tau_max))
        config.PREDICTION_HORIZON = int(np.ceil(3.0 * tau_max))
        config.STEPTEST_HOURS = int(np.ceil(6.0 * tau_max))

        # --- 2. constraint direction -------------------------------------
        lows, highs = [], []
        for choke, bag in ((25, lows), (75, highs)):
            self.sim.reset()
            for _ in range(config.SETTLE_HOURS):
                self.sim.step(choke)
            bag.append(np.mean([self.sim.step(choke)
                                for _ in range(config.RECORD_HOURS)], axis=0))
        lo_p, hi_p = lows[0][1:], highs[0][1:]      # WHP, FLP, BHP
        falling = bool(np.all(hi_p < lo_p))
        config.CONSTRAINT_DIRECTION = "lower" if falling else "upper"

        if verbose:
            print("Time constants (measured): " +
                  "  ".join(f"{k} {v:.1f} h" for k, v in taus.items()))
            print(f"  -> SETTLE_HOURS = {config.SETTLE_HOURS} (4 tau), "
                  f"PREDICTION_HORIZON = {config.PREDICTION_HORIZON} (3 tau)")
            arrow = "fall" if falling else "rise"
            print(f"Pressures {arrow} as the choke opens "
                  f"-> limits are {config.CONSTRAINT_DIRECTION} bounds")

        self.taus = taus
        return {"tau": taus, "direction": config.CONSTRAINT_DIRECTION}

    # ------------------------------------------------------------------
    # Safety limits
    # ------------------------------------------------------------------

    def compute_limits(self, verbose=True):
        """Derive the operating envelope by holding each choke until SETTLED.

        Called on every startup from main.run(), never cached. Previously this
        lived inside generate(), which train_if_needed() skipped whenever the
        model pickle existed - so the limits stayed None and the second run
        crashed inside safe().

        Each limit is the MEAN of the last RECORD_HOURS samples, not a single
        reading. One sample carries the full measurement noise straight into the
        limit, which then shifts the whole operating envelope by a random couple
        of psi.
        """
        for name, choke, idx in (
            ("WHP_LIMIT", config.LIMIT_CHOKE_WHP, 1),
            ("FLP_LIMIT", config.LIMIT_CHOKE_FLP, 2),
            ("BHP_LIMIT", config.LIMIT_CHOKE_BHP, 3),
        ):
            self.sim.reset()
            for _ in range(config.SETTLE_HOURS):
                self.sim.step(choke)
            tail = [self.sim.step(choke)[idx] for _ in range(config.RECORD_HOURS)]
            setattr(config, name, float(np.mean(tail)))

        if verbose:
            print("Pressure limits (settled, averaged)")
            print(f"  WHP >= {config.WHP_LIMIT:8.2f} psi   (at {config.LIMIT_CHOKE_WHP} % choke)")
            print(f"  FLP >= {config.FLP_LIMIT:8.2f} psi   (at {config.LIMIT_CHOKE_FLP} % choke)")
            print(f"  BHP >= {config.BHP_LIMIT:8.2f} psi   (at {config.LIMIT_CHOKE_BHP} % choke)")

        return config.WHP_LIMIT, config.FLP_LIMIT, config.BHP_LIMIT

    # ------------------------------------------------------------------
    # Measurement noise
    # ------------------------------------------------------------------

    def measure_noise(self, choke=50, verbose=True):
        """Estimate per-output noise sigma from a settled hold.

        Sizes the constraint back-off. Detrended so a slow residual drift is not
        counted as noise.
        """
        self.sim.reset()
        for _ in range(config.SETTLE_HOURS):
            self.sim.step(choke)

        rows = [self.sim.step(choke) for _ in range(config.NOISE_PROBE_HOURS)]
        arr = np.asarray(rows, dtype=float)

        sigma = {}
        x = np.arange(arr.shape[0])
        for i, key in enumerate(("Oil", "WHP", "FLP", "BHP")):
            y = arr[:, i]
            resid = y - np.polyval(np.polyfit(x, y, 1), x)
            sigma[key] = float(np.std(resid, ddof=2))

        if verbose:
            print("Measurement noise (1 sigma): " +
                  "  ".join(f"{k} {v:.3f}" for k, v in sigma.items()))
        return sigma

    # ------------------------------------------------------------------
    # Steady-state sweep
    # ------------------------------------------------------------------

    def generate(self, verbose=True):
        """Sweep the choke and record ONLY settled data."""

        if verbose:
            print(f"\nGenerating steady-state map "
                  f"({config.SETTLE_HOURS} h settle + {config.RECORD_HOURS} h record "
                  f"per level)...")

        with open(config.DATASET_FILENAME, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Time_hr", "Choke_pct", "OilRate_bbl_hr",
                             "WHP_psi", "FLP_psi", "BHP_psi"])

            for choke in range(config.MIN_CHOKE,
                               config.MAX_CHOKE + 1,
                               config.CHOKE_STEP):

                self.sim.reset()

                # settle - record nothing
                for _ in range(config.SETTLE_HOURS):
                    self.sim.step(choke)

                # now the well is at steady state; record
                for hour in range(config.RECORD_HOURS):
                    oil, whp, flp, bhp = self.sim.step(choke)
                    writer.writerow([hour, choke, oil, whp, flp, bhp])

        if verbose:
            print(f"Saved {config.DATASET_FILENAME}")

        self.generate_steptest(verbose=verbose)

    # ------------------------------------------------------------------
    # Step test for dynamics
    # ------------------------------------------------------------------

    def generate_steptest(self, verbose=True):
        """One open-loop step, full transient recorded, for time-constant fitting."""

        if verbose:
            print(f"Running step test {config.STEPTEST_FROM} -> "
                  f"{config.STEPTEST_TO} % ...")

        self.sim.reset()
        for _ in range(config.SETTLE_HOURS):
            self.sim.step(config.STEPTEST_FROM)

        with open(config.STEPTEST_FILENAME, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Time_hr", "Choke_pct", "OilRate_bbl_hr",
                             "WHP_psi", "FLP_psi", "BHP_psi"])

            # one sample at the old level so the step edge is captured
            oil, whp, flp, bhp = self.sim.step(config.STEPTEST_FROM)
            writer.writerow([0, config.STEPTEST_FROM, oil, whp, flp, bhp])

            for hour in range(1, config.STEPTEST_HOURS + 1):
                oil, whp, flp, bhp = self.sim.step(config.STEPTEST_TO)
                writer.writerow([hour, config.STEPTEST_TO, oil, whp, flp, bhp])

        if verbose:
            print(f"Saved {config.STEPTEST_FILENAME}")

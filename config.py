"""
config.py
---------
Configuration for the Honeywell Autonomous Choke Controller.

Every tunable lives here. No magic numbers anywhere else.
"""

# ==========================================================
# Plant / problem definition (given by the challenge brief)
# ==========================================================

TS_HOURS = 1.0            # control interval
MIN_CHOKE = 0
MAX_CHOKE = 100
MAX_CHOKE_MOVE = 5        # max |delta u| per control interval, %


# ==========================================================
# Dataset generation
# ==========================================================

CHOKE_STEP = 5
"""Choke increment for the steady-state sweep.

5 % rather than 1 % because each level now costs SETTLE_HOURS + RECORD_HOURS
simulator calls. 21 levels across 0-100 is ample for a smooth monotone map."""

SETTLE_HOURS = 80
"""Hours held at each choke BEFORE recording anything.

AUTO-TUNED. characterize() overwrites this with ceil(4 * tau_slowest) measured
from the PROVIDED simulator. The value here is only a pre-characterisation
default - never rely on it, because it was set against a development stand-in
and the real plant may be faster or far slower.

THIS IS THE MOST IMPORTANT NUMBER IN THE FILE. The previous version recorded
every hour from reset, so the training data averaged the settling transient in
with the steady state. Measured effect: the model believed BHP at 70 % choke was
2928 psi when it truly settles to 2874 - a 54 psi optimistic bias that grew to
79 psi at 100 % choke. The controller then parked at 74 % believing it was safe
and violated the limits on 200 out of 200 samples.

BHP has the slowest response (tau ~ 21 h), so 80 h is ~4 time constants."""

RECORD_HOURS = 20
"""Hours recorded per choke level, after settling. Averaging 20 settled samples
also beats the measurement noise down by sqrt(20)."""

# Step test used to identify the dynamics (time constants)
STEPTEST_FROM = 30
STEPTEST_TO = 60
STEPTEST_HOURS = 150


# ==========================================================
# MPC parameters
# ==========================================================

PREDICTION_HORIZON = 60
"""Prediction horizon in control intervals (hours).

AUTO-TUNED. characterize() overwrites this with ceil(3 * tau_slowest) measured
from the provided simulator.

Rule: N >= 3 * tau_slowest / Ts. BHP's tau is ~21 h, so 60 covers three time
constants. A short horizon looks fine on the oil-rate trace and then approves a
move that drives BHP under its floor 40 hours later."""

N_CANDIDATES = 41
"""Candidate moves spanning [-MAX_CHOKE_MOVE, +MAX_CHOKE_MOVE].
41 gives 0.25 % resolution. All candidates are evaluated in ONE batched model
call, so the cost is negligible."""


# ==========================================================
# Controller cost weights
# ==========================================================

WEIGHT_TRACKING = 1.0
WEIGHT_CHOKE_MOVEMENT = 0.6
WEIGHT_VIOLATION = 1_000_000.0
"""Applied only in the fallback path, when no candidate is feasible. Large
enough to dominate tracking so the controller always takes the least-unsafe move."""


# ==========================================================
# Safety limits
# ==========================================================
# The challenge statement names WHP / FLP / BHP as active constraints but gives
# no numeric values. We derive them by holding the choke at a maximum
# sustainable opening until SETTLED, then reading the pressure off.
#
# Filled by limits.compute_limits() on EVERY startup - never cached. The previous
# version computed them inside dataset generation, which was skipped whenever the
# model pickle already existed, so the second run always crashed with
# "'<' not supported between instances of 'float' and 'NoneType'".

CONSTRAINT_DIRECTION = "lower"
"""Whether the pressure limits are lower or upper bounds.

AUTO-DETECTED by characterize(): it compares settled pressures at a low and a
high choke opening. If pressures FALL as the choke opens (the physical case for
a naturally flowing well) the limits are lower bounds. Detected rather than
assumed, because getting this backwards is exactly what froze an earlier version
of this controller - every candidate was judged unsafe and the choke never moved.
"""

LIMIT_CHOKE_WHP = 70
LIMIT_CHOKE_FLP = 72
LIMIT_CHOKE_BHP = 75

WHP_LIMIT = None
FLP_LIMIT = None
BHP_LIMIT = None


# ==========================================================
# Constraint back-off
# ==========================================================

BACKOFF_SIGMA = 2.0
"""The controller steers to (limit + back-off), never to the limit itself.

Because each limit is defined AS the settled pressure at a chosen choke opening,
that opening sits exactly on the boundary by construction. With any measurement
noise at all, riding it is a coin flip on every sample. Back-off is sized in
multiples of the MEASURED noise sigma so it self-scales to whatever simulator we
are handed on test day."""

BACKOFF_FLOOR = {"WHP": 2.0, "FLP": 1.5, "BHP": 8.0}
NOISE_PROBE_HOURS = 40


# ==========================================================
# Scenarios   (target, duration_hours)
# ==========================================================

INITIAL_CHOKE = 30

SCENARIO_A = {
    "name": "Scenario A - Startup to Target",
    "start_choke": 20,
    "segments": [(0.85, 120)],
}

SCENARIO_B = {
    "name": "Scenario B - Target Tracking",
    "start_choke": 35,
    "segments": [(0.72, 70), (0.95, 70), (0.82, 70)],
}
"""Each segment is 70 h (>3 tau_BHP) so the well actually settles before the
next change. The old version gave each of four targets 25 h - about 1.2 time
constants - so nothing ever reached steady state. The final target is also kept
inside capacity; the old list ended on 180 bbl/hr, which is unreachable, so
'target tracking' was really testing infeasibility."""

SCENARIO_C = {
    "name": "Scenario C - Infeasible Target",
    "start_choke": 30,
    "segments": [(2.00, 150)],
}

SCENARIO_D = {
    "name": "Scenario D - Robustness (model gain -20%)",
    "start_choke": 35,
    "segments": [(0.72, 70), (0.95, 70)],
    "gain_mismatch": 0.80,
}

TARGETS_ARE_FRACTIONS = True
"""Scenario targets are FRACTIONS of the discovered maximum safe rate, not
absolute bbl/hr. Absolute numbers were tuned to a development stand-in whose
capacity happened to be ~161 bbl/hr; on the provided simulator they could be
trivially easy or permanently unreachable, which would silently invalidate every
scenario. Resolved to absolute values at runtime by scenarios.resolve_targets().

Scenario C is deliberately 2.00x - unreachable on any plant."""

MAX_SAFE_RATE = None      # discovered at runtime

SCENARIOS = {"A": SCENARIO_A, "B": SCENARIO_B, "C": SCENARIO_C, "D": SCENARIO_D}


# ==========================================================
# Plotting / IO
# ==========================================================

FIGURE_SIZE = (12, 6)
SAVE_PLOTS = True
SHOW_PLOTS = False          # True blocks the script between scenarios
OUTPUT_FOLDER = "outputs"

MODEL_FILENAME = "prediction_model.pkl"
DATASET_FILENAME = "train_dataset.csv"
STEPTEST_FILENAME = "steptest_dataset.csv"

RANDOM_SEED = 42

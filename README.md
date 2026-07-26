# Autonomous Production Choke Controller

Honeywell Campus Challenge — constraint-aware predictive control of a single
naturally flowing oil well.

```bash
pip install -r requirements.txt
python main.py            # full pipeline, ~9 seconds
```

## Results

| | A · Startup | B · Tracking | C · Infeasible | D · Robustness |
|---|---|---|---|---|
| Target (bbl/hr) | 150 | 120→155→135 | 300 | 120→155, model −20 % |
| Achieved | 149.9 | 135.0 | **160.5** | 155.0 |
| % of target | 99.9 | 100.0 | 53.5 *(unreachable)* | 100.0 |
| Steady-state offset | −0.10 | −0.01 | n/a | +0.02 |
| Settling (h) | 23 | 13 | 27 | 13 |
| Overshoot | 0.81 % | 0.57 % | 0.91 % | 0.78 % |
| **Violations** | **0** | **0** | **0** | **0** |
| Max \|Δu\| | 5.00 % | 5.00 % | 5.00 % | 5.00 % |

**Zero constraint violations across all scenarios.** Max choke move 5.000 %
against a 5 % limit. Scenario C settles at the maximum safe rate; WHP is the
binding constraint.

Model validation on a held-out step transient — RMSE by prediction horizon:

| Horizon | Oil | WHP | FLP | BHP |
|---|---|---|---|---|
| 1-step | 0.63 | 0.75 | 0.66 | 0.68 |
| 10-step | 1.17 | 0.81 | 0.73 | 0.95 |
| **60-step** | **1.62** | **1.23** | **0.83** | **1.97** |

The 60-step row is the accuracy the MPC actually relies on — it commits to a
move based on a prediction 60 intervals out.

## Running against the Honeywell simulator

**Drop Honeywell's `simulator.py` into this folder. That is the entire step.**

`well.py` resolves the simulator at import time and Honeywell's file always
wins:

```
1. simulator.py            <- provided by Honeywell. Takes precedence.
2. dev_stub_simulator.py   <- development stand-in, loudly announced.
```

There is deliberately **no `simulator.py` in this package**. The brief states
four times that the simulator is provided and is the source of process
behaviour, so shipping a file with that name would risk silently shadowing the
real one and would look like we had built our own. The stand-in is named
`dev_stub_simulator.py` so it can never be mistaken for a submission artefact,
and every run prints which simulator is active:

```
==============================================================
SIMULATOR IN USE: simulator.py  (PROVIDED BY HONEYWELL)
==============================================================
```

If the stand-in is ever used, the banner says so, a `RuntimeWarning` fires, and
the run ends with an explicit instruction to replace it.

### Nothing is hardcoded to a particular plant

`characterize()` runs first, on whatever simulator is loaded, and measures:

| Measured | Sets |
|---|---|
| time constants from an open-loop step | `SETTLE_HOURS` = 4τ, `PREDICTION_HORIZON` = 3τ |
| settled pressures at low vs high choke | constraint direction (lower or upper bounds) |
| noise σ from a settled hold | constraint back-off |
| max safe rate from the identified maps | resolves the fractional scenario targets |

Scenario targets are stored as **fractions of the discovered maximum safe rate**
(A 0.85, B 0.72→0.95→0.82, C 2.00, D 0.72→0.95), not absolute bbl/hr. Absolute
numbers would be tuned to whatever plant they were written against and could be
trivially easy or permanently unreachable on the real one.

`well.Simulator` also normalises the interface: `reset()` may take a starting
choke, take nothing, or not exist; `step()` may return a 4-tuple, a dict or a
namedtuple with any of the usual key spellings.

**Verified** against a deliberately different test plant (τ_BHP 8 h instead of
21 h, different capacity, different pressure levels, dict return with different
key names). The pipeline re-derived `SETTLE_HOURS` = 32 and
`PREDICTION_HORIZON` = 24 on its own and produced **zero violations at 99.0 %
of the maximum safe rate** with no code changes.

## Files

| File | Role |
|---|---|
| `config.py` | every tunable, each justified in place |
| `generate_dataset.py` | settled steady-state sweep + step test + limits + noise |
| `predictor.py` | static map + first-order dynamics, batched prediction, validation |
| `controller.py` | MPC: bias correction, constraint rejection, fallback |
| `scenarios.py` | A / B / C / D runners |
| `metrics.py` | KPIs |
| `plotting.py` | six required trends + ramp proof + diagnostics |
| `main.py` | pipeline |

## What was fixed

**Training data averaged transients with steady state.** The sweep recorded
every hour from reset, so the model learned a blend of the settling curve and
the steady state. Measured bias: BHP believed to be 2928 psi at 70 % choke when
it truly settles to 2874 — a 54 psi optimistic error growing to 79 psi at 100 %.
The controller parked at 74 % believing it was safe and violated on **200 of 200
samples**. Fixed by settling 80 h (≈4 τ_BHP) before recording.

**Limits were computed inside dataset generation**, which was skipped whenever
the model pickle existed — so the second run always died with
`'<' not supported between instances of 'float' and 'NoneType'`. Limits and
noise are now recomputed on every startup, independent of the model cache.

**`config.DATASET_HOURS` did not exist** → `AttributeError`.

**The controller received no measurements.** `compute(current_choke, target)` is
open-loop; the brief requires the controller to receive Q, WHP, FLP, BHP and the
choke position. It now does, and forms a bias estimate from them:
`bias = measured − predicted_for_now`. Without it there is no integral action,
so model gain error becomes permanent offset. Scenario D proves it works by
detuning the model gains −20 % and still tracking offset-free.

**The prediction horizon was dead code.** The static predictor returned the
identical value for all 20 horizon steps (verified:
`[2968.705, 2968.705, 2968.705]`), so the loop multiplied the cost by 20 and
cost 20× compute for nothing. The model now has real first-order dynamics
(τ identified from a step test: Oil 5.4 h, WHP/FLP 8.9 h, BHP 22.1 h) and the
horizon is 60 — three τ_BHP.

**No back-off.** Each limit is defined *as* the settled pressure at a chosen
choke, so that opening sits exactly on the boundary by construction. The
controller now steers to limit + 2σ of measured noise.

**Speed.** 880 single-row forest calls per control interval measured at 6.6 s
per step — 33 minutes for three scenarios. Prediction is now fully vectorised
across all candidates: **8.6 s for the entire pipeline.**

**Fallback** now takes the least-unsafe move instead of silently freezing;
candidates use `np.linspace` (0.25 % resolution, float-safe) instead of
`range()`; scenario segments are 70 h so the well actually settles; Scenario B
no longer ends on an unreachable 180 bbl/hr; `plt.show()` no longer blocks.

**Added:** `metrics.py` with violation counts and KPI export, multi-step model
validation, the six required per-scenario trends on one shared axis plus a
ramp-rate proof panel, and controller diagnostics.

## Assumptions

- Limits are derived by holding the choke at a maximum sustainable opening
  (WHP 70 %, FLP 72 %, BHP 75 %) until settled. The brief supplies no numeric
  limits; if the provided simulator publishes them, override in `config.py`.
- Linear dynamics with a nonlinear static gain over 0–100 % choke
- No dead time observable at Ts = 1 h
- No reservoir depletion, no GOR or water-cut change (per the brief)
- Tangent extrapolation outside the identified range

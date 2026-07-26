"""
main.py

Honeywell Autonomous Choke Controller - full pipeline.

    python main.py

Pipeline:
  1. compute safety limits and measurement noise      (EVERY startup)
  2. generate settled steady-state data + a step test (cached)
  3. train and validate the process model             (cached)
  4. run Scenarios A, B, C and robustness bonus D
  5. plots + KPI table
"""

import os

import config
import well
from generate_dataset import DatasetGenerator
from predictor import Predictor
from plotting import Plotter
from metrics import compute_kpis, print_table, save_csv
from scenarios import scenario_A, scenario_B, scenario_C, scenario_D


def prepare():
    """Limits and noise are recomputed on every run; the model is cached.

    Splitting these matters. Previously the limits were a side effect of dataset
    generation, so whenever prediction_model.pkl already existed the limits
    stayed None and the run died inside safe() with a TypeError.
    """
    print(well.banner())
    gen = DatasetGenerator()
    print('step(40) ->  Oil %.3f  WHP %.3f  FLP %.3f  BHP %.3f'
          % gen.sim.step(40.0))

    print("\n" + "=" * 62)
    print("PLANT CHARACTERISATION  (measured, not assumed)")
    print("=" * 62)
    gen.characterize()

    print("\n" + "=" * 62)
    print("OPERATING ENVELOPE")
    print("=" * 62)
    gen.compute_limits()
    sigma = gen.measure_noise()

    if os.path.exists(config.MODEL_FILENAME):
        print("\nModel already exists - loading.")
        predictor = Predictor().load()
    else:
        gen.generate()
        predictor = Predictor()
        predictor.train()
        predictor.save()

    predictor.validate()

    # Discover the maximum safe rate, then resolve the fractional scenario
    # targets against it. This is what makes the scenarios meaningful on a
    # simulator we have never seen.
    from controller import max_safe_rate
    backoff = {k: max(config.BACKOFF_SIGMA * float(sigma.get(k, 0.0)),
                      config.BACKOFF_FLOOR[k]) for k in ("WHP", "FLP", "BHP")}
    ms = max_safe_rate(predictor, backoff)
    config.MAX_SAFE_RATE = ms["Q"] if ms["feasible"] else None

    print("\n" + "=" * 62)
    print("MAXIMUM SAFE PRODUCTION RATE")
    print("=" * 62)
    if ms["feasible"]:
        print(f"  {ms['Q']:.2f} bbl/hr at {ms['u']:.2f} % choke, "
              f"limited by {ms['limiting']}")
        print("  (computed independently of the controller, by sweeping the")
        print("   identified steady-state maps - Scenario C is scored against it)")
    else:
        print("  No feasible operating point found within the choke range.")

    return predictor, sigma


def run():
    predictor, sigma = prepare()

    plotter = Plotter()
    plotter.steady_state_plot(predictor)

    runners = [("A", scenario_A), ("B", scenario_B),
               ("C", scenario_C), ("D", scenario_D)]

    kpis = []
    for key, fn in runners:
        print("\n" + "=" * 62)
        print(config.SCENARIOS[key]["name"])
        print("=" * 62)

        history = fn(sigma=sigma)
        plotter.all_plots(history, key)

        k = compute_kpis(history, label=key)
        kpis.append(k)

        print(f"  achieved {k['achieved_bbl_hr']:.1f} bbl/hr "
              f"({k['pct_of_target']:.1f} % of target) at choke {k['final_choke']:.2f} %")
        print(f"  violations {k['violations_total']}   "
              f"max |du| {k['max_abs_du']:.2f} %   "
              f"min BHP margin {k['min_margin_BHP']:+.2f} psi")

    print("\n" + "=" * 62)
    print("RESULTS")
    print("=" * 62)
    print_table(kpis)
    save_csv(kpis, os.path.join(config.OUTPUT_FOLDER, "kpis.csv"))

    total = sum(k["violations_total"] for k in kpis)
    max_du = max(k["max_abs_du"] for k in kpis)
    print("\n" + "=" * 62)
    print(f"  CONSTRAINT VIOLATIONS ACROSS ALL SCENARIOS : {total}")
    print(f"  MAXIMUM CHOKE MOVE OBSERVED                : {max_du:.3f} % "
          f"(limit {config.MAX_CHOKE_MOVE:.1f} %)")
    print("=" * 62)
    if config.MAX_SAFE_RATE:
        c = next(k for k in kpis if k["scenario"] == "C")
        print(f"  SCENARIO C reached {c['achieved_bbl_hr']:.1f} bbl/hr = "
              f"{100 * c['achieved_bbl_hr'] / config.MAX_SAFE_RATE:.1f} % "
              f"of the maximum safe rate")
    print(f"\nFigures and kpis.csv written to {config.OUTPUT_FOLDER}/")
    if not well.USING_PROVIDED_SIMULATOR:
        print("\n*** These results come from the DEVELOPMENT STAND-IN. Drop "
              "Honeywell's\n    simulator.py into this folder and re-run before "
              "submitting. ***")


if __name__ == "__main__":
    run()

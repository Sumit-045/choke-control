"""
scenarios.py

Runs the Honeywell challenge scenarios.

Changes from the previous version:
  * measurements are passed into the controller (the brief requires it)
  * the initial condition is recorded before the first move
  * durations come from config, sized at >3 tau_BHP per segment so the well
    actually settles; the old version reused SAMPLES_PER_CHOKE and gave each
    target 25 h, about 1.2 time constants
"""

import numpy as np

from well import Simulator
import config
from controller import MPCController
from predictor import Predictor


class ScenarioRunner:

    def __init__(self, predictor=None, sigma=None):
        self.sim = Simulator()
        if sigma:
            self.controller = MPCController.with_noise_backoff(sigma, predictor)
        else:
            self.controller = MPCController(predictor)

    def run(self, targets, start_choke):
        self.sim.reset()

        choke = float(start_choke)

        # settle at the starting condition so the run begins from steady state
        for _ in range(config.SETTLE_HOURS):
            oil, whp, flp, bhp = self.sim.step(choke)

        self.controller.initialise(
            {"Oil": oil, "WHP": whp, "FLP": flp, "BHP": bhp})

        history = {k: [] for k in
                   ("time", "target", "oil", "whp", "flp", "bhp", "choke")}

        for hour, target in enumerate(targets):
            # record the state the controller is about to act on
            history["time"].append(hour)
            history["target"].append(target)
            history["oil"].append(oil)
            history["whp"].append(whp)
            history["flp"].append(flp)
            history["bhp"].append(bhp)
            history["choke"].append(choke)

            choke = self.controller.compute(choke, target,
                                            oil=oil, whp=whp, flp=flp, bhp=bhp)
            oil, whp, flp, bhp = self.sim.step(choke)

        history["log"] = self.controller.log
        return history


def resolve_targets(segments):
    """Turn fractional targets into absolute bbl/hr.

    Scenario targets are stored as FRACTIONS of the discovered maximum safe
    rate. Absolute numbers would be tuned to whichever simulator they were
    written against; on the provided plant they could be trivially easy or
    permanently unreachable, silently invalidating every scenario.
    """
    if not config.TARGETS_ARE_FRACTIONS:
        return list(segments)
    if config.MAX_SAFE_RATE is None:
        raise RuntimeError("config.MAX_SAFE_RATE not set - run main.prepare() "
                           "so the maximum safe rate is discovered first.")
    return [(round(f * config.MAX_SAFE_RATE, 1), h) for f, h in segments]


def build_targets(segments):
    out = []
    for target, hours in resolve_targets(segments):
        out.extend([float(target)] * int(hours))
    return out


def run_scenario(key, sigma=None):
    spec = config.SCENARIOS[key]

    predictor = Predictor().load()
    if "gain_mismatch" in spec:
        # plant untouched; only the controller's belief about it is detuned
        predictor.scale_gains(spec["gain_mismatch"])

    runner = ScenarioRunner(predictor=predictor, sigma=sigma)
    targets = build_targets(spec["segments"])
    history = runner.run(targets, spec["start_choke"])
    history["name"] = spec["name"]
    return history


def scenario_A(sigma=None):
    return run_scenario("A", sigma)


def scenario_B(sigma=None):
    return run_scenario("B", sigma)


def scenario_C(sigma=None):
    return run_scenario("C", sigma)


def scenario_D(sigma=None):
    return run_scenario("D", sigma)

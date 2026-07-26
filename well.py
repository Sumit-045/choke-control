"""
well.py

Resolves WHICH simulator to use, and normalises its interface.

THE BRIEF IS EXPLICIT AND REPEATED: the simulator is PROVIDED by Honeywell and
must be treated as the source of process behaviour. Students do not build it.

Import order, deliberately:

    1. simulator.py            <- Honeywell's. Always wins if present.
    2. dev_stub_simulator.py   <- development stand-in, LOUDLY announced.

The stand-in exists only so the pipeline could be written and debugged before
the real simulator was available. It is named so that it can never be mistaken
for a submission artefact and can never silently shadow the provided file. Drop
Honeywell's simulator.py into this folder and it takes over automatically -
nothing else changes.
"""

import warnings

import numpy as np

USING_PROVIDED_SIMULATOR = False
SIMULATOR_SOURCE = None

try:
    from simulator import Simulator as _RawSimulator      # Honeywell's
    USING_PROVIDED_SIMULATOR = True
    SIMULATOR_SOURCE = "simulator.py  (PROVIDED BY HONEYWELL)"
except ImportError:
    from dev_stub_simulator import Simulator as _RawSimulator
    SIMULATOR_SOURCE = "dev_stub_simulator.py  (DEVELOPMENT STAND-IN)"


_ORDER = ("Oil", "WHP", "FLP", "BHP")

_ALIASES = {
    "Oil": {"q", "oil", "oilrate", "oil_rate", "rate", "flow", "flowrate",
            "oilflowrate", "oil_flow_rate", "production"},
    "WHP": {"whp", "wellhead_pressure", "p_wh", "pwh", "wellhead"},
    "FLP": {"flp", "flowline_pressure", "p_fl", "pfl", "flowline"},
    "BHP": {"bhp", "bottomhole_pressure", "bottom_hole_pressure", "p_bh",
            "pbh", "bottomhole"},
}


def _canonical(name):
    k = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    for canon, alts in _ALIASES.items():
        if k == canon.lower() or k in alts:
            return canon
    for canon, alts in _ALIASES.items():
        for a in sorted(alts, key=len, reverse=True):
            if len(a) >= 3 and a in k:
                return canon
    return None


def banner():
    line = "=" * 62
    tag = "" if USING_PROVIDED_SIMULATOR else "   <-- REPLACE BEFORE SUBMISSION"
    return (f"{line}\nSIMULATOR IN USE: {SIMULATOR_SOURCE}{tag}\n{line}")


class Simulator:
    """Thin adapter around whichever simulator was resolved above.

    Normalises two things we cannot assume about the provided file:

      * reset() may take a starting choke, take nothing, or not exist
      * step() may return a 4-tuple, a dict, or a namedtuple

    Everything downstream sees step(u) -> (Oil, WHP, FLP, BHP).
    """

    def __init__(self, *args, **kwargs):
        try:
            self.sim = _RawSimulator(*args, **kwargs)
        except TypeError:
            self.sim = _RawSimulator()

        self._step_fn = None
        for attr in ("step", "advance", "update", "simulate", "run_step"):
            fn = getattr(self.sim, attr, None)
            if callable(fn):
                self._step_fn = fn
                break
        if self._step_fn is None and callable(self.sim):
            self._step_fn = self.sim
        if self._step_fn is None:
            raise RuntimeError(
                "Could not find a step function on the provided simulator. "
                "Tried step/advance/update/simulate/run_step.")

        self._reset_fn = None
        for attr in ("reset", "initialize", "init", "restart"):
            fn = getattr(self.sim, attr, None)
            if callable(fn):
                self._reset_fn = fn
                break

    # ------------------------------------------------------------------

    def _normalise(self, raw):
        if isinstance(raw, dict):
            out = {}
            for k, v in raw.items():
                c = _canonical(k)
                if c is not None:
                    out[c] = float(v)
            if set(out) == set(_ORDER):
                return tuple(out[k] for k in _ORDER)
            raise RuntimeError(f"step() returned dict keys {list(raw)}; "
                               f"could not map onto {_ORDER}")

        if hasattr(raw, "_asdict"):
            return self._normalise(raw._asdict())

        vals = np.asarray(raw, dtype=float).ravel()
        if vals.size != 4:
            raise RuntimeError(f"step() returned {vals.size} values, expected 4 "
                               f"in the order {_ORDER}")
        return tuple(float(v) for v in vals)

    def step(self, choke):
        return self._normalise(self._step_fn(float(np.clip(choke, 0.0, 100.0))))

    def reset(self, choke=None):
        if self._reset_fn is not None:
            try:
                self._reset_fn() if choke is None else self._reset_fn(choke)
            except TypeError:
                try:
                    self._reset_fn()
                except Exception:
                    pass
        return self

    # ------------------------------------------------------------------

    def self_test(self, verbose=True):
        """Run this FIRST on test day."""
        self.reset()
        out = self.step(40.0)
        if verbose:
            print(banner())
            print("step(40) ->  Oil %.3f  WHP %.3f  FLP %.3f  BHP %.3f" % out)
            if not USING_PROVIDED_SIMULATOR:
                warnings.warn(
                    "Running against the development stand-in. Place "
                    "Honeywell's simulator.py in this folder before submitting.",
                    RuntimeWarning)
        return out

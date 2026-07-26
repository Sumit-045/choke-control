"""Stub simulator matching the documented Honeywell interface, calibrated to the
reference dataset (IPR + friction + first-order lags). Stands in for the real one."""
import numpy as np

class Simulator:
    TAU = {"Q": 5.0, "WHP": 8.0, "FLP": 8.0, "BHP": 21.0}
    def __init__(self, u0=30.0, ts=1.0, seed=0):
        self.ts, self.rng = ts, np.random.default_rng(seed)
        self.reset(u0)
    def _ss(self, u):
        q = 379.6 * max(u, 0.0) / (max(u, 0.0) + 92.4)
        return {"Q": q, "WHP": 345.15 - 0.8092*q,
                "FLP": 239.26 - 0.5371*q, "BHP": 3481.8 - 3.7113*q}
    def reset(self, u0=30.0):
        self.u = float(u0); self.state = self._ss(self.u)
        return tuple(self.state[k] for k in ("Q","WHP","FLP","BHP"))
    def step(self, choke_position):
        u = float(np.clip(choke_position, 0, 100)); self.u = u
        t = self._ss(u); out = {}
        for k in ("Q","WHP","FLP","BHP"):
            a = np.exp(-self.ts/self.TAU[k])
            self.state[k] = a*self.state[k] + (1-a)*t[k]
            out[k] = self.state[k] + self.rng.normal(0, 0.5)
        return out["Q"], out["WHP"], out["FLP"], out["BHP"]

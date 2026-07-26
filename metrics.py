"""
metrics.py

KPIs for each closed-loop run.

A results section with plots but no numbers is not a validation. These are what
let the report state "zero violations, 12 h settling, -0.05 bbl/hr offset"
instead of "the controller works well".
"""

import numpy as np

import config

CONSTRAINED = ("whp", "flp", "bhp")


def _limits():
    return {"whp": config.WHP_LIMIT, "flp": config.FLP_LIMIT,
            "bhp": config.BHP_LIMIT}


def settling_time(t, y, sp, tol=0.02):
    """Time after which |y - sp| stays inside the band for good.

    Measured from the LAST excursion outside the band, not the first entry into
    it - a trace that enters, overshoots back out and returns has not settled at
    the first crossing.
    """
    sp = np.asarray(sp, dtype=float)
    band = np.abs(tol * np.where(sp == 0, 1.0, sp))
    outside = np.abs(np.asarray(y) - sp) > band
    if not outside.any():
        return 0.0
    last = int(np.max(np.flatnonzero(outside)))
    if last >= len(t) - 1:
        return float("nan")
    return float(t[last + 1])


def compute_kpis(history, label="", tail=20):
    t = np.asarray(history["time"], dtype=float)
    q = np.asarray(history["oil"], dtype=float)
    sp = np.asarray(history["target"], dtype=float)
    u = np.asarray(history["choke"], dtype=float)
    lim = _limits()

    du = np.diff(u, prepend=u[0])

    # violations against the TRUE limits, not the backed-off ones
    viol = {}
    for k in CONSTRAINED:
        v = np.asarray(history[k], dtype=float)
        viol[k] = int(np.sum(v < lim[k]))
    viol_total = sum(viol.values())

    margins = {k: float(np.min(np.asarray(history[k], dtype=float)) - lim[k])
               for k in CONSTRAINED}

    # last setpoint segment only
    changes = np.flatnonzero(np.diff(sp) != 0)
    seg = int(changes[-1] + 1) if len(changes) else 0
    seg_t, seg_q, seg_sp = t[seg:] - t[seg], q[seg:], sp[seg:]

    achieved = float(np.mean(q[-tail:]))
    requested = float(sp[-1])
    feasible = achieved >= 0.98 * requested

    ref = requested if feasible else achieved
    direction = 1.0
    if seg > 0:
        direction = np.sign(sp[seg] - sp[seg - 1]) or 1.0
    elif len(seg_q):
        direction = np.sign(ref - seg_q[0]) or 1.0
    overshoot = 100.0 * float(np.max(direction * (seg_q - ref))) / ref if ref else np.nan

    return {
        "scenario": label,
        "final_target": round(requested, 1),
        "achieved_bbl_hr": round(achieved, 2),
        "pct_of_target": round(100.0 * achieved / requested, 1) if requested else np.nan,
        "target_feasible": bool(feasible),
        "settling_hr": round(settling_time(seg_t, seg_q, np.full_like(seg_q, achieved)), 1),
        "offset_bbl_hr": round(achieved - requested, 3) if feasible else np.nan,
        "overshoot_pct": round(overshoot, 2),
        "IAE": round(float(np.sum(np.abs(q - sp))), 1),
        "violations_total": viol_total,
        "violations_WHP": viol["whp"],
        "violations_FLP": viol["flp"],
        "violations_BHP": viol["bhp"],
        "min_margin_WHP": round(margins["whp"], 2),
        "min_margin_FLP": round(margins["flp"], 2),
        "min_margin_BHP": round(margins["bhp"], 2),
        "max_abs_du": round(float(np.max(np.abs(du))), 3),
        "ramp_ok": bool(np.max(np.abs(du)) <= config.MAX_CHOKE_MOVE + 1e-6),
        "choke_travel": round(float(np.sum(np.abs(du))), 1),
        "final_choke": round(float(u[-1]), 2),
    }


ROWS = ["final_target", "achieved_bbl_hr", "pct_of_target", "settling_hr",
        "offset_bbl_hr", "overshoot_pct", "violations_total",
        "min_margin_BHP", "max_abs_du", "ramp_ok", "final_choke"]


def print_table(kpis):
    keys = [k["scenario"] for k in kpis]
    width = max(22, max(len(str(x)) for x in keys) + 2)
    print(f"{'':<24}" + "".join(f"{k:>{width}}" for k in keys))
    for row in ROWS:
        print(f"{row:<24}" + "".join(f"{str(k[row]):>{width}}" for k in kpis))


def save_csv(kpis, path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + [k["scenario"] for k in kpis])
        for row in kpis[0]:
            if row == "scenario":
                continue
            w.writerow([row] + [k[row] for k in kpis])

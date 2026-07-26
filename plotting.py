"""
plotting.py

Figures for each scenario.

The brief requires, per scenario: Target Oil Rate, Actual Oil Rate, WHP, FLP,
BHP and Choke Position. combined_plot() produces all six on one shared time
axis, plus a seventh panel showing the per-interval choke move against the
+/-5 % band - the fastest way to PROVE ramp compliance rather than assert it.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config

C_ACT = "#1f4e79"
C_TGT = "#c00000"
C_CHK = "#2e7d32"
C_BAND = "#f4b183"


class Plotter:

    def __init__(self):
        if config.SAVE_PLOTS:
            os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)

    def save(self, fig, filename):
        if config.SAVE_PLOTS:
            fig.savefig(os.path.join(config.OUTPUT_FOLDER, filename),
                        dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------

    def combined_plot(self, history, scenario):
        """The six required trends plus the ramp-rate proof panel."""
        t = np.asarray(history["time"], dtype=float)
        lim = {"whp": config.WHP_LIMIT, "flp": config.FLP_LIMIT,
               "bhp": config.BHP_LIMIT}

        fig, ax = plt.subplots(6, 1, figsize=(11, 13), sharex=True)

        ax[0].plot(t, history["target"], "--", color=C_TGT, lw=1.6, label="Target oil rate")
        ax[0].plot(t, history["oil"], color=C_ACT, lw=1.4, label="Actual oil rate")
        ax[0].set_ylabel("Oil rate\n(bbl/hr)")
        ax[0].legend(loc="lower right", ncol=2, fontsize=8)
        ax[0].set_title(history.get("name", f"Scenario {scenario}"),
                        loc="left", fontweight="bold")

        for a, key, name in zip(ax[1:4], ("whp", "flp", "bhp"),
                                ("WHP", "FLP", "BHP")):
            y = np.asarray(history[key], dtype=float)
            a.plot(t, y, color=C_ACT, lw=1.3)
            a.axhline(lim[key], color=C_TGT, ls="--", lw=1.4,
                      label=f"{name} limit = {lim[key]:.1f}")
            span = max(y.max() - min(y.min(), lim[key]), 1.0)
            a.set_ylim(min(y.min(), lim[key]) - 0.08 * span, y.max() + 0.08 * span)
            a.set_ylabel(f"{name}\n(psi)")
            a.legend(loc="lower right", fontsize=8)

        ax[4].plot(t, history["choke"], color=C_CHK, lw=1.5)
        ax[4].set_ylabel("Choke\n(%)")
        ax[4].set_ylim(0, 100)

        du = np.diff(np.asarray(history["choke"], dtype=float),
                     prepend=history["choke"][0])
        ax[5].axhspan(-config.MAX_CHOKE_MOVE, config.MAX_CHOKE_MOVE,
                      color=C_BAND, alpha=0.35,
                      label=f"allowed +/-{config.MAX_CHOKE_MOVE} %/interval")
        ax[5].plot(t, du, color=C_CHK, lw=1.1)
        ax[5].set_ylabel("Choke move\n(%/interval)")
        ax[5].set_xlabel("Time (hours)")
        ax[5].set_ylim(-1.6 * config.MAX_CHOKE_MOVE, 1.6 * config.MAX_CHOKE_MOVE)
        ax[5].legend(loc="lower right", fontsize=8)

        for a in ax:
            a.grid(alpha=0.3)
        fig.align_ylabels(ax)
        fig.tight_layout()
        self.save(fig, f"scenario_{scenario}_trends.png")

    # ------------------------------------------------------------------

    def oil_plot(self, history, scenario):
        fig = plt.figure(figsize=config.FIGURE_SIZE)
        plt.plot(history["time"], history["oil"], color=C_ACT, lw=2,
                 label="Actual Oil Rate")
        plt.plot(history["time"], history["target"], "--", color=C_TGT,
                 label="Target")
        plt.xlabel("Time (Hours)")
        plt.ylabel("Oil Rate (bbl/hr)")
        plt.title(f"Scenario {scenario} - Oil Tracking")
        plt.grid(True)
        plt.legend()
        self.save(fig, f"oil_tracking_{scenario}.png")

    def choke_plot(self, history, scenario):
        fig = plt.figure(figsize=config.FIGURE_SIZE)
        plt.plot(history["time"], history["choke"], color=C_CHK, lw=2)
        plt.xlabel("Time (Hours)")
        plt.ylabel("Choke (%)")
        plt.title(f"Scenario {scenario} - Choke Position")
        plt.grid(True)
        self.save(fig, f"choke_{scenario}.png")

    def diagnostics_plot(self, history, scenario):
        """Controller internals - feasible candidate count and the bias estimate."""
        log = history.get("log")
        if not log:
            return
        n = len(log["n_feasible"])
        t = np.asarray(history["time"][:n], dtype=float)

        fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
        ax[0].plot(t, log["n_feasible"], color=C_ACT, lw=1.2)
        ax[0].set_ylabel("Feasible\ncandidates")
        ax[0].set_title(f"Scenario {scenario} - controller internals",
                        loc="left", fontweight="bold")

        ax[1].plot(t, log["bias_BHP"], color=C_TGT, lw=1.2)
        ax[1].axhline(0, color="grey", ls=":", lw=1)
        ax[1].set_ylabel("BHP model\nbias (psi)")

        ax[2].fill_between(t, 0, np.asarray(log["blocked"], dtype=int),
                           step="post", color=C_TGT, alpha=0.5)
        ax[2].set_ylabel("Opening blocked\nby constraint")
        ax[2].set_yticks([0, 1])
        ax[2].set_xlabel("Time (hours)")
        for a in ax:
            a.grid(alpha=0.3)
        fig.tight_layout()
        self.save(fig, f"diagnostics_{scenario}.png")

    def steady_state_plot(self, predictor, filename="model_steady_state.png"):
        """Identified steady-state maps with the limits overlaid."""
        u = np.linspace(0, 100, 300)
        lim = {"WHP": config.WHP_LIMIT, "FLP": config.FLP_LIMIT,
               "BHP": config.BHP_LIMIT}
        fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
        for a, key in zip(ax, ("Oil", "WHP", "FLP", "BHP")):
            a.plot(u, predictor.static(key, u), color=C_ACT, lw=1.6)
            if key in lim and lim[key] is not None:
                a.axhline(lim[key], color=C_TGT, ls="--", lw=1.2, label="limit")
                a.legend(fontsize=8)
            a.set_xlabel("Choke (%)")
            a.set_ylabel(key)
            a.grid(alpha=0.3)
        fig.suptitle("Identified steady-state maps", fontweight="bold")
        fig.tight_layout()
        self.save(fig, filename)

    # ------------------------------------------------------------------

    def all_plots(self, history, scenario):
        self.combined_plot(history, scenario)
        self.oil_plot(history, scenario)
        self.choke_plot(history, scenario)
        self.diagnostics_plot(history, scenario)
        if config.SHOW_PLOTS:
            plt.show()

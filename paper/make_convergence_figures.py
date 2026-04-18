"""Generate the three convergence / sensitivity / optimisation figures.

Numbers are baked in from the Modal sweeps whose driver scripts are
in examples/: 10_cd_convergence, 12_sharpness_sensitivity, and
11_optimisation_v2. Anyone can regenerate the data by re-running
those examples on an A100; this script just plots what those runs
produced, so it does not need a GPU.

Usage:
    python paper/make_convergence_figures.py

Output:
    paper/figures/cd_convergence.pdf
    paper/figures/sharpness_sensitivity.pdf
    paper/figures/optim_summary.pdf
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.3,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

CLIFT_REF = 1.09


def fig_cd_convergence():
    """Sphere Cd grid convergence at Re=100.

    Data from examples/10_cd_convergence.py run on A100:
    time-averaged Cd over the final 30% of each run.
    """
    D = np.array([8, 16, 32, 48])
    cd_mean = np.array([1.2903, 1.1863, 1.1854, 1.1679])
    cd_std = np.array([0.0190, 0.0406, 0.0705, 0.0769])

    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.6))

    ax.errorbar(
        D, cd_mean, yerr=cd_std,
        fmt="o-", color="C0", capsize=3, markersize=5,
        label=r"LBM, time-averaged $C_d \pm \sigma$",
    )
    ax.axhline(
        CLIFT_REF, color="k", linestyle="--", linewidth=1.0,
        label=f"Clift et al. 1978 ($C_d = {CLIFT_REF}$)",
    )
    ax.set_xlabel("sphere diameter $D$ (lattice cells)")
    ax.set_ylabel(r"$C_d$")
    ax.set_xticks(D)
    ax.set_xticklabels([str(d) for d in D])
    ax.legend(loc="upper right", frameon=True)
    ax.set_ylim(1.05, 1.35)

    fig.tight_layout()
    path = os.path.join(OUT, "cd_convergence.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def fig_sharpness():
    """Sensitivity of Cd and dCd/dr to the sigmoid sharpness alpha.

    Data from examples/12_sharpness_sensitivity.py on A100.
    """
    alpha = np.array([5, 10, 20, 40, 80])
    cd = np.array([3.7555, 1.3538, 0.8745, 0.3283, 0.5414])
    grad_ad = np.array([15.1634, 3.6649, 1.8524, 9.4889, 2.4426])
    grad_fd = np.array([15.4666, 3.7023, 2.2820, 10.3985, 3.3549])

    fig, ax1 = plt.subplots(1, 1, figsize=(3.4, 2.6))

    line_cd, = ax1.plot(alpha, cd, "o-", color="C0", label=r"$C_d$")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"sharpness $\alpha$")
    ax1.set_ylabel(r"$C_d$", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.set_xticks(alpha)
    ax1.set_xticklabels([str(a) for a in alpha])

    ax2 = ax1.twinx()
    ax2.grid(False)
    line_ad, = ax2.plot(
        alpha, grad_ad, "s--", color="C3",
        label=r"$|\partial C_d/\partial r|$ (AD)",
    )
    line_fd, = ax2.plot(
        alpha, grad_fd, "^:", color="C2",
        label=r"$|\partial C_d/\partial r|$ (FD)",
    )
    ax2.set_ylabel(r"gradient magnitude", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")

    handles = [line_cd, line_ad, line_fd]
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc="upper right", frameon=True)

    fig.tight_layout()
    path = os.path.join(OUT, "sharpness_sensitivity.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def fig_optim_summary():
    """Four-mode optimisation summary: base Cd vs best Cd.

    Data from examples/11_optimisation_v2.py on A100 (60 Adam iters,
    96x48x48 MRT, 2000 steps per iter, r_init = 0.5).
    """
    modes = ["halfway\nhard", "bouzidi\nhard", "halfway\nsoft", "bouzidi\nsoft"]
    base_cd = [1.85, 1.77, 0.43, 0.45]
    best_cd = [None, 1.66, 0.003, 0.002]  # halfway_hard has grad=0
    grad_mag = [0.0, 5.55, 6.30, 3.19]

    x = np.arange(len(modes))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=False)

    ax1.bar(
        x - w / 2, base_cd, w, color="C0", alpha=0.85,
        label="baseline $C_d$",
    )
    for i, v in enumerate(best_cd):
        if v is None:
            ax1.text(x[i] + w / 2, 0.05, "grad $= 0$", ha="center",
                     va="bottom", rotation=90, fontsize=7, color="C3")
        else:
            ax1.bar(x[i] + w / 2, v, w, color="C3", alpha=0.85,
                    label="best $C_d$" if i == 1 else None)
    ax1.set_xticks(x)
    ax1.set_xticklabels(modes)
    ax1.set_ylabel(r"$C_d$")
    ax1.axhline(CLIFT_REF, color="k", linestyle="--", linewidth=0.8,
                label=f"Clift ($C_d = {CLIFT_REF}$)")
    ax1.legend(loc="upper right", frameon=True, fontsize=7)
    ax1.set_title("(a) baseline vs best $C_d$")

    colors = ["C7" if g == 0 else "C0" for g in grad_mag]
    ax2.bar(x, grad_mag, 0.6, color=colors, alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(modes)
    ax2.set_ylabel(r"avg $|\partial C_d/\partial r|$")
    ax2.set_title("(b) average gradient magnitude")
    ax2.text(0, 0.3, "0.0", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    path = os.path.join(OUT, "optim_summary.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    fig_cd_convergence()
    fig_sharpness()
    fig_optim_summary()

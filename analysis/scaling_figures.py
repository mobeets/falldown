# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#     jupytext_version: 1.19.4
#   kernelspec:
#     display_name: analysis
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Scaling figures from cached summaries
#
# `scaling_analysis.py` trains DeepONet/strategy models and needs `torch` +
# `ssm`, which are not co-installed in either venv here, so it cannot be
# re-trained in this environment. This script regenerates the scaling figures
# directly from the (current) summary CSVs so the figures and the write-up
# reflect the saved results.
#
# Run with:
#   .venv-analysis\Scripts\python.exe analysis\scaling_figures.py

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parent.parent
FIG = _REPO / "analysis" / "figures"


def plot(name, title):
    d = pd.read_csv(_REPO / "analysis" / f"scaling_{name}_summary.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for ff, g in d.groupby("fit_frac"):
        s = g.groupby("N")["acc_mean"].agg(["mean", "sem"])
        ax.errorbar(s.index, s["mean"], yerr=s["sem"], marker="o",
                    capsize=3, label=f"fit_frac={ff:g}")
    logi = d.groupby("N")["logistic_matched"].mean()
    ax.axhline(logi.mean(), color="k", ls="--", lw=1,
               label=f"matched logistic ({logi.mean():.3f})")
    ax.set_xlabel("number of pool participants (N)")
    ax.set_ylabel("held-out choice accuracy")
    ax.set_title(title)
    ax.set_xticks(sorted(d["N"].unique()))
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG / f"scaling_accuracy_{name}.png"
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    plot("cognitive", "CognitiveDeepONet transfer scaling")
    plot("strategy", "HMM-gated StrategyDeepONet transfer scaling")

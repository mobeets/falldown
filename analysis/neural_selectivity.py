# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Neural selectivity: planning / lapse / death-attuned units
#
# For every unit (0.5 Hz QC floor), compute per-trial firing rates in two
# windows around the choice moment (pre [-1000,0] ms, post [0,+1000] ms) and
# test whether firing reliably differs between trial conditions. Optionally
# also computes whole-trial mean firing rate (trial_start_ms -> exit_time_ms,
# so no spike bleed from neighboring trials).
#
# Contrasts:
#   planning vs agree_optimal   (conflict, chose planning  vs  no-conflict, optimal)
#   planning vs greedy          (conflict, chose planning  vs  conflict, chose greedy)
#   lapse    vs agree_optimal   (agree,  chose worst       vs  agree,  chose best)
#   death    vs normal          (death-anchored firing      vs  choice-anchored firing)
#
# Statistics: modulation index (A-B)/(A+B) as effect size, two-sided
# permutation test (5000 shuffles), FDR (Benjamini-Hochberg) across units.
#
# Output: selectivity_results.csv (NO images are written; plot_* functions are
# provided for you to call yourself).
#
# Run with:
#   C:\Users\manik\AppData\Local\Programs\Python\Python311\python.exe analysis\neural_selectivity.py

# %%
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------- Configuration ----------------------------
OUT_DIR = Path(r"C:\Users\manik\Desktop\Obsidian\General Thoughts\Z Images and Files\Hennig Lab Project\falldown\analysis\spike_data_alignment_output")
SPIKES_UNITS = OUT_DIR / "spikes_units.csv"
UNIT_META = OUT_DIR / "unit_metadata.csv"
TRIAL_TABLE = OUT_DIR / "trial_table.csv"
TRIAL_LABELS = OUT_DIR / "trial_labels.csv"
DEATH_TIMES = OUT_DIR / "death_times.csv"

WINDOWS = {"pre": (-1000.0, 0.0), "post": (0.0, 1000.0)}
INCLUDE_WHOLE_TRIAL = True
N_PERM = 5000
RNG_SEED = 42

CONTRASTS = [
    ("planning_vs_agree_optimal", "planning", "agree_optimal"),
    ("planning_vs_greedy", "planning", "greedy"),
    ("lapse_vs_agree_optimal", "lapse", "agree_optimal"),
]
# -----------------------------------------------------------------------


# %%
def load_data():
    units = pd.read_csv(UNIT_META)
    trials = pd.read_csv(TRIAL_TABLE)
    labels = pd.read_csv(TRIAL_LABELS)
    spikes = pd.read_csv(SPIKES_UNITS)
    deaths = pd.read_csv(DEATH_TIMES)

    # per-unit spike times (behavioral ms), only for units that passed QC
    keep_ids = set(units["unit_id"])
    spikes = spikes[spikes["unit_id"].isin(keep_ids)]
    times_by_unit = {}
    for uid, grp in spikes.groupby("unit_id"):
        times_by_unit[int(uid)] = grp["spike_time_behavioral_ms"].to_numpy()

    table = trials.merge(labels, on=["trial_id", "block_index", "sequence_index"])
    return units, table, times_by_unit, deaths


def channel_label(source_file):
    """Electrode label from a source_file name, e.g. 'times_mLF1aCa01_2285.mat'
    -> 'mLF1aCa01'."""
    stem = Path(source_file).stem          # times_mLF1aCa01_2285
    stem = stem[len("times_"):] if stem.startswith("times_") else stem
    return stem.rsplit("_", 1)[0]


def unit_channel_labels(units):
    """{unit_id: electrode label} from unit_metadata."""
    return {int(r["unit_id"]): channel_label(r["source_file"])
            for _, r in units.iterrows()}


def count_in_window(spike_times, center, lo, hi):
    """Number of spikes in [center+lo, center+hi)."""
    a = np.searchsorted(spike_times, center + lo, side="left")
    b = np.searchsorted(spike_times, center + hi, side="left")
    return b - a


def trial_rates(times_by_unit, table, window_lo, window_hi):
    """Per-unit per-trial firing rate (Hz) in the given window."""
    choice = table["choice_time_ms"].to_numpy()
    dur_s = (window_hi - window_lo) / 1000.0
    rates = {}
    for uid, ts in times_by_unit.items():
        rates[uid] = np.array(
            [count_in_window(ts, c, window_lo, window_hi) / dur_s for c in choice])
    return rates


def death_rates(times_by_unit, deaths, window_lo, window_hi):
    """Per-unit per-death firing rate (Hz) anchored at death_time_ms."""
    dt = deaths["death_time_ms"].to_numpy(dtype=float)
    dur_s = (window_hi - window_lo) / 1000.0
    rates = {}
    for uid, ts in times_by_unit.items():
        rates[uid] = np.array(
            [count_in_window(ts, t, window_lo, window_hi) / dur_s for t in dt])
    return rates


def whole_trial_rates(times_by_unit, table):
    """Per-unit per-trial mean firing rate (Hz) over the full trial
    [trial_start_ms, exit_time_ms), divided by the actual trial duration.

    Trials are contiguous and non-overlapping, so each spike is counted in
    exactly one trial's interval and there is no bleed from neighboring
    trials.
    """
    starts = table["trial_start_ms"].to_numpy()
    exits = table["exit_time_ms"].to_numpy()
    durs_s = (exits - starts) / 1000.0
    rates = {}
    for uid, ts in times_by_unit.items():
        counts = np.array(
            [count_in_window(ts, s, 0.0, e - s) for s, e in zip(starts, exits)])
        rates[uid] = counts / durs_s
    return rates


# %%
def modulation_index(a, b):
    """(mean_a - mean_b) / (mean_a + mean_b); NaN if both means are 0."""
    ma, mb = np.mean(a), np.mean(b)
    denom = ma + mb
    if denom == 0:
        return np.nan
    return (ma - mb) / denom


def permutation_test(a, b, n_perm=N_PERM, seed=RNG_SEED):
    """Two-sided permutation test on the modulation index.

    Shuffles the pooled condition labels n_perm times, recomputes MI each
    time, and returns the observed MI and the two-sided p-value.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    pooled = np.concatenate([a, b])
    n_a = len(a)
    rng = np.random.default_rng(seed)
    obs = modulation_index(a, b)
    if np.isnan(obs):
        return obs, np.nan

    # precompute shuffled split indexes once (n_perm x n_total)
    n_total = len(pooled)
    idx = np.argsort(rng.random((n_perm, n_total)), axis=1)
    shuffled = pooled[idx]          # (n_perm, n_total)
    mA = shuffled[:, :n_a].mean(axis=1)
    mB = shuffled[:, n_a:].mean(axis=1)
    denom = mA + mB
    perm_mi = np.where(denom != 0, (mA - mB) / np.where(denom == 0, 1, denom), np.nan)
    perm_mi = perm_mi[~np.isnan(perm_mi)]
    if len(perm_mi) == 0:
        return obs, np.nan
    p = np.mean(np.abs(perm_mi) >= abs(obs))
    return obs, p


def fdr_bh(pvals):
    """Benjamini-Hochberg FDR across the p-value vector."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    q = np.full(n, np.nan)
    valid = ~np.isnan(pvals)
    if valid.sum() == 0:
        return q
    pv = pvals[valid]
    order = np.argsort(pv)
    ranked = np.arange(1, len(pv) + 1)
    qv = pv[order] * n / ranked
    qv = np.minimum.accumulate(qv[::-1])[::-1]
    qv = np.clip(qv, 0, 1)
    q[valid] = qv[np.argsort(order)]
    return q


# %%
def main():
    print("Loading data ...")
    units, table, times_by_unit, deaths = load_data()
    unit_ids = sorted(times_by_unit.keys())
    chan_labels = unit_channel_labels(units)
    print(f"  {len(unit_ids)} units, {len(table)} trials, {len(deaths)} deaths")

    condition = table.set_index("trial_id")["condition"].to_dict()

    rows = []
    for window_name, (lo, hi) in WINDOWS.items():
        print(f"\nWindow {window_name} [{lo:+.0f},{hi:+.0f}] ms")
        rates = trial_rates(times_by_unit, table, lo, hi)
        death_rate = death_rates(times_by_unit, deaths, lo, hi)
        run_window(rows, window_name, unit_ids, table, rates, deaths,
                   death_rate=death_rate, chan_labels=chan_labels)

    if INCLUDE_WHOLE_TRIAL:
        print("\nWindow whole_trial [trial_start_ms, exit_time_ms]")
        rates = whole_trial_rates(times_by_unit, table)
        run_window(rows, "whole_trial", unit_ids, table, rates, deaths,
                   death_rate=None, chan_labels=chan_labels)

    res = pd.DataFrame(rows)
    res["q_fdr"] = np.nan
    for key, grp in res.groupby(["contrast", "window"]):
        q = fdr_bh(grp["p_perm"].to_numpy())
        res.loc[grp.index, "q_fdr"] = q
    res["significant"] = res["q_fdr"] < 0.05
    res.to_csv(OUT_DIR / "selectivity_results.csv", index=False)

    # descriptive death report (N is too small for inference in this session)
    if len(deaths) > 0:
        death_rows = []
        for window_name, (lo, hi) in WINDOWS.items():
            dr = death_rates(times_by_unit, deaths, lo, hi)
            baseline = trial_rates(times_by_unit, table, lo, hi)
            for uid in unit_ids:
                vals = dr[uid]
                death_rows.append({
                    "unit_id": uid,
                    "channel": chan_labels.get(uid, "") if chan_labels else "",
                    "window": window_name,
                    "n_deaths": len(vals),
                    "mean_death_rate_Hz": float(np.mean(vals)) if len(vals) else np.nan,
                    "baseline_mean_rate_Hz": float(np.mean(baseline.get(uid, [np.nan]))),
                })
        pd.DataFrame(death_rows).to_csv(
            OUT_DIR / "death_locked_rates.csv", index=False)
        print("\nSaved death_locked_rates.csv (descriptive only, N=%d)" % len(deaths))

    print("\nSaved selectivity_results.csv")
    print(res.groupby(["contrast", "window"])["significant"].sum().to_string())
    for (contrast, window), grp in res.groupby(["contrast", "window"]):
        sig = grp[grp["significant"]]
        if len(sig):
            labels = sorted(set(sig["channel"]))
            print(f"  {contrast} / {window}: {len(sig)} significant -> "
                  f"{', '.join(labels)}")


def run_window(rows, window_name, unit_ids, table, rates, deaths,
               death_rate=None, chan_labels=None):
    """Run all trial contrasts for a per-trial rate array.

    The death_vs_normal contrast is only added when a death-anchored rate
    array is supplied (it has no natural anchor for the whole-trial window).
    """
    normal_idx = np.arange(len(table))

    # per-contrast trial indexes
    for contrast, cond_a, cond_b in CONTRASTS:
        ia = table.index[table["condition"] == cond_a].to_numpy()
        ib = table.index[table["condition"] == cond_b].to_numpy()
        rows, _ = run_contrast(rows, contrast, window_name, unit_ids, rates, ia, ib,
                               chan_labels=chan_labels)

    # death vs normal: only meaningful if there are enough genuine deaths.
    # With a single death (this session), a permutation contrast would be
    # degenerate (1 window vs the whole session), so we skip it and report
    # the death-locked firing descriptively instead.
    if death_rate is not None and len(deaths) >= 5:
        rows, _ = run_contrast(
            rows, "death_vs_normal", window_name, unit_ids,
            rates, [None], normal_idx, death_rates=death_rate,
            chan_labels=chan_labels)
    elif death_rate is not None:
        print(f"    (skipping death_vs_normal: only {len(deaths)} death "
              f"moment(s); N<5 is too few for a permutation contrast)")
    return rows


def run_contrast(rows, contrast, window_name, unit_ids, rates, ia, ib,
                 death_rates=None, chan_labels=None):
    """Compute MI + permutation p for one contrast across all units.

    ia, ib are trial indexes; when contrast is death_vs_normal, ia is [None]
    and ib is the normal-trial indexes, and death_rates supplies the "A" rates.
    """
    for uid in unit_ids:
        if contrast == "death_vs_normal":
            a = death_rates[uid]
            b = rates[uid][ib]
        else:
            a = rates[uid][ia]
            b = rates[uid][ib]
        mi, p = permutation_test(a, b)
        rows.append({
            "unit_id": uid,
            "channel": chan_labels.get(uid, "") if chan_labels else "",
            "contrast": contrast,
            "window": window_name,
            "n_A": len(a),
            "n_B": len(b),
            "mean_rate_A_Hz": float(np.nanmean(a)) if len(a) else np.nan,
            "mean_rate_B_Hz": float(np.nanmean(b)) if len(b) else np.nan,
            "modulation_index": mi,
            "p_perm": p,
        })
    return rows, None


# %%
# ----------------------- Plot functions (not saved) ---------------------
# Call these yourself in Jupyter to visualize the results.

def load_results():
    return pd.read_csv(OUT_DIR / "selectivity_results.csv")


def plot_bar_by_contrast(ax=None):
    """Bar chart: number of significant units per contrast (pre+post)."""
    import matplotlib.pyplot as plt
    res = load_results()
    if ax is None:
        ax = plt.gca()
    piv = res[res["significant"]].groupby(["contrast", "window"]).size().unstack(fill_value=0)
    piv.plot(kind="bar", ax=ax)
    ax.set_ylabel("significant units (FDR<0.05)")
    ax.set_title("Significant units by contrast")
    ax.legend(title="window")
    return ax


def plot_significant_units_table(contrast="planning_vs_agree_optimal",
                                 window="pre"):
    """Print the significant units for a given contrast/window."""
    res = load_results()
    sub = res[(res["contrast"] == contrast) & (res["window"] == window)]
    sig = sub[sub["significant"]].sort_values("modulation_index")
    print(f"{contrast} / {window}: {len(sig)} significant units")
    return sig


if __name__ == "__main__":
    main()

# %%

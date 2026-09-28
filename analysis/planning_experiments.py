# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: analysis
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Planning experiments: greedy vs planning across three studies
#
# One self-contained module that turns the raw behavioral JSON into a single
# per-sequence table and runs the three experiment analyses:
#
#   Exp 1  planning prevalence + temporal structure (stay/switch, GLM-HMM)
#   Exp 2  sensitivity of planning to environmental statistics
#   Exp 3  threat (drift / proximity to death) and planning + execution benefit
#
# It deliberately re-implements the small geometry/parsing helpers instead of
# importing `exploratory_data_analysis` (which pulls in torch) or
# `classify_trials` (which creates output dirs at import). No existing file is
# modified.
#
# ## Environment
#
# Requires the pinned analysis venv (numpy<2 + the lindermanlab/ssm fork with
# `input_driven_obs`), created with:
#
#   uv venv --python 3.11 .venv-analysis
#   uv pip install --python .venv-analysis/Scripts/python.exe \
#       "numpy<2" scipy pandas scikit-learn statsmodels matplotlib seaborn \
#       Cython setuptools wheel
#   uv pip install --python .venv-analysis/Scripts/python.exe \
#       --no-build-isolation \
#       "ssm @ git+https://github.com/lindermanlab/ssm.git@eb6c8aa"
#   # autograd>=1.6 moved `logsumexp` out of autograd.scipy.misc; if the
#   # installed autograd is newer, patch the 5 ssm imports to
#   # `from autograd.scipy.special import logsumexp`.
#
# Run:  .venv-analysis/Scripts/python.exe analysis/planning_experiments.py
#
# Outputs go to `analysis/planning_outputs/` (CSV + PNG).

# %%
import io
import json
import sys
import contextlib
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy import stats
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

warnings.filterwarnings("ignore")

_REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = _REPO_ROOT / "analysis" / "planning_outputs"

MIN_EXPERIMENT_LEVELS = 10  # experimental blocks have >= 10 levels; practice <= 4


# %% [markdown]
# ## Dataset registry

# %%
@dataclass(frozen=True)
class Dataset:
    name: str
    folder: str
    pattern: str
    environment: str      # balanced | high_agree
    threat: str           # follow | drift | alternating
    config: str = ""      # config path (relative to repo root) for reference
    experiment: str = ""  # exp1 / exp2 / exp3
    config_exp_blocks: int = 0   # experimental blocks (with >4 levels) in the config
    min_completion: float = 0.6  # fraction of experimental blocks required


# All online-study sessions (data/logs_sorted). Completion is measured as
# distinct experimental blocks actually played / config experimental blocks
# (Definition A). Exp 1 (nodrift) uses a relaxed 40% cutoff because the strict
# 60% rule would leave only 3 participants.
DATASETS = {
    "exp1_nodrift": Dataset(
        "exp1_nodrift", "data/logs_sorted/short_trials_experiment_nodrift",
        "*_cleaned.json", "balanced", "follow",
        "app/configs/short_trials_experiment_nodrift.json", "exp1",
        config_exp_blocks=40, min_completion=0.40),
    "exp3_altdrift": Dataset(
        "exp3_altdrift", "data/logs_sorted/short_trials_experiment",
        "*_cleaned.json", "high_agree", "alternating",
        "app/configs/short_trials_experiment.json", "exp3",
        config_exp_blocks=32, min_completion=0.60),
    "exp3_alldrift": Dataset(
        "exp3_alldrift", "data/logs_sorted/short_trials_experiment-7-10",
        "*_cleaned.json", "balanced", "drift",
        "app/configs/short_trials_experiment-7-10.json", "exp3",
        config_exp_blocks=40, min_completion=0.60),
}


# %% [markdown]
# ## Geometry helpers (canonical cost functions)

# %%
def greedy_cost(entry, hole, goal=None):
    """1-step cost: distance from the entry hole to the choice hole."""
    return abs(entry - hole)


def planning_cost(entry, hole, goal):
    """2-step cost: entry->choice plus choice->goal."""
    return abs(entry - hole) + abs(hole - goal)


def _block_drift(block):
    try:
        v = block["block_config"]["params"]["startCameraMode"]
        return int(v) if v is not None else 0
    except (KeyError, TypeError):
        return 0


def _relative_ball_y(block, t):
    gs = block.get("game_states") or {}
    st = np.asarray(gs.get("time", []))
    if len(st) == 0:
        return np.nan
    by = np.asarray(gs.get("ball_y", []))
    cy = np.asarray(gs.get("camera_y", []))
    i = int(np.argmin(np.abs(st - t)))
    return float(by[i] - cy[i])


def _event_time(trial):
    evs = trial.get("events") or []
    return evs[0]["time"] if evs else None


# %% [markdown]
# ## Per-participant sequence table

# %%
def build_sequence_table(data, participant, dataset):
    """Return one row per 3-level (entry -> choice -> goal) sequence.

    Columns mirror `exploratory_data_analysis.pre_proccess_data_from_choice_vs_no_choice`
    so downstream models can reuse the same feature definitions, plus
    condition labels, decision/execution RT decomposition, and per-block
    environment statistics.
    """
    rows = []
    for block_num, block in enumerate(data.get("blocks", [])):
        levels = block.get("block_config", {}).get("levels", [])
        is_experimental = len(levels) >= MIN_EXPERIMENT_LEVELS
        drift = _block_drift(block)
        trials = block.get("trials", [])

        for i in range(len(trials) // 3):
            t0, t1, t2 = trials[3 * i], trials[3 * i + 1], trials[3 * i + 2]
            t0e, t1e, t2e = _event_time(t0), _event_time(t1), _event_time(t2)
            if t0e is None or t1e is None or t2e is None:
                continue
            if not (t0.get("hole_locations") and t1.get("hole_locations")
                    and t2.get("hole_locations")):
                continue

            entry = t0["events"][0]["holeUsed"]
            goal = t2["events"][0]["holeUsed"]
            holes = list(t1["hole_locations"])
            if len(holes) != 2:
                continue
            chosen = t1["events"][0]["holeUsed"]
            if chosen not in holes:
                continue
            unchosen = holes[0] if holes[1] == chosen else holes[1]

            g = {h: greedy_cost(entry, h, goal) for h in holes}
            p = {h: planning_cost(entry, h, goal) for h in holes}
            g_best = min(g, key=g.get)
            p_best = min(p, key=p.get)
            agree = g_best == p_best

            chosen_1step = abs(chosen - entry)
            unchosen_1step = abs(unchosen - entry)
            chosen_2step = chosen_1step + abs(chosen - goal)
            unchosen_2step = unchosen_1step + abs(unchosen - goal)
            chosen_left = chosen < unchosen

            if agree:
                condition = "agree_optimal" if chosen == g_best else "lapse"
            else:
                if chosen == p_best:
                    condition = "planning"
                elif chosen == g_best:
                    condition = "greedy"
                else:
                    condition = "other"

            rows.append({
                "participant": participant,
                "dataset": dataset,
                "block_number": block_num,
                "sequence_index": i,
                "is_experimental": is_experimental,
                "block_drift": drift,
                "entry_hole": entry,
                "goal_hole": goal,
                "choice_holes": tuple(holes),
                "chosen_hole": chosen,
                "unchosen_hole": unchosen,
                "greedy_optimal_hole": g_best,
                "planning_optimal_hole": p_best,
                "conflict": int(not agree),
                "chose_greedy": int(chosen == g_best),
                "chose_planning": int(chosen == p_best),
                "condition": condition,
                "chosen_1step_dist": chosen_1step,
                "unchosen_1step_dist": unchosen_1step,
                "chosen_2step_dist": chosen_2step,
                "unchosen_2step_dist": unchosen_2step,
                "chosen_left": int(chosen_left),
                "plan_advantage": (planning_cost(entry, g_best, goal)
                                   - planning_cost(entry, p_best, goal)),
                "greedy_advantage": (greedy_cost(entry, p_best, goal)
                                     - greedy_cost(entry, g_best, goal)),
                "observed_rt": t2e - t0e,
                "rt_decision": t1e - t0e,
                "rt_exec": t2e - t1e,
                "entry_time_ms": t0e,
                "choice_time_ms": t1e,
                "exit_time_ms": t2e,
                "ball_y_at_top": _relative_ball_y(block, t0e),
                "scroll_speed": t0["events"][0].get("scrollSpeed", np.nan),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(["block_number", "sequence_index"]).reset_index(drop=True)

    # incoming direction: sign of (previous sequence's goal - this entry),
    # only valid for consecutive sequences inside the same block
    prev_goal = df["goal_hole"].shift(1)
    prev_block = df["block_number"].shift(1)
    prev_seq = df["sequence_index"].shift(1)
    valid = (prev_block == df["block_number"]) & (prev_seq + 1 == df["sequence_index"])
    direction = np.sign(prev_goal - df["entry_hole"])
    df["incoming_direction"] = np.where(valid, -direction, np.nan)

    # RT outlier trim (same 2.5*IQR rule used elsewhere)
    q1, q3 = df["observed_rt"].quantile(0.25), df["observed_rt"].quantile(0.75)
    iqr = q3 - q1
    df = df[(df["observed_rt"] >= q1 - 2.5 * iqr)
            & (df["observed_rt"] <= q3 + 2.5 * iqr)].reset_index(drop=True)

    # per-block environment statistics (computed on experimental decision trials)
    exp = df[df["is_experimental"]]
    if not exp.empty:
        env = exp.groupby("block_number").agg(
            block_conflict_rate=("conflict", "mean"),
            block_plan_advantage=("plan_advantage", "mean"),
            block_n_conflict=("conflict", "sum"),
            block_n_trials=("conflict", "size"),
        )
        df = df.merge(env, on="block_number", how="left")

    return df


# %% [markdown]
# ## Loading

# %%
def _participant_id(path):
    stem = path.stem
    return stem.replace("_cleaned", "")


def _count_experimental_blocks(data):
    """Distinct block_index values that are experimental (more than 4 trials).

    This is the numerator of completion Definition A, computed from the raw
    block records so it is unaffected by trials that produced no sequence.
    """
    return len({b.get("block_index") for b in data.get("blocks", [])
                if len(b.get("trials", [])) > 4})


def load_dataset(ds, verbose=True):
    """Return (sequence_table, completion_rows) for one dataset.

    Participants are kept only if they completed at least
    `ds.min_completion * ds.config_exp_blocks` experimental blocks
    (Definition A).
    """
    folder = _REPO_ROOT / ds.folder
    files = []
    for pat in ds.pattern.split(";"):
        files.extend(folder.glob(pat))
    files = sorted(set(files))
    frames = []
    completion = []
    for fp in files:
        with open(fp, encoding="utf-8") as fh:
            data = json.load(fh)
        if not data.get("blocks"):
            continue
        pid = _participant_id(fp)
        n_exp = _count_experimental_blocks(data)
        frac = n_exp / ds.config_exp_blocks if ds.config_exp_blocks else np.nan
        retained = bool(frac >= ds.min_completion)
        completion.append({
            "participant": pid, "dataset": ds.name, "experiment": ds.experiment,
            "exp_blocks_completed": n_exp,
            "config_exp_blocks": ds.config_exp_blocks,
            "completion_fraction": round(frac, 3),
            "min_completion": ds.min_completion,
            "retained": retained,
        })
        if not retained:
            continue
        t = build_sequence_table(data, pid, ds.name)
        if not t.empty:
            frames.append(t)
    if not frames:
        out = pd.DataFrame()
    else:
        out = pd.concat(frames, ignore_index=True)
        out["environment"] = ds.environment
        out["threat"] = ds.threat
        out["experiment"] = ds.experiment
    if verbose:
        kept = sum(c["retained"] for c in completion)
        print(f"  loaded {ds.name:16s} -> {kept:2d}/{len(completion):2d} participants "
              f"retained (>= {ds.min_completion:.0%} of {ds.config_exp_blocks} exp blocks), "
              f"{len(out):6d} sequences")
    return out, completion


def load_all(datasets=None):
    datasets = datasets or DATASETS
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames, completion = [], []
    for name, ds in datasets.items():
        t, comp = load_dataset(ds)
        frames.append(t)
        completion.extend(comp)
    comp_df = pd.DataFrame(completion)
    if not comp_df.empty:
        comp_df.to_csv(OUT_DIR / "completion_table.csv", index=False)
    return pd.concat(frames, ignore_index=True), comp_df


# %% [markdown]
# ## Shared feature derivations

# %%
def add_features(df):
    """L1/R1/L2/R2 and z-scored threat, computed exactly like the existing code."""
    df = df.copy()
    left = df["chosen_left"].astype(bool)
    df["L1"] = np.where(left, df["chosen_1step_dist"], df["unchosen_1step_dist"])
    df["R1"] = np.where(~left, df["chosen_1step_dist"], df["unchosen_1step_dist"])
    chosen_2 = df["chosen_2step_dist"] - df["chosen_1step_dist"]
    unchosen_2 = df["unchosen_2step_dist"] - df["unchosen_1step_dist"]
    df["L2"] = np.where(left, chosen_2, unchosen_2)
    df["R2"] = np.where(~left, chosen_2, unchosen_2)
    df["diff_1step"] = df["L1"] - df["R1"]
    df["diff_planning"] = df["L1"] + df["L2"] - df["R1"] - df["R2"]
    df["z_ball_y"] = (df["ball_y_at_top"] - df["ball_y_at_top"].mean()) / (
        df["ball_y_at_top"].std(ddof=0) or 1.0)
    df["z_block_conflict"] = (
        df["block_conflict_rate"] - df["block_conflict_rate"].mean()) / (
        df["block_conflict_rate"].std(ddof=0) or 1.0)
    return df


# %% [markdown]
# ## Mixture model (greedy vs planning + lapse + threat covariate)

# %%
def _p_right(params, d_greedy, d_plan, covariates):
    p_lapse, p_plan_base, w1, s_greedy, s_plan, bias_dir = params
    ball_y = covariates[:, 0] if covariates.ndim == 2 else covariates
    incoming = covariates[:, 1] if covariates.ndim == 2 else np.zeros_like(ball_y)
    prob_greedy = expit(d_greedy / s_greedy)
    prob_plan = expit(d_plan / s_plan)
    p0 = np.clip(p_plan_base, 1e-6, 1 - 1e-6)
    dyn_p_plan = expit(logit(p0) + w1 * ball_y)
    model_prob = (1 - dyn_p_plan) * prob_greedy + dyn_p_plan * prob_plan
    logit_prob = logit(np.clip(model_prob, 1e-10, 1 - 1e-10)) + bias_dir * incoming
    return p_lapse * 0.5 + (1 - p_lapse) * expit(logit_prob)


def fit_mixture_model(df, use_threat=True, n_starts=5, seed=0):
    """Fit the greedy/planning mixture to one participant's choice trials.

    Cost features and `ball_y_at_top` are z-scored so the inverse-temperature
    and threat parameters are on comparable, well-conditioned scales, and the
    parameters are bounded. Multiple restarts guard against local optima.

    Returns a dict of parameters or None. `w1` is the threat effect: negative
    means planning rises as `ball_y_at_top` falls (closer to death).
    """
    d = df.dropna(subset=["diff_1step", "diff_planning",
                          "ball_y_at_top", "incoming_direction"]).copy()
    if len(d) < 20:
        return None
    rng = np.random.default_rng(seed)

    def z(v):
        v = np.asarray(v, dtype=float)
        s = v.std()
        return (v - v.mean()) / (s if s > 0 else 1.0)

    dg = z(d["diff_1step"].values)
    dp = z(d["diff_planning"].values)
    ball_y = z(d["ball_y_at_top"].values)
    cov = np.column_stack([ball_y, d["incoming_direction"].values])
    y = d["chosen_left"].astype(int).values

    if use_threat:
        bounds = [(0.0, 0.6), (0.02, 0.98), (-8.0, 8.0),
                  (0.05, 20.0), (0.05, 20.0), (-8.0, 8.0)]
        keys = ["p_lapse", "p_plan_base", "w1", "s_greedy", "s_plan", "bias_dir"]
    else:
        bounds = [(0.0, 0.6), (0.02, 0.98), (0.05, 20.0), (0.05, 20.0), (-8.0, 8.0)]
        keys = ["p_lapse", "p_plan_base", "s_greedy", "s_plan", "bias_dir"]

    def nll(params):
        if use_threat:
            p = _p_right(params, dg, dp, cov)
        else:
            p = _p_right([params[0], params[1], 0.0, params[2], params[3], params[4]],
                         dg, dp, cov)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    best = None
    for _ in range(n_starts):
        if use_threat:
            x0 = [rng.uniform(0, 0.3), rng.uniform(0.2, 0.8), rng.uniform(-2, 2),
                  rng.uniform(0.3, 3), rng.uniform(0.3, 3), rng.uniform(-2, 2)]
        else:
            x0 = [rng.uniform(0, 0.3), rng.uniform(0.2, 0.8),
                  rng.uniform(0.3, 3), rng.uniform(0.3, 3), rng.uniform(-2, 2)]
        res = minimize(nll, x0, bounds=bounds, method="L-BFGS-B")
        if res.success and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        return None
    out = dict(zip(keys, best.x))
    out["nll"] = float(best.fun)
    out["n_trials"] = int(len(d))
    out["log_likelihood"] = -float(best.fun) / len(d)
    return out


# %% [markdown]
# ## GEE helper (population-level logistic with participant clusters)

# %%
def fit_gee(df, formula_cols, outcome, groups, family=None):
    """Binomial GEE with exchangeable working correlation, clustered by groups.

    `formula_cols` is a list of column names (interactions can be precomputed).
    Returns the fitted results object (or None).
    """
    d = df.dropna(subset=list(formula_cols) + [outcome, groups]).copy()
    if d[outcome].nunique() < 2 or len(d) < 20:
        return None
    X = sm.add_constant(d[list(formula_cols)].astype(float))
    fam = family or sm.families.Binomial()
    try:
        model = sm.GEE(d[outcome].astype(float), X, groups=d[groups].values,
                       family=fam, cov_struct=sm.cov_struct.Exchangeable())
        return model.fit()
    except Exception as exc:  # pragma: no cover - numerical guard
        print(f"    GEE failed: {exc}")
        return None


# %% [markdown]
# # Experiment 1 — planning prevalence and temporal structure

# %%
def exp1_prevalence(df, experiment=None, min_trials=20):
    """Per-participant planning prevalence on conflict and agreement trials.

    `experiment=None` pools all online participants; pass "exp1"/"exp3" to
    restrict. `p_plan = P(planning | conflict)`, `p_optimal = P(optimal |
    agreement)` (= `chose_greedy` on agreement trials), `p_lapse = 1 - p_optimal`.
    """
    d = df[df["is_experimental"]].copy()
    if experiment is not None:
        d = d[d["experiment"] == experiment]
    rows = []
    for pid, g in d.groupby("participant"):
        conf = g[g["conflict"] == 1]
        agree = g[g["conflict"] == 0]
        if len(conf) < min_trials:
            continue
        rows.append({
            "participant": pid,
            "experiment": g["experiment"].iloc[0],
            "dataset": g["dataset"].iloc[0],
            "environment": g["environment"].iloc[0],
            "threat": g["threat"].iloc[0],
            "n_conflict": len(conf), "n_agree": len(agree),
            "p_plan": conf["chose_planning"].mean(),
            "p_greedy": conf["chose_greedy"].mean(),
            "p_optimal": agree["chose_greedy"].mean() if len(agree) else np.nan,
            "p_lapse": (agree["condition"] == "lapse").mean() if len(agree) else np.nan,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        scope = experiment or "all"
        t, p = stats.ttest_1samp(out["p_plan"], 0.5)
        print(f"  [{scope}] P(plan|conflict): mean={out['p_plan'].mean():.3f}, "
              f"t={t:.2f}, p={p:.2e}, N={len(out)}")
    return out


def _strategy_runs(strategy):
    """Run lengths of a 0/1 sequence."""
    runs, cur, n = [], strategy[0], 1
    for s in strategy[1:]:
        if s == cur:
            n += 1
        else:
            runs.append(n)
            cur, n = s, 1
    runs.append(n)
    return runs


def exp1_transitions(df, experiment=None, n_perm=1000, seed=0, min_trials=20):
    """Empirical stay probability and run lengths on consecutive conflict trials.

    The null permutes the strategy labels *within each block* (preserving each
    block's length and base planning rate), then recomputes adjacent-pair stay
    probability. This isolates true sequential persistence from the trivial
    fact that a low/high base rate already inflates "stay".

    `experiment=None` pools all online participants.
    """
    d = df[df["is_experimental"]].copy()
    if experiment is not None:
        d = d[d["experiment"] == experiment]
    rng = np.random.default_rng(seed)
    rows = []
    for pid, g in d.groupby("participant"):
        g = g.sort_values(["block_number", "sequence_index"])
        conf = g[g["conflict"] == 1]
        blocks = [b["chose_planning"].values
                  for _, b in conf.groupby("block_number") if len(b) >= 2]
        strategy = np.concatenate(blocks) if blocks else np.array([])
        if len(strategy) < min_trials:
            continue

        def stay_of(arrs):
            vals = []
            for a in arrs:
                if len(a) >= 2:
                    vals.extend((a[:-1] == a[1:]).tolist())
            return np.mean(vals) if vals else np.nan

        stay = stay_of(blocks)
        base = strategy.mean()
        null_stay = base ** 2 + (1 - base) ** 2
        perm = []
        for _ in range(n_perm):
            shuffled = [rng.permutation(a) for a in blocks]
            perm.append(stay_of(shuffled))
        perm = np.array(perm)
        rows.append({"participant": pid,
                     "experiment": g["experiment"].iloc[0],
                     "dataset": g["dataset"].iloc[0],
                     "environment": g["environment"].iloc[0],
                     "n_conflict": len(strategy),
                     "p_plan": base, "stay_prob": stay,
                     "bernoulli_stay": null_stay,
                     "perm_mean": np.nanmean(perm),
                     "perm_lo": np.nanpercentile(perm, 2.5),
                     "perm_hi": np.nanpercentile(perm, 97.5),
                     "perm_p": np.nanmean(perm >= stay),
                     "mean_run": np.mean(_strategy_runs(strategy))})
    return pd.DataFrame(rows)


FEATURE_NAMES = ["L1-R1", "L1+L2-R1-R2", "incoming_dir", "bias"]


def _hmm_design(g):
    """Sorted choice trials -> (X, y, frame) with raw (unstandardized) features."""
    g = g.sort_values(["block_number", "sequence_index"])
    g = g.dropna(subset=["diff_1step", "diff_planning", "incoming_direction"])
    X = np.column_stack([g["diff_1step"].values, g["diff_planning"].values,
                         g["incoming_direction"].values])
    y = g["chosen_left"].astype(int).values
    return X, y, g


def _fit_glmhmm(Xtr, ytr, Xte, yte, K, num_iters=100, seed=0):
    """Fit a K-state GLM-HMM (K=1 -> logistic) on train, score on test.

    Returns dict with train/test log-likelihood, per-state emission weights,
    transition matrix, filtered posterior, occupancy and free-parameter count.
    """
    mu = Xtr[:, :2].mean(0)
    sd = Xtr[:, :2].std(0)
    sd[sd == 0] = 1.0

    def design(X):
        return np.column_stack([(X[:, :2] - mu) / sd, X[:, 2], np.ones(len(X))])

    Ztr, Zte = design(Xtr), design(Xte)
    M = Ztr.shape[1]
    ytr = ytr.astype(int)
    yte = yte.astype(int)

    if K == 1:
        try:
            m = sm.Logit(ytr, Ztr).fit(disp=0)
        except Exception:
            return None
        p = np.clip(m.predict(Zte), 1e-12, 1 - 1e-12)
        return dict(ll_train=float(m.llf),
                    ll_test=float(np.sum(yte * np.log(p) + (1 - yte) * np.log(1 - p))),
                    W=m.params.reshape(1, M), tm=np.ones((1, 1)),
                    post=np.ones((len(yte), 1)), occ=np.ones(1),
                    n_params=M, M=M)

    import ssm
    import autograd.numpy.random as npr
    np.random.seed(seed)
    npr.seed(seed)
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            model = ssm.HMM(K, 1, M, observations="input_driven_obs",
                            observation_kwargs=dict(C=2), transitions="standard")
            model.fit([ytr.reshape(-1, 1)], inputs=Ztr, method="em",
                      num_iters=num_iters, tolerance=1e-4)
        ll_train = float(model.log_likelihood([ytr.reshape(-1, 1)], inputs=Ztr))
        ll_test = float(model.log_likelihood([yte.reshape(-1, 1)], inputs=Zte))
        W = model.observations.params[:, 0, :]
        tm = np.exp(model.transitions.log_Ps)
        post = model.filter(yte.reshape(-1, 1), input=Zte)
        return dict(ll_train=ll_train, ll_test=ll_test, W=W, tm=tm, post=post,
                    occ=post.mean(0),
                    n_params=K * M + K * (K - 1) + (K - 1), M=M)
    except Exception as exc:  # pragma: no cover
        print(f"    GLM-HMM K={K} failed: {exc}")
        return None


def exp1_hmm_model_selection(df, experiment="exp1", Ks=(1, 2, 3, 4),
                             num_iters=100, seed=0, test_frac=0.2):
    """BIC-based selection of the number of HMM states (group-level K*).

    For each participant and K, fit on the first (1 - test_frac) choice trials
    and score the held-out tail. Returns (per_participant_df, K_star, per_opt).
    """
    exp = df[(df["is_experimental"]) & (df["experiment"] == experiment)]
    rows = []
    for pid, g in exp.groupby("participant"):
        X, y, _ = _hmm_design(g)
        T = len(y)
        if T < 60:
            continue
        cut = int((1 - test_frac) * T)
        Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]
        for K in Ks:
            st = _fit_glmhmm(Xtr, ytr, Xte, yte, K, num_iters, seed)
            if st is None:
                continue
            p = st["n_params"]
            rows.append({
                "experiment": experiment,
                "participant": pid, "K": K, "n_trials": T,
                "ll_train": st["ll_train"], "ll_test": st["ll_test"],
                "n_params": p,
                "bic": -2 * st["ll_train"] + p * np.log(len(ytr)),
                "aic": -2 * st["ll_train"] + 2 * p,
            })
    out = pd.DataFrame(rows)
    K_star, per_opt = None, pd.DataFrame()
    if not out.empty:
        sum_bic = out.groupby("K")["bic"].sum()
        K_star = int(sum_bic.idxmin())
        per_opt = (out.loc[out.groupby("participant")["bic"].idxmin()]
                   [["participant", "K"]].rename(columns={"K": "optimal_K"}))
        per_opt["experiment"] = experiment
        dist = per_opt["optimal_K"].value_counts().sort_index()
        print(f"  HMM model selection [{experiment}]: group-level K* = {K_star} "
              f"(min sum BIC); per-participant optimal K = {dist.to_dict()}")
    return out, K_star, per_opt


def exp1_glmhmm(df, experiment="exp1", num_states=2, num_iters=100, seed=0):
    """Per-participant GLM-HMM (fit on all trials) for state interpretation.

    Returns (summary_df, details, long_weights). With num_states=1 this is the
    maximum-likelihood logistic fit (a single "state"); num_states>=2 is used
    for the forced multi-state comparison.
    """
    exp = df[(df["is_experimental"]) & (df["experiment"] == experiment)]
    rows, details, weight_rows = [], {}, []
    for pid, g in exp.groupby("participant"):
        X, y, _ = _hmm_design(g)
        T = len(y)
        if T < 60:
            continue
        st = _fit_glmhmm(X, y, X, y, num_states, num_iters, seed)
        if st is None:
            continue
        W, tm, occ, post = st["W"], st["tm"], st["occ"], st["post"]
        plan_state = int(np.argmax(W[:, 1]))
        rows.append({
            "experiment": experiment,
            "participant": pid, "n_trials": T, "num_states": num_states,
            "stay_diag": float(np.mean(np.diag(tm))),
            "switch_rate": float(1 - np.trace(tm) / num_states),
            "occupancy": json.dumps(np.round(occ, 3).tolist()),
            "weights": json.dumps(np.round(W, 3).tolist()),
            "transition_matrix": json.dumps(np.round(tm, 3).tolist()),
            "plan_state": plan_state,
            "feature_names": json.dumps(FEATURE_NAMES),
        })
        details[pid] = dict(weights=W, tm=tm, occ=occ, posterior=post,
                            n_trials=T, features=FEATURE_NAMES)
        for k in range(W.shape[0]):
            for f_i, f_name in enumerate(FEATURE_NAMES):
                weight_rows.append({"experiment": experiment, "participant": pid,
                                    "num_states": num_states, "state": k,
                                    "feature": f_name, "weight": float(W[k, f_i])})
    return pd.DataFrame(rows), details, pd.DataFrame(weight_rows)


def _posterior_long(details):
    """Flatten state posteriors into a long DataFrame for timeline plots."""
    rows = []
    for pid, det in details.items():
        P = det["posterior"]
        for t in range(P.shape[0]):
            for k in range(P.shape[1]):
                rows.append({"participant": pid, "trial": t, "state": k,
                             "posterior": float(P[t, k])})
    return pd.DataFrame(rows)


# %% [markdown]
# # Experiment 2 — environmental statistics

# %%
def exp2_block_environment(df):
    """Per block: conflict rate, expected planning advantage, planning rate."""
    exp = df[(df["is_experimental"])].copy()
    rows = []
    for (ds, pid, blk), g in exp.groupby(["dataset", "participant", "block_number"]):
        dec = g
        if len(dec) < 5:
            continue
        conf = g[g["conflict"] == 1]
        rows.append({
            "dataset": ds, "participant": pid, "block_number": blk,
            "environment": g["environment"].iloc[0],
            "threat": g["threat"].iloc[0],
            "block_conflict_rate": g["conflict"].mean(),
            "block_plan_advantage": g["plan_advantage"].mean(),
            "planning_rate": conf["chose_planning"].mean() if len(conf) else np.nan,
            "n_trials": len(g), "n_conflict": len(conf),
            "block_drift": g["block_drift"].iloc[0],
        })
    return pd.DataFrame(rows)


def exp2_within_subject(block_env):
    """Does per-block planning rate track the block's conflict rate?

    Per participant OLS slope + Spearman correlation across their blocks, then a
    group-level test of the mean slope/correlation against 0.
    """
    rows = []
    for pid, g in block_env.groupby("participant"):
        g = g.dropna(subset=["planning_rate", "block_conflict_rate"])
        if len(g) < 6:
            continue
        slope, intercept, r, p, se = stats.linregress(
            g["block_conflict_rate"], g["planning_rate"])
        rho, prho = stats.spearmanr(g["block_conflict_rate"], g["planning_rate"])
        ga = g.dropna(subset=["block_plan_advantage"])
        if len(ga) >= 6 and ga["block_plan_advantage"].std() > 0:
            slope_adv, _, r_adv, p_adv, _ = stats.linregress(
                ga["block_plan_advantage"], ga["planning_rate"])
        else:
            slope_adv, r_adv, p_adv = np.nan, np.nan, np.nan
        rows.append({"participant": pid, "n_blocks": len(g), "slope": slope,
                     "pearson_r": r, "spearman_rho": rho, "p": p,
                     "slope_adv": slope_adv, "r_adv": r_adv, "p_adv": p_adv})
    out = pd.DataFrame(rows)
    if len(out) > 1:
        t, p = stats.ttest_1samp(out["slope"], 0.0)
        print(f"  Exp2 within-subject slope of planning~conflict rate: "
              f"mean={out['slope'].mean():.3f}, t={t:.2f}, p={p:.3f}, N={len(out)}")
        t2, p2 = stats.ttest_1samp(out["slope_adv"].dropna(), 0.0)
        print(f"  Exp2 within-subject slope of planning~plan advantage: "
              f"mean={out['slope_adv'].mean():.3f}, t={t2:.2f}, p={p2:.3f}")
    return out


def exp2_shift_case(df):
    """YFX default_experiment: planning rate before vs after the agreement shift."""
    d = df[(df["dataset"] == "exp2_shift") & (df["is_experimental"])].copy()
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()
    block = d.groupby("block_number").agg(
        conflict_rate=("conflict", "mean"),
        planning_rate=("chose_planning", lambda s: np.nan),
        n=("conflict", "size"))
    rows = []
    for blk, g in d.groupby("block_number"):
        conf = g[g["conflict"] == 1]
        rows.append({"block_number": blk, "conflict_rate": g["conflict"].mean(),
                     "planning_rate": conf["chose_planning"].mean() if len(conf) else np.nan,
                     "n_conflict": len(conf)})
    by_block = pd.DataFrame(rows).sort_values("block_number")
    hi = by_block[by_block["conflict_rate"] <= 0.5]
    lo = by_block[by_block["conflict_rate"] > 0.5]
    summary = pd.DataFrame([{
        "regime": "high_agreement(low conflict)", "n_blocks": len(hi),
        "mean_conflict_rate": hi["conflict_rate"].mean(),
        "mean_planning_rate": hi["planning_rate"].mean()},
        {"regime": "low_agreement(high conflict)", "n_blocks": len(lo),
         "mean_conflict_rate": lo["conflict_rate"].mean(),
         "mean_planning_rate": lo["planning_rate"].mean()}])
    return by_block, summary


def exp2_between_cohort(df):
    """Planning rate on conflict trials: balanced vs high-agreement cohorts."""
    exp = df[(df["is_experimental"]) & (df["experiment"].isin(["exp1", "exp2", "exp3"]))]
    conf = exp[exp["conflict"] == 1]
    rows = []
    for pid, g in conf.groupby("participant"):
        if len(g) < 20:
            continue
        rows.append({"participant": pid, "dataset": g["dataset"].iloc[0],
                     "environment": g["environment"].iloc[0],
                     "p_plan": g["chose_planning"].mean(), "n": len(g)})
    per = pd.DataFrame(rows)
    if per.empty:
        return per, None
    groups = [g["p_plan"].values for _, g in per.groupby("environment")]
    if len(groups) >= 2:
        t, p = stats.ttest_ind(groups[0], groups[1], equal_var=False)
        print(f"  Exp2 between-cohort planning rate: "
              f"{dict(per.groupby('environment')['p_plan'].mean())}, t={t:.2f}, p={p:.3f}")
    return per, None


# %% [markdown]
# # Experiment 3 — threat and planning

# %%
def exp3_mixture(df):
    """Fit the threat-aware mixture model per participant (datasets with drift)."""
    d = df[(df["is_experimental"]) & (df["threat"].isin(["drift", "alternating"]))]
    rows = []
    for pid, g in d.groupby("participant"):
        fit = fit_mixture_model(g, use_threat=True)
        if fit is None:
            continue
        fit["participant"] = pid
        fit["dataset"] = g["dataset"].iloc[0]
        rows.append(fit)
    out = pd.DataFrame(rows)
    if len(out) > 1:
        t, p = stats.ttest_1samp(out["w1"], 0.0)
        print(f"  Exp3 mixture w1 (threat effect on planning): "
              f"mean={out['w1'].mean():.3f}, t={t:.2f}, p={p:.3f}, N={len(out)}")
    return out


def exp3_gee(df):
    """Population logistic: planning choice ~ planning diff + threat + drift."""
    d = df[(df["is_experimental"]) & (df["conflict"] == 1)
           & (df["threat"].isin(["drift", "alternating"]))].copy()
    d = d.dropna(subset=["diff_planning", "z_ball_y", "incoming_direction"])
    d["ballXdrift"] = d["z_ball_y"] * d["block_drift"]
    res = fit_gee(d, ["diff_planning", "z_ball_y", "block_drift", "ballXdrift",
                      "incoming_direction"], "chose_planning", "participant")
    return res, d


def exp3_drift_contrast(df):
    """Within-subject planning rate in drift vs follow blocks (alternating cohort)."""
    d = df[(df["is_experimental"]) & (df["dataset"] == "exp3_altdrift")
           & (df["conflict"] == 1)]
    rows = []
    for pid, g in d.groupby("participant"):
        for drift in (0, 1):
            gg = g[g["block_drift"] == drift]
            if len(gg) < 15:
                continue
            rows.append({"participant": pid, "block_drift": drift,
                         "p_plan": gg["chose_planning"].mean(), "n": len(gg)})
    per = pd.DataFrame(rows)
    if per.empty:
        return per, None
    piv = per.pivot_table(index="participant", columns="block_drift",
                          values="p_plan").dropna()
    if len(piv) > 1:
        t, p = stats.ttest_rel(piv[1], piv[0])
        print(f"  Exp3 drift-vs-follow planning rate (N={len(piv)}): "
              f"drift={piv[1].mean():.3f}, follow={piv[0].mean():.3f}, "
              f"t={t:.2f}, p={p:.3f}")
    return per, piv


def exp3_logistic_slopes(df):
    """Per-participant logistic slope of planning on threat (conflict trials).

    Robust complement to the mixture model: on conflict trials the choice is a
    clean binary planning/greedy decision, so a simple logistic fit of
    `chose_planning ~ z(ball_y)` gives an interpretable threat coefficient.
    """
    d = df[(df["is_experimental"]) & (df["conflict"] == 1)
           & (df["threat"].isin(["drift", "alternating"]))].copy()
    d = d.dropna(subset=["z_ball_y"])
    rows = []
    for pid, g in d.groupby("participant"):
        if len(g) < 30 or g["chose_planning"].nunique() < 2:
            continue
        X = sm.add_constant(g[["z_ball_y"]].astype(float))
        try:
            m = sm.Logit(g["chose_planning"].astype(float), X).fit(disp=0)
            rows.append({"participant": pid, "dataset": g["dataset"].iloc[0],
                         "n_conflict": len(g),
                         "slope_ball_y": m.params["z_ball_y"],
                         "p_ball_y": m.pvalues["z_ball_y"]})
        except Exception:
            continue
    out = pd.DataFrame(rows)
    if len(out) > 1:
        t, p = stats.ttest_1samp(out["slope_ball_y"], 0.0)
        print(f"  Exp3 logistic slope of planning on z(ball_y): "
              f"mean={out['slope_ball_y'].mean():.3f}, t={t:.2f}, p={p:.3f}, N={len(out)}")
    return out


def exp3_threat_bins(df):
    """Planning rate in the closest-to-death vs safest third of sequences.

    `ball_y_at_top` is the ball's relative screen y at the sequence entry;
    it approaches 0 as the camera catches up to a dying ball (verified against
    the game-over condition `ball.y - cameraY < 0`). So LOW values = near death.
    """
    d = df[(df["is_experimental"]) & (df["conflict"] == 1)
           & (df["threat"].isin(["drift", "alternating"]))].dropna(
        subset=["ball_y_at_top"])
    rows = []
    for pid, g in d.groupby("participant"):
        if len(g) < 40:
            continue
        q1, q2 = g["ball_y_at_top"].quantile([1 / 3, 2 / 3])
        near = g[g["ball_y_at_top"] <= q1]
        safe = g[g["ball_y_at_top"] >= q2]
        if len(near) < 10 or len(safe) < 10:
            continue
        rows.append({"participant": pid, "dataset": g["dataset"].iloc[0],
                     "p_plan_near_death": near["chose_planning"].mean(),
                     "p_plan_safe": safe["chose_planning"].mean(),
                     "n_near": len(near), "n_safe": len(safe)})
    out = pd.DataFrame(rows)
    if len(out) > 1:
        t, p = stats.ttest_rel(out["p_plan_near_death"], out["p_plan_safe"])
        print(f"  Exp3 planning near-death vs safe: "
              f"near={out['p_plan_near_death'].mean():.3f}, "
              f"safe={out['p_plan_safe'].mean():.3f}, t={t:.2f}, p={p:.3f}, N={len(out)}")
    return out


def exp3_execution_benefit(df):
    """On conflict trials, does planning have a non-kinematic RT effect?

    Conflict trials guarantee the planning hole is farther from the entry
    (larger 1-step distance) and closer to the goal (smaller 2-step distance),
    so raw RT differences are dominated by kinematics. The decision model
    controls for `chosen_1step_dist`; the execution model controls for
    `chosen_2step_dist` (+ `chosen_1step_dist`). RTs are also returned
    residualized on the relevant kinematic distance for the adjusted figure.
    """
    d = df[(df["is_experimental"]) & (df["conflict"] == 1)
           & (df["threat"].isin(["drift", "alternating"]))].copy()
    d = d.dropna(subset=["rt_exec", "rt_decision", "chosen_2step_dist",
                         "chosen_1step_dist", "z_ball_y", "block_drift"])
    d = d[(d["rt_exec"] > 0) & (d["rt_decision"] > 0)].copy()
    d["chosen_second_step"] = d["chosen_2step_dist"] - d["chosen_1step_dist"]
    d["rt_total"] = d["rt_decision"] + d["rt_exec"]
    for col in ["rt_decision", "rt_exec", "rt_total"]:
        d["log_" + col] = np.log(d[col].clip(lower=1))

    res_dec = fit_gee(d, ["chose_planning", "chosen_1step_dist", "block_drift",
                          "z_ball_y"], "log_rt_decision", "participant",
                      family=sm.families.Gaussian())
    res_exec = fit_gee(d, ["chose_planning", "chosen_2step_dist",
                           "chosen_1step_dist", "block_drift", "z_ball_y"],
                       "log_rt_exec", "participant",
                       family=sm.families.Gaussian())
    res_total = fit_gee(d, ["chose_planning", "chosen_2step_dist", "block_drift",
                            "z_ball_y"], "log_rt_total", "participant",
                        family=sm.families.Gaussian())

    def _resid(y, cols):
        X = sm.add_constant(d[list(cols)].astype(float))
        return sm.OLS(y, X).fit().resid

    # residualize on the full kinematic set so the plotted residuals are the
    # non-geometric component of each RT
    d["rt_decision_adj"] = _resid(d["rt_decision"], ["chosen_1step_dist"])
    d["rt_exec_adj"] = _resid(d["rt_exec"],
                              ["chosen_1step_dist", "chosen_2step_dist"])
    d["rt_total_adj"] = _resid(d["rt_total"],
                               ["chosen_1step_dist", "chosen_2step_dist"])
    return res_exec, res_dec, res_total, d


# %% [markdown]
# ## Plotting

# %%
def _savefig(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def make_plots(prevalence, transitions, glmhmm, block_env, shift_by_block,
               mixture, drift_piv, exec_df, threat_bins=None):
    sns.set_style("whitegrid")

    if not prevalence.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        p = prevalence.sort_values("p_plan")
        ax.barh(p["participant"].str[:8], p["p_plan"], color="steelblue")
        ax.axvline(0.5, color="k", ls="--")
        ax.set_xlabel("P(plan | conflict)"); ax.set_xlim(0, 1)
        ax.set_title("Exp1: planning prevalence")
        _savefig(fig, "exp1_prevalence.png")

    if not transitions.empty:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(transitions["bernoulli_stay"], transitions["stay_prob"],
                   color="coral", zorder=3)
        lims = [0.4, 1.0]
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlabel("Bernoulli (independent) stay prob")
        ax.set_ylabel("Empirical stay prob")
        ax.set_title("Exp1: strategy persistence vs independence")
        _savefig(fig, "exp1_transitions.png")

    if not glmhmm.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        tm = np.array([json.loads(s) for s in glmhmm["transition_matrix"]])
        sns.heatmap(tm.mean(axis=0), annot=True, vmin=0, vmax=1, cmap="bone", ax=ax)
        ax.set_title(f"Exp1: mean GLM-HMM transition matrix (N={len(glmhmm)})")
        _savefig(fig, "exp1_glmhmm_transition.png")

    if not block_env.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        for env, g in block_env.groupby("environment"):
            ax.scatter(g["block_conflict_rate"], g["planning_rate"],
                       s=18, alpha=0.5, label=env)
        ax.set_xlabel("block conflict rate (planning vs greedy disagree)")
        ax.set_ylabel("P(plan | conflict)")
        ax.set_title("Exp2: planning vs environmental statistics")
        ax.legend(fontsize=8)
        _savefig(fig, "exp2_env_vs_planning.png")

    if shift_by_block is not None and not shift_by_block.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(shift_by_block["block_number"], shift_by_block["planning_rate"],
                "o-", color="purple", label="P(plan|conflict)")
        ax2 = ax.twinx()
        ax2.plot(shift_by_block["block_number"], shift_by_block["conflict_rate"],
                 "s--", color="gray", label="block conflict rate")
        ax.set_xlabel("block"); ax.set_ylabel("P(plan | conflict)", color="purple")
        ax2.set_ylabel("conflict rate", color="gray")
        ax.set_title("Exp2: YFX default_experiment environment shift")
        _savefig(fig, "exp2_shift_case.png")

    if not mixture.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(mixture["w1"], bins=12, color="indianred", edgecolor="k")
        ax.axvline(0, color="k", ls="--")
        ax.set_xlabel("w1 (threat effect on planning)")
        ax.set_title("Exp3: mixture-model threat coefficient")
        _savefig(fig, "exp3_w1.png")

    if drift_piv is not None and not drift_piv.empty:
        fig, ax = plt.subplots(figsize=(5, 5))
        for pid, row in drift_piv.iterrows():
            ax.plot([0, 1], [row[0], row[1]], "-o", alpha=0.6)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["follow", "drift"])
        ax.set_ylabel("P(plan | conflict)")
        ax.set_title("Exp3: within-subject threat contrast")
        _savefig(fig, "exp3_drift_contrast.png")

    if exec_df is not None and not exec_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=exec_df, x="chose_planning", y="log_rt_exec",
                    hue="block_drift", ax=ax)
        ax.set_xlabel("chose planning"); ax.set_ylabel("log execution RT")
        ax.set_title("Exp3: execution-time benefit of planning")
        _savefig(fig, "exp3_exec_benefit.png")

    if threat_bins is not None and not threat_bins.empty:
        fig, ax = plt.subplots(figsize=(5, 5))
        for _, row in threat_bins.iterrows():
            ax.plot([0, 1], [row["p_plan_near_death"], row["p_plan_safe"]],
                    "-o", alpha=0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["near death\n(low ball_y)", "safe\n(high ball_y)"])
        ax.set_ylabel("P(plan | conflict)")
        ax.set_title("Exp3: planning vs proximity to death")
        _savefig(fig, "exp3_threat_bins.png")


# %% [markdown]
# ## Main

# %%
def _summarize_gee(res, label):
    if res is None:
        print(f"  {label}: not fit")
        return
    print(f"  {label}:")
    for name, coef, p in zip(res.params.index, res.params.values, res.pvalues.values):
        print(f"      {name:32s} beta={coef:8.3f}  p={p:.3g}")


def _write_gee(res, path):
    if res is None:
        return
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(res.summary().tables[1].as_csv())


def main(from_cache=False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = OUT_DIR / "sequence_table.csv"
    if from_cache and cache.exists():
        print(f"Loading cached sequence table {cache} ...")
        all_df = pd.read_csv(cache)
    else:
        print("Loading datasets ...")
        all_df, comp_df = load_all()
        all_df = add_features(all_df)
        all_df.to_csv(cache, index=False)
    print(f"  total: {all_df['participant'].nunique()} participants, "
          f"{len(all_df)} sequences")

    print("\n=== Experiment 1: planning prevalence & temporal structure ===")
    prevalence = exp1_prevalence(all_df)
    transitions = exp1_transitions(all_df)
    sel1, K1, per1 = exp1_hmm_model_selection(all_df, experiment="exp1")
    sel3, K3, per3 = exp1_hmm_model_selection(all_df, experiment="exp3")
    sel_df = pd.concat([sel1, sel3], ignore_index=True)
    per_opt = pd.concat([per1, per3], ignore_index=True)
    K_use = K1 if K1 else 1
    glmhmm, hmm_details, state_weights = exp1_glmhmm(
        all_df, experiment="exp1", num_states=K_use)
    # forced 2-state comparison (not BIC-supported) for the state-coefficient viz
    glmhmm_k2, details_k2, weights_k2 = exp1_glmhmm(
        all_df, experiment="exp1", num_states=2)
    posteriors = _posterior_long(details_k2)
    state_weights_all = pd.concat([state_weights, weights_k2], ignore_index=True)
    exp1_mix = []
    for pid, g in all_df[all_df["experiment"] == "exp1"].groupby("participant"):
        fit = fit_mixture_model(g, use_threat=False)
        if fit:
            fit["participant"] = pid
            exp1_mix.append(fit)
    exp1_mix = pd.DataFrame(exp1_mix)
    prevalence.to_csv(OUT_DIR / "exp1_prevalence.csv", index=False)
    transitions.to_csv(OUT_DIR / "exp1_transitions.csv", index=False)
    sel_df.to_csv(OUT_DIR / "exp1_hmm_model_selection.csv", index=False)
    per_opt.to_csv(OUT_DIR / "exp1_hmm_optimal_k_per_participant.csv", index=False)
    glmhmm.to_csv(OUT_DIR / "exp1_glmhmm.csv", index=False)
    glmhmm_k2.to_csv(OUT_DIR / "exp1_glmhmm_k2.csv", index=False)
    state_weights_all.to_csv(OUT_DIR / "exp1_hmm_state_weights.csv", index=False)
    posteriors.to_csv(OUT_DIR / "exp1_hmm_state_posteriors.csv", index=False)
    exp1_mix.to_csv(OUT_DIR / "exp1_mixture.csv", index=False)
    if not glmhmm.empty:
        print(f"  GLM-HMM K*={K_use} (exp1); forced K=2 for comparison")
    if not exp1_mix.empty:
        print(f"  mixture baseline P(plan)={exp1_mix['p_plan_base'].mean():.3f}")

    print("\n=== Experiment 2: environmental statistics ===")
    block_env = exp2_block_environment(all_df)
    within = exp2_within_subject(block_env)
    shift_by_block, shift_summary = exp2_shift_case(all_df)
    per_cohort, _ = exp2_between_cohort(all_df)
    block_env.to_csv(OUT_DIR / "exp2_block_environment.csv", index=False)
    within.to_csv(OUT_DIR / "exp2_within_subject.csv", index=False)
    # exp2_shift (YFX default_experiment) is deprecated: that dataset is no
    # longer registered in DATASETS, so exp2_shift_case returns empty. Only write
    # the shift outputs if a real shift experiment is present.
    if not shift_by_block.empty:
        shift_by_block.to_csv(OUT_DIR / "exp2_shift_blocks.csv", index=False)
    if not shift_summary.empty:
        shift_summary.to_csv(OUT_DIR / "exp2_shift_summary.csv", index=False)
    per_cohort.to_csv(OUT_DIR / "exp2_between_cohort.csv", index=False)
    if not shift_summary.empty:
        print(shift_summary.to_string(index=False))

    print("\n=== Experiment 3: threat and planning ===")
    mixture = exp3_mixture(all_df)
    slopes = exp3_logistic_slopes(all_df)
    threat_bins = exp3_threat_bins(all_df)
    gee_res, gee_data = exp3_gee(all_df)
    drift_per, drift_piv = exp3_drift_contrast(all_df)
    exec_res, dec_res, total_res, exec_df = exp3_execution_benefit(all_df)
    mixture.to_csv(OUT_DIR / "exp3_mixture.csv", index=False)
    slopes.to_csv(OUT_DIR / "exp3_logistic_slopes.csv", index=False)
    threat_bins.to_csv(OUT_DIR / "exp3_threat_bins.csv", index=False)
    _write_gee(gee_res, OUT_DIR / "exp3_gee_planning.csv")
    _write_gee(exec_res, OUT_DIR / "exp3_gee_exec_rt.csv")
    _write_gee(dec_res, OUT_DIR / "exp3_gee_decision_rt.csv")
    _write_gee(total_res, OUT_DIR / "exp3_gee_total_rt.csv")
    drift_per.to_csv(OUT_DIR / "exp3_drift_contrast.csv", index=False)
    _summarize_gee(gee_res, "GEE: chose_planning")
    _summarize_gee(exec_res, "GEE: log execution RT (kinematic-controlled)")
    _summarize_gee(dec_res, "GEE: log decision RT (kinematic-controlled)")
    _summarize_gee(total_res, "GEE: log total RT")

    print("\nMaking plots ...")
    make_plots(prevalence, transitions, glmhmm, block_env, shift_by_block,
               mixture, drift_piv, exec_df, threat_bins)
    print(f"Done. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main(from_cache="--from-cache" in sys.argv)

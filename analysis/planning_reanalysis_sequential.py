# %% [markdown]
# # Sequential / "memoryless" re-analysis (claim 2)
#
# The synthesis says strategy choice is memoryless (GLM-HMM K*=1; 0/18 excess
# stay-probability). The level decomposition flagged `prev_plan` (previous
# conflict-trial planning) as the strongest trial-level predictor. This script
# decides whether that is real short-term dependence, a within-block-centering
# artifact, or noise.
#
#   * Lag Mundlak: `chose_planning ~ prev_plan_{w,b,p}` (+ controls).
#   * Artifact test: LPM with uncentered participant+block fixed effects vs
#     block-mean centering.
#   * Within-participant permutation of the lag association.
#   * GLM-HMM BIC with vs without a lag input (K = 1, 2).
#
# Outputs (analysis/planning_outputs/):
#   planning_reanalysis_sequential_coefs.csv
#   planning_reanalysis_sequential_perm.csv
#   planning_reanalysis_sequential_hmm.csv
#
# Run with:
#   python analysis/planning_reanalysis_sequential.py

# %%
import contextlib
import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parent.parent
XD = _REPO / "analysis" / "cross_dataset_outputs"
PO = _REPO / "analysis" / "planning_outputs"
N_PERM = 1000
RNG = np.random.default_rng(0)


def load_trials():
    d = pd.read_csv(XD / "cross_dataset_trials.csv")
    d = d[d["is_experimental"]].copy()
    d["conflict"] = d["conflict"].fillna(0).astype(int)
    return d


def build_conflict_lags(d):
    c = d[d.conflict == 1].copy()
    c = c.sort_values(["participant", "block_number", "sequence_index"])
    g = c.groupby(["participant", "block_number"])
    c["prev_plan"] = g["chose_planning"].shift(1)
    c["lag_index"] = g.cumcount()
    c["conflict_mag"] = (c["diff_planning"] - c["diff_1step"]).abs()
    return c


def add_levels(df, cols):
    df = df.copy()
    for c in cols:
        by = df.groupby(["participant", "block_number"])[c]
        df[c + "_w"] = df[c] - by.transform("mean")
        df[c + "_b"] = by.transform("mean")
        df[c + "_p"] = df.groupby("participant")[c].transform("mean")
    return df


def clogit(f, data):
    return smf.logit(f, data=data).fit(disp=0, cov_type="cluster",
                                       cov_kwds={"groups": data["participant"]})


# %% [markdown]
# ## Lag coefficients and the centering-artifact test

# %%
def lag_coefficients(c):
    c = add_levels(c.dropna(subset=["prev_plan", "conflict_mag"]),
                   ["prev_plan", "conflict_mag", "lag_index"])
    rows = []
    # pooled
    m = clogit("chose_planning ~ prev_plan", c)
    rows.append({"model": "pooled_prev", "term": "prev_plan",
                 "coef": m.params["prev_plan"], "se": m.bse["prev_plan"],
                 "p": m.pvalues["prev_plan"], "n": int(m.nobs)})
    # Mundlak with difficulty + time controls
    f = ("chose_planning ~ prev_plan_w + prev_plan_b + prev_plan_p"
         " + conflict_mag_w + conflict_mag_b + conflict_mag_p"
         " + lag_index_w + lag_index_b + lag_index_p")
    m = clogit(f, c)
    for term in ["prev_plan_w", "prev_plan_b", "prev_plan_p"]:
        rows.append({"model": "mundlak_prev", "term": term,
                     "coef": m.params[term], "se": m.bse[term],
                     "p": m.pvalues[term], "n": int(m.nobs)})
    return pd.DataFrame(rows), c


def artifact_test_lpm(c):
    """Compare the lag slope under (a) block-mean centering and (b) uncentered
    participant+block fixed effects."""
    d = c.dropna(subset=["prev_plan", "chose_planning"]).copy()
    y = d["chose_planning"].astype(float).to_numpy()
    rows = []
    # (a) centred Mundlak
    d = add_levels(d, ["prev_plan"])
    Xa = sm.add_constant(d[["prev_plan_w", "prev_plan_b", "prev_plan_p"]]
                         .astype(float).to_numpy())
    ma = sm.OLS(y, Xa).fit(cov_type="cluster",
                           cov_kwds={"groups": d["participant"]})
    rows.append({"spec": "lpm_centred", "term": "prev_plan_w",
                 "coef": ma.params[1], "se": ma.bse[1], "p": ma.pvalues[1]})
    # (b) uncentered participant + block fixed effects
    fb = "chose_planning ~ prev_plan + C(participant) + C(participant):C(block_number)"
    mb = smf.ols(fb, data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["participant"]})
    rows.append({"spec": "lpm_block_FE", "term": "prev_plan",
                 "coef": mb.params["prev_plan"], "se": mb.bse["prev_plan"],
                 "p": mb.pvalues["prev_plan"]})
    return pd.DataFrame(rows)


# %% [markdown]
# ## Within-participant permutation

# %%
def permutation_test(c, n_perm=N_PERM):
    """Permute chose_planning within participant; observed statistic = pooled
    within-block lag-1 logistic coefficient."""
    d = c.dropna(subset=["prev_plan", "chose_planning"]).copy()
    d = d.sort_values(["participant", "block_number", "sequence_index"])

    def stat(frame):
        try:
            m = smf.ols("chose_planning ~ prev_plan_w + C(participant)",
                        data=add_levels(frame, ["prev_plan"])).fit()
            return float(m.params["prev_plan_w"])
        except Exception:
            return np.nan

    obs = stat(d)
    null = np.empty(n_perm)
    for k in range(n_perm):
        dd = d.copy()
        dd["chose_planning"] = (
            dd.groupby("participant")["chose_planning"]
            .transform(lambda s: RNG.permutation(s.to_numpy())))
        null[k] = stat(dd)
    return {"observed": obs, "null_mean": float(np.nanmean(null)),
            "p_two_sided": float(np.mean(np.abs(null) >= abs(obs))),
            "p_negative": float(np.mean(null <= obs))}


# %% [markdown]
# ## GLM-HMM with vs without a lag input

# %%
def hmm_design_with_lag(g):
    g = g.sort_values(["block_number", "sequence_index"]).dropna(
        subset=["diff_1step", "diff_planning", "incoming_direction"])
    return np.column_stack([g["diff_1step"].values, g["diff_planning"].values,
                            g["incoming_direction"].values,
                            g["prev_plan"].fillna(0).values]), \
        g["chosen_left"].astype(int).values


def fit_hmm(Xtr, ytr, K, num_iters=50, seed=0):
    import ssm
    import autograd.numpy.random as npr
    np.random.seed(seed); npr.seed(seed)
    mu, sd = Xtr[:, :2].mean(0), Xtr[:, :2].std(0)
    sd[sd == 0] = 1.0
    Z = np.column_stack([(Xtr[:, :2] - mu) / sd, Xtr[:, 2:], np.ones(len(Xtr))])
    T_, M = Z.shape
    if K == 1:
        try:
            m = sm.Logit(ytr, Z).fit(disp=0)
        except Exception:
            return None
        ll, npar = float(m.llf), M
    else:
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                model = ssm.HMM(K, 1, M, observations="input_driven_obs",
                                observation_kwargs=dict(C=2),
                                transitions="standard")
                model.fit([ytr.reshape(-1, 1)], inputs=Z, method="em",
                          num_iters=num_iters, tolerance=1e-4)
            ll = float(model.log_likelihood([ytr.reshape(-1, 1)], inputs=Z))
            npar = K * M + K * (K - 1) + (K - 1)
        except Exception:
            return None
    return {"ll": ll, "n_params": npar, "bic": -2 * ll + npar * np.log(T_)}


def hmm_with_lag(c, Ks=(1, 2)):
    c = c.sort_values(["participant", "block_number", "sequence_index"])
    rows = []
    for pid, g in c.groupby("participant"):
        X, y = hmm_design_with_lag(g)
        if len(y) < 60 or y.min() == y.max():
            continue
        for use_lag in (False, True):
            Xk = X if use_lag else X[:, :3]
            for K in Ks:
                st = fit_hmm(Xk, y, K)
                if st is None:
                    continue
                rows.append({"participant": pid, "K": K, "lag_input": use_lag,
                             "ll": st["ll"], "n_params": st["n_params"],
                             "bic": st["bic"]})
    return pd.DataFrame(rows)


# %% [markdown]
# ## Main

# %%
def make_figure(coefs, perm, hmm):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    m = coefs[coefs.model == "mundlak_prev"]
    ax = axes[0]
    y = np.arange(len(m))
    ax.errorbar(m["coef"], y, xerr=1.96 * m["se"], fmt="o", color="k",
                ecolor="0.6", capsize=3)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(m["term"])
    ax.set_title(f"Lag Mundlak coefficients\n(perm p={perm['p_two_sided']:.3f})")
    ax.set_xlabel("logit coef")

    ax = axes[1]
    pooled = coefs[coefs.model == "pooled_prev"]
    ax.bar([0], pooled["coef"], color="#e07a5f")
    ax.errorbar([0], pooled["coef"], yerr=1.96 * pooled["se"], fmt="none",
                ecolor="k", capsize=4)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks([0]); ax.set_xticklabels(["pooled prev_plan"])
    ax.set_title("Pooled lag effect")

    ax = axes[2]
    if not hmm.empty:
        s = hmm.groupby(["K", "lag_input"])["bic"].sum().reset_index()
        for lag, g in s.groupby("lag_input"):
            ax.plot(g["K"], g["bic"], "-o", label=f"lag_input={lag}")
        ax.set_xlabel("K"); ax.set_ylabel("sum BIC")
        ax.set_title("GLM-HMM BIC with/without lag")
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = PO / "figures" / "fig_planning_reanalysis_sequential.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    PO.mkdir(parents=True, exist_ok=True)
    d = load_trials()
    c = build_conflict_lags(d[d.source == "cloud"])
    coefs, c2 = lag_coefficients(c)
    art = artifact_test_lpm(c)
    coefs = pd.concat([coefs, art.assign(term=art["term"])], ignore_index=True)
    perm = permutation_test(c)
    print("lag coefficients:\n", coefs.round(4).to_string(index=False))
    print("\npermutation:", {k: round(v, 4) if isinstance(v, float) else v
                             for k, v in perm.items()})
    print("\nfitting GLM-HMM with/without lag (may take a few minutes)...")
    hmm = hmm_with_lag(c)
    if not hmm.empty:
        piv = hmm.pivot_table(index=["K", "lag_input"], values="bic",
                              aggfunc="sum")
        print(piv.round(1).to_string())

    coefs.to_csv(PO / "planning_reanalysis_sequential_coefs.csv", index=False)
    pd.DataFrame([perm]).to_csv(PO / "planning_reanalysis_sequential_perm.csv",
                                index=False)
    hmm.to_csv(PO / "planning_reanalysis_sequential_hmm.csv", index=False)
    fig = make_figure(coefs, perm, hmm)
    print(f"\nwrote planning_reanalysis_sequential_*.csv and {fig}")


if __name__ == "__main__":
    main()

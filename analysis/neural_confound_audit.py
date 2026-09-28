# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#     jupytext_version: 1.19.4
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Neural confound audit
#
# Adversarial controls for the multi-run neural conclusions. Runs across all
# registered sessions and writes per-run CSVs plus aggregate copies. No
# existing script is modified; all helpers here are local or pure functions
# imported from the existing modules.
#
# Controls:
#   1. leakage_control        shuffled CV vs leave-block-out CV
#   2. position_control       base nuisances vs + ball_x/ball_y/camera_y/rel_y
#   3. region_count_matched   per-region decoding at the smallest region's N
#   4. death_matched          death-anchored vs block/time-matched normal
#   5. threat_associations    is the threat regressor just RT/position?
#   6. aggregate_fdr          BH-FDR across the headline metrics
#   7. order_diagnostics      lag-1 autocorrelation, condition x block V
#
# Run with:
#   C:\Users\manik\AppData\Local\Programs\Python\Python311\python.exe analysis\neural_confound_audit.py

# %%
import json
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler

from neural_common import RUN_ORDER, RUNS, out_dir, AGG_DIR, unit_regions
from neural_lda_decoding import (
    rate_features, pca_features, build_label_vectors, make_transform,
    RNG_SEED, N_FOLDS, PERM_FOLDS,
)
from neural_continuous_decoding import (
    residualize, residualize_features, cv_regression, permutation_corr_p,
    N_PERM,
)

AUDIT_REPEATS = 3
N_BOOT_REGION = 10
BOOT_FOLDS = 5
# -----------------------------------------------------------------------


# %%
def load_binned(out):
    z = np.load(out / "segmented_spikes_binned.npz", allow_pickle=True)
    return z["binned"], z["unit_ids"], z["bin_centers"]


def load_labels(out):
    """Label vectors; a dummy move_dir_vx column avoids the module-level
    path-dependent `add_move_dir_vx`."""
    lab = pd.read_csv(out / "trial_labels.csv")
    lab["move_dir_vx"] = 0
    return build_label_vectors(lab), lab


def build_regressors(out):
    lab = pd.read_csv(out / "trial_labels.csv")
    tt = pd.read_csv(out / "trial_table.csv")
    gL, gR = lab["greedy_cost_L"].to_numpy(float), lab["greedy_cost_R"].to_numpy(float)
    pL, pR = lab["planning_cost_L"].to_numpy(float), lab["planning_cost_R"].to_numpy(float)
    ch = lab["choice_hole"].to_numpy()

    def holes(s):
        try:
            return json.loads(s)
        except Exception:
            return [np.nan, np.nan]

    hl = tt["hole_locations"].map(holes)
    hA, hB = hl.str[0].to_numpy(), hl.str[1].to_numpy()
    isA, isB = ch == hA, ch == hB
    chosen_planning = np.where(isA, pL, np.where(isB, pR, np.nan))
    greedy_gap, planning_gap = gL - gR, pL - pR
    return {
        "greedy_gap": greedy_gap, "planning_gap": planning_gap,
        "model_conflict": planning_gap - greedy_gap,
        "conflict_mag": np.abs(planning_gap - greedy_gap),
        "chosen_greedy_cost": np.where(isA, gL, np.where(isB, gR, np.nan)),
        "chosen_planning_cost": chosen_planning,
    }


def build_nuisances(out):
    tt = pd.read_csv(out / "trial_table.csv")
    lab = pd.read_csv(out / "trial_labels.csv")
    df = tt[["trial_id", "block_index", "choice_hole"]].copy()
    df["rt_ms"] = tt["choice_time_ms"] - tt["entry_time_ms"]
    df["ball_time_ms"] = tt["exit_time_ms"] - tt["choice_time_ms"]
    df["trial_duration_ms"] = tt["exit_time_ms"] - tt["trial_start_ms"]
    df["side"] = (tt["choice_hole"] >= 6).astype(int)
    df = df.merge(lab[["trial_id", "entry_hole"]], on="trial_id", how="left")
    df["move_dir"] = np.sign(df["choice_hole"] - df["entry_hole"]).to_numpy()
    cols = ["rt_ms", "ball_time_ms", "trial_duration_ms", "block_index",
            "side", "move_dir"]
    X = np.nan_to_num(df[cols].to_numpy(float))
    return np.column_stack([np.ones(len(X)), X]), df


def build_position(run_id, out):
    """(n_trials, 4) ball_x, ball_y, camera_y, ball_y-camera_y at choice."""
    with open(RUNS[run_id].behavior_path, encoding="utf-8") as fh:
        data = json.load(fh)
    t, bx, by, cy = [], [], [], []
    for b in data.get("blocks", []):
        if b.get("block_index", 0) < 4:
            continue
        gs = b.get("game_states") or {}
        if not gs:
            continue
        t.append(np.asarray(gs["time"], float))
        bx.append(np.asarray(gs["ball_x"], float))
        by.append(np.asarray(gs["ball_y"], float))
        cy.append(np.asarray(gs["camera_y"], float))
    if not t:
        return np.zeros((0, 4))
    t = np.concatenate(t); bx = np.concatenate(bx)
    by = np.concatenate(by); cy = np.concatenate(cy)
    o = np.argsort(t)
    t, bx, by, cy = t[o], bx[o], by[o], cy[o]
    tt = pd.read_csv(out / "trial_table.csv")
    idx = np.clip(np.searchsorted(t, tt["choice_time_ms"].to_numpy(float)),
                  0, len(t) - 1)
    return np.nan_to_num(np.column_stack([bx[idx], by[idx], cy[idx],
                                          by[idx] - cy[idx]]))


def build_threat(run_id, out):
    return build_position(run_id, out)[:, 3]


# %%
def leave_block_out_acc(y, X, groups, transform, n_repeats=AUDIT_REPEATS):
    """Balanced accuracy with entire blocks held out (no temporal leakage)."""
    accs = []
    blocks = np.unique(groups)
    for rep in range(n_repeats):
        rng = np.random.default_rng(RNG_SEED + rep)
        for g in rng.permutation(blocks):
            te = groups == g
            tr = ~te
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            Xtr, Xte = transform(X[tr], y[tr], X[te])
            lda = LinearDiscriminantAnalysis(solver="eigen",
                                             shrinkage="auto").fit(Xtr, y[tr])
            accs.append(balanced_accuracy_score(y[te], lda.predict(Xte)))
    return float(np.mean(accs)) if accs else np.nan


def shuffled_acc(y, X, transform, n_repeats=AUDIT_REPEATS, n_folds=N_FOLDS):
    accs = []
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                              random_state=RNG_SEED + rep)
        for tr, te in skf.split(X, y):
            Xtr, Xte = transform(X[tr], y[tr], X[te])
            lda = LinearDiscriminantAnalysis(solver="eigen",
                                             shrinkage="auto").fit(Xtr, y[tr])
            accs.append(balanced_accuracy_score(y[te], lda.predict(Xte)))
    return float(np.mean(accs))


def leave_block_out_corr(y, X, groups):
    cs = []
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        if te.sum() < 5 or tr.sum() < 20:
            continue
        sc = StandardScaler().fit(X[tr])
        model = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr])
        pred = model.predict(sc.transform(X[te]))
        if np.std(pred) > 1e-12 and np.std(y[te]) > 1e-12:
            cs.append(float(np.corrcoef(y[te], pred)[0, 1]))
    return float(np.nanmean(cs)) if cs else np.nan


# %%
def leakage_control(run_id):
    out = out_dir(run_id)
    binned, unit_ids, bc = load_binned(out)
    lv, lab = load_labels(out)
    groups = lab["block_index"].to_numpy()
    lo, hi = 0.0, 1000.0
    X_rate = rate_features(binned, bc, lo, hi)
    X_pca = pca_features(binned, bc, lo, hi)
    rows = []
    specs = [("planning_vs_greedy", "rate", X_rate, "post"),
             ("planning_vs_greedy", "pca", X_pca, "post"),
             ("condition", "pca", X_pca, "post"),
             ("agree", "pca", X_pca, "post")]
    for hname, rep, X, win in specs:
        y, idx = lv[hname]
        keep = np.ones(len(y), bool) if idx is None else (
            idx if idx.dtype == bool else np.isin(np.arange(len(y)), idx))
        if keep.sum() < 40 or len(np.unique(y[keep])) < 2:
            continue
        tr = make_transform(rep)
        a_sh = shuffled_acc(y[keep], X[keep], tr)
        a_bl = leave_block_out_acc(y[keep], X[keep], groups[keep], tr)
        rows.append({"run_id": run_id, "metric": f"lda:{hname}", "rep": rep,
                     "window": win, "shuffled": a_sh,
                     "leave_block_out": a_bl, "delta": a_bl - a_sh})
    # ridge conflict_mag
    reg = build_regressors(out)["conflict_mag"]
    nuis, _ = build_nuisances(out)
    m = ~np.isnan(reg)
    y = reg[m]
    yc = residualize(y, nuis[m])
    Xc = residualize_features(X_pca[m], nuis[m])
    _, _, corr_sh, _ = cv_regression(yc, Xc, N_FOLDS, AUDIT_REPEATS, RNG_SEED)[0]
    corr_bl = leave_block_out_corr(yc, Xc, groups[m])
    rows.append({"run_id": run_id, "metric": "cont:conflict_mag", "rep": "pca",
                 "window": "post", "shuffled": corr_sh,
                 "leave_block_out": corr_bl, "delta": corr_bl - corr_sh})
    return rows


# %%
def position_control(run_id):
    out = out_dir(run_id)
    binned, unit_ids, bc = load_binned(out)
    X_pca = pca_features(binned, bc, 0.0, 1000.0)
    regs = build_regressors(out)
    nuis, _ = build_nuisances(out)
    pos = build_position(run_id, out)
    nuis_pos = np.column_stack([nuis, pos]) if len(pos) else nuis
    rows = []
    specs = [("conflict_mag", "both"), ("chosen_planning_cost", "none")]
    for rname, control in specs:
        y_reg = regs[rname]
        m = ~np.isnan(y_reg)
        X = X_pca[m]
        for label, nu in [("base", nuis[m]), ("base+position", nuis_pos[m])]:
            yc = residualize(y_reg[m], nu)
            Xc = residualize_features(X, nu) if control == "both" else X
            _, _, corr, _ = cv_regression(yc, Xc, N_FOLDS, AUDIT_REPEATS,
                                          RNG_SEED)[0]
            rows.append({"run_id": run_id, "regressor": rname,
                         "control": control, "nuisance_set": label,
                         "corr_mean": corr})
    # threat regressor
    threat = build_threat(run_id, out)
    m = ~np.isnan(threat)
    yc_base = residualize(threat[m], nuis[m])
    yc_pos = residualize(threat[m], nuis_pos[m])
    for label, yc in [("base", yc_base), ("base+position", yc_pos)]:
        _, _, corr, _ = cv_regression(yc, X_pca[m], N_FOLDS, AUDIT_REPEATS,
                                      RNG_SEED)[0]
        rows.append({"run_id": run_id, "regressor": "threat_rel_y",
                     "control": "target", "nuisance_set": label,
                     "corr_mean": corr})
    return rows


# %%
def region_count_matched(run_id):
    out = out_dir(run_id)
    binned, unit_ids, bc = load_binned(out)
    X = rate_features(binned, bc, 0.0, 1000.0)
    lv, _ = load_labels(out)
    meta = pd.read_csv(out / "unit_metadata.csv")
    reg = unit_regions(meta)
    masks = {r: np.array([reg.get(int(u), "") == r for u in unit_ids])
             for r in sorted(set(reg.values()))}
    sizes = {r: int(m.sum()) for r, m in masks.items() if m.sum() >= 8}
    if not sizes:
        return []
    n_min = min(sizes.values())
    tr = make_transform("rate")
    rows = []
    rng = np.random.default_rng(RNG_SEED)
    for hname in ["planning_vs_greedy", "condition"]:
        y, idx = lv[hname]
        keep = np.ones(len(y), bool) if idx is None else (
            idx if idx.dtype == bool else np.isin(np.arange(len(y)), idx))
        if keep.sum() < 40 or len(np.unique(y[keep])) < 2:
            continue
        for region, m in masks.items():
            if region not in sizes:
                continue
            accs = []
            for _ in range(N_BOOT_REGION):
                sub = rng.choice(np.flatnonzero(m), size=n_min, replace=False)
                accs.append(shuffled_acc(y[keep], X[keep][:, sub], tr,
                                         n_repeats=1, n_folds=BOOT_FOLDS))
            rows.append({"run_id": run_id, "region": region,
                         "hypothesis": hname, "n_units": sizes[region],
                         "n_matched": n_min, "acc_mean": float(np.mean(accs)),
                         "acc_sd": float(np.std(accs))})
    return rows


# %%
def death_matched(run_id):
    out = out_dir(run_id)
    dp = out / "death_times.csv"
    if not dp.exists():
        return []
    deaths = pd.read_csv(dp)
    if "death_time_ms" not in deaths.columns or len(deaths) < 5:
        return []
    tt = pd.read_csv(out / "trial_table.csv")
    units = pd.read_csv(out / "unit_metadata.csv")
    keep = set(units["unit_id"])
    spikes = pd.read_csv(out / "spikes_units.csv")
    spikes = spikes[spikes["unit_id"].isin(keep)]
    tbu = {int(u): np.sort(g["spike_time_behavioral_ms"].to_numpy())
           for u, g in spikes.groupby("unit_id")}
    unit_ids = sorted(tbu)
    d_times = deaths["death_time_ms"].to_numpy(float)
    d_blocks = deaths["block_index"].to_numpy()
    choice = tt["choice_time_ms"].to_numpy(float)
    blocks = tt["block_index"].to_numpy()

    def rmat(centers, lo, hi):
        dur = (hi - lo) / 1000.0
        X = np.zeros((len(centers), len(unit_ids)))
        for j, u in enumerate(unit_ids):
            ts = tbu[u]
            for i, c in enumerate(centers):
                a = np.searchsorted(ts, c + lo, "left")
                b = np.searchsorted(ts, c + hi, "left")
                X[i, j] = (b - a) / dur
        return np.sqrt(np.maximum(X, 0.0))

    rows = []
    lo, hi = -500.0, 0.0
    Xd = rmat(d_times, lo, hi)
    variants = {
        "all_normal": np.ones(len(tt), bool),
        "block_matched": np.isin(blocks, d_blocks),
    }
    for name, mask in variants.items():
        Xn = rmat(choice[mask], lo, hi)
        X = np.vstack([Xd, Xn])
        y = np.r_[np.ones(len(Xd), int), np.zeros(len(Xn), int)]
        accs = []
        for rep in range(AUDIT_REPEATS):
            skf = StratifiedKFold(n_splits=5, shuffle=True,
                                  random_state=RNG_SEED + rep)
            for tr, te in skf.split(X, y):
                sc = StandardScaler().fit(X[tr])
                lda = LinearDiscriminantAnalysis(solver="eigen",
                                                 shrinkage="auto").fit(
                    sc.transform(X[tr]), y[tr])
                accs.append(balanced_accuracy_score(
                    y[te], lda.predict(sc.transform(X[te]))))
        rows.append({"run_id": run_id, "window": f"[{lo:.0f},{hi:.0f}]",
                     "control": name, "n_deaths": len(Xd),
                     "n_normal": int(mask.sum()), "acc_mean": float(np.mean(accs))})
    return rows


# %%
def threat_associations(run_id):
    out = out_dir(run_id)
    threat = build_threat(run_id, out)
    _, df = build_nuisances(out)
    pos = build_position(run_id, out)
    df = df.copy()
    if len(pos):
        df["ball_x"] = pos[:, 0]; df["ball_y"] = pos[:, 1]
        df["camera_y"] = pos[:, 2]; df["rel_y"] = pos[:, 3]
    cols = ["rt_ms", "ball_time_ms", "trial_duration_ms", "block_index",
            "move_dir", "ball_x", "ball_y", "camera_y", "rel_y"]
    rows = []
    m0 = ~np.isnan(threat)
    for c in cols:
        if c not in df.columns:
            continue
        z = df[c].to_numpy(float)
        m = m0 & ~np.isnan(z)
        if m.sum() < 30 or np.std(z[m]) < 1e-12:
            continue
        r = float(np.corrcoef(threat[m], z[m])[0, 1])
        rows.append({"run_id": run_id, "nuisance": c, "pearson_r": r,
                     "n": int(m.sum())})
    return rows


# %%
def order_diagnostics(run_id):
    out = out_dir(run_id)
    lab = pd.read_csv(out / "trial_labels.csv")
    pvg = lab[lab["condition"].isin(["planning", "greedy"])].copy()
    pvg = pvg.sort_values(["block_index", "sequence_index"])
    s = (pvg["condition"] == "planning").astype(int).to_numpy()
    acs = []
    for _, g in pvg.groupby("block_index"):
        v = (g["condition"] == "planning").astype(int).to_numpy()
        if len(v) > 3 and v.std() > 0:
            acs.append(np.corrcoef(v[:-1], v[1:])[0, 1])
    tab = pd.crosstab(lab["condition"], lab["block_index"]).to_numpy(float)
    n = tab.sum()
    chi2 = sum((tab[i, j] - tab[i].sum() * tab[:, j].sum() / n) ** 2 /
               (tab[i].sum() * tab[:, j].sum() / n)
               for i in range(tab.shape[0]) for j in range(tab.shape[1])
               if tab[i].sum() * tab[:, j].sum() > 0)
    V = np.sqrt(chi2 / (n * min(tab.shape[0] - 1, tab.shape[1] - 1)))
    return {"run_id": run_id, "lag1_autocorr_planning": float(np.nanmean(acs)),
            "condition_block_cramers_v": float(V),
            "n_conflict": int(len(pvg))}


# %%
def fdr_bh(p):
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    if not ok.any():
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = np.arange(1, len(pv) + 1)
    qv = pv[order] * len(pv) / ranked
    qv = np.minimum.accumulate(qv[::-1])[::-1]
    q[ok] = np.clip(qv[np.argsort(order)], 0, 1)
    return q


def main():
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    leak, pos_c, region, death, thr_assoc, order = [], [], [], [], [], []
    for run in RUN_ORDER:
        print(f"=== {run} ===")
        leak += leakage_control(run)
        pos_c += position_control(run)
        region += region_count_matched(run)
        death += death_matched(run)
        thr_assoc += threat_associations(run)
        order.append(order_diagnostics(run))

    def dump(rows, name):
        df = pd.DataFrame(rows)
        if not df.empty:
            df.to_csv(AGG_DIR / name, index=False)
            print(f"  saved {name} ({len(df)} rows)")

    dump(leak, "leakage_control.csv")
    dump(pos_c, "position_control.csv")
    dump(region, "region_count_matched.csv")
    dump(death, "death_matched.csv")
    dump(thr_assoc, "threat_associations.csv")
    pd.DataFrame(order).to_csv(AGG_DIR / "order_diagnostics.csv", index=False)

    # FDR across the headline aggregate metrics
    summ = AGG_DIR / "aggregate_sign_consistency.csv"
    if summ.exists():
        s = pd.read_csv(summ)
        s["signflip_q_fdr"] = fdr_bh(s["signflip_p"].to_numpy())
        s.to_csv(AGG_DIR / "aggregate_fdr.csv", index=False)
        print("  saved aggregate_fdr.csv")

    print("\nLeakage (shuffled vs leave-block-out):")
    if leak:
        print(pd.DataFrame(leak)[["run_id", "metric", "rep", "shuffled",
                                  "leave_block_out"]].round(3).to_string(index=False))
    print("\nPosition control:")
    if pos_c:
        print(pd.DataFrame(pos_c).pivot_table(
            index=["regressor", "control"], columns="nuisance_set",
            values="corr_mean").round(3).to_string())
    print("\nOrder diagnostics:")
    print(pd.DataFrame(order).round(3).to_string(index=False))


if __name__ == "__main__":
    main()

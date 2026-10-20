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
# # Build the browsable figure index
#
# Creates, at the repo root, a clean themed symlink tree of every analysis
# figure, a MANIFEST.csv linking each figure to its generating script, its
# doc captions, its graphify concepts/community, and a status label, and an
# HTML contact sheet (`figures/index.html`) that carries the graphify tags and
# status badges (the symlink tree itself stays tag-free).
#
# Sources merged:
#   1. code scan  -> figure file  -> generating script (+line)
#   2. docs scan  -> figure file  -> caption (nearest markdown heading)
#   3. graphify   -> figure file  -> concept labels + community_name
#   4. curated    -> status labels (and a few manual captions)
#
# Run:  .venv-analysis/bin/python analysis/build_figure_index.py

# %%
import csv
import html
import json
import os
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIGROOT = REPO / "figures"
GRAPH = REPO / "graphify-out" / "graph.json"

THEMES = [
    ("00_overview", "Overview & task"),
    ("10_planning_behaviour", "Planning behaviour (online)"),
    ("20_cross_dataset", "Cross-dataset (cloud x EMU)"),
    ("30_neural_mtl", "Neural MTL decoding"),
    ("40_pass_timing", "Pass-timing & exit targeting"),
    ("50_scaling_modeling", "Scaling & model comparison"),
    ("90_problematic/superseded", "Not current - superseded"),
    ("90_problematic/stale_duplicates", "Not current - stale duplicates"),
    ("90_problematic/deprecated", "Not current - deprecated"),
    ("90_problematic/per_run_diagnostics", "Not current - per-run diagnostics"),
]

# %% ---------------------------------------------------------------- paths
def all_figures():
    return sorted(str(p.relative_to(REPO))
                  for p in (REPO / "analysis").rglob("*.png")
                  if "figures" in p.parts or p.parent.name == "figures"
                  or p.parent.name.startswith("aggregate")
                  or p.parent.name in {"planning_outputs", "superseded_figures"}
                  or re.match(r"y(fz|ga)_\d", p.parent.name)
                  or p.parent.name == "neural_outputs")


# %% ------------------------------------------------------- 1. code scan
def scan_code():
    """basename.png -> list of (script, line)."""
    out = {}
    pat = re.compile(r"""['"]([^'"]*\.png)['"]""")
    for f in sorted((REPO / "analysis").glob("*.py")):
        if f.name == "build_figure_index.py":
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if ".png" not in line:
                continue
            for name in pat.findall(line):
                out.setdefault(os.path.basename(name), []).append(
                    (str(f.relative_to(REPO)), i))
    return out


# %% ------------------------------------------------------- 2. docs scan
def _clean_heading(line, name):
    txt = line.lstrip("# ").strip()
    txt = txt.replace(name, " ")
    txt = re.sub(r"[`*]", "", txt)
    txt = re.sub(r"^[\s\-–—:.0-9]+", "", txt)
    txt = re.sub(r"[\s\-–—:]+$", "", txt)
    return re.sub(r"\s+", " ", txt).strip()


def scan_docs():
    """basename.png -> (doc, caption).

    Caption prefers a markdown heading that *names* the figure (the dedicated
    '### fig_x.png - ...' sections), then the nearest heading above its first
    mention, then a manual caption.
    """
    docs = sorted((REPO / "docs").rglob("*.md")) + sorted((REPO / "analysis").rglob("*.md"))
    named, fallback = {}, {}
    for f in docs:
        current = ""
        rel = str(f.relative_to(REPO))
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            is_head = line.lstrip().startswith("#")
            if is_head:
                current = line.lstrip("# ").strip().strip("`")
            for name in re.findall(r"[A-Za-z0-9_\-]+\.png", line):
                if is_head:
                    named.setdefault(name, (rel, _clean_heading(line, name)))
                else:
                    fallback.setdefault(name, (rel, current))
    cap = {}
    for name in set(named) | set(fallback):
        doc, c = (named.get(name) or fallback.get(name))
        if name in named and named[name][1]:
            doc, c = named[name]
        elif not c:
            doc, c = fallback.get(name, (doc, c))
        cap[name] = (doc, c)
    return cap


# %% ------------------------------------------------------- 3. graphify
def load_graphify():
    """basename.png -> dict(concepts=[...], community=...)."""
    if not GRAPH.exists():
        return {}
    g = json.loads(GRAPH.read_text())
    by = {}
    for n in g.get("nodes", []):
        sf = str(n.get("source_file", ""))
        if not sf.endswith(".png"):
            continue
        name = os.path.basename(sf)
        d = by.setdefault(name, {"concepts": [], "community": ""})
        label = n.get("label", "")
        if n.get("file_type") == "concept" and label and not label.endswith(".png"):
            if label not in d["concepts"]:
                d["concepts"].append(label)
        if not d["community"] and n.get("community_name"):
            d["community"] = n["community_name"]
    return by


# %% ------------------------------------------------------- 4. curated meta
STATUS = [
    (r"fig_planning_temporal", "superseded"),
    (r"^exp\d_", "stale duplicate"),
    (r"fig_xd_transfer|fig_xd_prevalence|fig_xd_prevalence_weight|fig_xd_individual",
     "robust (comparability)"),
    (r"fig_neural_confound_collapse", "robust"),
    (r"fig_neural_region", "robust (unit-matched)"),
    (r"fig_neural_bursty", "robust (control)"),
    (r"fig_neural_replication", "mixed (see rows)"),
    (r"fig_neural_window_profile", "null (pre) / consequence (post)"),
    (r"fig_neural_axis_alignment|fig_neural_sr_rsa|fig_neural_dpca_geometry",
     "weak / negative (exploratory)"),
    (r"fig_neural_death", "exploratory"),
    (r"fig_neural_threat|fig_xd_threat|fig_threat_bins5", "confounded (threat~height)"),
    (r"fig_entry_250ms|fig_choice_250ms|fig_entry_250ms_phase", "null (entry-anchored)"),
    (r"fig_pass_event", "exit only, small (kinematic)"),
    (r"fig_pass_time", "null (no non-kinematic code)"),
    (r"fig_exit_prospective|fig_exit_direction|fig_exit_lag_matched", "null/weak (no clock)"),
    (r"fig_exit_geometry", "robust (conflict_mag)"),
    (r"fig_exit_selectivity", "exit prevalent (kinematic)"),
    (r"fig_decoder_geometry", "conflict_mag distinct; chosen_cost~step2~rt_exec"),
    (r"fig_step_decomposition", "chosen_cost -> timing (collapses under rt_exec)"),
    (r"fig_motion_robustness", "survives interruptions, not timing control"),
    (r"fig_coverage_control", "HEADLINE CAVEAT: duration decodes = coverage artifact"),
    (r"fig_com_event|fig_com_presence", "null (no CoM signal)"),
    (r"fig_splice", "HEADLINE: decodes vanish without zero-padding"),
    (r"fig_rebin100", "100 ms replicates: planning->chance under splice"),
    (r"fig_planning_prevalence|fig_planning_reanalysis_reliability|fig_planning_reanalysis_sequential|fig_claim_audit",
     "solid / qualified"),
    (r"fig_exp3_threat|fig_exp2_within|fig_exp2_between", "qualified (confound)"),
    (r"fig_timing_", "kinematic (RT=transport)"),
    (r"fig_full_results_master", "summary"),
    (r"confirmation_plot", "diagnostic"),
]

CAPTION_MANUAL = {
    "RT.png": "Reaction-time distributions",
    "model_comparison_accuracy.png": "Model comparison: test accuracy",
    "model_comparison_ll.png": "Model comparison: test log-likelihood",
    "model_comparison_heatmap_accuracy.png": "Model comparison: accuracy heatmap",
    "model_comparison_heatmap_loglikelihood.png": "Model comparison: log-likelihood heatmap",
    "scaling_accuracy_cognitive.png": "Transfer scaling: CognitiveDeepONet accuracy",
    "scaling_accuracy_strategy.png": "Transfer scaling: StrategyDeepONet accuracy",
    "aggregate_forest.png": "Cross-run forest of headline neural effects",
    "confirmation_plot.png": "Per-run alignment verification (3 panels)",
    # pass-timing / 250 ms sweep (new)
    "fig_entry_250ms_sweep.png": "Entry-anchored single 250 ms window sweep (comparison labels)",
    "fig_choice_250ms_sweep.png": "Choice-anchored single 250 ms window sweep",
    "fig_entry_250ms_phase.png": "Phase composition of each entry-anchored 250 ms window",
    "fig_pass_event_detection.png": "Pass-event detection (+/-50 ms) per anchor/target",
    "fig_pass_event_transfer.png": "Pass-event temporal generalization (lag x lag)",
    "fig_pass_time_regression.png": "Per-bin pass-time decoding (B1)",
    "fig_pass_time_controls.png": "Window rt_resid decoding under controls (B2)",
    # exit targeting (tests 2-4)
    "fig_exit_prospective.png": "Exit-time decode: entry (pre-choice) vs choice anchored",
    "fig_exit_direction.png": "Exit-direction decode (goal side vs ball x)",
    "fig_exit_lag_matched.png": "Lag-to-exit-matched entry vs choice decode",
    "fig_exit_geometry.png": "Chosen-path-to-exit value/geometry (conflict_mag robust)",
    "fig_exit_selectivity.png": "Per-unit selectivity: exit-pass vs planning vs side",
    # decoder geometry / step decomposition / motion
    "fig_decoder_geometry.png": "Decoder-axis cosines among value/geometry/timing targets",
    "fig_step_decomposition.png": "Step decomposition (step1/step2/chosen-cost) + rt_exec control",
    "fig_motion_robustness.png": "Planning-cost decoding: motion interruption & timing control",
    "fig_coverage_control.png": "Coverage/censoring control: raw vs corrected decoding",
    "fig_com_event.png": "Change-of-mind event-locked population signal (null)",
    "fig_com_presence.png": "Change-of-mind presence decoding + coverage control (null)",
    "fig_splice.png": "Splice vs truncated segmentation: representation dependence",
    "fig_rebin100.png": "100 ms-bin re-run: truncated vs spliced vs coverage-controlled",
    # tier-1 neural
    "fig_neural_axis_alignment.png": "Value vs place/successor axis alignment",
    "fig_neural_choice_increment.png": "Neural choice increment",
    "fig_neural_dpca_geometry.png": "dPCA geometry vs policy axes",
    "fig_neural_entry_sliding_fine.png": "Fine entry-anchored pre-decision sliding decode",
    "fig_neural_sr_rsa.png": "Successor-representation RSA (null)",
    # reanalyses / validation
    "fig_claim_audit.png": "Claim-by-claim status board",
    "fig_level_decomposition.png": "Within-block / block / participant decomposition",
    "fig_participant_peak_location.png": "Per-participant ball-height peak location",
    "fig_planning_block_conflict_scatter.png": "Block conflict rate vs planning measures",
    "fig_planning_reanalysis_block.png": "Environment/threat block re-analysis",
    "fig_planning_reanalysis_reliability.png": "Split-half reliability of participant summaries",
    "fig_planning_reanalysis_sequential.png": "Within-block lag-1 / alternation re-analysis",
    "fig_shared_geometry_validation.png": "Shared-geometry variant validation",
    "fig_threat_bins5.png": "5-bin ball-height (threat) planning profile",
}

STALE_README = """# Stale duplicates

Older, loose PNGs written directly into `analysis/planning_outputs/` by
`planning_experiments.py`. Each duplicates (an earlier version of) a canonical
`analysis/planning_outputs/figures/fig_*.png`. Kept for provenance only; use the
`fig_*` version in `10_planning_behaviour/`.
"""

SUPERSEDED_README = """# Superseded figures

Replaced figures kept for provenance. See `analysis/superseded_figures/SUPERSEDED.md`.
`fig_planning_temporal.png` used a one-sided "excess stay" test blind to
alternation; replaced by `fig_planning_reanalysis_sequential.png`.
"""

DEPRECATED_README = """# Deprecated

Outputs the docs flag as deprecated (e.g. `exp2_shift_*`). None are currently on
disk; this folder exists so they land here if regenerated.
"""

PERRUN_README = """# Per-run diagnostics

Per-session alignment/QC plots (`confirmation_plot.png`), one per EMU run. Not
findings; kept so they do not clutter the themed sections.
"""


FALLBACK_SCRIPT = {
    "model_comparison": "analysis/model_comparison.py",
    "scaling_accuracy": "analysis/scaling_figures.py",
    "cross_dataset_outputs": "analysis/cross_dataset_figures.py",
    "planning_outputs/figures": "analysis/planning_figures.py",
    "neural_outputs/figures": "analysis/neural_figures.py",
}


def status_of(name):
    for pat, s in STATUS:
        if re.search(pat, name):
            return s
    return ""


def theme_of(path, name):
    p = path.replace("\\", "/")
    if "superseded_figures" in p:
        return "90_problematic/superseded"
    if "/planning_outputs/" in p and "/planning_outputs/figures/" not in p:
        return "90_problematic/stale_duplicates"
    if name == "confirmation_plot.png":
        return "90_problematic/per_run_diagnostics"
    if "cross_dataset_outputs" in p:
        return "20_cross_dataset"
    if "planning_outputs" in p:
        if name == "fig_full_results_master.png":
            return "00_overview"
        return "10_planning_behaviour"
    if "neural_outputs" in p:
        if re.search(r"fig_(entry_250ms|choice_250ms|entry_250ms_phase|pass_|exit_)", name):
            return "40_pass_timing"
        return "30_neural_mtl"
    if "analysis/figures/" in p:
        if name.startswith("scaling_") or name.startswith("model_comparison"):
            return "50_scaling_modeling"
        return "00_overview"
    return "00_overview"


# %% ------------------------------------------------------- build manifest
def build():
    figs = all_figures()
    code = scan_code()
    caps = scan_docs()
    gx = load_graphify()

    rows = []
    for path in figs:
        name = os.path.basename(path)
        theme = theme_of(path, name)
        script, line = (code.get(name) or [("", "")])[0]
        if not script:
            script = FALLBACK_SCRIPT.get(next(
                (k for k in FALLBACK_SCRIPT if k in path.replace("\\", "/")), ""), "")
        doc, heading = caps.get(name, ("", ""))
        g = gx.get(name, {})
        caption = CAPTION_MANUAL.get(name) or heading
        rows.append({
            "figure": name,
            "path": path,
            "theme": theme,
            "script": script,
            "line": line,
            "caption": caption,
            "status": status_of(name),
            "community": g.get("community", ""),
            "concepts": "; ".join(g.get("concepts", [])[:6]),
            "doc": doc,
        })
    rows.sort(key=lambda r: (r["theme"], r["figure"]))
    return rows


# %% ------------------------------------------------------- symlink tree
def build_tree(rows):
    if FIGROOT.exists():
        shutil.rmtree(FIGROOT)
    for theme, _label in THEMES:
        (FIGROOT / theme).mkdir(parents=True, exist_ok=True)
    # disambiguate identical basenames within a theme (e.g. per-run confirmation_plot)
    seen = {}
    for r in rows:
        key = (r["theme"], r["figure"])
        seen.setdefault(key, []).append(r)
    for (theme, name), rs in seen.items():
        for r in rs:
            link_name = name
            if len(rs) > 1:
                parent = Path(r["path"]).parent.name
                link_name = f"{parent}_{name}"
            r["link_name"] = link_name
            link = FIGROOT / theme / link_name
            target = os.path.relpath(REPO / r["path"], link.parent)
            os.symlink(target, link)

    (FIGROOT / "90_problematic" / "stale_duplicates" / "README.md").write_text(STALE_README)
    (FIGROOT / "90_problematic" / "superseded" / "README.md").write_text(SUPERSEDED_README)
    (FIGROOT / "90_problematic" / "deprecated" / "README.md").write_text(DEPRECATED_README)
    (FIGROOT / "90_problematic" / "per_run_diagnostics" / "README.md").write_text(PERRUN_README)


# %% ------------------------------------------------------- manifest csv
def write_csv(rows):
    cols = ["figure", "theme", "caption", "status", "script", "line",
            "community", "concepts", "doc", "path"]
    with open(FIGROOT / "MANIFEST.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


# %% ------------------------------------------------------- html
BADGE = {
    "robust": "#1a7f37", "robust (unit-matched)": "#1a7f37",
    "robust (control)": "#1a7f37", "robust (comparability)": "#1a7f37",
    "solid / qualified": "#1a7f37", "summary": "#57606a",
    "qualified (confound)": "#9a6700", "confounded (threat~height)": "#9a6700",
    "mixed (see rows)": "#9a6700", "exploratory": "#9a6700",
    "kinematic (RT=transport)": "#9a6700", "diagnostic": "#57606a",
    "null (entry-anchored)": "#8250df",
    "null (no non-kinematic code)": "#8250df",
    "null (pre) / consequence (post)": "#8250df",
    "exit only, small (kinematic)": "#8250df",
    "weak / negative (exploratory)": "#8250df",
    "superseded": "#cf222e", "stale duplicate": "#cf222e",
}


def badge(status):
    color = BADGE.get(status, "#57606a")
    if not status:
        return ""
    return f'<span class="badge" style="background:{color}">{html.escape(status)}</span>'


def write_html(rows):
    parts = ["""<!doctype html><html><head><meta charset="utf-8">
<title>Falldown - figure index</title><style>
body{font-family:system-ui,sans-serif;margin:24px;background:#f6f8fa;color:#1f2328}
h1{margin:0 0 4px}
h2{margin-top:32px;border-bottom:1px solid #d0d7de;padding-bottom:4px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px}
.card{background:#fff;border:1px solid #d0d7de;border-radius:8px;padding:10px}
.card img{width:100%;height:180px;object-fit:contain;background:#fff;border-bottom:1px solid #eee}
.cap{font-weight:600;font-size:13px;margin:6px 0 2px}
.meta{font-size:11px;color:#57606a;word-break:break-word}
.badge{display:inline-block;color:#fff;border-radius:10px;padding:1px 8px;font-size:11px;margin:4px 0}
.tags{font-size:11px;color:#8250df}
.sub{color:#57606a;font-size:13px}
</style></head><body>
<h1>Falldown - figure index</h1>
<p class="sub">Click a thumbnail to open the full PNG. Organised by theme; status and
graphify concept tags are shown here only (the folders on disk are tag-free).</p>
"""]
    from collections import OrderedDict
    groups = OrderedDict()
    for theme, label in THEMES:
        rs = [r for r in rows if r["theme"] == theme]
        if rs:
            groups[theme] = (label, rs)
    for theme, (label, rs) in groups.items():
        parts.append(f'<h2>{html.escape(label)} <span class="sub">({len(rs)})</span></h2><div class="grid">')
        for r in rs:
            src = f'{r["theme"]}/{r.get("link_name", r["figure"])}'
            cap = html.escape(r["caption"] or r["figure"])
            script = html.escape(os.path.basename(r["script"]) if r["script"] else "")
            parts.append(f"""<div class="card">
<a href="{src}"><img src="{src}" loading="lazy"></a>
<div class="cap">{cap}</div>
<div>{badge(r["status"])}</div>
<div class="meta">{html.escape(r["figure"])}<br>
{'script: ' + script if script else ''}
{' &middot; <a href="../' + html.escape(r["doc"]) + '">' + html.escape(r["doc"]) + '</a>' if r["doc"] else ''}</div>
<div class="tags">{html.escape(r["concepts"])}</div>
</div>""")
        parts.append("</div>")
    parts.append("</body></html>")
    (FIGROOT / "index.html").write_text("\n".join(parts))


def write_md(rows):
    lines = ["# Figure index", "",
             "Generated by `analysis/build_figure_index.py`. See `index.html` for the",
             "thumbnail contact sheet (status + graphify tags).", ""]
    for theme, label in THEMES:
        rs = [r for r in rows if r["theme"] == theme]
        if not rs:
            continue
        lines.append(f"## {label}")
        for r in rs:
            st = f" — _{r['status']}_" if r["status"] else ""
            lines.append(f"- `{r.get('link_name', r['figure'])}` — {r['caption'] or ''}{st}")
        lines.append("")
    (FIGROOT / "INDEX.md").write_text("\n".join(lines))


# %%
def main():
    rows = build()
    build_tree(rows)
    write_csv(rows)
    write_html(rows)
    write_md(rows)
    from collections import Counter
    by_theme = Counter(r["theme"] for r in rows)
    print(f"{len(rows)} figures -> {FIGROOT}")
    for theme, _ in THEMES:
        if by_theme[theme]:
            print(f"  {theme:34s} {by_theme[theme]}")
    no_script = [r["figure"] for r in rows if not r["script"]]
    print("figures without a scanned script:", no_script or "none")


if __name__ == "__main__":
    main()

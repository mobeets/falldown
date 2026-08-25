# Graphify Workflow — Adding Code and Papers

The project has **one** merged knowledge graph at `graphify-out/` (repo root). Papers live
in `papers/md/*.md` (converted from PDFs) and are ingested into this root graph — there is
**no** separate papers graph.

- `graphify query "..."` / `graphify path A B` / `graphify explain X` answer questions from `graphify-out/graph.json`.
- Raw PDFs go in `papers/queue/` and end up archived in `papers/archive/`. `papers/md/*.md` is the single source of truth for papers (`.graphifyignore` ignores `papers/*.pdf`, `papers/queue/`, `papers/archive/`).

All commands run from the **repo root**:
`C:\Users\manik\Desktop\Obsidian\General Thoughts\Z Images and Files\Hennig Lab Project\falldown`

---

## (a) New code only

```powershell
graphify update .
```

Free (AST only). Re-extracts changed code, reuses saved community labels, regenerates
`graph.json`, `GRAPH_REPORT.md`, `graph.html`. Use whenever you edit/add code or project docs.

---

## (b) New papers

```powershell
# 1. Convert (marker-pdf; ~5-15 min/paper on CPU, batch can take >1 hr)
C:\Users\manik\AppData\Local\Programs\Python\Python311\python.exe tools/pdf/preprocess_math_pdfs.py

# 2. Verify the new .md is substantial (>~1000 words) and math $$...$$ survived.
#    If it's near-empty, use the PyMuPDF fallback (see below).

# 3. Ingest — makes the paper queryable (only NEW papers cost, ~$0.02 each)
graphify extract . --backend gemini --max-concurrency 1

# 4. Refresh report + community names + html (optional but recommended)
graphify cluster-only . --backend gemini --max-concurrency 1 --missing-only
```

`--max-concurrency 1` is **mandatory** (Gemini free tier: 5 req/min). Existing
papers/docs/images are cache-hits, so only the new paper is extracted.

---

## (c) Both (new code + new papers)

Identical to (b): `graphify extract .` does **both** — AST for new code (free) and
semantic extraction for new papers — in one pass, then the same `cluster-only` line.
Don't run `graphify update .` first; it's redundant.

---

## Marker failure fallback (near-empty .md)

Some text-based PDFs (e.g. Acrobat Distiller) produce a near-empty `.md` through marker.
Recover the text layer with PyMuPDF (installed in Python 3.11):

```powershell
C:\Users\manik\AppData\Local\Programs\Python\Python311\python.exe -c "
import pymupdf, os
doc = pymupdf.open(r'papers/archive/<PAPER>.pdf')
md = ''
for page in doc:
    md += page.get_text('text') + '\n'
open(r'papers/md/<PAPER>.md', 'w', encoding='utf-8').write(md)
print('words:', len(md.split()))
"
```

Lossy (no LaTeX), but gets the content in. Then re-run step 3/4 above.

---

## What NOT to run

- ❌ `graphify update papers/md` — AST-only; does **no** semantic extraction for papers.
- ❌ `graphify extract papers/md` — recreates a duplicate papers graph (retired). Ingest against the repo root (`.`).
- ❌ `graphify extract .` without `--max-concurrency 1` — hits Gemini free-tier 429s.
- ❌ Dropping PDFs into `papers/` (not `papers/queue/`) — they're ignored and won't reach the graph.
- ❌ `/graphify .` as the routine ingest — works (agent path, free) but dispatches subagents; `graphify extract .` is the lighter routine. Pick one path per paper and stick with it.

---

## Which LLM runs the extraction?

- **Following the commands above (`graphify extract . --backend gemini ...`) → Gemini.**
  The `--backend gemini` flag sends extraction to the Gemini API using the
  `GEMINI_API_KEY` user environment variable — **not** the assistant. Coarse
  (~6 nodes/paper), ~$0.02/paper, fully autonomous.
- **The agent path (pasting the prompts below) → deepseek (the opencode assistant).**
  The assistant runs the `/graphify .` skill flow itself, using the session model
  (deepseek) to do a **dense** extraction (~30-80 nodes/paper) and then rebuilds the
  graph. Free in API dollars, but consumes session tokens. This matches the style of
  the existing 20 papers already in the graph.

Either is fine for a new paper — just don't run the other path on the same paper
afterwards (it churns the cache).

### Exact prompts to paste to the assistant

**Dense papers (md files only):**

> Run the graphify pipeline on the repo root with dense agent-based semantic extraction, the same style as the existing papers. Only newly added or changed files in papers/md should be extracted — everything else is a semantic cache hit. Then rebuild graph.json, regenerate GRAPH_REPORT.md and graph.html.

**Code only:**

> Run `graphify update .` to refresh the knowledge graph for the recent code changes.

**Both (new papers + new code):**

> Run the graphify pipeline on the repo root with dense agent-based semantic extraction for the new papers in papers/md, and refresh the AST for the new code. Only new or changed files should be extracted (cache-aware). Rebuild graph.json, regenerate GRAPH_REPORT.md and graph.html.

---

## Maintenance

- `graphify update .` after code changes keeps AST fresh at zero cost.
- Semantic cache lives at `graphify-out/cache/semantic/<prompt-fp>/`. It's keyed by
  content hash + prompt fingerprint; keep seeds and lookups under the same fingerprint
  or graphify will re-extract everything.
- See `.claude/skills/pdf-preprocess/SKILL.md` for the underlying pipeline and cache pitfalls.

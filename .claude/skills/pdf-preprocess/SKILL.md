---
name: pdf-preprocess
description: Converts research PDFs queued in papers/queue/ into math-preserving Markdown in papers/md/ (via marker-pdf), archives the source PDFs to papers/archive/, then ingests the new papers into the merged root knowledge graph (graphify-out/ at the repo root) via cache-aware extraction. Use whenever new research papers need to be added to the graph.
---

# Math-Preserving PDF Queue Processor

Use this skill when converting new research papers before updating Graphify.

## Workflow

1. **Queue the PDFs.** Place all new `.pdf` files into `papers/queue/`.

2. **Convert.** Run the queue processor with the Python 3.11 interpreter
   (marker-pdf is installed there, not in the uv venv):

   ```
   C:\Users\manik\AppData\Local\Programs\Python\Python311\python.exe tools/pdf/preprocess_math_pdfs.py
   ```

   This converts every PDF in `papers/queue/` to a `.md` file in `papers/md/`,
   then moves each source PDF to `papers/archive/`. The first run after a reboot
   spawns local llama.cpp OCR servers (CPU) — one paper can take ~5-15 min, and a
   batch can take over an hour. If it seems hung, check `~/.cache/datalab/surya/`
   for a live `llamacpp_server.log` before killing it.

3. **Verify.** Confirm the processed `.md` files landed in `papers/md/` with
   LaTeX formulas intact (`$ ... $` and `$$ ... $$`), and that the source PDFs
   moved to `papers/archive/`. Spot-check each new `.md` has real content
   (e.g. >1000 words); a near-empty file means the conversion failed (see
   fallback below).

4. **Ingest into Graphify (merged root graph).** Papers are part of the single
   root graph at the repo root (`graphify-out/`). The semantic cache is seeded,
   so this is incremental — only the new/changed papers are extracted:

   ```
   graphify extract . --backend gemini --max-concurrency 1
   ```

   `--max-concurrency 1` is required to stay under Gemini's free-tier quota
   (5 req/min). New papers cost roughly $0.02 each; unchanged files are cache
   hits (free). After it runs, the root `graphify-out/graph.json`,
   `GRAPH_REPORT.md` and `graph.html` are regenerated.

   If you want the **dense agent-based extraction** (free, ~30-80 nodes/paper,
   what the papers already in the graph use) instead of the coarse Gemini pass,
   run the `/graphify .` skill flow (or `graphify extract .` followed by
   `graphify cluster-only .` for community names). Prefer one path per paper
   and stick with it — mixing dense and coarse nodes for the same file just
   churns the cache.

## If marker fails on a PDF (near-empty .md)

Some text-based PDFs (e.g. Acrobat Distiller output) can produce a nearly empty
`.md` through marker (known issue). Fallback with PyMuPDF (already installed in
Python 3.11):

```
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

Lossy (no LaTeX), but gets the content in. Then run the ingest step again.

## Notes

- Conversion is fully local (marker-pdf + torch on CPU) — no API keys or tokens
  are consumed by the PDF→MD step. Tokens are only spent later by graphify's
  semantic extraction.
- The **papers graph is merged into the root graph** — there is no separate
  `papers/md/graphify-out/`. Do NOT run `graphify update papers/md` or
  `graphify extract papers/md` (it would recreate a duplicate papers graph).
  Ingest against the repo root (`.`).
- `papers/md/` is gitignored (generated artifacts) but is explicitly re-included
  for graphify via `.graphifyignore`, and readable by opencode/Claude via
  `.opencodeignore`. Raw PDFs in `papers/` and `papers/queue|archive/` are
  ignored — `papers/md/*.md` is the single source of truth for papers.

## Semantic-cache pitfalls (learned the hard way)

- `save_semantic_cache(..., cache_root=<corpus-root>/graphify-out)` writes to a
  NESTED `graphify-out/graphify-out/cache/...` and future runs MISS → full
  re-extraction. Always pass `root=<corpus-root>` (or omit `cache_root`), so the
  cache lands at `<corpus-root>/graphify-out/cache/semantic/<prompt-fp>/`.
- The cache is keyed by (content hash + extraction-prompt fingerprint). The
  agent/skill path uses `extraction-spec.md` (fp `pd5fd89c46bb5`); the Gemini
  CLI uses a different prompt fp. Keep seeds and lookups under the SAME fp.
- When agent extraction truncates JSON, pull the full output from
  `~/.local/share/opencode/tool-output/` and extract the JSON programmatically —
  never re-type it from context (huge token waste). Prefer `general`-purpose
  subagents (they can Write the chunk JSON directly) over `explore` (read-only).
- Small extraction groups (<=4 files) with node caps and "compact JSON,
  one-sentence rationales" instructions keep subagent responses under the
  truncation limit.

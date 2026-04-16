# SEBI PDF Reference Extraction Agent

Extract every reference to external documents (circulars, regulations, acts, gazette notifications, consultation papers, master circulars) from SEBI regulatory PDFs. Output structured JSON with verbatim evidence.

## Architecture — 6-stage pipeline

1. **Text extraction** (`src/extract.py`) — PyMuPDF per-page with zone detection (body / footnote / table / annexure).
2. **Candidate generation** (`src/candidates.py`) — regex seed hints for the LLM (SEBI circular IDs, "Circular dated", "SEBI (X) Regulations, YYYY", Acts, Gazette Notifications, Consultation Papers, etc.). Self-references filtered before the LLM sees them.
3. **LLM extraction** (`src/agent.py`) — per-page Gemini 2.5 Flash call. Receives page text + candidate hints. Returns structured JSON with verbatim `evidence_text`.
4. **Alias resolution** (`src/resolve.py` — `AliasResolver`) — "the said Regulations" → full title from earlier page. Tracks `(hereinafter referred to as "X")` definitions and last-of-type mentions.
5. **Title resolution** (`src/resolve.py` — `TitleResolver`) — optional CSV lookup for canonical SEBI doc metadata.
6. **Verification** (`src/verify.py`) — validates evidence text appears verbatim on claimed page, validates ISO date format, filters self-references, deduplicates within page.

## Hard rules — non-negotiable

- **Evidence required.** Every extraction carries `evidence_text` copied verbatim from the source page. Stage 6 drops anything whose evidence text cannot be located on its claimed page.
- **No hallucinated titles.** If unknown, `title` must be `null` — never a guess.
- **Exclude self-references.** "this circular", "the present circular", "this Master Circular", "this notification" — filtered at both candidate and verify stages.
- **Both page numbers.** Every mention carries `source_page` (0-based index) AND `display_page` (1-based).
- **No frameworks.** CLI only. No FastAPI, Streamlit, LangChain. Plain Python + PyMuPDF + google-generativeai SDK + pytest.

## Output schemas

### `mentions.json` — one entry per reference occurrence
```json
{
  "source_page": 0,
  "display_page": 1,
  "mention_text": "SEBI (LODR) Regulations, 2015",
  "evidence_text": "... verbatim snippet copied from the page ...",
  "doc_type": "circular | master_circular | regulation | act | consultation_paper | gazette_notification | other",
  "doc_id": "SEBI/HO/CFD/DIL1/CIR/P/2020/37 or null",
  "date": "YYYY-MM-DD or null",
  "title": "Full canonical title or null",
  "relation_type": "references | modifies | supersedes | issued_under",
  "confidence": "high | medium | low",
  "resolution_status": "resolved_alias | resolved_contextual | resolved_index | unresolved",
  "extraction_source": "body | footnote | table | annexure"
}
```

### `documents.json` — deduplicated canonical references
```json
{
  "canonical_title": "...",
  "doc_type": "...",
  "doc_id": "...",
  "date": "YYYY-MM-DD",
  "mentioned_on_pages": [0, 3, 5],
  "times_mentioned": 4,
  "relation_types": ["references", "supersedes"]
}
```

### `edges.json` — graph-ready export
```json
{
  "source": "<source-pdf-stem>",
  "relation": "references",
  "target": "SEBI (LODR) Regulations, 2015",
  "target_type": "regulation",
  "source_page": 0,
  "evidence": "..."
}
```

## Commands

```
pip install -r requirements.txt
python -m src.run --pdf path/to/file.pdf --out output/
python -m src.eval --gold tests/gold --predictions output/
pytest tests/ -v
python scripts/draft_gold.py --pdf path/to/file.pdf --save tests/gold/<name>.json
python scripts/compare_and_merge.py --agent-output output/documents.json --draft-gold tests/gold/<name>.json --save diff.json
```

## Environment

Set `GEMINI_API_KEY` in your environment or pass `--api-key`. Without a key, `src.run` still executes stages 1, 2, 4, 5, 6 end-to-end and writes empty mention lists — useful for dry runs and pipeline testing.

## File map

- `src/run.py` — CLI entry, orchestrates all 6 stages
- `src/extract.py` — PyMuPDF text extraction + zone detection
- `src/candidates.py` — regex patterns + self-reference filter
- `src/agent.py` — Gemini agent, prompts, JSON parsing, deduplication
- `src/resolve.py` — `AliasResolver` + `TitleResolver`
- `src/verify.py` — evidence validation + within-page dedup
- `src/eval.py` — P/R/F1, title accuracy, hallucination rate, sliced analysis
- `tests/test_core.py` — unit tests (no network / no PDF required)
- `scripts/draft_gold.py` — full-doc single-pass gold drafting (independent strategy)
- `scripts/compare_and_merge.py` — agent-vs-draft diff
- `scripts/scrape_sebi_index.py` — v2 metadata scraper stub

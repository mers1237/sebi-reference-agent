# SEBI Reference Extraction Agent

A CLI agent that extracts every reference to external documents from Securities and Exchange Board of India (SEBI) regulatory PDFs, producing graph-ready JSON with verbatim evidence.

## Architecture

```
+-----------+  +-----------+  +-----------+  +-----------+  +-----------+  +-----------+
| 1 Extract |->| 2 Regex   |->| 3 Gemini  |->| 4 Alias   |->| 5 Title   |->| 6 Verify  |
| PyMuPDF   |  |candidates |  | per-page  |  |resolve    |  |CSV lookup |  |+ dedupe   |
|+ zones    |  |+ hints    |  |JSON out   |  |("said X") |  |(optional) |  |           |
+-----------+  +-----------+  +-----------+  +-----------+  +-----------+  +-----------+
     |                                                                          |
     v                                                                          v
 pages[]                                                             mentions.json
                                                                     documents.json
                                                                     edges.json
```

The pipeline is intentionally **hybrid**. Regex seeds the LLM with likely candidates (improving recall on long PDFs, anchoring the model's attention to the right spans), but the LLM is the final judgment on whether a match is a genuine external reference and what its canonical form is. Every extraction is **verifiable** -- the agent must copy `evidence_text` verbatim from the page, and stage 6 drops anything whose evidence cannot be located on its claimed page.

## Quick start

```bash
git clone https://github.com/mers1237/sebi-reference-agent.git
cd sebi-reference-agent
pip install -r requirements.txt
cp .env.example .env                            # then paste your Gemini API key into .env
```

Get a Gemini API key at https://aistudio.google.com/apikey (free tier works). The `.env` file is gitignored -- each developer keeps their own.

### Run on a PDF

```bash
python -m src.run --pdf tests/fixtures/pdfs/circular_a_mf_borrowing_addendum.pdf --out output/demo/
```

Sample output (3 mentions extracted from a 2-page SEBI circular):

```json
[
  {
    "doc_type": "circular",
    "doc_id": "HO/(92)2026-IMD-POD-2/I/6961/2026",
    "date": "2026-03-13",
    "title": "SEBI Circular on Borrowing by Mutual Funds",
    "evidence_text": "SEBI vide Circular No. HO/(92)2026-IMD-POD-2/I/6961/2026 dated March 13, 2026...",
    "relation_type": "references",
    "source_page": 0,
    "display_page": 1
  },
  {
    "doc_type": "master_circular",
    "date": "2026-03-20",
    "title": "SEBI Master Circular for Mutual Funds",
    "relation_type": "references"
  },
  {
    "doc_type": "regulation",
    "title": "SEBI (Mutual Funds) Regulations",
    "relation_type": "issued_under"
  }
]
```

### Visualize the reference graph

```bash
python scripts/visualize.py --edges output/demo/edges.json
```

Generates a Mermaid diagram showing which documents the circular references, color-coded by type (circular, regulation, act, etc.). Paste the output into https://mermaid.live to view.

### Run tests

```bash
pytest tests/ -v    # 31 tests, no API key or PDF required
```

## Results

Evaluated on 3 real SEBI PDFs (2 circulars + 1 amendment notification) against independently-generated gold labels. Three measured iterations:

| Version | Precision | Recall | **F1** | Hallucination | Key change |
|---------|-----------|--------|--------|---------------|------------|
| v1 | 83% | 42% | **56%** | 0% | Baseline agent |
| v2 | 100% | 67% | **80%** | 0% | Regex for non-SEBI regulations, dedup fix, date-grounding bug fix |
| v3 | 100% | 83% | **91%** | 0% | Prompt: SEBI Act recognition, notification handling, title/date conventions |

**F1: 56% → 91% over three iterations. 0% hallucination throughout.**

### Final (v3) per-document-type

| Type | Precision | Recall | F1 |
|------|-----------|--------|-----|
| circular | 100% | 100% | 100% |
| master_circular | 100% | 100% | 100% |
| regulation | 100% | 100% | 100% |
| act | 100% | 100% | 100% |
| notification | 100% | 100% | 100% |

### Lessons from iterating

1. **Eval bugs mimic agent bugs.** An early reading of baseline showed F1=40%. Investigation found eval-side normalization issues (case, plural/singular, "SEBI" prefix stripping) were causing false negatives against gold. Fixing the eval moved the baseline measurement without changing the agent. A broken eval gives you the wrong optimization signals — always audit the eval before tuning the agent.

2. **Aggregate metrics can hide per-PDF failures.** The initial aggregate deduplicated gold docs across PDFs by key, so "SEBI Act 1992" missed in 3 PDFs only counted as 1 false negative instead of 3. Summing per-PDF TP/FP/FN gives an honest view.

3. **LLMs need explicit guidance on ambiguous cases.** The agent consistently skipped "Securities and Exchange Board of India Act, 1992" because SEBI is the issuing agency — the model treated it as a self-reference rather than a citation to the Act. An explicit prompt clause (*"'Securities and Exchange Board of India Act, 1992' IS an external reference even though SEBI is named"*) fixed it.

4. **Date hallucinations have a signature.** When the LLM sees "Regulations, 2018" and no full date, it often fabricates "2018-01-01". A cheap verifier (require the extracted year to co-occur with a month name or numeric date pattern in the evidence) catches these without extra model calls. Subtle bug: the first version matched "March" from a *different* date in the same evidence; fixed by pairing month with year.

### Remaining gaps

- **Per-PDF uniformity.** Amendment PDF hits F1=100%; circular_a and circular_b still miss SEBI Act 1992 from their authority clauses despite the prompt improvement (amendment PDF picks it up correctly). Suggests the LLM makes this inference unreliably — a few-shot example would likely close the gap.
- **Test set is small** (3 PDFs, 12 gold documents). Larger corpus would tighten confidence intervals on per-type F1.

## Eval methodology

`src/eval.py` computes document-level P/R/F1, page accuracy, title accuracy, hallucination rate, unresolved rate, and per-doc_type sliced analysis.

### Independent dual-strategy evaluation

Gold annotations live in `tests/gold/*.json`. To avoid circular evaluation (writing gold with the same prompt the agent uses), gold drafts are generated by a **different prompting strategy**:

1. **Agent** (per-page): Processes one page at a time with regex candidate hints. Fast, scalable, but misses cross-page context.
2. **Gold drafter** (full-document): `scripts/draft_gold.py` sends the entire document in a single pass with a stricter system prompt emphasizing cross-page coreference resolution.

Agreement between two independent strategies is a stronger signal than agreement with the agent's own output.

### Automated eval cycle

```bash
# Run agent + generate gold + evaluate + report (one command):
python scripts/run_eval_cycle.py --version baseline --delay 13

# Re-evaluate existing predictions (no API calls):
python scripts/run_eval_cycle.py --version baseline --eval-only

# Compare versions side-by-side:
python scripts/run_eval_cycle.py --compare baseline v1 v2
```

## Output schema

See [`CLAUDE.md`](CLAUDE.md) for the full field reference. Short version:

- **`mentions.json`** -- one row per occurrence (a document cited three times = three rows). Carries `source_page`, `display_page`, `mention_text`, `evidence_text`, `doc_type`, `doc_id`, `date`, `title`, `relation_type`, `confidence`, `resolution_status`, `extraction_source`.
- **`documents.json`** -- deduplicated canonical references with `mentioned_on_pages`, `times_mentioned`, `relation_types`.
- **`edges.json`** -- flat graph export: `(source, relation, target, target_type, source_page, evidence)`.

## Scaling to a full SEBI knowledge graph

To extend from single-PDF extraction to a complete regulatory knowledge graph:

1. **Batch ingestion**: Scrape all circulars from sebi.gov.in (`scripts/scrape_sebi_index.py` stub exists), run the pipeline on each, merge all `edges.json` into a unified graph.
2. **Graph database**: Load edges into Neo4j or similar. Enable traversal queries like "Which circulars are affected if SEBI amends LODR Regulations?"
3. **Incremental updates**: When new circulars are published, run the agent on just the new PDF and merge its edges into the existing graph.
4. **Specialist model**: The gold annotations are training data. Fine-tune a smaller model (e.g. Gemma) for faster, cheaper, on-premise extraction -- no external API dependency for data-sensitive compliance teams.

## Limitations

- **OCR not supported.** Scanned PDFs without a text layer yield empty pages.
- **Zone detection is heuristic.** Footnote / table detection uses whitespace patterns, not visual layout.
- **Alias resolution is local.** "The said Regulations" resolves from the last mention of type `regulation` -- cannot follow long-distance coreference across annexures.
- **Title normalization is minimal.** `SEBI (LODR) Regulations, 2015` and `SEBI Listing Obligations and Disclosure Requirements Regulations, 2015` may not deduplicate without the SEBI metadata CSV.
- **Per-page LLM extraction misses cross-page context.** A reference defined on page 3 and cited by shorthand on page 40 resolves only if the first-mention title was captured earlier.
- **Free-tier rate limits.** Gemini 2.5 Flash free tier allows ~20 requests/day. Use `--delay 13` to stay within RPM limits, or enable billing for unrestricted access.

## v2 roadmap

- OCR fallback (Tesseract) for scanned PDFs
- Real `scrape_sebi_index.py` populating canonical IDs/titles from sebi.gov.in
- Visual zone detection using PyMuPDF block coordinates
- Title normalizer (expand abbreviations, unify punctuation) before dedup
- Cross-page coreference memory for long annexures
- Migrate from deprecated `google-generativeai` to `google-genai` SDK
- Confidence calibration from held-out gold

## License / use

This tool is for research and compliance automation on public SEBI PDFs. SEBI's site and documents are subject to their own terms of use.

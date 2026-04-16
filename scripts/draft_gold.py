"""Draft-gold generator: full-document single-pass prompting for eval independence.

Uses a DIFFERENT prompting strategy from src/agent.py (per-page). Full-document
input lets the model reason about cross-page coreference ("the said Regulations")
and pick canonical titles using the whole context. Agreement between this strategy
and the page-based agent carries independent signal — that's what keeps the eval
honest.

Output matches the tests/gold/*.json schema consumed by src/eval.py.

Usage:
    python scripts/draft_gold.py --pdf path/to/file.pdf --save tests/gold/<name>.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Allow running as `python scripts/draft_gold.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.extract import extract_pages  # noqa: E402


DRAFT_SYSTEM_PROMPT = """You are an annotator building a GOLD reference list from a SEBI PDF.

You will receive the ENTIRE document at once. Extract every reference to an external
document (circular, regulation, act, gazette notification, consultation paper, master
circular, etc.).

DIFFERENCES FROM PER-PAGE EXTRACTION:
- Read the whole document before deciding what counts.
- Resolve coreferences ("the said Regulations", "the aforementioned circular") using
  the full context — do NOT emit unresolved shorthands.
- Pair every mention with its source_page (0-based, taken from the `--- PAGE N ---`
  markers in the input).

HARD RULES:
1. Every mention MUST include verbatim `evidence_text` from the document.
2. Never guess a title — use null if unknown.
3. Exclude self-references (this circular, the present circular, etc.).
4. `date` must be YYYY-MM-DD or null.
5. Return JSON only. No prose, no markdown fences.

Output schema:
{
  "documents": [
    {"canonical_title": "...", "doc_type": "...", "doc_id": "... or null", "date": "YYYY-MM-DD or null"}
  ],
  "mentions": [
    {"source_page": 0, "display_page": 1, "mention_text": "...", "evidence_text": "...",
     "title": "...", "doc_type": "...", "doc_id": "...", "date": "...",
     "relation_type": "references", "confidence": "high"}
  ]
}
"""


def _build_document_text(pages) -> str:
    parts = []
    for p in pages:
        parts.append(f"--- PAGE {p.index} ---")
        parts.append(p.text)
    return '\n'.join(parts)


def main():
    ap = argparse.ArgumentParser(description='Draft gold-label file via full-document prompt')
    ap.add_argument('--pdf', required=True)
    ap.add_argument('--save', required=True)
    ap.add_argument('--api-key', default=None)
    ap.add_argument('--model', default='gemini-2.5-flash')
    args = ap.parse_args()

    pages = extract_pages(args.pdf)
    doc_text = _build_document_text(pages)

    api_key = args.api_key or os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("[draft_gold] no API key — writing empty draft skeleton", file=sys.stderr)
        draft = {'documents': [], 'mentions': [], '_note': 'no api key at draft time'}
    else:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name=args.model,
                system_instruction=DRAFT_SYSTEM_PROMPT,
            )
            resp = model.generate_content(
                doc_text,
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.0,
                },
            )
            raw = resp.text if hasattr(resp, 'text') else ''
            try:
                draft = json.loads(raw)
            except json.JSONDecodeError:
                draft = {'documents': [], 'mentions': [], '_raw': raw}
        except Exception as e:
            print(f"[draft_gold] error: {e}", file=sys.stderr)
            draft = {'documents': [], 'mentions': [], '_error': str(e)}

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(draft, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"[draft_gold] wrote {save_path}")


if __name__ == '__main__':
    main()

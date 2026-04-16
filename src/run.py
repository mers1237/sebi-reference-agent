"""CLI entry point. Orchestrates the 6-stage SEBI reference extraction pipeline.

Usage:
    python -m src.run --pdf path/to/file.pdf --out output/ [--delay 1.0]
                      [--sebi-index metadata.csv] [--skip-verify] [--api-key KEY]
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .agent import GeminiAgent, dedupe_mentions
from .candidates import filter_self_references, find_candidates
from .extract import extract_pages
from .resolve import AliasResolver, TitleResolver
from .verify import verify_mentions


def build_documents(mentions: List[Dict]) -> List[Dict]:
    """Collapse mention list into deduplicated canonical documents."""
    docs: Dict[str, Dict] = {}
    for m in mentions:
        doc_id = (m.get('doc_id') or '').strip()
        title = (m.get('title') or '').strip()
        dtype = m.get('doc_type') or 'other'
        if doc_id:
            key = f"id:{doc_id.lower()}"
        elif title:
            key = f"t:{dtype}:{title.lower()}"
        else:
            continue
        entry = docs.setdefault(key, {
            'canonical_title': title or None,
            'doc_type': dtype,
            'doc_id': doc_id or None,
            'date': m.get('date'),
            'mentioned_on_pages': [],
            'times_mentioned': 0,
            'relation_types': [],
        })
        if title and not entry['canonical_title']:
            entry['canonical_title'] = title
        if m.get('date') and not entry['date']:
            entry['date'] = m['date']
        sp = m.get('source_page')
        if isinstance(sp, int) and sp not in entry['mentioned_on_pages']:
            entry['mentioned_on_pages'].append(sp)
        entry['times_mentioned'] += 1
        rt = m.get('relation_type')
        if rt and rt not in entry['relation_types']:
            entry['relation_types'].append(rt)
    for entry in docs.values():
        entry['mentioned_on_pages'] = sorted(entry['mentioned_on_pages'])
    return list(docs.values())


def build_edges(mentions: List[Dict], source_name: str) -> List[Dict]:
    """Flat graph export: (source -> target) per mention."""
    edges: List[Dict] = []
    for m in mentions:
        target = m.get('title') or m.get('doc_id') or m.get('mention_text')
        if not target:
            continue
        edges.append({
            'source': source_name,
            'relation': m.get('relation_type') or 'references',
            'target': target,
            'target_type': m.get('doc_type') or 'other',
            'source_page': m.get('source_page'),
            'evidence': m.get('evidence_text'),
        })
    return edges


def _classify_zone(evidence: str, page) -> str:
    ev = (evidence or '').lower()
    if ev and ev in (page.footnote_text or '').lower():
        return 'footnote'
    if ev and ev in (page.table_text or '').lower():
        return 'table'
    if page.is_annexure:
        return 'annexure'
    return 'body'


def main():
    ap = argparse.ArgumentParser(description='SEBI PDF reference extraction pipeline')
    ap.add_argument('--pdf', required=True, help='Path to input PDF')
    ap.add_argument('--out', required=True, help='Output directory')
    ap.add_argument('--delay', type=float, default=1.0, help='Delay between Gemini calls (seconds)')
    ap.add_argument('--sebi-index', default=None, help='Optional CSV of SEBI doc metadata (doc_id,title,date,doc_type)')
    ap.add_argument('--skip-verify', action='store_true', help='Skip stage 6 evidence verification')
    ap.add_argument('--api-key', default=None, help='Gemini API key (defaults to $GEMINI_API_KEY)')
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    api_key = args.api_key or os.environ.get('GEMINI_API_KEY')

    # --- Stage 1: extract pages ---
    pages = extract_pages(str(pdf_path))
    pages_text = [p.text for p in pages]
    print(f"[run] stage 1: extracted {len(pages)} pages")

    # --- Stages 2 + 3: candidates + per-page LLM extraction ---
    agent = GeminiAgent(api_key=api_key, delay=args.delay)
    alias_resolver = AliasResolver()
    all_mentions: List[Dict] = []

    for page in pages:
        alias_resolver.register_page_text(page.text)
        cands = filter_self_references(find_candidates(page.text))
        raw_mentions = agent.extract_page(page.text, cands, page.index)
        for m in raw_mentions:
            if not isinstance(m, dict):
                continue
            m.setdefault('source_page', page.index)
            m.setdefault('display_page', page.display_page)
            m['extraction_source'] = _classify_zone(m.get('evidence_text'), page)
            all_mentions.append(m)
    all_mentions = dedupe_mentions(all_mentions)
    print(f"[run] stages 2-3: collected {len(all_mentions)} raw mentions")

    # --- Stage 4: alias resolution ---
    for m in all_mentions:
        alias_resolver.register_mention(m)
    all_mentions = [alias_resolver.resolve(m) for m in all_mentions]

    # --- Stage 5: title resolution from optional SEBI metadata CSV ---
    title_resolver = TitleResolver.from_csv(args.sebi_index)
    all_mentions = [title_resolver.resolve(m) for m in all_mentions]

    # --- Stage 6: verification ---
    if not args.skip_verify:
        before = len(all_mentions)
        all_mentions = verify_mentions(all_mentions, pages_text, strict_evidence=True)
        print(f"[run] stage 6: verified {len(all_mentions)}/{before} mentions")

    documents = build_documents(all_mentions)
    edges = build_edges(all_mentions, source_name=pdf_path.stem)

    (out_dir / 'mentions.json').write_text(
        json.dumps(all_mentions, indent=2, ensure_ascii=False), encoding='utf-8')
    (out_dir / 'documents.json').write_text(
        json.dumps(documents, indent=2, ensure_ascii=False), encoding='utf-8')
    (out_dir / 'edges.json').write_text(
        json.dumps(edges, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f"[run] done: {len(all_mentions)} mentions -> {len(documents)} unique documents -> {out_dir}")


if __name__ == '__main__':
    main()

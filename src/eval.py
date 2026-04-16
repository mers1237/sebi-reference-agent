"""Eval harness: document-level P/R/F1, page accuracy, title accuracy,
hallucination rate, unresolved rate, and per-doc_type sliced analysis.
"""
import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple


def _norm(s) -> str:
    return re.sub(r'[^a-z0-9]+', ' ', (s or '').lower()).strip()


def title_match(pred: str, gold: str, fuzzy_threshold: float = 0.85) -> Tuple[bool, bool]:
    """Return (strict_equality, fuzzy_match)."""
    p, g = _norm(pred), _norm(gold)
    if not p or not g:
        return (False, False)
    strict = p == g
    ratio = SequenceMatcher(None, p, g).ratio()
    return (strict, ratio >= fuzzy_threshold)


def _norm_doc_type(dt: str) -> str:
    """Normalize doc_type to snake_case for consistent matching."""
    return re.sub(r'[^a-z0-9]+', '_', (dt or '').lower()).strip('_')


def doc_key(doc: Dict) -> str:
    """Canonical key for set-matching documents.

    Prefers `doc_id` when present; falls back to (doc_type, normalized title).
    """
    did = (doc.get('doc_id') or '').lower().strip()
    if did:
        return f"id:{did}"
    title = _norm(doc.get('canonical_title') or doc.get('title') or '')
    dtype = _norm_doc_type(doc.get('doc_type') or '')
    return f"t:{dtype}:{title}"


def match_documents(pred_docs: List[Dict], gold_docs: List[Dict]) -> Dict:
    pred_keys = {doc_key(d): d for d in pred_docs}
    gold_keys = {doc_key(d): d for d in gold_docs}
    tp = set(pred_keys) & set(gold_keys)
    fp = set(pred_keys) - set(gold_keys)
    fn = set(gold_keys) - set(pred_keys)
    precision = len(tp) / (len(tp) + len(fp)) if (tp or fp) else 0.0
    recall = len(tp) / (len(tp) + len(fn)) if (tp or fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        'tp': len(tp), 'fp': len(fp), 'fn': len(fn),
        'precision': precision, 'recall': recall, 'f1': f1,
        'tp_keys': sorted(tp),
        'fp_keys': sorted(fp),
        'fn_keys': sorted(fn),
    }


def page_accuracy(pred_mentions: List[Dict], gold_mentions: List[Dict]) -> float:
    """Fraction of matched-document mentions where source_page is correct."""
    gold_by_key: Dict[str, set] = {}
    for g in gold_mentions:
        gold_by_key.setdefault(doc_key(g), set()).add(g.get('source_page'))
    if not gold_by_key:
        return 0.0
    hits = total = 0
    for p in pred_mentions:
        k = doc_key(p)
        if k not in gold_by_key:
            continue
        total += 1
        if p.get('source_page') in gold_by_key[k]:
            hits += 1
    return hits / total if total else 0.0


def title_accuracy(pred_docs: List[Dict], gold_docs: List[Dict]) -> Dict:
    gold_by_key = {doc_key(g): g for g in gold_docs}
    strict = fuzzy = n = 0
    for p in pred_docs:
        g = gold_by_key.get(doc_key(p))
        if not g:
            continue
        n += 1
        s, fz = title_match(p.get('canonical_title') or '', g.get('canonical_title') or '')
        strict += int(s)
        fuzzy += int(fz)
    return {
        'n': n,
        'strict': strict / n if n else 0.0,
        'fuzzy': fuzzy / n if n else 0.0,
    }


def hallucination_rate(pred_mentions: List[Dict], pages_text: List[str]) -> float:
    """Fraction of mentions whose evidence_text cannot be located on the claimed page."""
    if not pred_mentions:
        return 0.0
    bad = 0
    for m in pred_mentions:
        ev = m.get('evidence_text') or ''
        sp = m.get('source_page')
        if not ev or not isinstance(sp, int) or not (0 <= sp < len(pages_text)):
            bad += 1
            continue
        if _norm(ev) not in _norm(pages_text[sp]):
            bad += 1
    return bad / len(pred_mentions)


def unresolved_rate(pred_mentions: List[Dict]) -> float:
    """Fraction of mentions where `title` is missing/null."""
    if not pred_mentions:
        return 0.0
    unresolved = sum(1 for m in pred_mentions if not m.get('title'))
    return unresolved / len(pred_mentions)


def slice_by_doc_type(pred_docs: List[Dict], gold_docs: List[Dict]) -> Dict[str, Dict]:
    by_type: Dict[str, Dict] = {}
    types = {d.get('doc_type') for d in pred_docs + gold_docs if d.get('doc_type')}
    for t in sorted(types):
        p_slice = [d for d in pred_docs if d.get('doc_type') == t]
        g_slice = [d for d in gold_docs if d.get('doc_type') == t]
        by_type[t] = match_documents(p_slice, g_slice)
    return by_type


def evaluate(pred_dir: Path, gold_dir: Path) -> Dict:
    """Evaluate predictions against gold labels.

    Supports two layouts:
    1. Flat: pred_dir/documents.json + gold_dir/*.json (legacy)
    2. Per-PDF: pred_dir/{stem}/documents.json matched to gold_dir/{stem}.json
    """
    report: Dict = {}
    for gold_file in sorted(gold_dir.glob('*.json')):
        gold = json.loads(gold_file.read_text(encoding='utf-8'))
        gold_docs = gold.get('documents', [])
        gold_mentions = gold.get('mentions', [])
        if not gold_docs and not gold_mentions:
            continue

        per_pdf_dir = pred_dir / gold_file.stem
        if per_pdf_dir.is_dir():
            pd = per_pdf_dir / 'documents.json'
            pm = per_pdf_dir / 'mentions.json'
        else:
            pd = pred_dir / 'documents.json'
            pm = pred_dir / 'mentions.json'

        pred_docs = json.loads(pd.read_text(encoding='utf-8')) if pd.exists() else []
        pred_mentions = json.loads(pm.read_text(encoding='utf-8')) if pm.exists() else []

        report[gold_file.stem] = {
            'document_match': match_documents(pred_docs, gold_docs),
            'page_accuracy': page_accuracy(pred_mentions, gold_mentions),
            'title_accuracy': title_accuracy(pred_docs, gold_docs),
            'unresolved_rate': unresolved_rate(pred_mentions),
            'by_doc_type': slice_by_doc_type(pred_docs, gold_docs),
        }
    return report


def evaluate_multi(pred_dirs: Dict[str, Path], gold_dir: Path, pages_by_pdf: Dict[str, List[str]] = None) -> Dict:
    """Evaluate multiple per-PDF prediction dirs against matched gold files.

    Args:
        pred_dirs: {pdf_stem: Path} mapping each PDF name to its output dir
        gold_dir: directory of gold JSON files
        pages_by_pdf: {pdf_stem: [page_texts]} for hallucination rate (optional)
    """
    per_pdf: Dict[str, Dict] = {}
    agg_pred_docs, agg_gold_docs = [], []
    agg_pred_mentions, agg_gold_mentions = [], []
    agg_pages: List[str] = []

    for gold_file in sorted(gold_dir.glob('*.json')):
        stem = gold_file.stem
        gold = json.loads(gold_file.read_text(encoding='utf-8'))
        gold_docs = gold.get('documents', [])
        gold_mentions = gold.get('mentions', [])
        if not gold_docs and not gold_mentions:
            continue

        pred_dir = pred_dirs.get(stem)
        if not pred_dir:
            continue
        pd = pred_dir / 'documents.json'
        pm = pred_dir / 'mentions.json'
        pred_docs = json.loads(pd.read_text(encoding='utf-8')) if pd.exists() else []
        pred_mentions = json.loads(pm.read_text(encoding='utf-8')) if pm.exists() else []

        per_pdf[stem] = {
            'document_match': match_documents(pred_docs, gold_docs),
            'title_accuracy': title_accuracy(pred_docs, gold_docs),
            'unresolved_rate': unresolved_rate(pred_mentions),
            'by_doc_type': slice_by_doc_type(pred_docs, gold_docs),
            'pred_mention_count': len(pred_mentions),
            'gold_doc_count': len(gold_docs),
        }
        if pages_by_pdf and stem in pages_by_pdf:
            per_pdf[stem]['hallucination_rate'] = hallucination_rate(pred_mentions, pages_by_pdf[stem])

        agg_pred_docs.extend(pred_docs)
        agg_gold_docs.extend(gold_docs)
        agg_pred_mentions.extend(pred_mentions)
        agg_gold_mentions.extend(gold_mentions)

    aggregate = {
        'document_match': match_documents(agg_pred_docs, agg_gold_docs),
        'title_accuracy': title_accuracy(agg_pred_docs, agg_gold_docs),
        'unresolved_rate': unresolved_rate(agg_pred_mentions),
        'by_doc_type': slice_by_doc_type(agg_pred_docs, agg_gold_docs),
        'total_pred_mentions': len(agg_pred_mentions),
        'total_gold_docs': len(agg_gold_docs),
    }
    return {'per_pdf': per_pdf, 'aggregate': aggregate}


def main():
    ap = argparse.ArgumentParser(description='Evaluate SEBI reference extraction output')
    ap.add_argument('--gold', required=True, help='Directory of gold JSON files')
    ap.add_argument('--predictions', required=True, help='Directory containing documents.json and mentions.json')
    args = ap.parse_args()
    report = evaluate(Path(args.predictions), Path(args.gold))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

"""Automated eval cycle: run agent, generate gold, evaluate, report.

One command per version. Compares across versions for progression tables.

Usage:
    # Run agent + eval for a version (requires GEMINI_API_KEY):
    python scripts/run_eval_cycle.py --version baseline --delay 13

    # Re-eval existing predictions without calling the API:
    python scripts/run_eval_cycle.py --version baseline --eval-only

    # Generate gold drafts (only needed once):
    python scripts/run_eval_cycle.py --gold-only --delay 13

    # Compare versions side-by-side:
    python scripts/run_eval_cycle.py --compare baseline v1 v2

    # Full cycle: run + eval + compare against prior versions:
    python scripts/run_eval_cycle.py --version v1 --delay 13 --compare baseline v1
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

PDF_DIR = Path('tests/fixtures/pdfs')
GOLD_DIR = Path('tests/gold')
OUTPUT_ROOT = Path('output')

TEST_PDFS = [
    'circular_a_mf_borrowing_addendum',
    'circular_b_eodb_stock_brokers',
    'amendment_lock_in_pledged_shares_icdr',
]


def find_test_pdfs():
    found = {}
    for stem in TEST_PDFS:
        pdf = PDF_DIR / f'{stem}.pdf'
        if pdf.exists():
            found[stem] = pdf
        else:
            print(f'[cycle] WARNING: {pdf} not found, skipping')
    return found


def run_agent(pdfs, version, delay):
    version_dir = OUTPUT_ROOT / version
    results = {}
    for stem, pdf_path in pdfs.items():
        out_dir = version_dir / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f'[cycle] running agent on {stem} -> {out_dir}')
        cmd = [
            sys.executable, '-m', 'src.run',
            '--pdf', str(pdf_path),
            '--out', str(out_dir),
            '--delay', str(delay),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        for line in (result.stdout + result.stderr).strip().split('\n'):
            if line.strip():
                print(f'  {line}')
        mentions_file = out_dir / 'mentions.json'
        if mentions_file.exists():
            mentions = json.loads(mentions_file.read_text(encoding='utf-8'))
            results[stem] = len(mentions)
        else:
            results[stem] = 0
    return results


def generate_gold(pdfs, delay):
    for stem, pdf_path in pdfs.items():
        gold_file = GOLD_DIR / f'{stem}.json'
        if gold_file.exists():
            existing = json.loads(gold_file.read_text(encoding='utf-8'))
            if existing.get('documents'):
                print(f'[cycle] gold already exists for {stem} ({len(existing["documents"])} docs), skipping')
                continue
        print(f'[cycle] drafting gold for {stem}')
        cmd = [
            sys.executable, 'scripts/draft_gold.py',
            '--pdf', str(pdf_path),
            '--save', str(gold_file),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        for line in (result.stdout + result.stderr).strip().split('\n'):
            if line.strip():
                print(f'  {line}')


def run_eval(version):
    from src.eval import evaluate_multi
    from src.extract import extract_pages

    version_dir = OUTPUT_ROOT / version
    if not version_dir.exists():
        print(f'[cycle] ERROR: {version_dir} does not exist')
        return None

    pred_dirs = {}
    pages_by_pdf = {}
    for stem in TEST_PDFS:
        pred_dir = version_dir / stem
        if pred_dir.exists() and (pred_dir / 'mentions.json').exists():
            mentions = json.loads((pred_dir / 'mentions.json').read_text(encoding='utf-8'))
            if mentions:
                pred_dirs[stem] = pred_dir
        pdf_path = PDF_DIR / f'{stem}.pdf'
        if pdf_path.exists():
            pages = extract_pages(str(pdf_path))
            pages_by_pdf[stem] = [p.text for p in pages]

    if not pred_dirs:
        print(f'[cycle] WARNING: no valid predictions found in {version_dir}')
        return None

    report = evaluate_multi(pred_dirs, GOLD_DIR, pages_by_pdf)
    report_path = version_dir / 'eval_report.json'
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[cycle] eval report saved to {report_path}')
    return report


def format_report(version, report):
    if not report:
        return f'## {version}\n\nNo results available.\n'

    lines = [f'## {version}\n']
    agg = report['aggregate']
    dm = agg['document_match']

    lines.append(f'**Aggregate**: P={dm["precision"]:.0%}  R={dm["recall"]:.0%}  F1={dm["f1"]:.0%}')
    lines.append(f'  TP={dm["tp"]}  FP={dm["fp"]}  FN={dm["fn"]}')
    ta = agg['title_accuracy']
    lines.append(f'  Title accuracy: strict={ta["strict"]:.0%}  fuzzy={ta["fuzzy"]:.0%}')
    lines.append(f'  Unresolved rate: {agg["unresolved_rate"]:.0%}')
    lines.append('')

    if dm.get('fp_keys'):
        lines.append('**False positives** (agent found, gold didn\'t):')
        for k in dm['fp_keys']:
            lines.append(f'  - `{k}`')
        lines.append('')

    if dm.get('fn_keys'):
        lines.append('**False negatives** (gold has, agent missed):')
        for k in dm['fn_keys']:
            lines.append(f'  - `{k}`')
        lines.append('')

    if agg.get('by_doc_type'):
        lines.append('**Per doc_type**:')
        lines.append('| Type | P | R | F1 | TP | FP | FN |')
        lines.append('|------|---|---|----|----|----|----|')
        for dtype, dm_slice in sorted(agg['by_doc_type'].items()):
            lines.append(
                f'| {dtype} | {dm_slice["precision"]:.0%} | {dm_slice["recall"]:.0%} | '
                f'{dm_slice["f1"]:.0%} | {dm_slice["tp"]} | {dm_slice["fp"]} | {dm_slice["fn"]} |'
            )
        lines.append('')

    for stem, pdf_report in report.get('per_pdf', {}).items():
        dm_pdf = pdf_report['document_match']
        lines.append(f'### {stem}')
        lines.append(f'P={dm_pdf["precision"]:.0%}  R={dm_pdf["recall"]:.0%}  F1={dm_pdf["f1"]:.0%}')
        lines.append(f'  Mentions: {pdf_report["pred_mention_count"]}  Gold docs: {pdf_report["gold_doc_count"]}')
        if 'hallucination_rate' in pdf_report:
            lines.append(f'  Hallucination rate: {pdf_report["hallucination_rate"]:.0%}')
        lines.append('')

    return '\n'.join(lines)


def format_progression(versions):
    """Load eval reports for multiple versions and print a comparison table."""
    reports = {}
    for v in versions:
        report_path = OUTPUT_ROOT / v / 'eval_report.json'
        if report_path.exists():
            reports[v] = json.loads(report_path.read_text(encoding='utf-8'))
        else:
            print(f'[cycle] WARNING: no eval report for {v} at {report_path}')

    if not reports:
        return 'No eval reports found.'

    lines = ['## Progression: ' + ' -> '.join(versions), '']

    header = '| Metric |' + '|'.join(f' {v} ' for v in versions) + '|'
    sep = '|--------|' + '|'.join('------' for _ in versions) + '|'
    lines.extend([header, sep])

    metrics = [
        ('Precision', lambda r: r['aggregate']['document_match']['precision']),
        ('Recall', lambda r: r['aggregate']['document_match']['recall']),
        ('F1', lambda r: r['aggregate']['document_match']['f1']),
        ('Title (fuzzy)', lambda r: r['aggregate']['title_accuracy']['fuzzy']),
        ('Unresolved rate', lambda r: r['aggregate']['unresolved_rate']),
    ]

    all_types = set()
    for r in reports.values():
        all_types.update(r['aggregate'].get('by_doc_type', {}).keys())

    for dtype in sorted(all_types):
        metrics.append((
            f'{dtype} recall',
            lambda r, dt=dtype: r['aggregate'].get('by_doc_type', {}).get(dt, {}).get('recall', 0.0),
        ))

    for label, getter in metrics:
        vals = []
        for v in versions:
            if v in reports:
                try:
                    val = getter(reports[v])
                    vals.append(f'{val:.0%}')
                except (KeyError, TypeError):
                    vals.append('—')
            else:
                vals.append('—')
        lines.append(f'| {label} |' + '|'.join(f' {val} ' for val in vals) + '|')

    lines.append('')

    for v in versions:
        if v in reports:
            dm = reports[v]['aggregate']['document_match']
            if dm.get('fn_keys'):
                lines.append(f'**{v} false negatives**: {", ".join(f"`{k}`" for k in dm["fn_keys"])}')
            if dm.get('fp_keys'):
                lines.append(f'**{v} false positives**: {", ".join(f"`{k}`" for k in dm["fp_keys"])}')

    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description='Automated eval cycle')
    ap.add_argument('--version', help='Version name (baseline, v1, v2, ...)')
    ap.add_argument('--delay', type=float, default=13.0, help='Delay between Gemini calls (default 13s for free tier)')
    ap.add_argument('--eval-only', action='store_true', help='Re-eval existing predictions, no API calls')
    ap.add_argument('--gold-only', action='store_true', help='Only generate gold drafts')
    ap.add_argument('--compare', nargs='+', metavar='VERSION', help='Compare versions side-by-side')
    ap.add_argument('--skip-gold', action='store_true', help='Skip gold generation even if missing')
    args = ap.parse_args()

    pdfs = find_test_pdfs()
    if not pdfs:
        print('[cycle] ERROR: no test PDFs found')
        sys.exit(1)

    if args.gold_only:
        generate_gold(pdfs, args.delay)
        return

    if args.compare:
        print(format_progression(args.compare))
        return

    if not args.version:
        ap.error('--version is required unless using --gold-only or --compare')

    if not args.eval_only:
        if not args.skip_gold:
            generate_gold(pdfs, args.delay)
        run_agent(pdfs, args.version, args.delay)

    report = run_eval(args.version)
    if report:
        print()
        print(format_report(args.version, report))

        report_md_path = OUTPUT_ROOT / args.version / 'eval_report.md'
        report_md_path.write_text(format_report(args.version, report), encoding='utf-8')
        print(f'[cycle] markdown report saved to {report_md_path}')


if __name__ == '__main__':
    main()

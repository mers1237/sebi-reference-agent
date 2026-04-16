"""Generate a Mermaid graph diagram from edges.json.

Usage:
    python scripts/visualize.py --edges output/baseline/circular_a/edges.json
    python scripts/visualize.py --edges-dir output/baseline/  # all PDFs
"""
import argparse
import json
import re
from pathlib import Path


def _sanitize(s: str) -> str:
    """Make a string safe for Mermaid node IDs."""
    return re.sub(r'[^a-zA-Z0-9]', '_', s)[:60]


def _label(s: str, max_len: int = 45) -> str:
    """Truncate for readable labels."""
    s = s.replace('"', "'")
    return s[:max_len] + '...' if len(s) > max_len else s


_RELATION_STYLES = {
    'references': '-->',
    'modifies': '-. modifies .->',
    'supersedes': '== supersedes ==>',
    'issued_under': '-. issued under .->',
}

_TYPE_SHAPES = {
    'circular': ('([', '])', ':::circular'),
    'master_circular': ('([', '])', ':::master'),
    'regulation': ('[[', ']]', ':::regulation'),
    'act': ('{{', '}}', ':::act'),
    'gazette_notification': ('>', ']', ':::notification'),
    'other': ('[', ']', ''),
}


def edges_to_mermaid(all_edges: list, title: str = 'SEBI Reference Graph') -> str:
    lines = [
        f'---',
        f'title: {title}',
        f'---',
        'graph LR',
        '    classDef circular fill:#e1f5fe,stroke:#0288d1',
        '    classDef master fill:#f3e5f5,stroke:#7b1fa2',
        '    classDef regulation fill:#e8f5e9,stroke:#388e3c',
        '    classDef act fill:#fff3e0,stroke:#f57c00',
        '    classDef notification fill:#fce4ec,stroke:#c62828',
        '    classDef source fill:#f5f5f5,stroke:#616161,stroke-width:2px',
    ]

    sources = set()
    targets = {}

    for e in all_edges:
        src = e.get('source', 'unknown')
        tgt = e.get('target', 'unknown')
        tgt_type = e.get('target_type', 'other')
        relation = e.get('relation', 'references')

        sources.add(src)
        if tgt not in targets:
            targets[tgt] = tgt_type

        src_id = _sanitize(src)
        tgt_id = _sanitize(tgt)
        arrow = _RELATION_STYLES.get(relation, '-->')

        lines.append(f'    {src_id} {arrow} {tgt_id}')

    for src in sources:
        sid = _sanitize(src)
        lines.append(f'    {sid}["{_label(src)}"]:::source')

    for tgt, ttype in targets.items():
        tid = _sanitize(tgt)
        open_b, close_b, cls = _TYPE_SHAPES.get(ttype, _TYPE_SHAPES['other'])
        lines.append(f'    {tid}{open_b}"{_label(tgt)}"{close_b}{cls}')

    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description='Generate Mermaid graph from edges.json')
    ap.add_argument('--edges', help='Path to a single edges.json')
    ap.add_argument('--edges-dir', help='Directory containing per-PDF subdirs with edges.json')
    ap.add_argument('--save', help='Save Mermaid markdown to file (default: stdout)')
    args = ap.parse_args()

    all_edges = []
    if args.edges:
        all_edges = json.loads(Path(args.edges).read_text(encoding='utf-8'))
    elif args.edges_dir:
        base = Path(args.edges_dir)
        for ef in sorted(base.rglob('edges.json')):
            all_edges.extend(json.loads(ef.read_text(encoding='utf-8')))
    else:
        ap.error('Provide --edges or --edges-dir')

    mermaid = edges_to_mermaid(all_edges)
    output = f'```mermaid\n{mermaid}\n```\n'

    if args.save:
        Path(args.save).write_text(output, encoding='utf-8')
        print(f'[visualize] saved to {args.save}')
    else:
        print(output)


if __name__ == '__main__':
    main()

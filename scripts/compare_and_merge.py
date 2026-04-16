"""Compare agent output against a draft-gold file.

Auto-accepts documents both sides agree on; surfaces only disagreements for a
human reviewer. Keyed by src.eval.doc_key so matching is consistent with the
eval harness.

Usage:
    python scripts/compare_and_merge.py \\
        --agent-output output/documents.json \\
        --draft-gold tests/gold/<name>.json \\
        --save disagreements.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval import doc_key  # noqa: E402


def compare(agent_docs, draft_docs):
    agent_keys = {doc_key(d): d for d in agent_docs}
    draft_keys = {doc_key(d): d for d in draft_docs}
    agree = sorted(set(agent_keys) & set(draft_keys))
    only_agent = sorted(set(agent_keys) - set(draft_keys))
    only_draft = sorted(set(draft_keys) - set(agent_keys))
    return {
        'agreements': [agent_keys[k] for k in agree],
        'only_in_agent': [agent_keys[k] for k in only_agent],
        'only_in_draft': [draft_keys[k] for k in only_draft],
        'stats': {
            'agreements': len(agree),
            'only_in_agent': len(only_agent),
            'only_in_draft': len(only_draft),
        },
    }


def main():
    ap = argparse.ArgumentParser(description='Diff agent output vs draft gold')
    ap.add_argument('--agent-output', required=True, help='documents.json from src.run')
    ap.add_argument('--draft-gold', required=True, help='JSON from scripts/draft_gold.py')
    ap.add_argument('--save', required=True)
    args = ap.parse_args()

    agent = json.loads(Path(args.agent_output).read_text(encoding='utf-8'))
    draft_raw = json.loads(Path(args.draft_gold).read_text(encoding='utf-8'))
    draft_docs = draft_raw.get('documents', []) if isinstance(draft_raw, dict) else []

    diff = compare(agent, draft_docs)
    Path(args.save).write_text(json.dumps(diff, indent=2, ensure_ascii=False), encoding='utf-8')
    s = diff['stats']
    print(
        f"[compare] agreements={s['agreements']} "
        f"only_agent={s['only_in_agent']} "
        f"only_draft={s['only_in_draft']} -> {args.save}"
    )


if __name__ == '__main__':
    main()

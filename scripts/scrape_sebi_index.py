"""Placeholder: scrape SEBI index pages for canonical doc metadata.

Intended v2 behavior:
- Walk sebi.gov.in listing pages for circulars, master circulars, regulations,
  notifications, and consultation papers.
- Extract (doc_id, title, date, doc_type) tuples.
- Write CSV consumable by src.resolve.TitleResolver.

A real implementation must respect robots.txt, throttle requests, and handle
the site's navigation structure (which changes periodically). This file is a
stub so that the pipeline's --sebi-index CSV contract is documented and can
be pointed at a hand-curated CSV in the meantime.

Usage (stub):
    python scripts/scrape_sebi_index.py --out sebi_index.csv
"""
import argparse
import csv
import sys


def main():
    ap = argparse.ArgumentParser(description='SEBI metadata scraper (v2 stub)')
    ap.add_argument('--out', required=True, help='Output CSV path')
    ap.add_argument('--base-url', default='https://www.sebi.gov.in/')
    args = ap.parse_args()

    print(
        "[scrape_sebi_index] stub — not implemented yet. "
        "Writing empty CSV with expected header so downstream code has a valid file.",
        file=sys.stderr,
    )
    with open(args.out, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['doc_id', 'title', 'date', 'doc_type'])
    print(f"[scrape_sebi_index] wrote header-only CSV to {args.out}")


if __name__ == '__main__':
    main()

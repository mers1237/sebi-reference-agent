"""Alias and title resolution (stages 4 and 5)."""
import csv
import re
from dataclasses import dataclass, field
from typing import Dict, Optional


_HEREINAFTER_RE = re.compile(
    r'\(\s*hereinafter\s+(?:referred\s+to\s+as|called)\s+["\']?([^)"\']{3,80})["\']?\s*\)',
    re.IGNORECASE,
)


@dataclass
class AliasResolver:
    """Tracks aliases and last-of-type mentions for later shorthand resolution.

    Resolution sources, in priority order:
      1. Explicit `(hereinafter referred to as "X")` definitions captured from
         page text via `register_page_text`.
      2. "The said Regulations / said Circular / said Act" → last registered
         mention of that type.
    """
    aliases: Dict[str, str] = field(default_factory=dict)
    first_mentions: Dict[str, Dict] = field(default_factory=dict)
    last_by_type: Dict[str, Dict] = field(default_factory=dict)

    def register_page_text(self, page_text: str) -> None:
        """Scan a page for `hereinafter referred to as "X"` definitions and
        store them as aliases pointing at the preceding phrase (a rough title
        guess from the line immediately before the parenthetical)."""
        for m in _HEREINAFTER_RE.finditer(page_text or ''):
            alias = m.group(1).strip().lower()
            if not alias:
                continue
            prefix = page_text[max(0, m.start() - 200):m.start()]
            title_guess = prefix.strip().split('\n')[-1].strip(' ,.;:')
            if title_guess:
                self.aliases[alias] = title_guess

    def register_mention(self, mention: Dict) -> None:
        title = mention.get('title')
        doc_type = mention.get('doc_type')
        if title:
            key = title.lower().strip()
            if key and key not in self.first_mentions:
                self.first_mentions[key] = mention
        if doc_type and title:
            self.last_by_type[doc_type] = mention

    def resolve(self, mention: Dict) -> Dict:
        """If a mention has no title but uses a shorthand, fill it in from history."""
        if mention.get('title'):
            return mention
        text = (mention.get('mention_text') or '').lower()

        for alias, title in self.aliases.items():
            if alias and alias in text:
                return {**mention, 'title': title, 'resolution_status': 'resolved_alias'}

        if 'said regulation' in text and 'regulation' in self.last_by_type:
            prev = self.last_by_type['regulation']
            return {**mention, 'title': prev.get('title'), 'resolution_status': 'resolved_contextual'}
        if 'said circular' in text and 'circular' in self.last_by_type:
            prev = self.last_by_type['circular']
            return {**mention, 'title': prev.get('title'), 'resolution_status': 'resolved_contextual'}
        if 'said act' in text and 'act' in self.last_by_type:
            prev = self.last_by_type['act']
            return {**mention, 'title': prev.get('title'), 'resolution_status': 'resolved_contextual'}
        if 'said notification' in text and 'gazette_notification' in self.last_by_type:
            prev = self.last_by_type['gazette_notification']
            return {**mention, 'title': prev.get('title'), 'resolution_status': 'resolved_contextual'}

        return {**mention, 'resolution_status': mention.get('resolution_status') or 'unresolved'}


@dataclass
class TitleResolver:
    """Optional CSV-backed lookup for canonical SEBI doc metadata.

    Expected CSV columns: doc_id, title, date, doc_type. If the CSV is missing
    or cannot be read, resolution is a no-op.
    """
    index: Dict[str, Dict] = field(default_factory=dict)

    @classmethod
    def from_csv(cls, csv_path: Optional[str]) -> 'TitleResolver':
        idx: Dict[str, Dict] = {}
        if not csv_path:
            return cls(index=idx)
        try:
            with open(csv_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    doc_id = (row.get('doc_id') or row.get('id') or '').strip()
                    if not doc_id:
                        continue
                    idx[doc_id.lower()] = {
                        'title': row.get('title') or row.get('subject'),
                        'date': row.get('date'),
                        'doc_type': row.get('doc_type'),
                    }
        except FileNotFoundError:
            pass
        except OSError:
            pass
        return cls(index=idx)

    def resolve(self, mention: Dict) -> Dict:
        doc_id = (mention.get('doc_id') or '').lower()
        if not doc_id or doc_id not in self.index:
            return mention
        rec = self.index[doc_id]
        out = dict(mention)
        if not out.get('title') and rec.get('title'):
            out['title'] = rec['title']
            out['resolution_status'] = 'resolved_index'
        if not out.get('date') and rec.get('date'):
            out['date'] = rec['date']
        return out

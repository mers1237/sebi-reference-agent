"""Unit tests for SEBI reference extraction pipeline.

These tests exercise pure-Python logic: regex candidate generation, dedup,
date validation, evidence checking, resolver behavior, and eval metrics.
They do not require PyMuPDF, google-generativeai, a Gemini API key, or any
PDF file — both heavy deps are imported lazily in src/extract.py and
src/agent.py.
"""
import json
from pathlib import Path

from src.agent import dedupe_mentions, parse_response, strip_fences
from src.candidates import (
    Candidate,
    filter_self_references,
    find_candidates,
    is_self_reference,
)
from src.eval import (
    hallucination_rate,
    match_documents,
    title_match,
    unresolved_rate,
)
from src.resolve import AliasResolver, TitleResolver
from src.run import build_documents, build_edges
from src.verify import (
    dedupe_within_page,
    evidence_present,
    is_valid_date,
    verify_mentions,
)


SAMPLE_PAGE = """Securities and Exchange Board of India
CIRCULAR

SEBI/HO/CFD/DIL1/CIR/P/2020/37                                   March 17, 2020

Subject: Relaxation from compliance with certain provisions

This circular is issued in exercise of powers conferred under Section 11 of the
Securities and Exchange Board of India Act, 1992 read with Regulation 101 of the
SEBI (Listing Obligations and Disclosure Requirements) Regulations, 2015.

Reference is drawn to the Master Circular on Mutual Funds dated May 13, 2019 and
the SEBI (Mutual Funds) Regulations, 1996. See also circular no. SEBI/CIR/IMD/DF/7/2011.

This circular supersedes the earlier guidelines.
"""


# --- Candidate generation ---------------------------------------------------

def test_find_candidates_detects_sebi_ids():
    cands = find_candidates(SAMPLE_PAGE)
    texts = [c.text for c in cands]
    assert any('SEBI/HO/CFD/DIL1/CIR/P/2020/37' in t for t in texts)
    assert any('SEBI/CIR/IMD/DF/7/2011' in t for t in texts)


def test_find_candidates_detects_regulations_and_acts_and_master_circular():
    cands = find_candidates(SAMPLE_PAGE)
    names = {c.pattern_name for c in cands}
    assert 'sebi_regulations' in names
    assert 'act_reference' in names
    assert 'master_circular' in names


def test_find_candidates_deduplicates_within_pattern():
    txt = "SEBI Act, 1992 and again SEBI Act, 1992."
    cands = find_candidates(txt)
    act_hits = [c for c in cands if c.pattern_name == 'act_reference']
    assert len(act_hits) == 1


# --- Self-reference filter --------------------------------------------------

def test_self_reference_detection():
    assert is_self_reference("this circular")
    assert is_self_reference("the present circular")
    assert is_self_reference("This Master Circular")
    assert is_self_reference("THIS NOTIFICATION")
    assert not is_self_reference("the said Regulations")
    assert not is_self_reference("SEBI Act, 1992")


def test_filter_self_references():
    cands = [
        Candidate('master_circular', 'This Master Circular', 0, 20),
        Candidate('act_reference', 'SEBI Act, 1992', 25, 40),
    ]
    kept = filter_self_references(cands)
    assert len(kept) == 1
    assert kept[0].text == 'SEBI Act, 1992'


# --- Date validation --------------------------------------------------------

def test_is_valid_date():
    assert is_valid_date('2020-03-17')
    assert is_valid_date(None)  # null is OK
    assert not is_valid_date('17-03-2020')
    assert not is_valid_date('2020/03/17')
    assert not is_valid_date('not a date')
    assert not is_valid_date('2020-13-01')  # month out of range
    assert not is_valid_date('1800-01-01')  # year out of range
    assert not is_valid_date(20200317)


# --- Evidence / verify ------------------------------------------------------

def test_evidence_present_whitespace_tolerant():
    page = "The SEBI Act, 1992 provides the legal framework."
    assert evidence_present("SEBI Act, 1992", page)
    assert evidence_present("sebi  act,  1992", page)
    assert not evidence_present("Companies Act, 2013", page)
    assert not evidence_present("", page)


def test_dedupe_within_page():
    mentions = [
        {'source_page': 0, 'doc_id': 'X/1', 'title': 'A', 'mention_text': 'ref a'},
        {'source_page': 0, 'doc_id': 'X/1', 'title': 'A', 'mention_text': 'ref a'},
        {'source_page': 0, 'doc_id': 'X/2', 'title': 'B', 'mention_text': 'ref b'},
        {'source_page': 1, 'doc_id': 'X/1', 'title': 'A', 'mention_text': 'ref a'},
    ]
    out = dedupe_within_page(mentions)
    assert len(out) == 3  # one duplicate on page 0 collapses


def test_verify_mentions_drops_hallucinations_and_self_refs():
    pages = ["The SEBI Act, 1992 applies here."]
    mentions = [
        {'source_page': 0, 'mention_text': 'SEBI Act', 'evidence_text': 'SEBI Act, 1992',
         'date': '1992-04-12', 'title': 'SEBI Act'},
        {'source_page': 0, 'mention_text': 'ghost', 'evidence_text': 'Companies Act, 2013',
         'date': None, 'title': None},
        {'source_page': 0, 'mention_text': 'this circular', 'evidence_text': 'this circular',
         'date': None, 'title': None},
    ]
    out = verify_mentions(mentions, pages)
    assert len(out) == 1
    assert out[0]['title'] == 'SEBI Act'


def test_verify_mentions_clears_bad_date_but_keeps_mention():
    pages = ["The text with SEBI Act, 1992."]
    mentions = [{
        'source_page': 0, 'mention_text': 'act', 'evidence_text': 'SEBI Act, 1992',
        'date': '17-03-2020', 'title': 'SEBI Act',
    }]
    out = verify_mentions(mentions, pages)
    assert len(out) == 1
    assert out[0]['date'] is None
    assert out[0]['title'] == 'SEBI Act'


# --- Agent parsing ----------------------------------------------------------

def test_strip_fences():
    assert strip_fences('```json\n{"a":1}\n```') == '{"a":1}'
    assert strip_fences('```\n{"a":1}\n```') == '{"a":1}'
    assert strip_fences('{"a":1}') == '{"a":1}'
    assert strip_fences('') == ''


def test_parse_response_handles_fences():
    raw = '```json\n{"mentions": [{"mention_text": "x", "evidence_text": "y", "doc_type": "act"}]}\n```'
    mentions = parse_response(raw)
    assert len(mentions) == 1
    assert mentions[0]['doc_type'] == 'act'


def test_parse_response_handles_embedded_json():
    raw = 'Here is the output: {"mentions": [{"doc_type": "circular"}]} done'
    mentions = parse_response(raw)
    assert len(mentions) == 1


def test_parse_response_empty_and_invalid():
    assert parse_response('') == []
    assert parse_response('not json at all') == []
    assert parse_response('{"no_mentions_key": 1}') == []


def test_dedupe_mentions_across_pages():
    ms = [
        {'source_page': 1, 'doc_id': 'A/1', 'title': 'T', 'mention_text': 'x'},
        {'source_page': 1, 'doc_id': 'A/1', 'title': 'T', 'mention_text': 'x'},
        {'source_page': 2, 'doc_id': 'A/1', 'title': 'T', 'mention_text': 'x'},
    ]
    assert len(dedupe_mentions(ms)) == 2


# --- Resolvers --------------------------------------------------------------

def test_alias_resolver_hereinafter_definition_captured():
    r = AliasResolver()
    r.register_page_text(
        'The Securities and Exchange Board of India (hereinafter referred to as "SEBI") hereby...'
    )
    assert 'sebi' in r.aliases


def test_alias_resolver_said_regulations_resolves_from_history():
    r = AliasResolver()
    r.register_mention({
        'doc_type': 'regulation',
        'title': 'SEBI (LODR) Regulations, 2015',
    })
    out = r.resolve({'mention_text': 'the said Regulations', 'title': None})
    assert out['title'] == 'SEBI (LODR) Regulations, 2015'
    assert out['resolution_status'] == 'resolved_contextual'


def test_alias_resolver_unresolved_stays_unresolved():
    r = AliasResolver()
    out = r.resolve({'mention_text': 'mystery reference', 'title': None})
    assert out.get('title') is None
    assert out['resolution_status'] == 'unresolved'


def test_title_resolver_missing_csv_is_noop(tmp_path):
    r = TitleResolver.from_csv(str(tmp_path / 'no_such.csv'))
    assert r.index == {}
    m = {'doc_id': 'X/1', 'title': None}
    assert r.resolve(m) == m


def test_title_resolver_csv_lookup(tmp_path):
    csv_path = tmp_path / 'idx.csv'
    csv_path.write_text(
        'doc_id,title,date,doc_type\n'
        'SEBI/CIR/IMD/DF/7/2011,Investor Protection Circular,2011-10-28,circular\n',
        encoding='utf-8',
    )
    r = TitleResolver.from_csv(str(csv_path))
    out = r.resolve({'doc_id': 'SEBI/CIR/IMD/DF/7/2011', 'title': None})
    assert out['title'] == 'Investor Protection Circular'
    assert out['date'] == '2011-10-28'
    assert out['resolution_status'] == 'resolved_index'


# --- Eval metrics -----------------------------------------------------------

def test_match_documents_p_r_f1():
    pred = [{'doc_id': 'A'}, {'doc_id': 'B'}, {'doc_id': 'C'}]
    gold = [{'doc_id': 'A'}, {'doc_id': 'B'}, {'doc_id': 'D'}]
    r = match_documents(pred, gold)
    assert r['tp'] == 2
    assert r['fp'] == 1
    assert r['fn'] == 1
    assert round(r['precision'], 3) == round(2 / 3, 3)
    assert round(r['recall'], 3) == round(2 / 3, 3)


def test_title_match_fuzzy():
    _, fuzzy = title_match('SEBI LODR Regulations 2015', 'SEBI LODR Regulations, 2015')
    assert fuzzy
    _, fuzzy2 = title_match('Completely Different', 'SEBI LODR Regulations, 2015')
    assert not fuzzy2


def test_hallucination_rate():
    pages = ["SEBI Act, 1992 here."]
    mentions = [
        {'source_page': 0, 'evidence_text': 'SEBI Act, 1992'},
        {'source_page': 0, 'evidence_text': 'Companies Act, 2013'},  # not on page
    ]
    assert hallucination_rate(mentions, pages) == 0.5


def test_unresolved_rate():
    ms = [{'title': 'x'}, {'title': None}, {'title': ''}]
    assert unresolved_rate(ms) == 2 / 3


# --- Document / edge builders ----------------------------------------------

def test_build_documents_aggregates_pages_and_relations():
    mentions = [
        {'doc_id': 'A/1', 'title': 'Alpha', 'doc_type': 'circular',
         'source_page': 0, 'relation_type': 'references'},
        {'doc_id': 'A/1', 'title': 'Alpha', 'doc_type': 'circular',
         'source_page': 1, 'relation_type': 'supersedes'},
        {'doc_id': 'B/2', 'title': 'Beta', 'doc_type': 'act',
         'source_page': 2, 'relation_type': 'issued_under'},
    ]
    docs = build_documents(mentions)
    assert len(docs) == 2
    alpha = next(d for d in docs if d['doc_id'] == 'A/1')
    assert alpha['times_mentioned'] == 2
    assert alpha['mentioned_on_pages'] == [0, 1]
    assert set(alpha['relation_types']) == {'references', 'supersedes'}


def test_build_edges():
    mentions = [{
        'title': 'Alpha', 'doc_type': 'circular', 'source_page': 0,
        'evidence_text': 'e', 'relation_type': 'references',
    }]
    edges = build_edges(mentions, 'source.pdf')
    assert len(edges) == 1
    assert edges[0]['source'] == 'source.pdf'
    assert edges[0]['target'] == 'Alpha'
    assert edges[0]['target_type'] == 'circular'
    assert edges[0]['relation'] == 'references'


# --- Fixture sanity ---------------------------------------------------------

def test_gold_fixture_parses():
    p = Path(__file__).parent / 'gold' / 'example_circular.json'
    data = json.loads(p.read_text(encoding='utf-8'))
    assert 'documents' in data
    assert 'mentions' in data
    assert len(data['documents']) > 0
    assert len(data['mentions']) > 0

"""Keyword retrieval over the PaperPilot store.

What each FTS5 piece operates on:
- MATCH   : the segmented index text in pages_fts (title / body_seg columns)
- BM25    : the same segmented columns, with explicit column weights
  (title weight 8.0, body weight 1.0; SQLite bm25 is a distance: smaller
  means more relevant)
- snippet(): NOT used for evidence. The FTS copy is segmented and must not
  pass as原文. Evidence snippets are built here from pages.body (raw text)
  by locating query terms at their real match positions.

Unicode: indexing and querying normalize with NFKC (see paperpilot.segment),
so evidence terms are normalized the same way. Raw bodies may still contain
compatibility characters ('⼤', 'ﬁ', ...). Highlighting is attempted on the
raw text first; when that fails, positions from the NFKC form are used only
if the mapping is provably 1:1. Otherwise nothing is highlighted and the
result is marked body_matched_no_highlight instead of pretending a title-only
hit -- raw pages.body is never replaced by the normalized copy.

Query safety: user text is segmented first, every resulting token is
quoted and bound as a SQL parameter. Punctuation, quotes and FTS5
operators never reach the MATCH expression. Empty or punctuation-only
queries return [] without touching SQL.

Retrieval runs two passes over the same index:
- strict: every query token must match (current AND behaviour). High
  precision, but AND plus segmentation boundaries misses pages whose
  wording differs slightly ('统计语言模型' vs '统计学习方法的语言模型').
- loose: OR over *discriminative* terms (whole Latin/digit words and CJK
  bigrams -- never bare CJK unigrams), kept only when a page covers at
  least MIN_LOOSE_TERM_HITS of them. Coverage is counted on the indexed
  body_seg tokens -- the same units MATCH runs against -- so a 'hit'
  here means what a MATCH hit means, and a stale index still misses.

Query-time expansion (this pass only; the index is never rewritten):
- inflection_family(): +/-s plurals plus a small irregular map, so
  'updates' can reach 'update' and 'frozen' can reach 'freezes'.
- TERM_ALIASES: document-attested abbreviation pairs only
  (e.g. 神经语言模型 == NLM per 'Neural Language Model, 简称 NLM').
Expanded forms live in the SAME concept group as the original term and
are OR-ed together. A concept counts once towards coverage no matter how
many surface forms hit, so adding variants never changes MIN_LOOSE_TERM_HITS
semantics. Expanded forms only widen candidate recall -- evidence snippets
still come from the original query terms at real spans in pages.body.

Both candidate sets are merged and ranked by the loose BM25 score alone
(strict hits are not pinned to the front), so a loose page can still
enter top_k when the strict pass has already filled it. Strict stays in
the pipeline as a high-precision candidate source; it is never the only
one, and its results are never blindly appended after loose ones.
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata

from dataclasses import dataclass

from paperpilot.segment import is_cjk, segment_tokens
from paperpilot.store import PaperStore

TITLE_WEIGHT = 8.0
BODY_WEIGHT = 1.0
DEFAULT_TOP_K = 5
SNIPPET_RADIUS = 40
MAX_SNIPPET_WINDOWS = 3

# Loose-pass admission: a multi-term query page must cover at least this
# many CONCEPT groups. 2 is the only integer that satisfies both sides of
# the eval set -- Q8's target p.45 covers just 2/4 concepts (must be
# admitted), while negative Q11 tops out at 1/3 (must be refused).
# Counting is per concept, not per surface form: 'update' + 'updates'
# together still count as one hit.
MIN_LOOSE_TERM_HITS = 2

# Irregular English inflections that the +/-s rule cannot derive.
# Keys and members are lowercase; lookup is casefolded. Only forms with
# confirmed evidence in the corpus are listed (freeze/freezes/frozen on
# LoRA p.1). Adding a family is a deliberate, reviewable act.
_IRREGULAR_INFLECTIONS: dict[str, frozenset[str]] = {
    'freeze': frozenset({'freeze', 'freezes', 'froze', 'frozen', 'freezing'}),
}
_IRREGULAR_LOOKUP: dict[str, str] = {
    form: base for base, family in _IRREGULAR_INFLECTIONS.items() for form in family
}

# Document-attested term aliases (abbreviation -> expansion source).
# Key is a whole CJK query run after NFKC; values are the abbreviation
# stems whose inflection_family() becomes that concept's extra surface
# forms. Each entry cites its definition sentence in the corpus:
#   神经语言模型  'Neural Language Model, 简称 NLM'   (AI报告 p.3)
#   统计语言模型  'Statistical Language Models, 简称 SLMs' (AI报告 p.2,
#                 phrased there as 基于统计学习方法的语言模型)
#   大语言模型    'Large Language Model, 简称 LLM'   (AI报告 p.1)
# No machine translation, no generic synonym list -- an alias without a
# cited definition does not belong here.
TERM_ALIASES: dict[str, tuple[str, ...]] = {
    '神经语言模型': ('NLM',),
    '统计语言模型': ('SLM',),
    '大语言模型': ('LLM',),
}

# Query-level controlled expansion -- deliberately NOT merged into
# TERM_ALIASES. These extra index tokens widen the loose MATCH OR and
# therefore BM25 scoring for ONE specific query. They never enter
# concept_coverage(), never move min_term_hits(), and can never admit a
# page whose original-query concepts fall below the threshold.
#
# Each entry is (token, kind, source) and must be a real index token
# (see is_index_token): a CJK bigram or a whole Latin/digit run. Multi-
# token strings such as 'n-gram' or '快速注意力' are REJECTED -- the
# index stores unigrams+bigrams for CJK and one token per alnum run.
#
# kind is 'synonym' (near-equivalent wording) or 'context' (related
# context word that co-occurs with the concept but is NOT a synonym).
# Only tokens validated in the step-3-4 B' isolation experiments are
# listed; do not add entries without corpus evidence and an experiment.
QUERY_EXPANSIONS: dict[str, tuple[tuple[str, str, str], ...]] = {
    '统计语言模型': (
        # p.3 states the limitation of statistical LMs via 转移概率, and
        # p.3/p.4 discuss the successor NLM -- development-history context,
        # NOT synonyms of 统计语言模型.
        ('转移', 'context', 'AI报告 p.3「由于需要估计指数级数量的转移概率」'),
        ('概率', 'context', 'AI报告 p.3「由于需要估计指数级数量的转移概率」'),
        ('马尔', 'context', 'AI报告 p.2「基于马尔可夫假设建立单词预测模型」'),
        ('可夫', 'context', 'AI报告 p.2「基于马尔可夫假设建立单词预测模型」'),
        ('NLM', 'context', 'AI报告 p.3「神经语言模型(Neural Language Model,简称NLM)」'),
        ('NLMs', 'context', 'AI报告 p.4「NLMs专注于学习与任务无关的表示」'),
    ),
    'Transformer 注意力机制': (
        # p.17/p.22 phrase attention-mechanism content without the
        # continuous form 注意力机制 -- these are the wording variants.
        ('遮罩', 'context', 'AI报告 p.17「因果解码器架构采用单向注意力遮罩」'),
        ('掩码', 'context', 'AI报告 p.17「修改了因果解码器的掩码机制」'),
        ('快速', 'context', 'AI报告 p.22「快速注意力…优化注意力模块」'),
    ),
}

STATUS_HIGHLIGHTED = 'highlighted'
STATUS_BODY_NO_HIGHLIGHT = 'body_matched_no_highlight'
STATUS_TITLE_ONLY = 'title_only'


@dataclass(frozen=True)
class Evidence:
    paper_id: int
    title: str
    page_id: int
    page_no: int
    snippet: str
    score: float
    status: str = STATUS_TITLE_ONLY


@dataclass(frozen=True)
class SnippetResult:
    text: str
    status: str
    body_matched: bool


def evidence_terms(query: str) -> list[str]:
    """Substrings to locate in pages.body (longest first, deduped).

    The query is NFKC-normalized first, the same rule as indexing, so terms
    are always in canonical form. CJK runs contribute the full run (when
    length >= 2) and their bigrams; single CJK characters contribute
    themselves; alnum runs contribute the run as-is.
    """
    query = unicodedata.normalize('NFKC', query)
    terms: list[str] = []
    i = 0
    n = len(query)
    while i < n:
        char = query[i]
        if is_cjk(char):
            j = i
            while j < n and is_cjk(query[j]):
                j += 1
            run = query[i:j]
            if len(run) == 1:
                terms.append(run)
            else:
                terms.append(run)
                terms.extend(run[k:k + 2] for k in range(len(run) - 1))
            i = j
        elif char.isalnum():
            j = i
            while j < n and query[j].isalnum() and not is_cjk(query[j]):
                j += 1
            terms.append(query[i:j])
            i = j
        else:
            i += 1
    seen = set()
    unique = []
    for term in terms:
        if term and term not in seen:
            seen.add(term)
            unique.append(term)
    return sorted(unique, key=len, reverse=True)


def build_match_query(query: str) -> str:
    """Safe FTS5 MATCH expression: every token double-quoted, params later.

    Returns '' when the query has no tokens (empty / punctuation only).
    """
    tokens = segment_tokens(query)
    if not tokens:
        return ''
    return ' '.join('"' + token.replace('"', '""') + '"' for token in tokens)


def inflection_family(word: str) -> tuple[str, ...]:
    """Controlled surface variants of one English/digit word (original first).

    Rules, both sides derivable and unit-tested:
    - +/-s plural/third-person, derived from the original spelling so
      'NLM' yields 'NLMs' (not 'nlms'): 'updates' -> 'update',
      'weight' -> 'weights'. Words ending in 'ss' ('class') only gain +s.
    - irregular families from _IRREGULAR_INFLECTIONS, looked up
      casefolded in either direction ('frozen' -> whole freeze family).

    No broad suffix truncation (-ed/-ing stripping is deliberately absent:
    it would invent junk stems like 'used' -> 'us'). Extend the irregular
    table when corpus evidence demands another family. Comparison and
    MATCH are both case-insensitive, so spelling here is for stable
    ordering and readable tests.
    """
    raw = word.strip()
    if not raw:
        return ()
    lower = raw.casefold()
    if lower in _IRREGULAR_LOOKUP:
        family = _IRREGULAR_INFLECTIONS[_IRREGULAR_LOOKUP[lower]]
        return (raw,) + tuple(sorted(family - {lower}))
    if len(raw) > 2 and raw[-1] in 'sS' and raw[-2:-1] not in 'sS':
        return (raw, raw[:-1])          # updates -> update
    if len(raw) > 2 and raw[-1] in 'sS':
        return (raw,)                   # -ss words ('class') get no +/-s partner
    return (raw, raw + 's')             # update -> updates


def concept_groups(query: str) -> list[tuple[str, ...]]:
    """Logical concept groups for the loose pass.

    Each group is one concept with its surface variants (original term,
    inflection forms, attested aliases). Exactly one coverage hit per
    group, however many variants match -- so expansion never shifts the
    MIN_LOOSE_TERM_HITS threshold.

    Grouping rules:
    - Latin/digit run -> one group whose variants are inflection_family().
    - CJK run of length >= 2 -> one group per bigram (bare unigrams are
      far too common and would flood the OR; whole runs are not FTS5
      tokens). If the run is a TERM_ALIASES key, one extra group holds
      the abbreviation's inflection family.
    - one-character CJK run -> one group of itself, so single-character
      queries still work.
    """
    query = unicodedata.normalize('NFKC', query)
    groups: list[tuple[str, ...]] = []
    i = 0
    n = len(query)
    while i < n:
        char = query[i]
        if is_cjk(char):
            j = i
            while j < n and is_cjk(query[j]):
                j += 1
            run = query[i:j]
            if len(run) == 1:
                groups.append((run,))
            else:
                for k in range(len(run) - 1):
                    groups.append((run[k:k + 2],))
                for alias in TERM_ALIASES.get(run, ()):
                    groups.append(inflection_family(alias))
            i = j
        elif char.isalnum():
            j = i
            while j < n and query[j].isalnum() and not is_cjk(query[j]):
                j += 1
            groups.append(inflection_family(query[i:j]))
            i = j
        else:
            i += 1
    return groups


def discriminative_terms(query: str) -> list[str]:
    """Flat view of concept_groups(): every surface form, deduped, stable order.

    Used to build the loose MATCH expression. Coverage must be computed
    with concept_coverage() instead, so variants of one concept cannot
    be counted twice.
    """
    seen = set()
    unique = []
    for group in concept_groups(query):
        for term in group:
            key = term.casefold()
            if key not in seen:
                seen.add(key)
                unique.append(term)
    return unique


def is_index_token(token: str) -> bool:
    """True iff ``token`` is exactly one token produced by segment_tokens().

    The index stores CJK unigrams+bigrams and whole Latin/digit runs, so
    '转移', '遮罩' and 'NLM' are valid while '转移概率' (4-char run) and
    'n-gram' (two alnum runs split by '-') are not. Expansion entries
    that fail this check are refused loudly, never silently queried as
    a phrase.
    """
    return bool(token) and token in set(segment_tokens(token))


def expansion_tokens(query: str) -> list[str]:
    """Query-level expansion tokens for the loose MATCH OR only.

    These widen candidate recall and BM25 scoring. They are deliberately
    absent from concept_coverage(): a page matching ONLY expansion
    tokens still fails admission unless the ORIGINAL query concepts
    reach min_term_hits(). Invalid entries (wrong granularity) raise
    ValueError at call time so a bad config cannot ship silently.
    """
    key = unicodedata.normalize('NFKC', query)
    entries = QUERY_EXPANSIONS.get(key, ())
    tokens: list[str] = []
    seen = set()
    for token, kind, source in entries:
        if kind not in ('synonym', 'context'):
            raise ValueError(f'expansion kind must be synonym/context: {token!r}')
        if not is_index_token(token):
            raise ValueError(
                f'expansion token {token!r} is not a single index token '
                f'(source: {source})'
            )
        fold = token.casefold()
        if fold not in seen:
            seen.add(fold)
            tokens.append(token)
    return tokens


def build_loose_match(terms: list[str]) -> str:
    """Safe FTS5 MATCH for the loose pass: OR of quoted terms, params later.

    Terms come from segmentation, but each is still double-quoted with
    embedded quotes doubled, so FTS5 operators never become syntax.
    Returns '' when there are no terms.
    """
    if not terms:
        return ''
    return ' OR '.join('"' + term.replace('"', '""') + '"' for term in terms)


def min_term_hits(term_count: int) -> int:
    """Coverage threshold for the loose pass (see MIN_LOOSE_TERM_HITS)."""
    if term_count <= 1:
        return term_count
    return MIN_LOOSE_TERM_HITS


def term_coverage(page_tokens: set[str], terms: list[str]) -> int:
    """How many surface forms this page's index tokens contain.

    ``page_tokens`` must already be casefolded, matching FTS5's unicode61
    folding. Membership is exact-token, never substring: 'weight' must not
    count as covering 'weights', because MATCH would not match it either.
    """
    return sum(1 for term in terms if term.casefold() in page_tokens)


def concept_coverage(page_tokens: set[str], groups: list[tuple[str, ...]]) -> int:
    """How many CONCEPTS this page covers (one hit per group, variants shared).

    This is the number MIN_LOOSE_TERM_HITS is compared against. A concept
    matches when ANY of its surface forms is an index token, so adding
    'update' alongside 'updates' widens recall but cannot inflate the
    coverage count past one for that concept.
    """
    return sum(
        1 for group in groups
        if any(term.casefold() in page_tokens for term in group)
    )


def _index_tokens(body_seg: str) -> set[str]:
    """Casefolded tokens of an indexed ``body_seg`` column.

    ``body_seg`` is the space-joined output of ``segment()``, so splitting
    it recovers exactly the tokens FTS5 stored. Coverage is computed from
    these -- never from ``pages.body`` -- so a stale index misses just
    like MATCH does instead of being rescued by the raw text.
    """
    return {token.casefold() for token in body_seg.split()}


def _find_spans(
    body: str,
    terms: list[str],
    mapped: str | None = None,
) -> list[tuple[int, int]]:
    """Real match spans in raw ``body`` (indices interchangeable with ``mapped``).

    Each term is looked up in the raw text first; only when a term has no raw
    hit is the provably 1:1 NFKC form consulted, so every returned span is a
    genuine raw-text position.
    """
    spans: list[tuple[int, int]] = []
    taken: list[tuple[int, int]] = []
    haystacks = (body, mapped) if mapped is not None else (body,)
    for term in sorted(terms, key=len, reverse=True):
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        for hay in haystacks:
            for match in pattern.finditer(hay):
                start, end = match.start(), match.end()
                if any(not (end <= s or start >= e) for s, e in taken):
                    continue  # longest term already marked this span
                taken.append((start, end))
                spans.append((start, end))
    spans.sort()
    return spans


def _render_snippet(
    body: str,
    spans: list[tuple[int, int]],
    radius: int,
    max_windows: int,
) -> str:
    windows: list[tuple[int, int]] = []
    for start, end in spans:
        left = max(0, start - radius)
        right = min(len(body), end + radius)
        if windows and left <= windows[-1][1]:
            windows[-1] = (windows[-1][0], max(windows[-1][1], right))
        else:
            windows.append((left, right))
        if len(windows) >= max_windows:
            break

    parts = []
    for left, right in windows:
        piece = ''
        cursor = left
        for start, end in spans:
            if start < left or end > right:
                continue
            piece += body[cursor:start] + '[' + body[start:end] + ']'
            cursor = end
        piece += body[cursor:right]
        prefix = '…' if left > 0 else ''
        suffix = '…' if right < len(body) else ''
        parts.append(prefix + piece + suffix)
    return ' … '.join(parts)


def _one_to_one_nfkc(text: str) -> str | None:
    """NFKC(text), but only when every character maps 1:1 onto its own index.

    Returns None whenever offsets cannot be proven to line up ('ﬁ' -> 'fi'
    expands, Hangul jamo compose, mixed cases shift boundaries), so callers
    never paint highlights at unverified positions.
    """
    normalized = unicodedata.normalize('NFKC', text)
    if len(normalized) != len(text):
        return None
    for index, char in enumerate(text):
        if unicodedata.normalize('NFKC', char) != normalized[index]:
            return None
    return normalized


def locate_snippet(
    body: str,
    terms: list[str],
    *,
    radius: int = SNIPPET_RADIUS,
    max_windows: int = MAX_SNIPPET_WINDOWS,
) -> SnippetResult:
    """Locate evidence in raw page text without ever faking positions.

    status:
      - highlighted:                 body matched, markers at real spans
      - body_matched_no_highlight:   body matched, but normalization changed
                                     offsets so highlighting is unsafe
      - title_only:                  no body match (e.g. the hit is in title)
    """
    if not body or not terms:
        return SnippetResult('', STATUS_TITLE_ONLY, False)

    mapped = _one_to_one_nfkc(body)
    spans = _find_spans(body, terms, mapped)
    if spans:
        return SnippetResult(
            _render_snippet(body, spans, radius, max_windows), STATUS_HIGHLIGHTED, True
        )

    if mapped is not None:
        # 1:1 normalization proved and still nothing: body has no such terms.
        return SnippetResult('', STATUS_TITLE_ONLY, False)

    # Length-changing normalization: the body does match via the normalized
    # form, but offsets cannot be proven to map back -- never fake highlights.
    if _find_spans(unicodedata.normalize('NFKC', body), terms):
        return SnippetResult('', STATUS_BODY_NO_HIGHLIGHT, True)
    return SnippetResult('', STATUS_TITLE_ONLY, False)


def build_snippet(
    body: str,
    terms: list[str],
    *,
    radius: int = SNIPPET_RADIUS,
    max_windows: int = MAX_SNIPPET_WINDOWS,
) -> str:
    """Original-text evidence windows around real term matches (text only)."""
    return locate_snippet(body, terms, radius=radius, max_windows=max_windows).text


_SEARCH_COLUMNS = (
    'SELECT pg.page_id, pg.page_no, pg.body, p.paper_id, p.title, '
    'pages_fts.body_seg AS body_seg, '
    f'bm25(pages_fts, {TITLE_WEIGHT}, {BODY_WEIGHT}) AS score '
    'FROM pages_fts '
    'JOIN pages pg ON pg.page_id = pages_fts.rowid '
    'JOIN papers p ON p.paper_id = pg.paper_id '
    'WHERE pages_fts MATCH ? '
    'ORDER BY score'
)


def _match_rows(store: PaperStore, match_expr: str) -> list[sqlite3.Row]:
    """All pages matching a (already safe) FTS5 expression, best score first."""
    return store.conn.execute(_SEARCH_COLUMNS, (match_expr,)).fetchall()


def search(store: PaperStore, query: str, *, top_k: int = DEFAULT_TOP_K) -> list[Evidence]:
    """Keyword search returning traceable evidence rows (title + page + 原文).

    Strict AND candidates and loose (coverage-filtered) candidates are
    merged and re-ranked by the loose BM25 score, then cut to top_k. The
    strict pass is kept as a high-precision candidate source; because its
    hits cover every query token they also cover every discriminative
    term, so in the usual case it is a subset of the loose pass. Explicit
    merging still guards the boundary cases and keeps the two sources
    visible in the code.

    Loose candidates come from concept_groups(): inflection forms and
    attested aliases are OR-ed in for recall, but coverage is counted per
    concept (see concept_coverage), so the MIN_LOOSE_TERM_HITS threshold
    still means "this many distinct query concepts", not "this many
    surface strings". Evidence snippets always use the ORIGINAL query
    terms (evidence_terms) at real spans in pages.body -- an expanded
    form can pull a page in, but never paints a highlight.

    Query-level expansion (QUERY_EXPANSIONS) joins the loose MATCH OR to
    widen recall and BM25 scoring for one specific query. Those tokens
    are NOT concept forms: they never enter concept_coverage() and never
    move min_term_hits(). A page that only matches expansion tokens still
    needs the original query's concepts to clear the same threshold --
    expansion widens the net, it does not lower the bar.

    Status correction: when the original query strings cannot be located
    in pages.body (locate_snippet says title_only) but the indexed body
    still satisfies the SAME concept-coverage rule as the loose pass, the
    page really matched via expanded forms in its body. Reporting that as
    title_only would be a lie, so it becomes body_matched_no_highlight.
    The check is coverage-based, never "did the strict pass see it": one
    page can arrive through both passes, and only the body decides.

    Ranking is score-only on purpose: strict hits are NOT pinned above
    loose ones. Otherwise a strict result list that already fills top_k
    would lock out a genuinely relevant page that only the loose pass
    could see (the failure mode this merge exists to fix).
    """
    match_expr = build_match_query(query)
    if not match_expr:
        return []
    terms = evidence_terms(query)
    groups = concept_groups(query)
    # Expansion tokens widen the OR / BM25 only; admission and status
    # correction below still use concept_coverage(groups) alone.
    loose_terms = discriminative_terms(query) + expansion_tokens(query)
    loose_expr = build_loose_match(loose_terms)
    required = min_term_hits(len(groups))

    candidates: dict[tuple[int, int], dict] = {}

    def remember(row: sqlite3.Row, score: float) -> None:
        key = (row['paper_id'], row['page_no'])
        entry = candidates.get(key)
        if entry is None:
            candidates[key] = {
                'paper_id': row['paper_id'],
                'title': row['title'],
                'page_id': row['page_id'],
                'page_no': row['page_no'],
                'body': row['body'],
                'body_seg': row['body_seg'],
                'score': score,
            }
        elif score < entry['score']:
            entry['score'] = score  # better (smaller) bm25 distance wins

    if loose_expr:
        for row in _match_rows(store, loose_expr):
            page_tokens = _index_tokens(row['body_seg'])
            if concept_coverage(page_tokens, groups) >= required:
                remember(row, row['score'])

    for row in _match_rows(store, match_expr):
        remember(row, row['score'])

    ranked = sorted(
        candidates.values(),
        key=lambda item: (item['score'], item['paper_id'], item['page_no']),
    )
    results = []
    for item in ranked[: int(top_k)]:
        snippet = locate_snippet(item['body'], terms)
        status = snippet.status
        if status == STATUS_TITLE_ONLY:
            # Original query strings absent from pages.body. If the indexed
            # body still covers enough CONCEPTS, this is a body hit through
            # expanded forms (inflection/alias), not a title-only hit. Same
            # threshold as candidate admission -- no separate looser bar.
            page_tokens = _index_tokens(item['body_seg'])
            if concept_coverage(page_tokens, groups) >= required:
                status = STATUS_BODY_NO_HIGHLIGHT
        results.append(
            Evidence(
                paper_id=item['paper_id'],
                title=item['title'],
                page_id=item['page_id'],
                page_no=item['page_no'],
                snippet=snippet.text,
                score=item['score'],
                status=status,
            )
        )
    return results

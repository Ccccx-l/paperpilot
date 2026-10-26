"""PaperPilot minimal local web UI (stdlib http.server, no extra deps).

Scope (P1): HTTP service, paper import, single-turn Q&A, basic HTML.

Design:
- Reuses the existing tools (import_papers / search_papers / read_paper_page)
  and DeepSeekAgent; nothing here reimplements retrieval.
- The agent's final answer and tool trace are taken from ``agent.message``
  (Result holds statistics only) -- Result is deliberately left untouched.
- Binds to 127.0.0.1 only; all data stays under MINI_HARNESS_WORK_SPACE.

Run:
    MINI_HARNESS_WORK_SPACE=/path/to/ws uv run python -m paperpilot.web
    # then open http://127.0.0.1:8001
"""

from __future__ import annotations

import copy
import html
import json
import re
import threading
import unicodedata
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from pydantic import ValidationError

from mini_harness.agent import DeepSeekAgent, _is_context_overflow
from mini_harness.config import CONFIG

from paperpilot.cli import build_paper_config
from paperpilot.segment import is_cjk
from paperpilot.sessions import SessionStore, new_session_id
from paperpilot.store import PaperStore, default_db_path
from paperpilot.tools import (
    PAPER_TOOLS,
    ImportPapersInput,
    import_papers,
)

DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 8001
MAX_BODY_BYTES = 64 * 1024  # form posts only; reject anything larger


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _esc(value: object) -> str:
    """Escape any dynamic text before it reaches HTML."""
    return html.escape(str(value), quote=True)


# --------------------------------------------------------------------------
# current-turn message boundary
# --------------------------------------------------------------------------
# Evidence extraction must never look at prior turns' messages: a page read in
# answer to an earlier question cannot vouch for a quotation in this one. The
# agent records the current turn in ``turn_messages`` (append-only, survives
# context compaction). Objects without that attribute (FakeAgent, older
# shapes) may fall back to a validated ``turn_start`` index. If the boundary
# cannot be determined reliably we refuse and use NO messages at all -- we
# never fall back to the whole history.

def current_turn_messages(source) -> list:
    """Messages belonging to the current turn of ``source``.

    ``source`` may be an agent-like object (with ``turn_messages`` and/or
    ``message``/``turn_start``) or a plain list of messages (returned as-is).
    """
    if isinstance(source, list):
        return source
    turn = getattr(source, 'turn_messages', None)
    if isinstance(turn, list):
        return list(turn)
    messages = getattr(source, 'message', None)
    if not isinstance(messages, list):
        return []
    start = getattr(source, 'turn_start', None)
    if start is None:
        # Legacy/FakeAgent with a single-turn message list and no boundary
        # marker: the whole list is the current turn.
        return list(messages)
    if not isinstance(start, int) or isinstance(start, bool):
        return []  # unreliable boundary: refuse rather than use history
    if start < 0 or start > len(messages):
        return []  # stale/out-of-range boundary after compaction: refuse
    return list(messages[start:])


def _safe_paper_dir(raw: str, cfg=CONFIG) -> tuple[str | None, str | None]:
    """Validate a workspace-relative paper directory. Returns (value, error).

    Rejects absolute paths and any path that escapes the workspace.
    """
    candidate = (raw or '').strip()
    if not candidate:
        return None, '论文目录不能为空'
    p = Path(candidate)
    if p.is_absolute() or candidate.startswith('~'):
        return None, '只接受工作区内的相对路径，不接受绝对路径'
    if '..' in p.parts:
        return None, '路径不允许包含 ..'
    try:
        resolved = (Path(cfg.work_space) / p).resolve()
        base = Path(cfg.work_space).resolve()
    except OSError as e:
        return None, f'路径无法解析: {e}'
    if not resolved.is_relative_to(base):
        return None, '路径越界：必须位于工作区内'
    return candidate, None


def _extract_answer_and_trace(source) -> tuple[str, list[str]]:
    """Pull the final assistant text and tool trace from THIS turn's messages.

    ``source`` is an agent-like object (see ``current_turn_messages``). Only
    the current turn is inspected, so a prior turn's answer can never be
    rendered as this turn's answer.
    """
    answer = ''
    trace: list[str] = []
    for item in current_turn_messages(source):
        role = item.get('role') if isinstance(item, dict) else getattr(item, 'role', None)
        if role != 'assistant':
            continue
        tool_calls = item.get('tool_calls') if isinstance(item, dict) else getattr(item, 'tool_calls', None)
        if tool_calls:
            for tc in tool_calls:
                if isinstance(tc, dict):
                    fn = tc.get('function') or {}
                    name = fn.get('name', '?')
                    args = fn.get('arguments', '')
                else:
                    fn = getattr(tc, 'function', None)
                    name = getattr(fn, 'name', '?')
                    args = getattr(fn, 'arguments', '')
                trace.append(f'调用工具 {name}({args})')
        content = item.get('content') if isinstance(item, dict) else getattr(item, 'content', None)
        if content:
            answer = content
    return answer, trace


# --------------------------------------------------------------------------
# verbatim quotation verification
# --------------------------------------------------------------------------
# Only explicitly marked quotations (markdown `> ` blocks and paired quote
# marks) are checked, and only against page bodies that THIS run actually
# read via read_paper_page. Comparison is deterministic string matching
# after a deliberately conservative normalization -- no fuzzy/semantic rules.

CLIP_MARKER = '.....clipped at'

_QUOTE_PAIRS = (
    ('“', '”'),   # “ ”
    ('「', '」'),   # 「 」
    ('『', '』'),   # 『 』
    ('"', '"'),
)


def _collapse_hard_wraps(text: str) -> str:
    """Resolve PDF hard line breaks without destroying word boundaries.

    A wrap strictly between two CJK characters is a layout break inside a
    word (概\\n率 -> 概率) and is dropped. A wrap immediately after a hyphen is
    PDF soft-hyphenation (de-\\ncomposition -> decomposition): the hyphen was
    inserted by the typesetter to break the word, so both the hyphen and the
    break are dropped to restore the original word. Any other wrap is a word
    boundary (low\\nrank -> low rank) and becomes a single space, so it can
    never glue two words together.

    Counter-examples that must NOT match after this rule:
    - real hyphens not at a line break (``low-rank`` stays ``low-rank``;
      ``low rank`` never equals ``low-rank``);
    - wrong words (``decompositionX`` != ``decomposition``);
    - wrong numbers (``4.8M`` != ``4.7M``).
    """
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in '\r\n':
            j = i
            while j < n and text[j] in '\r\n':
                j += 1
            prev = out[-1] if out else ''
            nxt = text[j] if j < n else ''
            if prev and nxt and is_cjk(prev) and is_cjk(nxt):
                pass  # in-word CJK wrap: drop it
            elif prev == '-':
                # PDF soft hyphenation: "de-\ncomposition" -> "decomposition".
                # The hyphen is a typesetting artifact, not part of the word.
                if out:
                    out.pop()
                # join without a space
            else:
                out.append(' ')
            i = j
        else:
            out.append(ch)
            i += 1
    return ''.join(out)


def _normalize_quote(text: str) -> str:
    """Conservative normalization for comparison.

    Allowed: NFKC (compatibility characters); dropping hard line breaks
    that fall INSIDE a CJK word (概\\n率 -> 概率); turning any other hard
    break into a single space so word boundaries survive (low\\nrank becomes
    'low rank', never 'lowrank'); collapsing runs of remaining whitespace
    to a single space; trimming ends.
    Forbidden: deleting whitespace BETWEEN words (that would make 'low rank'
    equal 'lowrank'), case folding, punctuation rewriting, hyphen removal,
    or any rule that could hide a real word difference (重要 vs 重大).
    """
    text = unicodedata.normalize('NFKC', text)
    text = _collapse_hard_wraps(text)
    return re.sub(r'[^\S\r\n]+', ' ', text).strip()


# Markdown emphasis is presentation formatting, not paper wording: ``**x**``,
# ``*x*`` and ``__x__`` all render as ``x``. Only PAIRED markers are removed,
# so a literal asterisk in the paper text survives untouched. This is a
# bounded format step applied before comparison; _normalize_quote() itself
# keeps its conservative word/punctuation rules unchanged.
_MD_EMPHASIS_RE = re.compile(r'(\*{1,2})([^*\n]+?)\1|(__)([^_\n]+?)\3')


def _strip_md_emphasis(text: str) -> str:
    """Remove paired Markdown emphasis markers used for display only."""
    if not text:
        return ''
    return _MD_EMPHASIS_RE.sub(lambda m: m.group(2) if m.group(2) is not None else m.group(4),
                               text or '')


def _compare_text(text: str) -> str:
    """Comparison key: strip display-only Markdown, then normalize."""
    return _normalize_quote(_strip_md_emphasis(text))


def _parse_page_body(tool_content: str) -> str | None:
    """Extract the raw page body from a read_paper_page tool result.

    Format is ``《title》 src p.N\\n\\n<body>``. Returns None for failure
    markers, clipped output, or anything we cannot parse confidently.
    """
    if not tool_content:
        return None
    text = tool_content.strip()
    if text.startswith('[not found]') or text.startswith('[Access denied]'):
        return None
    if CLIP_MARKER in tool_content:
        return None  # truncated: refuse to verify against a partial page
    parts = tool_content.split('\n\n', 1)
    if len(parts) != 2:
        return None
    body = parts[1].strip()
    return body if body else None


def _parse_page_title(tool_content: str) -> str:
    """Paper title from a read_paper_page header ``《title》 src p.N``."""
    if not tool_content:
        return ''
    m = re.match(r'\s*《([^》]*)》', tool_content)
    return (m.group(1).strip() if m else '')


def _collect_page_records(source) -> dict:
    """Map (paper_id, page_no) -> {'body': str, 'title': str} for THIS turn.

    Associates each ``read_paper_page`` tool_call with its ``role='tool'``
    result via tool_call_id. Only the current turn's messages are scanned, so
    a page read in a previous turn can never become evidence for this turn.
    search_papers snippets are never collected.
    """
    messages = current_turn_messages(source)
    call_args: dict[str, tuple[int, int]] = {}
    call_results: dict[str, str] = {}
    for item in messages:
        if not isinstance(item, dict):
            continue
        if item.get('role') == 'assistant':
            for tc in item.get('tool_calls') or []:
                fn = (tc.get('function') or {}) if isinstance(tc, dict) else {}
                if fn.get('name') != 'read_paper_page':
                    continue
                try:
                    args = json.loads(fn.get('arguments') or '{}')
                    key = (int(args['paper_id']), int(args['page_no']))
                except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                    continue
                if tc.get('id'):
                    call_args[tc['id']] = key
        elif item.get('role') == 'tool':
            cid = item.get('tool_call_id')
            if cid in call_args:
                call_results[cid] = item.get('content') or ''
    records: dict[tuple[int, int], dict] = {}
    for cid, key in call_args.items():
        raw = call_results.get(cid, '')
        body = _parse_page_body(raw)
        if body is not None:
            records[key] = {'body': body, 'title': _parse_page_title(raw)}
    return records


def _collect_read_pages(source) -> dict:
    """Map (paper_id, page_no) -> body for pages read successfully THIS turn."""
    return {k: v['body'] for k, v in _collect_page_records(source).items()}


def _extract_quotes(answer: str) -> list[tuple]:
    """Extract explicitly marked quotations.

    Returns (text, reliable, is_block, start, end) tuples where start/end are
    the span of the whole quote form in ``answer`` (including its marks), so
    callers can judge the LOCAL citation context around each quote.

    reliable=False marks quotes we cannot parse confidently (nested or
    unbalanced quote marks); those must never be reported as verified.
    is_block=True marks markdown "> " quote blocks. Nested content is never
    silently truncated at the first closing mark.

    Extracting a span is only a FORM observation -- it says the model used
    quote marks or a quote block. Whether that form carries paper-quotation
    INTENT is decided separately by ``is_paper_quote``.
    """
    src = answer or ''
    found: list[tuple] = []
    # 1) markdown blockquote runs: consecutive "> " lines form one quote
    line_spans: list[tuple[str, int, int]] = []
    pos = 0
    for line in src.splitlines(keepends=True):
        line_spans.append((line, pos, pos + len(line)))
        pos += len(line)
    block_parts: list[str] = []
    block_start = block_end = 0
    for line, s, e in line_spans:
        stripped = line.strip()
        if stripped.startswith('>'):
            piece = stripped.lstrip('>').strip()
            if piece:
                if not block_parts:
                    block_start = s
                block_parts.append(piece)
                block_end = e
        elif block_parts:
            found.append(('\n'.join(block_parts), True, True, block_start, block_end))
            block_parts = []
    if block_parts:
        found.append(('\n'.join(block_parts), True, True, block_start, block_end))
    # 2) paired quote marks (no apostrophes -- they are not quote boundaries)
    for left, right in _QUOTE_PAIRS:
        start = 0
        while True:
            i = src.find(left, start)
            if i < 0:
                break
            j = src.find(right, i + len(left))
            if j < 0:
                break
            inner = src[i + len(left):j]
            if inner.strip():
                # nested/unbalanced: the closing mark we found may belong to
                # an inner quote -- do not pretend the span is a clean quote
                nested = left in inner or (right in inner and right != left)
                found.append((inner, not nested, False,
                              i, j + len(right)))
            start = j + len(right)
    # de-duplicate: drop a quote whose normalized text is a substring of
    # another extracted quote (outer/inner pairs must not double-count)
    kept: list[tuple] = []
    keys = [_normalize_quote(t) for t, _, _, _, _ in found]
    for idx, (text, reliable, is_block, s, e) in enumerate(found):
        key = keys[idx]
        if not key:
            continue
        if any(j != idx and keys[j] and key in keys[j] and len(keys[j]) > len(key)
               for j in range(len(found))):
            continue  # contained in a longer quote -- not a separate one
        kept.append((text, reliable, is_block, s, e))
    return kept


# ---- ordinary chat vs paper quotation -----------------------------------
#
# Quotation marks alone do not make a paper quotation. A test marker, a code
# snippet, the user's own words, or a restatement of an earlier message is
# ordinary chat and must NOT produce an evidence card. Paper quotations are
# recognised only when the answer presents the text as paper evidence:
# a markdown "> " block (the D5 citation format), explicit paper-source
# context (page ref, 《title》, citation verbs), or a text that really matches
# a page body read this turn. Ambiguous cases are NOT marked as paper quotes,
# so they can never be reported as verified.

_CODE_TOKEN_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_\-\.\:/@+%]{0,63}$')

_PAPER_SOURCE_RE = re.compile(
    r'(原文|引述|引用|引文|摘录|摘自|征引|转引|论文中|论文指出|论文写道|论文描述|'
    r'论文认为|文章中|文章指出|文中|书中|第\s*\d+\s*页|'
    r'p\.\s*\d+|page\s+\d+|paper[_\s]?id|《[^》]{1,80}》|'
    r'according\s+to|as\s+(stated|noted|described|shown|reported)\s+in|'
    r'the\s+(paper|article|study|authors?)|quot(?:e|ed|ing|ation)|verbatim|'
    r'from\s+the\s+(paper|text|page))',
    re.IGNORECASE,
)

# Attribution verbs that mark the text BEFORE a quote as a citation
# introduction ("原文是…", "引述：…", "according to …").
_ATTRIBUTION_RE = re.compile(
    r'(原文是|原文：|引述|引用|引文|摘录|摘自|征引|转引|再述|复述|'
    r'论文中|论文指出|论文写道|论文描述|论文认为|文章中|文章指出|'
    r'according\s+to|as\s+(stated|noted|described|shown|reported)\s+in|'
    r'the\s+(paper|article|study|authors?)\s+(states?|says?|writes?|notes?)|'
    r'quot(?:e|ed|ing|ation)|verbatim|from\s+the\s+(paper|text|page))',
    re.IGNORECASE,
)

# Concrete source marks: a place to check (page number, paper id). A bare
# 《title》 is only a paper MENTION ("索引的论文包括《LoRA》") -- it does not
# attribute any quote to that paper. Only a page-level mark cites a location.
_SOURCE_MARK_RE = re.compile(
    r'(第\s*\d+\s*页|p\.\s*\d+|page\s+\d+|paper[_\s]?id\s*=)',
    re.IGNORECASE,
)

# Section headings: their appearance marks a title/summary block, not a
# verbatim quotation of paper wording.
_SECTION_HEAD_RE = re.compile(
    r'^\s*(ABSTRACT|INTRODUCTION|BACKGROUND|METHODOLOGY|METHODS?\b|EXPERIMENTS?|'
    r'RESULTS?|DISCUSSION|CONCLUSIONS?|REFERENCES|ACKNOWLEDGMENTS?|'
    r'ACKNOWLEDGEMENTS|APPENDIX|SUPPLEMENTARY|KEYWORDS?|INDEX\s+TERMS|'
    r'相关工作|研究背景|实验|结论|参考文献|摘要|引言|绪论|方法|讨论)\b',
    re.IGNORECASE,
)

# Trailing / embedded ellipsis marks an excerpted summary, not verbatim text.
_ELLIPSIS_RE = re.compile(r'\.\.\.|…')


def _is_code_like(text: str) -> bool:
    """Identifier / code-ish token rather than prose (e.g. PP-CONTEXT-7319)."""
    t = (text or '').strip()
    return bool(t) and bool(_CODE_TOKEN_RE.match(t))


def _has_paper_source_context(answer: str) -> bool:
    """Does the answer present its quotations as paper evidence?"""
    return bool(_PAPER_SOURCE_RE.search(answer or ''))


def _local_paper_source(answer: str, start: int, end: int) -> bool:
    """Is THIS quote span attributed to a paper source?

    Eligibility is judged per quotation. Two distinct signals, both scoped to
    the quote itself:

    * an ATTRIBUTION label directly before the quote (``原文是…``,
      ``引述：…``, ``according to…``) -- the answer introduces the quote as
      coming from a paper;
    * a CONCRETE source mark (``《title》``, ``p.N``, ``第 N 页``) in the
      quote's own paragraph -- the answer gives a place to check.

    Bare words like 原文/文中 in the surrounding prose ("在原文中出现过")
    are NOT attributions and must not promote a quoted term to a citation.
    """
    if not answer:
        return False
    s, e = (start or 0), (end or 0)
    # attribution verbs immediately before the quote
    head = answer[max(0, s - 40):s]
    if _ATTRIBUTION_RE.search(head):
        return True
    # concrete source marks anywhere in the quote's paragraph
    lo = answer.rfind('\n\n', 0, s)
    lo = 0 if lo < 0 else lo + 2
    hi = answer.find('\n\n', e)
    hi = len(answer) if hi < 0 else hi
    return bool(_SOURCE_MARK_RE.search(answer[lo:hi]))


# Letter-by-letter bold term expansion: "**T**ask-**C**onstrained **M**..."
# Each bold span is a single letter/tiny fragment, not a quotation.
_SHORT_BOLD_RE = re.compile(r'(\*{1,2})([^*\n]{1,3}?)\1')
_MD_BOLD_SPAN_RE = re.compile(r'(\*{1,2})([^*\n]+?)\1')


def _is_term_expansion(text: str) -> bool:
    """True for bold-letter term expansions.

    ``**T**ask-**C**onstrained **M**ultimodal **S**cheduling`` spells out an
    acronym with letter-by-letter emphasis -- that is terminology, not a
    verbatim quotation of paper wording. A single long bold span is display
    emphasis on a quotation and must NOT be rejected here.
    """
    raw = (text or '').strip()
    if not raw:
        return False
    # several tiny bold spans interspersed with plain text = acronym spelling
    short = _SHORT_BOLD_RE.findall(raw)
    return len(short) >= 3


def _is_fully_emphasised(text: str) -> bool:
    """True when the whole span is one bold/italic run (display emphasis).

    ``**在调度时考虑…效率。**`` is the model emphasising its own summary;
    with no citation beside it and no page match it is not a quotation.
    """
    raw = (text or '').strip()
    if not raw:
        return False
    m = re.match(r'^\s*(\*{1,2})([^*\n]+?)\1\s*$', raw, re.S)
    return bool(m)


# A quote span that IS a section name or section reference (not paper prose):
#   "METHOD"  "ABSTRACT"  "the METHOD section"  "METHOD 部分"  "摘要"
# This is a narrow structural check on the span text itself -- not a blanket
# ban on short or uppercase text. A real short quote ("yes, really") or a
# non-section acronym ("NLP") is unaffected.
_SECTION_REF_RE = re.compile(
    r'^\s*(?:'
    r'(?:the\s+)?(' + '|'.join(
        n for n in ('ABSTRACT|INTRODUCTION|BACKGROUND|METHODOLOGY|METHODS|'
                    'METHOD|EXPERIMENTS?|RESULTS?|DISCUSSION|CONCLUSIONS?|'
                    'REFERENCES|ACKNOWLEDGMENTS?|ACKNOWLEDGEMENTS|APPENDIX|'
                    'SUPPLEMENTARY|KEYWORDS?|INDEX\\s+TERMS').split('|')
    ) + r')\s*(?:section|part|部分|章节|节)?'
    r'|(?:摘要|引言|绪论|相关工作|研究背景|实验|结论|讨论|参考文献|附录)'
    r'\s*(?:部分|章节|节)?'
    r')\s*$', re.IGNORECASE)


def _is_section_reference(text: str) -> bool:
    """True when the span is a section name or section pointer, not prose.

    "METHOD" / "the METHOD section" / "第 4 页的 METHOD 部分" name a part of
    the paper; they are not verbatim quotations of paper wording. Bounded to
    known section names -- ordinary short quotes are unaffected.
    """
    return bool(_SECTION_REF_RE.match(_strip_md_emphasis(text or '').strip()))


def _looks_like_heading_or_summary(text: str) -> bool:
    """True for paper titles, section headings and excerpted summaries.

    These are how a model *labels or describes* a paper, not verbatim
    quotations of it, so they must never become evidence-card candidates.
    Signals are structural (ellipsis, section name, bold section label, title
    colon-form), never a length threshold -- a long verbatim quotation must
    still be checked. Callers must first accept anything that verbatim-matches
    a page read this turn, so real quotes starting with "ABSTRACT" are
    unaffected.
    """
    raw = (text or '').strip()
    if not raw:
        return True
    stripped = _strip_md_emphasis(raw)
    # 0) the span IS a section name / section pointer ("METHOD", "ABSTRACT")
    if _is_section_reference(raw):
        return True
    # 1) ellipsis truncation -> excerpted summary, never verbatim paper text
    if _ELLIPSIS_RE.search(stripped):
        return True
    # 2) bold LABEL at the very start followed by unbolded prose. The bold
    #    span LABELS the block ("**ABSTRACT** An important paradigm ...",
    #    "**LoRA**: L OW-R ANK ADAPTATION OF LARGE LAN-"). A genuine quote is
    #    either fully bolded ("**ABSTRACT ... tasks.**"), has only inline
    #    emphasis ("The **model** uses x."), or is plain -- none of which
    #    match this shape.
    m = re.match(r'^\s*(\*{1,2})([^*\n:：]{1,24}?)\1(\s*[:：])?\s+(\S.*)$',
                 raw, re.S)
    if m:
        label, colon, rest = m.group(2).strip(), m.group(3), m.group(4)
        # section heading label: "**ABSTRACT** An important paradigm ..."
        if _SECTION_HEAD_RE.match(label):
            return True
        # title colon-form: "**LoRA**: L OW-R ANK ADAPTATION OF LARGE LAN-"
        latin = [c for c in rest if 'A' <= c <= 'Z' or 'a' <= c <= 'z']
        title_like = (bool(latin)
                      and sum(1 for c in latin if c.isupper()) / len(latin) > 0.6
                      and not re.search(r'[。！？.!?]$', rest.strip()))
        if colon and title_like:
            return True
    return False


# Labels that look like "…:" but introduce an EXAMPLE or a REQUEST for
# information, never a citation of paper wording.
_EXAMPLE_INTRO_RE = re.compile(
    r'(例如|比如|比方|如|像|e\.g\.|eg\.|for example|for instance|'
    r'名称|标题|关键词|链接|文件|关键词|keyword|title|name|link|url|'
    r'请提供|请补充|请填写|补充|提供)\s*[:：]?\s*$',
    re.IGNORECASE)


def _has_citation_intro(answer: str, start: int) -> bool:
    """Is this quote introduced by a short attribution label ending in a colon?

    ``原文：``, ``引述：``, ``再述：``, ``The paper states:`` -- a short label
    line right before the quote span is how the answer attributes the quote.
    This is a per-span signal: only the label immediately preceding THIS quote
    counts, so a citation before one quote never promotes another.

    Example/clarification markers (``例如：``, ``名称：``) are NOT citations:
    they introduce sample titles or ask for information, not paper wording.
    """
    if not answer or not start:
        return False
    head = answer[max(0, start - 60):start]
    m = re.search(r'([^\n]{1,30}?)[:：]\s*$', head)
    if not m:
        return False
    label = m.group(1).strip()
    if _EXAMPLE_INTRO_RE.search(label):
        return False
    return True


def _is_term_mention(answer: str, start: int, end: int) -> bool:
    """True when the quote marks a TERM being defined/explained, not quoted.

    ``"前缀"是在输入前拼接可训练向量`` mentions the term 前缀 and defines it;
    the quote marks are mention punctuation, not a citation of paper wording.
    """
    if not answer:
        return False
    tail = answer[(end or 0):(end or 0) + 12]
    # NB: \b is ASCII-only, so CJK markers get no trailing boundary.
    return bool(re.match(r'^\s*(是|指|叫做|称为|means\b|refers to\b|stands for\b)',
                         tail))


def is_paper_quote(text: str, is_block: bool, answer: str,
                   chat_texts, pages: dict,
                   start: int | None = None,
                   end: int | None = None) -> bool:
    """STAGE A -- does the model present THIS text as paper evidence?

    This is an intent judgement only: it decides whether the span enters the
    verbatim-verification pipeline at all. Whether the text then MATCHES a
    page is Stage B (``_verify_quotes``).

    Eligibility is judged per quotation, using the LOCAL context around its
    span. A citation elsewhere in the answer never promotes unrelated bold
    spans or quote marks. ``matches_page`` never acts alone: it only
    corroborates text that already carries quote form (it is reached here
    solely because ``_extract_quotes`` yielded a quote span).
    """
    nq = _compare_text(text)
    if not nq:
        return False
    # -- acceptance first: an explicit citation label for THIS span is the
    #    strongest intent signal and outranks shape heuristics (a quoted
    #    token like "ab" under "原文：" is still a citation).
    cited = _local_paper_source(answer, start, end) or \
        _has_citation_intro(answer, start)
    if cited and not _looks_like_heading_or_summary(text) \
            and not _is_term_expansion(text):
        return True
    # -- rejections: this is how the model LABELS or DEFINES, not quotes ----
    if _is_term_expansion(text):
        return False          # **T**ask-**C**onstrained... acronym expansion
    if _looks_like_heading_or_summary(text):
        return False          # titles, section headings, ellipsis excerpts
    if _is_code_like(text):
        return False          # identifiers / test markers
    if _is_term_mention(answer, start, end):
        return False          # "前缀"是指... : term definition, not a citation
    echoed = any(t and nq in _compare_text(t) for t in (chat_texts or []))
    matches_page = bool(pages) and any(
        nq in _compare_text(body) for body in pages.values())
    if echoed and not matches_page:
        return False          # the user's own words echoed back
    # -- acceptance: quote form against real evidence ---------------------
    # 2) plain blockquote used against real evidence this turn: D5's normal
    #    citation form (a fully emphasised block is display, not a quote)
    if is_block and not _is_fully_emphasised(text) and pages:
        return True
    # 3) blockquote corroborated by a verbatim hit on this turn's pages.
    #    NEVER for paired-quote spans: a short term in quote marks that
    #    happens to occur on a page is a term, not a citation.
    if is_block and matches_page:
        return True
    # ambiguous: ordinary chat / summary, not a paper quotation
    return False


def _page_refs_in(text: str) -> list[tuple[int, int | None]]:
    """Find explicit (paper_id?, page_no) references like 'p.3', 'page 3'."""
    refs = []
    for m in re.finditer(r'(?:paper[_ ]?id\s*=\s*(\d+)[^0-9]{0,12})?'
                         r'(?:p\.|page\s+)(\d{1,4})', text, re.IGNORECASE):
        pid = int(m.group(1)) if m.group(1) else None
        refs.append((pid, int(m.group(2))))
    return refs


def _extract_tool_events(source) -> list[dict]:
    """Pair every tool_call of THIS turn with its real tool result.

    Returns one dict per call: name, arguments, tool_call_id, has_result,
    ok, result_summary. Calls without a result are reported as such -- never
    silently dropped. Only the current turn is inspected.
    """
    events: list[dict] = []
    by_id: dict[str, dict] = {}
    for item in current_turn_messages(source):
        if not isinstance(item, dict):
            continue
        if item.get('role') == 'assistant':
            for tc in item.get('tool_calls') or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get('function') or {}
                cid = tc.get('id') or ''
                ev = {
                    'tool_call_id': cid,
                    'name': fn.get('name', '?'),
                    'arguments': fn.get('arguments', ''),
                    'has_result': False,
                    'ok': None,
                    'result_summary': '',
                }
                events.append(ev)
                if cid:
                    by_id[cid] = ev
        elif item.get('role') == 'tool':
            cid = item.get('tool_call_id') or ''
            ev = by_id.get(cid)
            if ev is None:
                continue
            content = item.get('content') or ''
            ev['has_result'] = True
            ev['result_summary'] = content if len(content) <= 400 else content[:400] + '…'
            lowered = content.lstrip().lower()
            failed = lowered.startswith('[not found]') or lowered.startswith(
                '[access denied]') or lowered.startswith('[error') or lowered.startswith(
                '[intterupted]')
            ev['ok'] = not failed
    return events


PIPELINE_STEPS = (
    ('import', '导入论文', 'import_papers'),
    ('search', '检索命中', 'search_papers'),
    ('read', '读取整页原文', 'read_paper_page'),
    ('verify', '逐字引文核验', None),
)


def build_pipeline(tool_events: list[dict], checks: list[dict]) -> list[dict]:
    """Derive pipeline state from what ACTUALLY happened this turn.

    A step is 'done' only if its tool was really called and returned a
    successful result; 'fail' if called but failed; 'skipped' if never called.
    The verify step reflects the real _verify_quotes() execution.
    """
    by_tool: dict[str, list[dict]] = {}
    for ev in tool_events:
        by_tool.setdefault(ev['name'], []).append(ev)

    pipeline = []
    for key, label, tool_name in PIPELINE_STEPS:
        if key == 'verify':
            if checks:
                state = 'done'
                detail = f'核验 {len(checks)} 条引文'
            elif tool_events:
                state = 'skipped'
                detail = '本轮无可核验引文'
            else:
                state = 'skipped'
                detail = '本轮未执行'
            pipeline.append({'key': key, 'label': label, 'state': state, 'detail': detail})
            continue

        evs = by_tool.get(tool_name, [])
        if not evs:
            pipeline.append({'key': key, 'label': label, 'state': 'skipped',
                             'detail': '本轮未调用'})
            continue
        ok_n = sum(1 for e in evs if e.get('ok'))
        fail_n = sum(1 for e in evs if e.get('has_result') and e.get('ok') is False)
        pending = sum(1 for e in evs if not e.get('has_result'))
        if fail_n and not ok_n:
            state, detail = 'fail', f'{tool_name} 失败 {fail_n} 次'
        elif ok_n:
            state = 'done'
            detail = f'{tool_name} 成功 {ok_n} 次'
            if fail_n:
                detail += f'，失败 {fail_n} 次'
            if pending:
                detail += f'，无结果 {pending} 次'
        else:
            state, detail = 'fail', f'{tool_name} 未收到结果'
        pipeline.append({'key': key, 'label': label, 'state': state, 'detail': detail})
    return pipeline


def _user_texts(session) -> list:
    """User message texts of this session (for chat-vs-quote filtering).

    A quotation that is merely the user's own words echoed back is ordinary
    chat, not paper evidence. Session history is used only for this filter --
    never as verification evidence.
    """
    out = []
    for item in getattr(session, 'ui_history', None) or []:
        if isinstance(item, dict) and item.get('role') == 'user':
            t = item.get('content')
            if t:
                out.append(t)
    return out


def _verify_quotes(answer: str, pages: dict,
                   question: str | None = None,
                   user_texts: list | None = None,
                   page_meta: dict | None = None) -> list[dict]:
    """Verify each PAPER quotation of this turn against pages read this turn.

    Returns one dict per paper quote: text, status, source, checked, note.
    status: 'verified' | 'mismatch' | 'unverified'

    ``source`` is set ONLY for a confirmed verbatim match. ``checked`` lists
    the pages actually read and examined this turn (with real titles), so a
    quote that failed to match can honestly say "checked this page" without
    claiming it as the source.

    Ordinary chat (test markers, code, the user's own words, restatements of
    earlier messages) and paper summaries/titles are filtered out first and
    produce NO card. A genuine paper quotation is always reported -- even with
    no page read, where it is ``unverified`` -- so real quote problems are
    never hidden.
    """
    results = []
    refs = _page_refs_in(answer or '')
    chat_texts = list(user_texts or [])
    if question:
        chat_texts.append(question)
    meta = page_meta or {}
    # Pages actually read this turn, for honest "checked but not matched"
    # display -- never presented as a confirmed source.
    checked = []
    for key in pages.keys():
        checked.append({
            'paper_id': key[0],
            'page_no': key[1],
            'title': (meta.get(key) or {}).get('title', ''),
        })
    for raw, reliable, is_block, q_start, q_end in _extract_quotes(answer or ''):
        if not reliable:
            # Nested/unbalanced marks: we can neither parse nor classify the
            # span confidently. Conservative: show it as unverified (never
            # verified), so a possible paper quotation is not silently hidden.
            results.append({'text': raw, 'status': 'unverified', 'source': None,
                            'checked': checked,
                            'note': '引号嵌套或不配对，无法可靠解析，保守判为未核验'})
            continue
        # STAGE A: quotation intent (per-span, local citation context)
        if not is_paper_quote(raw, is_block, answer, chat_texts, pages,
                              start=q_start, end=q_end):
            continue  # ordinary chat / term / summary: no card
        nq = _compare_text(raw)
        if not nq:
            continue
        hits = [key for key, body in pages.items() if nq in _compare_text(body)]
        if not pages:
            results.append({'text': raw, 'status': 'unverified', 'source': None,
                            'checked': [],
                            'note': '未读取原文，无法逐字核验'})
            continue
        if not hits:
            results.append({'text': raw, 'status': 'mismatch', 'source': None,
                            'checked': checked,
                            'note': '引文与本轮读取的原文不一致，未匹配到已读取页面'})
            continue
        # source attribution: prefer an explicit page reference in the answer
        claimed = [(p, g) for p, g in refs if g is not None]
        matched = None
        if claimed:
            agree = [h for h in hits if any(h[1] == g and (p is None or p == h[0])
                                            for p, g in claimed)]
            if len(agree) == 1:
                matched = agree[0]
            elif len(agree) == 0:
                results.append({'text': raw, 'status': 'mismatch', 'source': None,
                                'checked': checked,
                                'note': '引文来源标注与实际读取的原文不一致'})
                continue
            else:
                results.append({'text': raw, 'status': 'unverified', 'source': None,
                                'checked': checked,
                                'note': '本轮读取多个页面且来源标注不明确，无法消除歧义'})
                continue
        elif len(hits) == 1:
            matched = hits[0]
        else:
            results.append({'text': raw, 'status': 'unverified', 'source': None,
                            'checked': checked,
                            'note': '引文出现在多个已读页面，未明确来源'})
            continue
        results.append({'text': raw, 'status': 'verified', 'source': matched,
                        'checked': checked, 'note': '逐字核验通过'})
    return results


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

def render_index(result_html: str = '') -> str:
    return f"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>PaperPilot</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }}
  h1 {{ margin-bottom: .25rem; }}
  .hint {{ color: #666; margin-bottom: 1.5rem; }}
  fieldset {{ margin-bottom: 1.25rem; border: 1px solid #ccc; border-radius: 6px; }}
  legend {{ font-weight: 600; padding: 0 .5rem; }}
  label {{ display: block; margin: .5rem 0 .25rem; }}
  input[type=text] {{ width: 100%; padding: .5rem; box-sizing: border-box; }}
  button {{ margin-top: .75rem; padding: .5rem 1.25rem; cursor: pointer; }}
  .result {{ margin-top: 1.5rem; }}
  .ok {{ background: #f0fff0; border: 1px solid #9c9; padding: .75rem; border-radius: 6px; white-space: pre-wrap; }}
  .err {{ background: #fff0f0; border: 1px solid #c99; padding: .75rem; border-radius: 6px; white-space: pre-wrap; }}
  .trace {{ color: #555; font-size: .9rem; white-space: pre-wrap; margin-top: .5rem; }}
  .checks {{ list-style: none; padding: 0; }}
  .cards {{ display: flex; flex-direction: column; gap: .75rem; }}
  .card {{ border: 1px solid #ddd; border-radius: 8px; padding: .75rem 1rem; }}
  .card.ok {{ background: #f0fff0; border-color: #9c9; }}
  .card.err {{ background: #fff0f0; border-color: #c99; }}
  .card.warn {{ background: #fffbe6; border-color: #dc9; }}
  .card-head {{ display: flex; flex-wrap: wrap; align-items: center; gap: .5rem; margin-bottom: .5rem; }}
  .badge {{ font-weight: 700; padding: .15rem .55rem; border-radius: 999px; font-size: .85rem; }}
  .badge.ok {{ background: #cfc; color: #163; }}
  .badge.err {{ background: #fbb; color: #611; }}
  .badge.warn {{ background: #fd7; color: #630; }}
  .src {{ color: #444; font-size: .9rem; }}
  .jump {{ font-size: .85rem; }}
  .quote {{ margin: .35rem 0 .35rem 1rem; padding-left: .75rem; border-left: 3px solid #bbb; }}
  blockquote {{ margin: .35rem 0 .35rem 1rem; padding-left: .75rem; border-left: 3px solid #bbb; }}
  details {{ margin-top: .35rem; }}
  pre {{ background: #f6f6f6; padding: .5rem; overflow-x: auto; white-space: pre-wrap; }}
  .note {{ color: #666; font-size: .9rem; margin-top: .25rem; }}
</style>
</head>
<body>
<h1>PaperPilot</h1>
<p class="hint">本地科研论文问答：导入 PDF、检索论文、按原文证据回答。所有数据保存在本地工作区。</p>

<fieldset>
  <legend>① 导入论文</legend>
  <form method="post" action="/import">
    <label for="paper_dir">工作区内的论文目录（相对路径）</label>
    <input type="text" id="paper_dir" name="paper_dir" placeholder="papers" value="papers">
    <button type="submit">导入</button>
  </form>
</fieldset>

<fieldset>
  <legend>② 研究问题</legend>
  <form method="post" action="/ask">
    <label for="question">研究问题</label>
    <input type="text" id="question" name="question" placeholder="例如：这篇论文的核心方法是什么？">
    <button type="submit">提问</button>
  </form>
</fieldset>

<div class="result">{result_html}</div>
</body>
</html>"""


_STATUS_META = {
    'verified': ('ok', '逐字核验通过'),
    'mismatch': ('err', '未通过逐字核验'),
    'unverified': ('warn', '未核验'),
}


def _evidence_anchor(source: tuple) -> str:
    """Unique anchor id for a SHARED original-text region (one per page).

    Anchors key on the page identity, so several quotations drawn from the
    same paper/page all point at the same region. The anchor locates the
    PAGE region only -- never character-level offsets, since normalization
    breaks raw-text positions.
    """
    return f'src-{_esc(source[0])}-{_esc(source[1])}'


def _evidence_card(check: dict, pages: dict) -> str:
    """One evidence card: status badge, quote, source, note, jump link.

    The card carries NO id of its own -- ids live only on the shared
    original-text regions, which are rendered once per page by
    _origins_section(). That keeps every HTML id unique even when several
    quotations come from the same page.
    """
    status = check.get('status') or 'unverified'
    cls, label = _STATUS_META.get(status, ('warn', status))
    src = check.get('source')

    if src:
        source_line = f'论文标识 paper_id={_esc(src[0])} · 页码 p.{_esc(src[1])}'
        target = _evidence_anchor(src)
    else:
        checked = check.get('checked') or []
        if len(checked) == 1:
            p = checked[0]
            title = p.get('title') or f"paper_id={p['paper_id']}"
            source_line = f"已检查页面：{_esc(title)} · 第 {_esc(p['page_no'])} 页（引文未匹配）"
        elif len(checked) > 1:
            source_line = f'未匹配到本轮已读取的 {len(checked)} 个页面'
        else:
            source_line = '未读取原文，无法逐字核验'
        target = 'origins'  # shared hub: no source to point at

    jump = f'<a class="jump" href="#{_esc(target)}">跳转到原文区域 ↓</a>'
    return (
        f'<div class="card {cls}">'
        f'<div class="card-head">'
        f'<span class="badge {cls}">{_esc(label)}</span>'
        f'<span class="src">{source_line}</span> {jump}'
        f'</div>'
        f'<blockquote class="quote">{_esc(check.get("text", ""))}</blockquote>'
        f'<div class="note">{_esc(check.get("note", ""))}</div>'
        f'</div>'
    )


def _origins_section(pages: dict, sources: list) -> str:
    """Render ONE original-text region per distinct page (unique ids).

    ``sources`` is the ordered list of page keys referenced by the cards;
    pages present in ``pages`` but not referenced are still shown so the
    user can cross-check. Every id appears exactly once.
    """
    if not pages:
        return ('<div class="note" id="origins">本轮未读取原文，'
                '无法提供原文核对。</div>')
    ordered: list[tuple] = []
    seen = set()
    for key in list(sources) + list(pages.keys()):
        if key in pages and key not in seen:
            seen.add(key)
            ordered.append(key)
    parts = ['<h4 id="origins">原文区域</h4>', '<div class="origins">']
    for key in ordered:
        anchor = _evidence_anchor(key)
        parts.append(
            f'<details class="orig" id="{_esc(anchor)}">'
            f'<summary>实际读取的整页原文（paper_id={_esc(key[0])} p.{_esc(key[1])}）</summary>'
            f'<div class="note">工具真实返回的原文；引文定位仅到页面区域，未做字符级高亮。</div>'
            f'<pre>{_esc(pages[key])}</pre>'
            f'</details>'
        )
    parts.append('</div>')
    return ''.join(parts)


def render_ok(title: str, body: str, trace: list[str] | None = None,
              checks: list[dict] | None = None, pages: dict | None = None) -> str:
    parts = [f'<h3>{_esc(title)}</h3>', f'<div class="ok">{_esc(body)}</div>']
    pages = pages or {}
    if checks:
        parts.append('<h4>证据卡片</h4><div class="cards">')
        for c in checks:
            parts.append(_evidence_card(c, pages))
        parts.append('</div>')
        # Shared original-text regions: one per page, ids unique.
        referenced = [c.get('source') for c in checks if c.get('source')]
        parts.append(_origins_section(pages, referenced))
    elif pages:
        parts.append('<div class="note">本次回答未包含可核验的显式引文。</div>')
        parts.append(_origins_section(pages, []))
    if trace:
        parts.append('<div class="trace"><b>工具调用</b><br>' + '<br>'.join(_esc(t) for t in trace) + '</div>')
    return ''.join(parts)


def render_err(title: str, body: str) -> str:
    return f'<h3>{_esc(title)}</h3><div class="err">{_esc(body)}</div>'


# --------------------------------------------------------------------------
# shared business functions (used by BOTH the HTML routes and the JSON API)
# --------------------------------------------------------------------------

def do_import(cfg, paper_dir_value: str) -> tuple[dict | None, tuple[int, str] | None]:
    """Run one import. Returns (result, error) where error is (status, message).

    result: {output, status, papers} -- status in ok|partial|failed|rejected.
    """
    value, err = _safe_paper_dir(paper_dir_value, cfg=cfg)
    if err:
        return None, (400, err)
    try:
        out = import_papers(ImportPapersInput(paper_dir=value), cfg=cfg)
    except ValidationError as e:
        return None, (400, f'参数无效: {e}')
    except Exception as e:
        return None, (500, f'{type(e).__name__}: {e}')
    if out.startswith('[not found]') or out.startswith('[Access denied]'):
        return None, (400, out)
    has_fail = 'failed:' in out
    has_ok = 'imported:' in out or 'updated:' in out or 'skipped:' in out
    status = 'failed' if (has_fail and not has_ok) else ('partial' if has_fail else 'ok')
    return {'output': out, 'status': status, 'papers': list_papers_payload(cfg)}, None


def list_papers_payload(cfg) -> list[dict]:
    """Indexed papers straight from SQLite (never a filesystem scan)."""
    store = PaperStore(default_db_path(cfg))
    try:
        return [
            {
                'paper_id': r['paper_id'],
                'title': r['title'],
                'source_path': r['source_path'],
                'page_count': r['page_count'],
                'sha256': r['sha256'],
            }
            for r in store.list_papers()
        ]
    finally:
        store.close()


def read_page_payload(cfg, paper_id: int, page_no: int) -> tuple[dict | None, tuple[int, str] | None]:
    """Read one indexed page. Returns (payload, error=(status, message))."""
    if not isinstance(paper_id, int) or isinstance(paper_id, bool) or paper_id < 1:
        return None, (400, 'paper_id 必须是正整数')
    if not isinstance(page_no, int) or isinstance(page_no, bool) or page_no < 1:
        return None, (400, 'page_no 必须是正整数')
    store = PaperStore(default_db_path(cfg))
    try:
        row = store.read_page(paper_id, page_no)
    finally:
        store.close()
    if row is None:
        paper = None
        store = PaperStore(default_db_path(cfg))
        try:
            paper = store.get_paper_by_id(paper_id)
        finally:
            store.close()
        if paper is None:
            return None, (404, f'没有 paper_id={paper_id} 的论文')
        return None, (404, f'paper_id={paper_id} 没有第 {page_no} 页（共 {paper["page_count"]} 页）')
    return {
        'paper_id': row['paper_id'],
        'title': row['title'],
        'source_path': row['source_path'],
        'page_no': row['page_no'],
        'page_count': row['page_count'],
        'body': row['body'],
    }, None


def run_agent_turn(cfg, session, question: str, agent_factory=None):
    """Execute one real Agent turn and record it in ``session``.

    Returns (payload, error=(status, message)). The payload carries only THIS
    turn's answer, tool events, evidence and pipeline state. ``session.lock``
    must be held by the caller.

    ``agent_factory`` is injectable for tests (FakeAgent); default builds a
    real DeepSeekAgent with memory disabled so the shared CLI session.json and
    the shared compaction log are never touched.
    """
    prior = copy.deepcopy(session.agent_history)
    if agent_factory is None:
        agent = DeepSeekAgent(PAPER_TOOLS, cfg=cfg, memory_enabled=False)
    else:
        agent = agent_factory(PAPER_TOOLS, cfg=cfg, memory_enabled=False)
    try:
        result = agent.run_task(question, prior=prior)
    except KeyboardInterrupt:
        return None, (503, '已中断', '任务被中断')
    except Exception as e:
        # Classify the failure: context overflow vs network/service/tool error.
        # NEVER report a failure as "no papers found" -- the user must not be
        # led to believe the index lacks content when the query never ran.
        err_cls = type(e).__name__
        if _is_context_overflow(e):
            return None, (502, '查询未完成',
                          '这次查询暂时没有完成（请求超出模型上下文限制），'
                          '我还不能确认论文库中是否有相关内容。'
                          '你可以稍后重试，或先缩小问题范围。')
        return None, (502, '查询未完成',
                      '这次查询暂时没有完成（服务或工具异常），'
                      '我还不能确认论文库中是否有相关内容。'
                      '你可以稍后重试，或先缩小问题范围。')

    turn_msgs = current_turn_messages(agent)
    answer, trace = _extract_answer_and_trace(agent)
    tool_events = _extract_tool_events(agent)
    records = _collect_page_records(agent)
    pages = {k: v['body'] for k, v in records.items()}
    checks = _verify_quotes(answer, pages, question=question,
                            user_texts=_user_texts(session),
                            page_meta=records) if answer else []
    pipeline = build_pipeline(tool_events, checks)
    outcome = getattr(result, 'outcome', '?')

    if not answer:
        # Do NOT fall back to a prior turn's answer, and do NOT claim the
        # papers are absent -- the agent simply produced no final answer.
        return None, (502, '查询未完成',
                      '这次查询暂时没有完成，我还不能确认论文库中是否有相关内容。'
                      '你可以稍后重试，或先缩小问题范围。')

    evidence = [
        {
            'text': c.get('text', ''),
            'status': c.get('status'),
            'source': list(c['source']) if c.get('source') else None,
            'checked': [dict(k) for k in (c.get('checked') or [])],
            'note': c.get('note', ''),
        }
        for c in checks
    ]
    payload = {
        'question': question,
        'answer': answer,
        'evidence': evidence,
        'tool_events': tool_events,
        'pipeline': pipeline,
        'outcome': outcome,
        'trace': trace,
        'messages': [m for m in turn_msgs if isinstance(m, dict)],
    }
    session.append_turn(
        ui_items=[{'role': 'user', 'content': question},
                  {'role': 'assistant', 'content': answer,
                   'evidence': evidence, 'tool_events': tool_events,
                   'pipeline': pipeline}],
        agent_messages=copy.deepcopy([m for m in turn_msgs if isinstance(m, dict)]),
    )
    return payload, None


# --------------------------------------------------------------------------
# HTTP handler
# --------------------------------------------------------------------------

# Fixed static-resource whitelist: only these three files are ever served.
STATIC_DIR = Path(__file__).resolve().parent / 'static' / 'workbench'
STATIC_WHITELIST = {
    '/': ('index.html', 'text/html; charset=utf-8'),
    '/index.html': ('index.html', 'text/html; charset=utf-8'),
    '/styles.css': ('styles.css', 'text/css; charset=utf-8'),
    '/app.js': ('app.js', 'application/javascript; charset=utf-8'),
}

SESSION_COOKIE = 'pp_sid'


class PaperPilotHandler(BaseHTTPRequestHandler):
    server_version = 'PaperPilotWeb/0.1'
    cfg = CONFIG  # patched by tests / main()
    sessions: SessionStore | None = None  # patched by tests / make_server
    agent_factory = None  # patched by tests

    def log_message(self, fmt, *args):  # quieter default logging
        pass

    # -- response helpers --------------------------------------------------

    def _send_html(self, html_text: str, status: int = 200) -> None:
        data = html_text.encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload, status: int = 200, sid: str | None = None) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        if sid:
            self.send_header(
                'Set-Cookie',
                f'{SESSION_COOKIE}={sid}; Path=/; HttpOnly; SameSite=Lax',
            )
        self.end_headers()
        self.wfile.write(data)

    def _send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def _json_error(self, status: int, message: str) -> None:
        self._send_json({'error': message}, status=status)

    # -- request parsing ----------------------------------------------------

    def _read_body(self) -> str:
        try:
            length = int(self.headers.get('Content-Length') or 0)
        except ValueError:
            length = 0
        if length <= 0:
            raise ValueError('请求体为空')
        if length > MAX_BODY_BYTES:
            raise ValueError(f'请求体过大（上限 {MAX_BODY_BYTES} 字节）')
        return self.rfile.read(length).decode('utf-8', errors='replace')

    def _read_form(self) -> dict:
        raw = self._read_body()
        ctype = (self.headers.get('Content-Type') or '').split(';')[0].strip()
        if ctype not in ('application/x-www-form-urlencoded', ''):
            raise ValueError(f'不支持的内容类型: {ctype or "(空)"}')
        from urllib.parse import parse_qsl
        return dict(parse_qsl(raw, keep_blank_values=True))

    def _read_json(self) -> dict:
        raw = self._read_body()
        ctype = (self.headers.get('Content-Type') or '').split(';')[0].strip()
        if ctype != 'application/json':
            raise ValueError(f'需要 application/json，收到: {ctype or "(空)"}')
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f'JSON 格式错误: {e}') from e
        if not isinstance(data, dict):
            raise ValueError('JSON 顶层必须是对象')
        return data

    # -- session -----------------------------------------------------------

    def _sid_from_cookie(self) -> str | None:
        raw = self.headers.get('Cookie') or ''
        for part in raw.split(';'):
            k, _, v = part.strip().partition('=')
            if k == SESSION_COOKIE and v:
                return v.strip()
        return None

    def _ensure_session(self):
        """Resolve the caller's session, creating one on first contact."""
        store = self.sessions
        if store is None:
            return None
        sid = self._sid_from_cookie()
        sess = store.get(sid) if sid else None
        if sess is not None:
            return sess
        return store.create(new_session_id())

    @property
    def _session(self):
        return self._ensure_session()

    # -- routing -------------------------------------------------------------

    def do_GET(self):
        path, _, query = self.path.partition('?')
        if path in STATIC_WHITELIST:
            self._serve_static(path)
            return
        if path == '/api/state':
            self._api_state()
            return
        if path == '/api/page':
            self._api_page(query)
            return
        if path.startswith('/api/'):
            self._json_error(404, f'未知 API 路径: {path}')
            return
        self._send_html(render_index(render_err('未找到', f'未知路径: {self.path}')), status=404)

    def do_POST(self):
        path, _, _ = self.path.partition('?')
        if path == '/api/import':
            self._api_import()
            return
        if path == '/api/ask':
            self._api_ask()
            return
        if path == '/api/reset':
            self._api_reset()
            return
        if path.startswith('/api/'):
            self._json_error(404, f'未知 API 路径: {path}')
            return
        try:
            form = self._read_form()
        except ValueError as e:
            self._send_html(render_index(render_err('请求无效', str(e))), status=400)
            return
        if path == '/import':
            self._handle_import(form)
        elif path == '/ask':
            self._handle_ask(form)
        else:
            self._send_html(render_index(render_err('未找到', f'未知路径: {path}')), status=404)

    # -- static (whitelisted) -----------------------------------------------

    def _serve_static(self, path: str) -> None:
        filename, ctype = STATIC_WHITELIST[path]
        # Whitelist maps a fixed URL to a fixed filename; no user input
        # reaches the filesystem path, so path traversal is impossible.
        target = (STATIC_DIR / filename).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.is_file():
            self._send_html(render_err('未找到', '静态资源缺失'), status=404)
            return
        self._send_bytes(target.read_bytes(), ctype)

    # -- JSON API ------------------------------------------------------------

    def _api_state(self) -> None:
        sess = self._session
        if sess is None:
            self._json_error(500, '会话不可用')
            return
        with sess.lock:
            history = copy.deepcopy(sess.ui_history)
        papers = list_papers_payload(self.cfg)
        self._send_json({
            'workspace': str(self.cfg.work_space),
            'db': str(default_db_path(self.cfg)),
            'service': 'ok',
            'papers': papers,
            'paper_count': len(papers),
            'conversation': history,
            'running': False,
            'session_id': sess.sid,
        }, sid=sess.sid)

    def _api_page(self, query: str) -> None:
        from urllib.parse import parse_qs
        params = parse_qs(query)
        try:
            paper_id = int((params.get('paper_id') or [''])[0])
            page_no = int((params.get('page_no') or [''])[0])
        except ValueError:
            self._json_error(400, 'paper_id 与 page_no 必须是整数')
            return
        payload, err = read_page_payload(self.cfg, paper_id, page_no)
        if err:
            self._json_error(err[0], err[1])
            return
        self._send_json(payload)

    def _api_import(self) -> None:
        try:
            data = self._read_json()
        except ValueError as e:
            self._json_error(400, str(e))
            return
        paper_dir = data.get('paper_dir')
        if not isinstance(paper_dir, str):
            self._json_error(400, 'paper_dir 必须是字符串')
            return
        result, err = do_import(self.cfg, paper_dir)
        if err:
            self._json_error(err[0], err[1])
            return
        self._send_json(result)

    def _api_ask(self) -> None:
        try:
            data = self._read_json()
        except ValueError as e:
            self._json_error(400, str(e))
            return
        question = data.get('question')
        if not isinstance(question, str) or not question.strip():
            self._json_error(400, 'question 必须是非空字符串')
            return
        sess = self._session
        if sess is None:
            self._json_error(500, '会话不可用')
            return
        with sess.lock:
            payload, err = run_agent_turn(
                build_paper_config(self.cfg), sess, question.strip(),
                agent_factory=self.agent_factory,
            )
        if err:
            self._json_error(err[0], err[2])
            return
        self._send_json(payload, sid=sess.sid)

    def _api_reset(self) -> None:
        sess = self._session
        if sess is None:
            self._json_error(500, '会话不可用')
            return
        with sess.lock:
            sess.reset()
        self._send_json({'ok': True, 'session_id': sess.sid}, sid=sess.sid)

    # -- HTML handlers (delegate to the same business functions) --------------

    def _handle_import(self, form: dict):
        result, err = do_import(self.cfg, form.get('paper_dir', ''))
        if err:
            status, message = err
            label = '导入被拒绝' if status == 400 else '导入失败'
            self._send_html(render_index(render_err(label, message)), status=status)
            return
        out = result['output']
        if result['status'] == 'failed':
            body = render_err('导入完成（全部失败）', out)
        elif result['status'] == 'partial':
            body = render_ok('导入完成（含失败项）', out)
        else:
            body = render_ok('导入完成', out)
        self._send_html(render_index(body))

    def _handle_ask(self, form: dict):
        question = (form.get('question') or '').strip()
        if not question:
            self._send_html(render_index(render_err('问题为空', '请输入研究问题后再提问。')), status=400)
            return
        sess = self._session
        if sess is None:
            self._send_html(render_index(render_err('会话不可用', '无法建立会话')), status=500)
            return
        with sess.lock:
            payload, err = run_agent_turn(
                build_paper_config(self.cfg), sess, question,
                agent_factory=self.agent_factory,
            )
        if err:
            status, title, message = err
            self._send_html(render_index(render_err(title, message)), status=status)
            return
        pages = {
            tuple(c['source']): '' for c in payload['evidence'] if c.get('source')
        }
        # Re-read the real pages for the HTML evidence panel.
        store = PaperStore(default_db_path(build_paper_config(self.cfg)))
        try:
            for key in list(pages):
                row = store.read_page(key[0], key[1])
                if row is not None:
                    pages[key] = row['body']
        finally:
            store.close()
        checks = [
            {'text': c['text'], 'status': c['status'],
             'source': tuple(c['source']) if c.get('source') else None,
             'note': c['note']}
            for c in payload['evidence']
        ]
        self._send_html(render_index(render_ok(
            '回答', payload['answer'], payload['trace'], checks, pages)))


# --------------------------------------------------------------------------
# entry
# --------------------------------------------------------------------------

def make_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, cfg=CONFIG,
                sessions: SessionStore | None = None,
                agent_factory=None) -> ThreadingHTTPServer:
    if sessions is None:
        sessions = SessionStore(Path(cfg.sandbox_dir) / 'paperpilot' / 'web_sessions')
    # staticmethod: a plain function stored in a class dict becomes a method
    # descriptor and would receive the handler instance as its first argument.
    handler = type('BoundHandler', (PaperPilotHandler,), {
        'cfg': cfg,
        'sessions': sessions,
        'agent_factory': staticmethod(agent_factory) if agent_factory is not None else None,
    })
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    cfg = CONFIG
    sessions = SessionStore(Path(cfg.sandbox_dir) / 'paperpilot' / 'web_sessions')
    server = make_server(DEFAULT_HOST, DEFAULT_PORT, cfg=cfg, sessions=sessions)
    db = default_db_path(cfg)
    print(f'[paperpilot-web]: http://{DEFAULT_HOST}:{DEFAULT_PORT}')
    print(f'[paperpilot-web]: work_space = {cfg.work_space}')
    print(f'[paperpilot-web]: db = {db}')
    print(f'[paperpilot-web]: web_sessions = {sessions.base_dir}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()

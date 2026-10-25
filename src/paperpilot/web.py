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

import html
import json
import re
import threading
import unicodedata
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from pydantic import ValidationError

from mini_harness.agent import DeepSeekAgent
from mini_harness.config import CONFIG

from paperpilot.cli import build_paper_config
from paperpilot.segment import is_cjk
from paperpilot.store import default_db_path
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


def _extract_answer_and_trace(agent) -> tuple[str, list[str]]:
    """Pull the final assistant text and a readable tool trace from messages.

    ``message`` entries are dicts in the current agent loop, but tolerate
    attribute-style objects as well.
    """
    answer = ''
    trace: list[str] = []
    for item in getattr(agent, 'message', []) or []:
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
    word (概\\n率 -> 概率) and is dropped. Any other wrap is a word boundary
    (low\\nrank -> low rank) and becomes a single space, so it can never
    glue two words together.
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


def _collect_read_pages(agent) -> dict:
    """Map (paper_id, page_no) -> page body for pages read successfully this run.

    Associates each ``read_paper_page`` tool_call with its ``role='tool'``
    result via tool_call_id. search_papers snippets are never collected.
    """
    messages = getattr(agent, 'message', []) or []
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
    pages: dict[tuple[int, int], str] = {}
    for cid, key in call_args.items():
        body = _parse_page_body(call_results.get(cid, ''))
        if body is not None:
            pages[key] = body
    return pages


def _extract_quotes(answer: str) -> list[tuple[str, bool]]:
    """Extract explicitly marked quotations. Returns (text, reliable).

    reliable=False marks quotes we cannot parse confidently (nested or
    unbalanced quote marks); those must never be reported as verified.
    Nested content is never silently truncated at the first closing mark.
    """
    found: list[tuple[str, bool]] = []
    # 1) markdown blockquote runs: consecutive "> " lines form one quote
    block: list[str] = []
    for line in (answer or '').splitlines():
        stripped = line.strip()
        if stripped.startswith('>'):
            block.append(stripped.lstrip('>').strip())
        elif block:
            found.append(('\n'.join(b for b in block if b), True))
            block = []
    if block:
        found.append(('\n'.join(b for b in block if b), True))
    # 2) paired quote marks (no apostrophes -- they are not quote boundaries)
    for left, right in _QUOTE_PAIRS:
        start = 0
        while True:
            i = (answer or '').find(left, start)
            if i < 0:
                break
            j = (answer or '').find(right, i + len(left))
            if j < 0:
                break
            inner = (answer or '')[i + len(left):j]
            if inner.strip():
                # nested/unbalanced: the closing mark we found may belong to
                # an inner quote -- do not pretend the span is a clean quote
                nested = left in inner or (right in inner and right != left)
                found.append((inner, not nested))
            start = j + len(right)
    # de-duplicate: drop a quote whose normalized text is a substring of
    # another extracted quote (outer/inner pairs must not double-count)
    kept: list[tuple[str, bool]] = []
    keys = [_normalize_quote(t) for t, _ in found]
    for idx, (text, reliable) in enumerate(found):
        key = keys[idx]
        if not key:
            continue
        if any(j != idx and keys[j] and key in keys[j] and len(keys[j]) > len(key)
               for j in range(len(found))):
            continue  # contained in a longer quote -- not a separate one
        kept.append((text, reliable))
    return kept


def _page_refs_in(text: str) -> list[tuple[int, int | None]]:
    """Find explicit (paper_id?, page_no) references like 'p.3', 'page 3'."""
    refs = []
    for m in re.finditer(r'(?:paper[_ ]?id\s*=\s*(\d+)[^0-9]{0,12})?'
                         r'(?:p\.|page\s+)(\d{1,4})', text, re.IGNORECASE):
        pid = int(m.group(1)) if m.group(1) else None
        refs.append((pid, int(m.group(2))))
    return refs


def _verify_quotes(answer: str, pages: dict) -> list[dict]:
    """Verify each marked quotation against pages read this run.

    Returns one dict per quote: text, status, source, note.
    status: 'verified' | 'mismatch' | 'unverified'
    """
    results = []
    refs = _page_refs_in(answer or '')
    for raw, reliable in _extract_quotes(answer or ''):
        if not reliable:
            results.append({'text': raw, 'status': 'unverified', 'source': None,
                            'note': '引号嵌套或不配对，无法可靠解析，保守判为未核验'})
            continue
        nq = _normalize_quote(raw)
        if not nq:
            continue
        hits = [key for key, body in pages.items() if nq in _normalize_quote(body)]
        if not pages:
            results.append({'text': raw, 'status': 'unverified', 'source': None,
                            'note': '未读取整页原文，引文未核验'})
            continue
        if not hits:
            results.append({'text': raw, 'status': 'mismatch', 'source': None,
                            'note': '引文与本轮读取的整页原文不一致'})
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
                                'note': '引文来源标注与实际读取的原文不一致'})
                continue
            else:
                results.append({'text': raw, 'status': 'unverified', 'source': None,
                                'note': '本轮读取多个页面且来源标注不明确，无法消除歧义'})
                continue
        elif len(hits) == 1:
            matched = hits[0]
        else:
            results.append({'text': raw, 'status': 'unverified', 'source': None,
                            'note': '引文出现在多个已读页面，未明确来源'})
            continue
        results.append({'text': raw, 'status': 'verified', 'source': matched,
                        'note': '逐字核验通过'})
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
  .checks li {{ margin: .5rem 0; padding: .5rem; border-radius: 6px; border: 1px solid #ddd; }}
  .checks li.ok {{ background: #f0fff0; border-color: #9c9; }}
  .checks li.err {{ background: #fff0f0; border-color: #c99; }}
  .checks li.warn {{ background: #fffbe6; border-color: #dc9; }}
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


def render_ok(title: str, body: str, trace: list[str] | None = None,
              checks: list[dict] | None = None, pages: dict | None = None) -> str:
    parts = [f'<h3>{_esc(title)}</h3>', f'<div class="ok">{_esc(body)}</div>']
    if checks:
        parts.append('<h4>引文逐字核验</h4><ul class="checks">')
        for c in checks:
            status = c.get('status')
            cls = {'verified': 'ok', 'mismatch': 'err'}.get(status, 'warn')
            label = {'verified': '逐字核验通过', 'mismatch': '未通过逐字核验',
                     'unverified': '未核验'}.get(status, status)
            src = c.get('source')
            src_txt = f'（来源：paper_id={src[0]} p.{src[1]}）' if src else ''
            parts.append(
                f'<li class="{cls}"><b>{_esc(label)}</b>{_esc(src_txt)}<br>'
                f'<blockquote>{_esc(c.get("text", ""))}</blockquote>'
                f'<div class="note">{_esc(c.get("note", ""))}</div>'
            )
            if src and pages and src in pages:
                parts.append(
                    '<details><summary>核对实际读取的整页原文</summary>'
                    f'<pre>{_esc(pages[src])}</pre></details>'
                )
            elif pages and status != 'verified':
                for key, body_text in pages.items():
                    parts.append(
                        f'<details><summary>实际读取的页面 paper_id={_esc(key[0])} '
                        f'p.{_esc(key[1])} 原文</summary><pre>{_esc(body_text)}</pre></details>'
                    )
                    break
            parts.append('</li>')
        parts.append('</ul>')
    elif pages:
        parts.append('<div class="note">本次回答未包含可核验的显式引文。</div>')
    if trace:
        parts.append('<div class="trace"><b>工具调用</b><br>' + '<br>'.join(_esc(t) for t in trace) + '</div>')
    return ''.join(parts)


def render_err(title: str, body: str) -> str:
    return f'<h3>{_esc(title)}</h3><div class="err">{_esc(body)}</div>'


# --------------------------------------------------------------------------
# HTTP handler
# --------------------------------------------------------------------------

class PaperPilotHandler(BaseHTTPRequestHandler):
    server_version = 'PaperPilotWeb/0.1'
    cfg = CONFIG  # patched by tests / main()

    def log_message(self, fmt, *args):  # quieter default logging
        pass

    def _send_html(self, html_text: str, status: int = 200) -> None:
        data = html_text.encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def _read_form(self) -> dict:
        try:
            length = int(self.headers.get('Content-Length') or 0)
        except ValueError:
            length = 0
        if length <= 0:
            raise ValueError('请求体为空')
        if length > MAX_BODY_BYTES:
            raise ValueError(f'请求体过大（上限 {MAX_BODY_BYTES} 字节）')
        raw = self.rfile.read(length).decode('utf-8', errors='replace')
        ctype = (self.headers.get('Content-Type') or '').split(';')[0].strip()
        if ctype not in ('application/x-www-form-urlencoded', ''):
            raise ValueError(f'不支持的内容类型: {ctype or "(空)"}')
        from urllib.parse import parse_qsl
        return dict(parse_qsl(raw, keep_blank_values=True))

    def do_GET(self):
        if self.path.split('?', 1)[0] in ('/', '/index.html'):
            self._send_html(render_index())
            return
        self._send_html(render_index(render_err('未找到', f'未知路径: {self.path}')), status=404)

    def do_POST(self):
        path = self.path.split('?', 1)[0]
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

    # -- handlers ----------------------------------------------------------

    def _handle_import(self, form: dict):
        cfg = self.cfg
        value, err = _safe_paper_dir(form.get('paper_dir', ''), cfg=cfg)
        if err:
            self._send_html(render_index(render_err('导入被拒绝', err)), status=400)
            return
        try:
            out = import_papers(ImportPapersInput(paper_dir=value), cfg=cfg)
        except ValidationError as e:
            self._send_html(render_index(render_err('参数无效', str(e))), status=400)
            return
        except Exception as e:  # tool layer reports most cases as text; this is the net
            self._send_html(render_index(render_err('导入失败', f'{type(e).__name__}: {e}')), status=500)
            return
        if out.startswith('[not found]') or out.startswith('[Access denied]'):
            self._send_html(render_index(render_err('导入被拒绝', out)), status=400)
            return
        has_fail = 'failed:' in out
        has_ok = 'imported:' in out or 'updated:' in out or 'skipped:' in out
        if has_fail and not has_ok:
            body = render_err('导入完成（全部失败）', out)
        elif has_fail:
            body = render_ok('导入完成（含失败项）', out)
        else:
            body = render_ok('导入完成', out)
        self._send_html(render_index(body))

    def _handle_ask(self, form: dict):
        question = (form.get('question') or '').strip()
        if not question:
            self._send_html(render_index(render_err('问题为空', '请输入研究问题后再提问。')), status=400)
            return
        cfg = build_paper_config(self.cfg)
        try:
            agent = DeepSeekAgent(PAPER_TOOLS, cfg=cfg)
            result = agent.run_task(question)
        except KeyboardInterrupt:
            self._send_html(render_index(render_err('已中断', '任务被中断。')), status=503)
            return
        except Exception as e:
            self._send_html(render_index(render_err('模型调用失败', f'{type(e).__name__}: {e}')), status=502)
            return

        answer, trace = _extract_answer_and_trace(agent)
        if not answer:
            note = f'（outcome={getattr(result, "outcome", "?")}）'
            self._send_html(render_index(render_err(
                '没有可用回答', f'Agent 未返回最终回答 {note}。未生成任何答案或证据。')), status=502)
            return
        pages = _collect_read_pages(agent)
        checks = _verify_quotes(answer, pages)
        self._send_html(render_index(render_ok('回答', answer, trace, checks, pages)))


# --------------------------------------------------------------------------
# entry
# --------------------------------------------------------------------------

def make_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, cfg=CONFIG) -> ThreadingHTTPServer:
    handler = type('BoundHandler', (PaperPilotHandler,), {'cfg': cfg})
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    cfg = CONFIG
    server = make_server(DEFAULT_HOST, DEFAULT_PORT, cfg=cfg)
    db = default_db_path(cfg)
    print(f'[paperpilot-web]: http://{DEFAULT_HOST}:{DEFAULT_PORT}')
    print(f'[paperpilot-web]: work_space = {cfg.work_space}')
    print(f'[paperpilot-web]: db = {db}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()

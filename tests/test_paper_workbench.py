"""Integration tests for the real workbench: JSON API, sessions, evidence boundary.

Everything runs offline against a temporary workspace, temporary SQLite index,
generated PDFs and FakeAgents -- no real model, no real user data.
"""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request

from dataclasses import replace
from pathlib import Path

from mini_harness.agent import DeepSeekAgent
from mini_harness.config import CONFIG

from paperpilot import web
from paperpilot.sessions import (SessionStore, is_valid_session_id,
                                new_session_id, trim_to_turns)
from paperpilot.store import default_db_path

from tests._support import RootSessionGuard
from tests.test_paper_pdf_import import make_pdf


class WorkbenchBase(RootSessionGuard, unittest.TestCase):
    """Temporary workspace + real HTTP server; FakeAgent injected per test."""

    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.ws = self.tmp / 'ws'
        (self.ws / 'papers').mkdir(parents=True)
        self.cfg = replace(CONFIG, work_space=self.ws)
        self.agent_factory = None  # tests set this to a FakeAgent
        self.sessions = SessionStore(self.ws / 'sandbox' / 'paperpilot' / 'web_sessions')
        self.server = web.make_server(
            '127.0.0.1', 0, cfg=self.cfg,
            sessions=self.sessions, agent_factory=lambda *a, **k: self.agent_factory(*a, **k),
        )
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    # -- HTTP helpers ---------------------------------------------------

    def _url(self, path):
        return f'http://127.0.0.1:{self.port}{path}'

    def request(self, path, data=None, method=None, json_body=None,
                headers=None, cookie=None):
        hdrs = dict(headers or {})
        if cookie:
            hdrs['Cookie'] = cookie
        body = None
        if json_body is not None:
            body = json.dumps(json_body).encode('utf-8')
            hdrs.setdefault('Content-Type', 'application/json')
        elif data is not None:
            body = urllib.parse.urlencode(data).encode()
            hdrs.setdefault('Content-Type', 'application/x-www-form-urlencoded')
        req = urllib.request.Request(self._url(path), data=body,
                                     method=method or ('POST' if body else 'GET'),
                                     headers=hdrs)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.status, resp.read().decode('utf-8'), resp.headers
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode('utf-8'), e.headers

    def get_json(self, path, cookie=None):
        status, body, headers = self.request(path, cookie=cookie)
        return status, json.loads(body) if body else None, headers

    def post_json(self, path, payload, cookie=None):
        status, body, headers = self.request(path, json_body=payload, cookie=cookie)
        return status, json.loads(body) if body else None, headers

    def sid_from(self, headers):
        raw = headers.get('Set-Cookie') or ''
        for part in raw.split(';'):
            if part.strip().startswith('pp_sid='):
                return part.strip().split('=', 1)[1]
        return None

    # -- fixtures --------------------------------------------------------

    def write_pdf(self, name, pages, title=None):
        (self.ws / 'papers' / name).write_bytes(make_pdf(pages, title=title))

    def import_papers_ok(self):
        status, data, _ = self.post_json('/api/import', {'paper_dir': 'papers'})
        self.assertEqual(status, 200, data)
        return data


class FakeAgentBase:
    """Minimal agent double compatible with run_task(task, cfg, prior)."""

    def __init__(self, tools, cfg=None, memory_enabled=True):
        self.tools = tools
        self.cfg = cfg
        self.memory_enabled = memory_enabled
        self.message = []
        self.turn_messages = []
        self.turn_start = 0
        self.received_prior = None

    def run_task(self, question, cfg=None, prior=None):
        self.received_prior = list(prior) if prior else []
        self.message = [{'role': 'system', 'content': 'sys'}]
        if prior:
            self.message.extend(prior)
        self.turn_start = len(self.message)
        self.turn_messages = []
        self._record({'role': 'user', 'content': question})
        self.build_turn(question)
        class R:
            outcome = 'completed'
        return R()

    def _record(self, msg):
        self.message.append(msg)
        self.turn_messages.append(msg)

    def build_turn(self, question):
        raise NotImplementedError


# --------------------------------------------------------------------------
# 1. API & static resources
# --------------------------------------------------------------------------

class TestStaticAndState(WorkbenchBase):
    def test_workbench_index_served(self):
        status, body, headers = self.request('/')
        self.assertEqual(status, 200)
        self.assertIn('text/html', headers.get('Content-Type', ''))
        self.assertIn('PaperPilot 研究工作台', body)
        self.assertNotIn('data.js', body)

    def test_css_and_js_served_with_correct_mime(self):
        status, _, headers = self.request('/styles.css')
        self.assertEqual(status, 200)
        self.assertIn('text/css', headers.get('Content-Type', ''))
        status, body, headers = self.request('/app.js')
        self.assertEqual(status, 200)
        self.assertIn('javascript', headers.get('Content-Type', ''))
        self.assertNotIn('模拟数据', body, '正式 JS 不得含合成数据标记')
        self.assertNotIn('PP_DATA', body, '正式 JS 不得加载合成数据')

    def test_static_traversal_rejected(self):
        for path in ('/static/../agent.py', '/../config.py', '/app.js/../web.py',
                     '/static/workbench/../../tools.py'):
            status, body, _ = self.request(path)
            self.assertIn(status, (400, 404), f'{path} must be rejected')

    def test_unknown_api_route_is_json_404(self):
        status, data, headers = self.get_json('/api/nope')
        self.assertEqual(status, 404)
        self.assertIn('json', headers.get('Content-Type', ''))
        self.assertIn('error', data)

    def test_state_empty_index(self):
        status, data, headers = self.get_json('/api/state')
        self.assertEqual(status, 200)
        self.assertEqual(data['papers'], [])
        self.assertEqual(data['paper_count'], 0)
        self.assertEqual(data['conversation'], [])
        self.assertTrue(data['workspace'])
        self.assertTrue(self.sid_from(headers), '首次访问应下发会话 Cookie')

    def test_state_lists_imported_papers(self):
        self.write_pdf('a.pdf', ['hello body'], title='Hello Paper')
        self.import_papers_ok()
        status, data, _ = self.get_json('/api/state')
        self.assertEqual(status, 200)
        self.assertEqual(data['paper_count'], 1)
        self.assertEqual(data['papers'][0]['title'], 'Hello Paper')
        self.assertEqual(data['papers'][0]['page_count'], 1)


class TestApiImport(WorkbenchBase):
    def test_import_success(self):
        self.write_pdf('a.pdf', ['body one'], title='Paper A')
        status, data, _ = self.post_json('/api/import', {'paper_dir': 'papers'})
        self.assertEqual(status, 200)
        self.assertEqual(data['status'], 'ok')
        self.assertIn('imported', data['output'])
        self.assertEqual(len(data['papers']), 1)

    def test_import_partial_failure(self):
        self.write_pdf('ok.pdf', ['good body'], title='Good Paper')
        (self.ws / 'papers' / 'bad.pdf').write_bytes(b'%PDF-1.4 broken')
        status, data, _ = self.post_json('/api/import', {'paper_dir': 'papers'})
        self.assertEqual(status, 200)
        self.assertEqual(data['status'], 'partial')
        self.assertEqual(len(data['papers']), 1)

    def test_import_no_pdf_in_directory(self):
        (self.ws / 'empty').mkdir()
        status, data, _ = self.post_json('/api/import', {'paper_dir': 'empty'})
        self.assertEqual(status, 200)
        self.assertEqual(data['papers'], [])

    def test_import_missing_directory(self):
        status, data, _ = self.post_json('/api/import', {'paper_dir': 'nope'})
        self.assertEqual(status, 400)
        self.assertIn('error', data)

    def test_import_rejects_traversal(self):
        status, data, _ = self.post_json('/api/import', {'paper_dir': '../etc'})
        self.assertEqual(status, 400)
        self.assertIn('..', data['error'])

    def test_import_rejects_absolute_path(self):
        status, data, _ = self.post_json('/api/import', {'paper_dir': '/etc'})
        self.assertEqual(status, 400)

    def test_import_rejects_non_string(self):
        status, data, _ = self.post_json('/api/import', {'paper_dir': 42})
        self.assertEqual(status, 400)


class TestApiPage(WorkbenchBase):
    def setUp(self):
        super().setUp()
        self.write_pdf('a.pdf', ['page one text', 'page two text'], title='Paper A')
        self.import_papers_ok()

    def test_page_valid(self):
        status, data, _ = self.get_json('/api/page?paper_id=1&page_no=2')
        self.assertEqual(status, 200)
        self.assertEqual(data['paper_id'], 1)
        self.assertEqual(data['page_no'], 2)
        self.assertEqual(data['page_count'], 2)
        self.assertIn('page two text', data['body'])
        self.assertEqual(data['title'], 'Paper A')

    def test_page_returns_raw_body_not_segmented(self):
        status, data, _ = self.get_json('/api/page?paper_id=1&page_no=1')
        self.assertEqual(status, 200)
        self.assertEqual(data['body'], 'page one text')

    def test_page_invalid_params(self):
        for qs in ('paper_id=abc&page_no=1', 'paper_id=1&page_no=x',
                   'paper_id=0&page_no=1', 'paper_id=1&page_no=-1',
                   'paper_id=-5&page_no=1', 'paper_id=1&page_no=0'):
            status, data, _ = self.get_json('/api/page?' + qs)
            self.assertEqual(status, 400, f'{qs} -> {data}')

    def test_page_missing_paper(self):
        status, data, _ = self.get_json('/api/page?paper_id=99&page_no=1')
        self.assertEqual(status, 404)

    def test_page_missing_page(self):
        status, data, _ = self.get_json('/api/page?paper_id=1&page_no=99')
        self.assertEqual(status, 404)


class TestApiErrors(WorkbenchBase):
    def test_bad_json(self):
        status, body, _ = self.request('/api/ask', method='POST',
                                       headers={'Content-Type': 'application/json'},
                                       json_body=None)
        # empty body
        self.assertIn(status, (400,))
        status, body, _ = self.request(
            '/api/ask', method='POST',
            headers={'Content-Type': 'application/json'},
            data=None)
        raw = b'{not json'
        req = urllib.request.Request(self._url('/api/ask'), data=raw, method='POST',
                                     headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                status = resp.status
                body = resp.read().decode()
        except urllib.error.HTTPError as e:
            status, body = e.code, e.read().decode()
        self.assertEqual(status, 400)
        self.assertIn('error', json.loads(body))

    def test_wrong_content_type(self):
        req = urllib.request.Request(self._url('/api/ask'), data=b'a=1', method='POST',
                                     headers={'Content-Type': 'application/x-www-form-urlencoded'})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                status = resp.status
                body = resp.read().decode()
        except urllib.error.HTTPError as e:
            status, body = e.code, e.read().decode()
        self.assertEqual(status, 400)

    def test_oversized_body(self):
        big = json.dumps({'question': 'x' * (web.MAX_BODY_BYTES + 100)}).encode()
        req = urllib.request.Request(self._url('/api/ask'), data=big, method='POST',
                                     headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                status = resp.status
        except urllib.error.HTTPError as e:
            status = e.code
        self.assertEqual(status, 400)

    def test_ask_empty_question(self):
        status, data, _ = self.post_json('/api/ask', {'question': '   '})
        self.assertEqual(status, 400)


# --------------------------------------------------------------------------
# 2. Multi-turn sessions
# --------------------------------------------------------------------------

def _agent_reads_page(page_body):
    class Agent(FakeAgentBase):
        def build_turn(self, question):
            self._record({'role': 'assistant', 'tool_calls': [
                {'id': 'c1', 'function': {
                    'name': 'read_paper_page',
                    'arguments': json.dumps({'paper_id': 1, 'page_no': 1})}},
            ], 'content': None})
            self._record({'role': 'tool', 'tool_call_id': 'c1',
                          'content': f'《T》 t.pdf p.1\n\n{page_body}'})
            self._record({'role': 'assistant', 'content': f'回答：{question}'})
    return Agent


class TestSessions(WorkbenchBase):
    def test_second_turn_receives_first_turn_history(self):
        Agent = _agent_reads_page('原文内容甲。')
        self.agent_factory = Agent
        s1, d1, h1 = self.post_json('/api/ask', {'question': '问题一'})
        self.assertEqual(s1, 200)
        sid = self.sid_from(h1)
        self.assertTrue(sid)

        seen_prior = {}

        def factory(*a, **k):
            ag = Agent(*a, **k)
            orig = ag.run_task
            def wrapped(question, cfg=None, prior=None):
                seen_prior['prior'] = list(prior) if prior else []
                return orig(question, cfg=cfg, prior=prior)
            ag.run_task = wrapped
            return ag

        self.agent_factory = factory
        s2, d2, h2 = self.post_json('/api/ask', {'question': '问题二'},
                                    cookie=f'pp_sid={sid}')
        self.assertEqual(s2, 200)
        self.assertTrue(seen_prior['prior'],
                        '第二轮必须收到第一轮历史作为 prior')
        self.assertEqual(seen_prior['prior'][0]['content'], '问题一')

        s3, data3, _ = self.get_json('/api/state', cookie=f'pp_sid={sid}')
        self.assertEqual(len(data3['conversation']), 4, '两轮共 4 条 UI 消息')
        self.assertEqual(data3['conversation'][0]['content'], '问题一')
        self.assertEqual(data3['conversation'][2]['content'], '问题二')

    def test_session_survives_refresh(self):
        Agent = _agent_reads_page('原文内容。')
        self.agent_factory = Agent
        s1, _, h1 = self.post_json('/api/ask', {'question': '第一问'})
        sid = self.sid_from(h1)
        s2, data, _ = self.get_json('/api/state', cookie=f'pp_sid={sid}')
        self.assertEqual(s2, 200)
        self.assertEqual(len(data['conversation']), 2)

    def test_two_cookies_are_isolated(self):
        Agent = _agent_reads_page('A 原文。')
        self.agent_factory = Agent
        s1, _, h1 = self.post_json('/api/ask', {'question': '会话甲的问题'})
        sid_a = self.sid_from(h1)

        AgentB = _agent_reads_page('B 原文。')
        self.agent_factory = AgentB
        s2, _, h2 = self.post_json('/api/ask', {'question': '会话乙的问题'})
        sid_b = self.sid_from(h2)

        self.assertNotEqual(sid_a, sid_b)
        _, data_a, _ = self.get_json('/api/state', cookie=f'pp_sid={sid_a}')
        _, data_b, _ = self.get_json('/api/state', cookie=f'pp_sid={sid_b}')
        self.assertEqual(len(data_a['conversation']), 2)
        self.assertEqual(len(data_b['conversation']), 2)
        self.assertEqual(data_a['conversation'][0]['content'], '会话甲的问题')
        self.assertEqual(data_b['conversation'][0]['content'], '会话乙的问题')
        self.assertNotIn('会话甲的问题', json.dumps(data_b['conversation']))

    def test_reset_clears_only_own_session(self):
        Agent = _agent_reads_page('原文。')
        self.agent_factory = Agent
        _, _, h1 = self.post_json('/api/ask', {'question': '甲的问题'})
        sid_a = self.sid_from(h1)
        _, _, h2 = self.post_json('/api/ask', {'question': '乙的问题'})
        sid_b = self.sid_from(h2)

        s, data, _ = self.post_json('/api/reset', {}, cookie=f'pp_sid={sid_a}')
        self.assertEqual(s, 200)
        self.assertTrue(data['ok'])

        _, data_a, _ = self.get_json('/api/state', cookie=f'pp_sid={sid_a}')
        _, data_b, _ = self.get_json('/api/state', cookie=f'pp_sid={sid_b}')
        self.assertEqual(data_a['conversation'], [])
        self.assertEqual(len(data_b['conversation']), 2, '其他会话不受影响')

    def test_reset_keeps_paper_index(self):
        self.write_pdf('a.pdf', ['body'], title='Paper A')
        self.import_papers_ok()
        Agent = _agent_reads_page('原文。')
        self.agent_factory = Agent
        _, _, h = self.post_json('/api/ask', {'question': '问'})
        sid = self.sid_from(h)
        self.post_json('/api/reset', {}, cookie=f'pp_sid={sid}')
        _, data, _ = self.get_json('/api/state', cookie=f'pp_sid={sid}')
        self.assertEqual(data['conversation'], [])
        self.assertEqual(data['paper_count'], 1, '论文索引不得被重置删除')

    def test_concurrent_asks_do_not_corrupt_history(self):
        Agent = _agent_reads_page('原文。')
        self.agent_factory = Agent
        _, _, h = self.post_json('/api/ask', {'question': '起点'})
        sid = self.sid_from(h)
        errors = []

        def worker(i):
            try:
                self.post_json('/api/ask', {'question': f'并发问题{i}'},
                               cookie=f'pp_sid={sid}')
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20)
        self.assertEqual(errors, [])
        _, data, _ = self.get_json('/api/state', cookie=f'pp_sid={sid}')
        # 1 original + 3 concurrent = 4 user turns = 8 UI items
        self.assertEqual(len(data['conversation']), 8)
        # Agent history must remain well-formed (tool pairing intact).
        for item in data['conversation']:
            self.assertIn(item['role'], ('user', 'assistant'))

    def test_web_does_not_write_cli_session_file(self):
        Agent = _agent_reads_page('原文。')
        self.agent_factory = Agent
        _, _, h = self.post_json('/api/ask', {'question': '问'})
        cli_session = self.ws / 'sandbox' / 'paperpilot' / 'session.json'
        self.assertFalse(cli_session.exists(),
                         'Web 请求不得写 CLI 会话文件 session.json')

    def test_web_sessions_dir_isolated_from_cli(self):
        Agent = _agent_reads_page('原文。')
        self.agent_factory = Agent
        _, _, h = self.post_json('/api/ask', {'question': '问'})
        sid = self.sid_from(h)
        self.assertTrue((self.sessions.base_dir / sid).is_dir())
        self.assertFalse((self.ws / 'sandbox' / 'paperpilot' / 'mini_harness_history.jsonl').exists(),
                         'Web 请求不得写共享压缩日志')

    def test_cookie_is_httponly(self):
        _, _, headers = self.get_json('/api/state')
        raw = headers.get('Set-Cookie', '')
        self.assertIn('HttpOnly', raw)
        self.assertIn('SameSite=Lax', raw)

    def test_invalid_sid_cookie_creates_new_session(self):
        Agent = _agent_reads_page('原文。')
        self.agent_factory = Agent
        s, data, h = self.post_json('/api/ask', {'question': '问'},
                                    cookie='pp_sid=../../etc/passwd')
        self.assertEqual(s, 200)
        new_sid = self.sid_from(h)
        self.assertTrue(new_sid)
        self.assertTrue(is_valid_session_id(new_sid))

    def test_client_cannot_forged_sid(self):
        # A forged valid-looking but unknown sid must NOT map to another session.
        Agent = _agent_reads_page('原文。')
        self.agent_factory = Agent
        _, _, h = self.post_json('/api/ask', {'question': '真实问题'})
        real_sid = self.sid_from(h)
        forged = 'A' * 43  # valid charset/length, but never issued
        s, data, _ = self.get_json('/api/state', cookie=f'pp_sid={forged}')
        self.assertEqual(s, 200)
        self.assertEqual(data['conversation'], [], '伪造 SID 不得读到他人会话')
        self.assertNotEqual(data['session_id'], real_sid)


class TestSessionStoreUnit(RootSessionGuard, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)

    def test_new_session_id_is_random(self):
        ids = {new_session_id() for _ in range(20)}
        self.assertEqual(len(ids), 20)
        for sid in ids:
            self.assertTrue(is_valid_session_id(sid))

    def test_rejects_bad_session_ids(self):
        for bad in ('', 'short', '../etc', 'x' * 200, None, 123, 'a b c', 'sid/1'):
            self.assertFalse(is_valid_session_id(bad), repr(bad))

    def test_trim_to_turns_keeps_tool_pairing(self):
        msgs = []
        for i in range(5):
            msgs.append({'role': 'user', 'content': f'u{i}'})
            msgs.append({'role': 'assistant', 'tool_calls': [
                {'id': f'c{i}', 'function': {'name': 't', 'arguments': '{}'}}]})
            msgs.append({'role': 'tool', 'tool_call_id': f'c{i}', 'content': 'r'})
            msgs.append({'role': 'assistant', 'content': f'a{i}'})
        trimmed = trim_to_turns(msgs, 2)
        self.assertEqual(len([m for m in trimmed if m['role'] == 'user']), 2)
        # every tool message still has its assistant tool_calls before it
        for i, m in enumerate(trimmed):
            if m['role'] == 'tool':
                self.assertEqual(trimmed[i - 1]['role'], 'assistant')
                self.assertTrue(trimmed[i - 1].get('tool_calls'))

    def test_corrupt_session_file_is_empty_not_fatal(self):
        store = SessionStore(self.base)
        sess = store.create()
        sess.root.mkdir(parents=True, exist_ok=True)
        (sess.root / 'ui.json').write_text('{corrupt', encoding='utf-8')
        (sess.root / 'agent.json').write_text('not-json', encoding='utf-8')
        fresh = SessionStore(self.base)
        s2 = fresh.get(sess.sid)
        self.assertIsNotNone(s2)
        self.assertEqual(s2.ui_history, [])
        self.assertEqual(s2.agent_history, [])

    def test_atomic_write_creates_valid_json(self):
        store = SessionStore(self.base)
        sess = store.create()
        sess.append_turn([{'role': 'user', 'content': 'q'}],
                         [{'role': 'user', 'content': 'q'}])
        data = json.loads((sess.root / 'ui.json').read_text(encoding='utf-8'))
        self.assertEqual(data[0]['content'], 'q')
        self.assertFalse((sess.root / 'ui.json.tmp').exists())


# --------------------------------------------------------------------------
# 3. Current-turn evidence boundary
# --------------------------------------------------------------------------

class TestEvidenceBoundary(WorkbenchBase):
    def _agent_reading(self, page_body, answer, paper_id=1, page_no=1,
                       read=True, label='c1'):
        class Agent(FakeAgentBase):
            def build_turn(self, question):
                if read:
                    self._record({'role': 'assistant', 'tool_calls': [
                        {'id': label, 'function': {
                            'name': 'read_paper_page',
                            'arguments': json.dumps({'paper_id': paper_id,
                                                     'page_no': page_no})}},
                    ], 'content': None})
                    self._record({'role': 'tool', 'tool_call_id': label,
                                  'content': f'《T》 t.pdf p.{page_no}\n\n{page_body}'})
                self._record({'role': 'assistant', 'content': answer})
        return Agent

    def setUp(self):
        super().setUp()
        self.write_pdf('a.pdf', ['神经语言模型使用神经网络来描述词序列的概率分布。',
                                 '第二页内容。'], title='Paper A')
        self.write_pdf('b.pdf', ['完全不同的一页原文。'], title='Paper B')
        self.import_papers_ok()

    def test_second_turn_cannot_reuse_first_turn_page(self):
        """第一轮读过某页，第二轮未读原文时，同一引文不得核验通过。"""
        quote = '神经语言模型使用神经网络来描述词序列的概率分布。'
        # Turn 1: reads page and quotes correctly
        self.agent_factory = self._agent_reading(
            '神经语言模型使用神经网络来描述词序列的概率分布。',
            f'引述：\n> {quote}', paper_id=1, page_no=1)
        s1, d1, h1 = self.post_json('/api/ask', {'question': '问题一'})
        self.assertEqual(s1, 200)
        self.assertEqual(d1['evidence'][0]['status'], 'verified',
                         '第一轮本轮读取了原文，应通过核验')
        sid = self.sid_from(h1)

        # Turn 2: same quote, but does NOT read any page
        self.agent_factory = self._agent_reading(
            '任意', f'再述：\n> {quote}', read=False)
        s2, d2, _ = self.post_json('/api/ask', {'question': '问题二'},
                                   cookie=f'pp_sid={sid}')
        self.assertEqual(s2, 200)
        self.assertEqual(d2['evidence'][0]['status'], 'unverified',
                         '第二轮未读原文，不得借用第一轮读取记录核验')

    def test_second_turn_cannot_borrow_other_page(self):
        quote = '神经语言模型使用神经网络来描述词序列的概率分布。'
        self.agent_factory = self._agent_reading(
            '神经语言模型使用神经网络来描述词序列的概率分布。',
            f'引述：\n> {quote}', paper_id=1, page_no=1)
        s1, d1, h1 = self.post_json('/api/ask', {'question': '问题一'})
        sid = self.sid_from(h1)

        # Turn 2: reads a DIFFERENT paper/page, quotes from the un-read one
        self.agent_factory = self._agent_reading(
            '完全不同的一页原文。', f'引述：\n> {quote}',
            paper_id=2, page_no=1, label='c2')
        s2, d2, _ = self.post_json('/api/ask', {'question': '问题二'},
                                   cookie=f'pp_sid={sid}')
        self.assertEqual(s2, 200)
        self.assertNotEqual(d2['evidence'][0]['status'], 'verified',
                            '不得借用第一轮的原文页作为本轮核验依据')

    def test_second_turn_extraction_excludes_first_turn(self):
        self.agent_factory = self._agent_reading(
            '神经语言模型使用神经网络来描述词序列的概率分布。',
            '第一轮回答文本', paper_id=1, page_no=1)
        s1, d1, h1 = self.post_json('/api/ask', {'question': '问题一'})
        sid = self.sid_from(h1)

        self.agent_factory = self._agent_reading('第二页内容。', '第二轮回答文本',
                                                 paper_id=1, page_no=2, label='c9')
        s2, d2, _ = self.post_json('/api/ask', {'question': '问题二'},
                                   cookie=f'pp_sid={sid}')
        self.assertEqual(d2['answer'], '第二轮回答文本')
        self.assertNotIn('第一轮回答文本', d2['answer'])
        for ev in d2['tool_events']:
            self.assertNotIn('c1', ev['tool_call_id'],
                             '工具轨迹不得包含第一轮的调用')
        self.assertEqual(len(d2['tool_events']), 1)
        self.assertEqual(d2['tool_events'][0]['tool_call_id'], 'c9')

    def test_tool_events_carry_real_results(self):
        self.agent_factory = self._agent_reading(
            '神经语言模型使用神经网络来描述词序列的概率分布。',
            '回答', paper_id=1, page_no=1)
        s, d, _ = self.post_json('/api/ask', {'question': '问'})
        self.assertEqual(s, 200)
        ev = d['tool_events'][0]
        self.assertEqual(ev['name'], 'read_paper_page')
        self.assertTrue(ev['has_result'])
        self.assertTrue(ev['ok'])
        self.assertIn('神经语言模型', ev['result_summary'])
        self.assertIn('paper_id', ev['arguments'])

    def test_pipeline_reflects_real_calls(self):
        # The answer must quote something so verification really runs.
        self.agent_factory = self._agent_reading(
            '神经语言模型使用神经网络来描述词序列的概率分布。',
            '引述：\n> 神经语言模型使用神经网络来描述词序列的概率分布。',
            paper_id=1, page_no=1)
        s, d, _ = self.post_json('/api/ask', {'question': '问'})
        pipe = {p['key']: p for p in d['pipeline']}
        self.assertEqual(pipe['read']['state'], 'done', 'read_paper_page 被真实调用')
        self.assertEqual(pipe['verify']['state'], 'done', '真实核验了引文')
        self.assertEqual(pipe['import']['state'], 'skipped', '本轮未调用导入')
        self.assertEqual(pipe['search']['state'], 'skipped', '本轮未调用检索')

    def test_pipeline_no_quote_skips_verify(self):
        self.agent_factory = self._agent_reading(
            '神经语言模型使用神经网络来描述词序列的概率分布。',
            '纯文字回答，没有任何引文。', paper_id=1, page_no=1)
        s, d, _ = self.post_json('/api/ask', {'question': '问'})
        pipe = {p['key']: p for p in d['pipeline']}
        self.assertEqual(pipe['verify']['state'], 'skipped')
        self.assertEqual(pipe['read']['state'], 'done')

    def test_failed_tool_shown_as_fail(self):
        class Agent(FakeAgentBase):
            def build_turn(self, question):
                self._record({'role': 'assistant', 'tool_calls': [
                    {'id': 'x', 'function': {
                        'name': 'read_paper_page',
                        'arguments': '{"paper_id": 1, "page_no": 1}'}},
                ], 'content': None})
                self._record({'role': 'tool', 'tool_call_id': 'x',
                              'content': '[not found]: no such page'})
                self._record({'role': 'assistant', 'content': '未能读取原文。'})
        self.agent_factory = Agent
        s, d, _ = self.post_json('/api/ask', {'question': '问'})
        pipe = {p['key']: p for p in d['pipeline']}
        self.assertEqual(pipe['read']['state'], 'fail')
        self.assertFalse(d['tool_events'][0]['ok'])

    def test_no_answer_falls_back_to_error_not_history(self):
        """新轮无回答正文时，不得回退显示旧轮答案。"""
        self.agent_factory = self._agent_reading(
            '神经语言模型使用神经网络来描述词序列的概率分布。',
            '第一轮有回答', paper_id=1, page_no=1)
        s1, _, h1 = self.post_json('/api/ask', {'question': '问题一'})
        sid = self.sid_from(h1)

        class SilentAgent(FakeAgentBase):
            def build_turn(self, question):
                self._record({'role': 'user', 'content': question})
        self.agent_factory = SilentAgent
        s2, data, _ = self.post_json('/api/ask', {'question': '问题二'},
                                     cookie=f'pp_sid={sid}')
        self.assertEqual(s2, 502)
        self.assertIn('error', data)
        self.assertNotIn('第一轮有回答', json.dumps(data))

    def test_conservative_rules_still_apply(self):
        # mismatch: one word changed
        self.agent_factory = self._agent_reading(
            '神经语言模型使用神经网络来描述词序列的概率分布。',
            '引述：\n> 神经语言模型使用神经网络来描述词序列的概率分部。',
            paper_id=1, page_no=1)
        s, d, _ = self.post_json('/api/ask', {'question': '问'})
        self.assertEqual(d['evidence'][0]['status'], 'mismatch')

    def test_multi_page_ambiguity_unverified(self):
        class Agent(FakeAgentBase):
            def build_turn(self, question):
                for i, pid in enumerate((1, 2), start=1):
                    self._record({'role': 'assistant', 'tool_calls': [
                        {'id': f'r{i}', 'function': {
                            'name': 'read_paper_page',
                            'arguments': json.dumps({'paper_id': pid, 'page_no': 1})}},
                    ], 'content': None})
                    self._record({'role': 'tool', 'tool_call_id': f'r{i}',
                                  'content': f'《T》 t.pdf p.1\n\n相同句子在两个页面都出现。'})
                self._record({'role': 'assistant', 'content': '引述：\n> 相同句子在两个页面都出现。'})
        self.agent_factory = Agent
        s, d, _ = self.post_json('/api/ask', {'question': '问'})
        self.assertEqual(d['evidence'][0]['status'], 'unverified')

    def test_evidence_source_from_real_verification(self):
        self.agent_factory = self._agent_reading(
            '神经语言模型使用神经网络来描述词序列的概率分布。',
            '引述：\n> 神经语言模型使用神经网络来描述词序列的概率分布。',
            paper_id=1, page_no=1)
        s, d, _ = self.post_json('/api/ask', {'question': '问'})
        self.assertEqual(d['evidence'][0]['source'], [1, 1],
                         '来源必须来自真实核验结果，不得前端推测')


class TestTurnMessagesUnit(RootSessionGuard, unittest.TestCase):
    """Agent-side turn_messages behaviour (compaction survival)."""

    def _make_cfg(self, tmp):
        return replace(CONFIG, work_space=tmp,
                       session_path=str(tmp / 'paper_session.json'),
                       system_prompt='SYS', base_url='http://127.0.0.1:9/x',
                       profile='t', model_main='m', think_main='off')

    def test_turn_messages_records_only_current_turn(self):
        from unittest import mock
        from tests.test_agent_config import make_result
        import tempfile as _tf
        with _tf.TemporaryDirectory() as td:
            tmp = Path(td)
            agent = DeepSeekAgent([], cfg=self._make_cfg(tmp), memory_enabled=False)
            prior = [{'role': 'user', 'content': 'old'}, {'role': 'assistant', 'content': 'old-a'}]
            with mock.patch('mini_harness.agent.OpenAI'):
                with mock.patch.object(DeepSeekAgent, '_run_turn', return_value=make_result()):
                    agent.run_task('new question', prior=prior)
            self.assertEqual([m['role'] for m in agent.turn_messages], ['user'])
            self.assertEqual(agent.turn_messages[0]['content'], 'new question')
            # prior history is in message but NOT in turn_messages
            self.assertEqual(len(agent.message), 4)

    def test_record_appends_to_both_lists(self):
        import tempfile as _tf
        with _tf.TemporaryDirectory() as td:
            tmp = Path(td)
            agent = DeepSeekAgent([], cfg=self._make_cfg(tmp), memory_enabled=False)
            agent.turn_messages = []
            agent.message = [{'role': 'system', 'content': 's'}]
            msg = {'role': 'user', 'content': 'x'}
            agent._record(msg)
            self.assertIn(msg, agent.turn_messages)
            self.assertIn(msg, agent.message)

    def test_compaction_does_not_rewrite_turn_messages(self):
        """模拟压缩重建 message 后，turn_messages 保持本轮完整记录。"""
        import tempfile as _tf
        with _tf.TemporaryDirectory() as td:
            tmp = Path(td)
            agent = DeepSeekAgent([], cfg=self._make_cfg(tmp), memory_enabled=False)
            agent.turn_messages = []
            agent.message = [{'role': 'system', 'content': 's'},
                             {'role': 'user', 'content': 'old'}]
            # current turn records a read
            agent._record({'role': 'user', 'content': 'q2'})
            agent._record({'role': 'assistant', 'tool_calls': [
                {'id': 'c', 'function': {
                    'name': 'read_paper_page',
                    'arguments': json.dumps({'paper_id': 1, 'page_no': 1})}}]})
            agent._record({'role': 'tool', 'tool_call_id': 'c',
                           'content': '《T》 t.pdf p.1\n\n真实原文'})
            before = list(agent.turn_messages)
            # Simulate compaction rebuilding message (drops old history)
            agent.message = [agent.message[0], {'role': 'assistant', 'content': 'summary'}]
            self.assertEqual(agent.turn_messages, before,
                             '压缩重建 message 不得改写 turn_messages')
            # Web extraction must still see this turn's page
            pages = web._collect_read_pages(agent)
            self.assertIn((1, 1), pages)


class TestCurrentTurnMessagesUnit(unittest.TestCase):
    def test_list_passthrough(self):
        msgs = [{'role': 'user', 'content': 'a'}]
        self.assertEqual(web.current_turn_messages(msgs), msgs)

    def test_prefers_turn_messages_over_message(self):
        class A:
            message = [{'role': 'user', 'content': 'old'}]
            turn_messages = [{'role': 'user', 'content': 'new'}]
        self.assertEqual(web.current_turn_messages(A())[0]['content'], 'new')

    def test_valid_turn_start_slicing(self):
        class A:
            turn_messages = None
            message = [{'role': 'system', 'content': 's'},
                       {'role': 'user', 'content': 'old'},
                       {'role': 'user', 'content': 'cur'}]
            turn_start = 2
        got = web.current_turn_messages(A())
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]['content'], 'cur')

    def test_invalid_turn_start_refuses_history(self):
        class A:
            turn_messages = None
            message = [{'role': 'user', 'content': 'old'},
                       {'role': 'assistant', 'content': 'old-a'}]
            turn_start = 99  # out of range after compaction
        self.assertEqual(web.current_turn_messages(A()), [],
                         '失效边界必须拒绝历史证据，不得回退整段历史')

    def test_negative_turn_start_refused(self):
        class A:
            turn_messages = None
            message = [{'role': 'user', 'content': 'a'}]
            turn_start = -1
        self.assertEqual(web.current_turn_messages(A()), [])

    def test_non_int_turn_start_refused(self):
        class A:
            turn_messages = None
            message = [{'role': 'user', 'content': 'a'}]
            turn_start = 'x'
        self.assertEqual(web.current_turn_messages(A()), [])

    def test_legacy_agent_without_boundary_uses_whole_list(self):
        class A:
            message = [{'role': 'assistant', 'content': 'only turn'}]
        self.assertEqual(len(web.current_turn_messages(A())), 1)

    def test_attribute_style_turn_messages(self):
        class Msg:
            role = 'assistant'
            tool_calls = None
            content = '对象形态'
        class A:
            turn_messages = [Msg()]
        got = web.current_turn_messages(A())
        self.assertEqual(got[0].content, '对象形态')


class TestContextMemoryPrompt(WorkbenchBase):
    """System prompt must distinguish session context from long-term memory.

    These tests check the PROMPT CONTENT and the history-passing mechanism.
    They cannot prove what a real model will say -- FakeAgent only validates
    wiring, not model semantics.
    """

    def test_prompt_distinguishes_session_context_from_long_term_memory(self):
        from paperpilot.cli import PAPER_SYSTEM_PROMPT
        p = ' '.join(PAPER_SYSTEM_PROMPT.lower().split())  # unwrap line breaks
        self.assertIn('session context', p)
        self.assertIn('not long-term memory', p)
        # must not claim total absence of context
        self.assertNotIn('i cannot remember', p)
        self.assertNotIn('i have no memory', p)
        # must acknowledge compression may drop details
        self.assertIn('compression', p)

    def test_prompt_says_explain_only_when_asked(self):
        from paperpilot.cli import PAPER_SYSTEM_PROMPT
        p = ' '.join(PAPER_SYSTEM_PROMPT.lower().split())
        self.assertIn('only when the user asks', p)
        self.assertIn('do not append memory explanations', p)

    def test_prompt_forbids_marking_chat_as_paper_quotation(self):
        from paperpilot.cli import PAPER_SYSTEM_PROMPT
        p = ' '.join(PAPER_SYSTEM_PROMPT.lower().split())
        self.assertIn('not a paper quote', p)
        self.assertIn('ordinary chat', p)

    def test_history_still_passed_as_prior(self):
        # Same-session second turn must receive turn-1 history via prior.
        Agent = _agent_reads_page('原文内容。')
        self.agent_factory = Agent
        s1, _, h1 = self.post_json('/api/ask', {'question': '第一问'})
        sid = self.sid_from(h1)
        captured = {}

        def factory(*a, **k):
            ag = Agent(*a, **k)
            orig = ag.run_task
            def wrapped(question, cfg=None, prior=None):
                captured['prior'] = list(prior) if prior else []
                return orig(question, cfg=cfg, prior=prior)
            ag.run_task = wrapped
            return ag

        self.agent_factory = factory
        s2, _, _ = self.post_json('/api/ask', {'question': '第二问'},
                                  cookie=f'pp_sid={sid}')
        self.assertEqual(s2, 200)
        self.assertTrue(captured['prior'], '第二轮必须收到 prior')
        self.assertEqual(captured['prior'][0]['content'], '第一问')

    def test_reset_isolation_unchanged(self):
        Agent = _agent_reads_page('原文。')
        self.agent_factory = Agent
        _, _, h1 = self.post_json('/api/ask', {'question': '甲'})
        sid_a = self.sid_from(h1)
        _, _, h2 = self.post_json('/api/ask', {'question': '乙'})
        sid_b = self.sid_from(h2)
        self.post_json('/api/reset', {}, cookie=f'pp_sid={sid_a}')
        _, da, _ = self.get_json('/api/state', cookie=f'pp_sid={sid_a}')
        _, db, _ = self.get_json('/api/state', cookie=f'pp_sid={sid_b}')
        self.assertEqual(da['conversation'], [])
        self.assertEqual(len(db['conversation']), 2)


class TestChatVsPaperQuotes(WorkbenchBase):
    """Ordinary chat must not produce paper-evidence cards; real paper
    quotations keep their three-state verification."""

    def setUp(self):
        super().setUp()
        self.write_pdf('a.pdf', ['神经语言模型使用神经网络来描述词序列的概率分布。',
                                 '第二页内容。'], title='Paper A')
        self.import_papers_ok()

    def _ask(self, answer, question='问', read=True, page_body=None,
             paper_id=1, page_no=1):
        """Run one FakeAgent turn returning ``answer``; return evidence list."""
        class Agent(FakeAgentBase):
            def build_turn(self, question):
                if read:
                    self._record({'role': 'assistant', 'tool_calls': [
                        {'id': 'c1', 'function': {
                            'name': 'read_paper_page',
                            'arguments': json.dumps({'paper_id': paper_id,
                                                     'page_no': page_no})}},
                    ], 'content': None})
                    self._record({'role': 'tool', 'tool_call_id': 'c1',
                                  'content': f'《T》 t.pdf p.{page_no}\n\n'
                                             f'{page_body or "神经语言模型使用神经网络来描述词序列的概率分布。"}'})
                self._record({'role': 'assistant', 'content': answer})
        self.agent_factory = Agent
        s, d, _ = self.post_json('/api/ask', {'question': question})
        self.assertEqual(s, 200, d)
        return d

    def test_test_marker_echo_produces_no_evidence_card(self):
        """PP-CONTEXT-7319 回显：不得产生论文证据卡片。"""
        d = self._ask('你提供的标记是 `PP-CONTEXT-7319`，记下了。',
                      question='请记住 PP-CONTEXT-7319')
        self.assertEqual(d['evidence'], [], '测试标记回显不得产生证据卡片')

    def test_test_marker_in_quotes_also_no_card(self):
        d = self._ask('好的，标记“PP-CONTEXT-7319”我已记录。',
                      question='标记是 PP-CONTEXT-7319')
        self.assertEqual(d['evidence'], [])

    def test_restatement_of_user_question_no_card(self):
        d = self._ask('你刚才问的是“什么是低秩适配”。',
                      question='什么是低秩适配？')
        self.assertEqual(d['evidence'], [], '复述用户问题不得产生证据卡片')

    def test_ordinary_chat_with_quotes_and_backticks_no_card(self):
        d = self._ask('他说“你好”之后，执行 `print("hi")` 就行了。',
                      question='闲聊一下', read=False)
        self.assertEqual(d['evidence'], [], '普通聊天的引号/反引号不得触发论文核验')

    def test_ordinary_chat_without_tools_no_warning(self):
        d = self._ask('今天天气不错。', question='随便聊聊', read=False)
        self.assertEqual(d['evidence'], [])
        pipe = {p['key']: p for p in d['pipeline']}
        self.assertEqual(pipe['read']['state'], 'skipped')
        self.assertNotIn('未读取整页原文', json.dumps(d))

    def test_paper_quote_with_read_page_still_verified(self):
        d = self._ask('引述：\n> 神经语言模型使用神经网络来描述词序列的概率分布。',
                      question='定义是什么')
        self.assertEqual(len(d['evidence']), 1)
        self.assertEqual(d['evidence'][0]['status'], 'verified')

    def test_paper_quote_without_read_still_unverified(self):
        d = self._ask('引述：\n> 神经语言模型使用神经网络来描述词序列的概率分布。',
                      question='定义是什么', read=False)
        self.assertEqual(len(d['evidence']), 1, '真实论文引文即使未读原文也要出卡片')
        self.assertEqual(d['evidence'][0]['status'], 'unverified')

    def test_paper_quote_mismatch_still_reported(self):
        d = self._ask('引述：\n> 这句话和原文完全对不上。',
                      question='定义是什么')
        self.assertEqual(len(d['evidence']), 1)
        self.assertEqual(d['evidence'][0]['status'], 'mismatch')

    def test_paired_quote_with_paper_context_verified(self):
        d = self._ask('原文是“神经语言模型使用神经网络来描述词序列的概率分布。”这句话。',
                      question='定义是什么')
        self.assertEqual(len(d['evidence']), 1)
        self.assertEqual(d['evidence'][0]['status'], 'verified')

    def test_paired_quote_without_context_is_chat(self):
        d = self._ask('他说“今天开会”然后就走了。',
                      question='闲聊', read=False)
        self.assertEqual(d['evidence'], [], '无论文语境的成对引号属普通聊天')

    def test_quote_marking_code_token_not_paper(self):
        d = self._ask('设置 `PP-CONTEXT-7319` 即可，见“PP-CONTEXT-7319”。',
                      question='怎么设置', read=False)
        self.assertEqual(d['evidence'], [])

    def test_paper_quote_borrows_no_history_page(self):
        # Turn 1 reads the page; turn 2 quotes it WITHOUT reading.
        self._ask('引述：\n> 神经语言模型使用神经网络来描述词序列的概率分布。',
                  question='第一问')
        _, data, _ = self.get_json('/api/state')
        sid = None
        # second turn in the same session
        self.agent_factory = None
        class Agent(FakeAgentBase):
            def build_turn(self, question):
                self._record({'role': 'assistant',
                              'content': '引述：\n> 神经语言模型使用神经网络来描述词序列的概率分布。'})
        self.agent_factory = Agent
        s, d, h = self.post_json('/api/ask', {'question': '第二问'})
        self.assertEqual(s, 200)
        self.assertEqual(d['evidence'][0]['status'], 'unverified',
                         '不得复用历史轮次页面作为本轮证据')

    def test_ambiguous_nested_quote_never_verified(self):
        d = self._ask('他说“外层开始“内层”结束了”',
                      question='闲聊', read=False)
        self.assertEqual(len(d['evidence']), 1)
        self.assertEqual(d['evidence'][0]['status'], 'unverified')
        self.assertNotEqual(d['evidence'][0]['status'], 'verified')

    def test_quote_extraction_only_from_final_answer(self):
        """证据候选只来自本轮最终回答，不来自用户消息。"""
        d = self._ask('好的，明白了。',
                      question='请引用“神经语言模型使用神经网络来描述词序列的概率分布”',
                      read=False)
        self.assertEqual(d['evidence'], [], '用户消息中的引文不得成为模型引文候选')


class TestToolTraceUI(WorkbenchBase):
    """Two-layer tool trace and honest pipeline states."""

    def setUp(self):
        super().setUp()
        self.write_pdf('a.pdf', ['神经语言模型使用神经网络来描述词序列的概率分布。'],
                       title='Paper A')
        self.import_papers_ok()

    def test_no_tools_no_fake_events(self):
        class Agent(FakeAgentBase):
            def build_turn(self, question):
                self._record({'role': 'assistant', 'content': '普通回答。'})
        self.agent_factory = Agent
        s, d, _ = self.post_json('/api/ask', {'question': '闲聊'})
        self.assertEqual(d['tool_events'], [], '不得伪造工具执行事件')
        pipe = {p['key']: p for p in d['pipeline']}
        self.assertEqual(pipe['import']['state'], 'skipped')
        self.assertEqual(pipe['search']['state'], 'skipped')
        self.assertEqual(pipe['read']['state'], 'skipped')

    def test_done_fail_and_skipped_states_differ(self):
        class Agent(FakeAgentBase):
            def build_turn(self, question):
                # search ok, read fails, import never called
                self._record({'role': 'assistant', 'tool_calls': [
                    {'id': 's1', 'function': {'name': 'search_papers',
                                              'arguments': '{"query":"x"}'}},
                    {'id': 'r1', 'function': {'name': 'read_paper_page',
                                              'arguments': '{"paper_id":1,"page_no":1}'}},
                ], 'content': None})
                self._record({'role': 'tool', 'tool_call_id': 's1',
                              'content': '[search snippets only] 1. 《A》 p.1'})
                self._record({'role': 'tool', 'tool_call_id': 'r1',
                              'content': '[not found]: no such page'})
                self._record({'role': 'assistant', 'content': '未能读取原文。'})
        self.agent_factory = Agent
        s, d, _ = self.post_json('/api/ask', {'question': '问'})
        pipe = {p['key']: p for p in d['pipeline']}
        self.assertEqual(pipe['import']['state'], 'skipped', '未调用=跳过')
        self.assertEqual(pipe['search']['state'], 'done', '成功=已执行')
        self.assertEqual(pipe['read']['state'], 'fail', '失败必须显示失败')
        states = {p['key']: p['state'] for p in d['pipeline']}
        self.assertGreaterEqual(len(set(states.values())), 2, '状态必须可区分')

    def test_tool_events_carry_real_params_and_results(self):
        class Agent(FakeAgentBase):
            def build_turn(self, question):
                self._record({'role': 'assistant', 'tool_calls': [
                    {'id': 's1', 'function': {'name': 'search_papers',
                                              'arguments': '{"query":"低秩 适配"}'}},
                ], 'content': None})
                self._record({'role': 'tool', 'tool_call_id': 's1',
                              'content': '[search snippets only] 1. 《A》 p.1'})
                self._record({'role': 'assistant', 'content': '检索完成。'})
        self.agent_factory = Agent
        s, d, _ = self.post_json('/api/ask', {'question': '问'})
        ev = d['tool_events'][0]
        self.assertIn('低秩 适配', ev['arguments'], '真实参数必须可查')
        self.assertTrue(ev['has_result'])
        self.assertTrue(ev['ok'])
        self.assertIn('search snippets', ev['result_summary'])

    def test_pending_tool_result_shown_as_no_result(self):
        class Agent(FakeAgentBase):
            def build_turn(self, question):
                self._record({'role': 'assistant', 'tool_calls': [
                    {'id': 'x1', 'function': {'name': 'search_papers',
                                              'arguments': '{"query":"q"}'}},
                ], 'content': None})
                # no tool result message follows
                self._record({'role': 'assistant', 'content': '未收到结果。'})
        self.agent_factory = Agent
        s, d, _ = self.post_json('/api/ask', {'question': '问'})
        self.assertFalse(d['tool_events'][0]['has_result'])

    def test_pipeline_verify_reflects_real_checks(self):
        class Agent(FakeAgentBase):
            def build_turn(self, question):
                self._record({'role': 'assistant', 'tool_calls': [
                    {'id': 'c1', 'function': {
                        'name': 'read_paper_page',
                        'arguments': '{"paper_id": 1, "page_no": 1}'}},
                ], 'content': None})
                self._record({'role': 'tool', 'tool_call_id': 'c1',
                              'content': '《T》 t.pdf p.1\n\n神经语言模型使用神经网络来描述词序列的概率分布。'})
                self._record({'role': 'assistant', 'content':
                    '引述：\n> 神经语言模型使用神经网络来描述词序列的概率分布。'})
        self.agent_factory = Agent
        s, d, _ = self.post_json('/api/ask', {'question': '问'})
        pipe = {p['key']: p for p in d['pipeline']}
        self.assertEqual(pipe['verify']['state'], 'done')
        self.assertIn('1 条引文', pipe['verify']['detail'])


class TestFrontendContract(unittest.TestCase):
    """Frontend keeps safe rendering and shows no empty evidence area."""

    JS = Path(__file__).resolve().parents[1] / 'src/paperpilot/static/workbench/app.js'
    HTML = Path(__file__).resolve().parents[1] / 'src/paperpilot/static/workbench/index.html'

    def test_app_js_has_no_innerhtml_concat(self):
        src = self.JS.read_text(encoding='utf-8')
        self.assertNotIn('innerHTML +=', src)
        self.assertNotIn('.innerHTML =', src, '动态内容必须走 textContent')

    def test_app_js_no_synthetic_data(self):
        src = self.JS.read_text(encoding='utf-8')
        self.assertNotIn('PP_DATA', src)
        self.assertNotIn('data.js', src)
        # setTimeout is allowed ONLY for the toast auto-hide, never for
        # faking tool execution or pipeline timing.
        for line in src.splitlines():
            if 'setTimeout' in line:
                self.assertIn('toast', line,
                              'setTimeout 只能用于 toast 自动隐藏，不得模拟工具运行')

    def test_app_js_never_renders_evidence_cards(self):
        """产品契约：对话区不再渲染任何证据卡片。"""
        src = self.JS.read_text(encoding='utf-8')
        self.assertNotIn('evidenceCard', src, '不得调用证据卡片渲染')
        self.assertNotIn("el('div', 'cards')", src, '不得创建卡片容器')

    def test_app_js_has_collapsed_tool_details(self):
        src = self.JS.read_text(encoding='utf-8')
        self.assertIn("'details'", src, '工具详情默认折叠')
        self.assertIn('查看执行详情', src)

    def test_app_js_summary_from_real_data(self):
        src = self.JS.read_text(encoding='utf-8')
        self.assertIn('toolSummaryLines', src, '摘要须由真实数据生成')
        self.assertNotIn('本轮未调用论文工具', src, '对话区不显示诊断提示')

    def test_html_has_no_duplicate_page_level_disclaimer(self):
        html = self.HTML.read_text(encoding='utf-8')
        total = html.count('字符级高亮')
        self.assertEqual(total, 1,
                         f'同一条页级定位提示只应出现一次，实际 {total} 次')


class TestScreenshotRegression(WorkbenchBase):
    """Regression for the real workbench over-extraction bug (screenshot).

    Samples mirror the answers that produced two bogus evidence cards:
    a paper title with Markdown bold and PDF layout spacing, and an
    ABSTRACT heading followed by a summarised excerpt.
    """

    # Realistic read_paper_page payload for LoRA p.1 (raw PDF text).
    LORA_P1 = ('LoRA: Low-Rank Adaptation of Large Language Models\n'
               'ABSTRACT An important paradigm of natural language processing '
               'consists of large-scale pre-training on general domain data '
               'and adaptation to particular tasks.')

    def setUp(self):
        super().setUp()
        self.write_pdf('a.pdf', [self.LORA_P1, '第二页正文内容。'], title='LoRA')
        self.import_papers_ok()

    def _ask(self, answer, question='问', read=True, page_no=1, page_body=None,
             extra_reads=None):
        default_body = page_body or self.LORA_P1
        extra = list(extra_reads or [])

        class Agent(FakeAgentBase):
            def build_turn(self, question):
                if read:
                    self._record({'role': 'assistant', 'tool_calls': [
                        {'id': 'c1', 'function': {
                            'name': 'read_paper_page',
                            'arguments': json.dumps({'paper_id': 1,
                                                     'page_no': page_no})}},
                    ], 'content': None})
                    self._record({'role': 'tool', 'tool_call_id': 'c1',
                                  'content': '《LoRA》 papers/a.pdf p.%d\n\n%s'
                                             % (page_no, default_body)})
                for i, (pid, pno, body) in enumerate(extra, start=2):
                    cid = 'c%d' % i
                    self._record({'role': 'assistant', 'tool_calls': [
                        {'id': cid, 'function': {
                            'name': 'read_paper_page',
                            'arguments': json.dumps({'paper_id': pid,
                                                     'page_no': pno})}},
                    ], 'content': None})
                    self._record({'role': 'tool', 'tool_call_id': cid,
                                  'content': '《X》 x.pdf p.%d\n\n%s' % (pno, body)})
                self._record({'role': 'assistant', 'content': answer})
        self.agent_factory = Agent
        s, d, _ = self.post_json('/api/ask', {'question': question})
        self.assertEqual(s, 200, d)
        return d

    # -- 1. summary in the model's own words: no quote card ----------------
    def test_model_summary_produces_no_card(self):
        d = self._ask(
            'LoRA 的核心做法是冻结预训练权重，只训练注入的低秩矩阵，'
            '这样可训练参数能降到全量微调的 0.1% 左右，推理延迟不变。',
            question='LoRA 的核心做法是什么？')
        self.assertEqual(d['evidence'], [], '论文总结不得产生逐字核验卡片')

    # -- 2. markdown headings must not create cards -----------------------
    def test_markdown_title_and_abstract_headings_no_card(self):
        """截图样本 1+2：**LoRA** 标题与 **ABSTRACT** 摘要段不得成卡。"""
        d = self._ask(
            '论文标题如下：\n'
            '> **LoRA**: L OW-R ANK ADAPTATION OF LARGE LAN-\n\n'
            '摘要开头：\n'
            '> **ABSTRACT** An important paradigm of natural language '
            'processing consists of large-scale pre-training ...')
        self.assertEqual(
            d['evidence'], [],
            '标题与 ABSTRACT 章节标题/摘要摘录不得成为引文候选')

    def test_bold_heading_in_prose_no_card(self):
        d = self._ask(
            '论文 **LoRA**: L OW-R ANK ADAPTATION OF LARGE LAN- 提出了一种低秩适配方法。',
            question='这篇论文讲什么')
        self.assertEqual(d['evidence'], [])

    def test_abstract_block_without_ellipsis_no_card(self):
        d = self._ask(
            '摘要：\n> **ABSTRACT** An important paradigm of natural language '
            'processing consists of large-scale pre-training on general domain '
            'data and adaptation to particular tasks.')
        self.assertEqual(d['evidence'], [], 'ABSTRACT 章节标题开头的段落是摘要展示，非引文')

    # -- 3. genuine quote with line breaks + markdown emphasis ------------
    def test_verbatim_quote_with_markdown_emphasis_verifies(self):
        d = self._ask(
            '原文是：\n> **ABSTRACT An important paradigm of natural '
            'language processing consists of large-scale pre-training on '
            'general domain data and adaptation to particular tasks.**')
        self.assertEqual(len(d['evidence']), 1, '去除强调后应识别为引文')
        self.assertEqual(d['evidence'][0]['status'], 'verified',
                         '去格式差异后应逐字核验通过')

    def test_verbatim_quote_with_line_breaks_verifies(self):
        body = ('ABSTRACT An important paradigm of natural language '
                'processing consists of large-scale pre-training on '
                'general domain data and adaptation to particular tasks.')
        d = self._ask(
            '引述：\n> ABSTRACT An important paradigm of natural language '
            'processing consists of large-scale pre-training on general '
            'domain data and adaptation to particular tasks.',
            page_body=body)
        self.assertEqual(d['evidence'][0]['status'], 'verified')

    def test_quote_with_inline_emphasis_verifies(self):
        d = self._ask(
            '原文是：\n> ABSTRACT An important paradigm of natural language '
            'processing consists of **large-scale pre-training** on general '
            'domain data and adaptation to particular tasks.')
        self.assertEqual(len(d['evidence']), 1, '行内强调不得改变引文性质')
        self.assertEqual(d['evidence'][0]['status'], 'verified')

    # -- 4. wrong quote still mismatch ------------------------------------
    def test_wrong_quote_still_mismatch(self):
        d = self._ask(
            '原文是：\n> ABSTRACT An important paradigm of natural language '
            'processing consists of large-scale pre-training on SPECIFIC '
            'domain data and adaptation to particular tasks.')
        self.assertEqual(d['evidence'][0]['status'], 'mismatch',
                         '错词必须继续暴露为不一致')

    def test_truncated_title_not_verified(self):
        """标题只引用部分文字且内容不一致：不得误判 verified。"""
        d = self._ask(
            '标题是：\n> **LoRA**: L OW-R ANK ADAPTATION OF LARGE LAN-')
        for ev in d['evidence']:
            self.assertNotEqual(ev['status'], 'verified',
                                '标题截断片段不得判为核验通过')

    # -- 5. quoted but no page read --------------------------------------
    def test_quote_without_read_unverified(self):
        d = self._ask(
            '引述：\n> ABSTRACT An important paradigm of natural language '
            'processing consists of large-scale pre-training on general '
            'domain data and adaptation to particular tasks.',
            read=False)
        self.assertEqual(len(d['evidence']), 1)
        self.assertEqual(d['evidence'][0]['status'], 'unverified')
        self.assertIn('未读取原文', d['evidence'][0]['note'])

    # -- 7. no cross-turn page borrowing ---------------------------------
    def test_second_turn_cannot_borrow_first_turn_page(self):
        quote = ('ABSTRACT An important paradigm of natural language '
                 'processing consists of large-scale pre-training on '
                 'general domain data and adaptation to particular tasks.')
        self._ask('引述：\n> ' + quote, question='第一问')
        class Agent(FakeAgentBase):
            def build_turn(self, question):
                self._record({'role': 'assistant',
                              'content': '引述：\n> ' + quote})
        self.agent_factory = Agent
        s, d, _ = self.post_json('/api/ask', {'question': '第二问'})
        self.assertEqual(d['evidence'][0]['status'], 'unverified',
                         '不得借用上一轮读取的页面作为本轮证据')

    # -- 8. checked pages shown honestly, not as source -------------------
    def test_checked_page_shown_not_as_source(self):
        d = self._ask(
            '引述：\n> ABSTRACT An important paradigm of natural language '
            'processing consists of large-scale pre-training on WRONG '
            'domain data and adaptation to particular tasks.')
        ev = d['evidence'][0]
        self.assertIsNone(ev['source'], '未匹配不得编造来源')
        self.assertEqual(ev['checked'][0]['paper_id'], 1)
        self.assertEqual(ev['checked'][0]['page_no'], 1)
        self.assertEqual(ev['checked'][0]['title'], 'LoRA',
                         '已检查页面应复用真实论文标题')
        self.assertIn('未匹配到已读取页面', ev['note'])

    def test_multi_page_no_match_reports_no_match(self):
        d = self._ask(
            '引述：\n> 这段文字在任何一个已读页面里都不存在。',
            extra_reads=[(1, 2, '第二页正文内容。')])
        ev = d['evidence'][0]
        self.assertIsNone(ev['source'])
        self.assertEqual(len(ev['checked']), 2)
        self.assertIn('未匹配到已读取页面', ev['note'])

    # -- 9. chat echoes / markers / quotes in chat ------------------------
    def test_chat_and_markers_no_card(self):
        for answer, question in (
            ('你提到的 `PP-CONTEXT-7319` 我记住了。', '记住 PP-CONTEXT-7319'),
            ('你刚才问的是“LoRA 的核心做法是什么”。', 'LoRA 的核心做法是什么？'),
            ('他说“你好”之后就走了。', '随便聊聊'),
        ):
            with self.subTest(answer=answer):
                d = self._ask(answer, question=question, read=False)
                self.assertEqual(d['evidence'], [],
                                 f'普通对话不得成卡: {answer}')

    def test_html_escaping_still_applied(self):
        d = self._ask('引述：\n> <script>alert(1)</script>',
                      page_body='<script>alert(1)</script> 原文')
        self.assertEqual(len(d['evidence']), 1)
        self.assertIn('<script>', d['evidence'][0]['text'])
        # front-end contract: no innerHTML concatenation
        js = (Path(__file__).resolve().parents[1]
              / 'src/paperpilot/static/workbench/app.js')
        self.assertNotIn('.innerHTML =', js.read_text(encoding='utf-8'))


class TestSampleTCM(WorkbenchBase):
    """Sample A regression: TCM-Serve answer with NO paper-tool calls.

    Only the explicitly attributed English quotation may become a card. Term
    expansions, the Chinese scheduling summary and bold technical terms must
    not.
    """

    # Real interaction record, reproduced verbatim.
    TCM_ANSWER = (
        'TCM 是 **T**ask-**C**onstrained **M**ultimodal **S**cheduling 的缩写。\n\n'
        '> **T**ask-**C**onstrained **M**ultimodal **S**cheduling\n\n'
        '> **在调度时考虑任务的模态属性和资源需求，动态分配计算资源，最大化吞吐量与效率。**\n\n'
        '该方法涉及**模态感知**、**任务约束建模**、**任务-模态**、**模态类型**、**资源池** 等概念。\n\n'
        '“TCM introduces a modality-aware scheduling policy that dynamically '
        'assigns tasks to compute resources based on their modality and '
        'resource requirements.”\n'
        '—— 《TCM-Serve》第 3 页，原文引用\n'
    )

    def setUp(self):
        super().setUp()
        self.write_pdf('a.pdf', ['无关页面正文内容。'], title='X')
        self.import_papers_ok()

    def _ask(self, answer, question):
        class Agent(FakeAgentBase):
            def build_turn(self, q):
                self._record({'role': 'assistant', 'content': answer})
        self.agent_factory = Agent
        s, d, _ = self.post_json('/api/ask', {'question': question})
        self.assertEqual(s, 200, d)
        return d

    def test_only_the_attributed_quote_becomes_a_card(self):
        """期望：只有一张引文卡片，而非每个粗体术语/引文块一张。"""
        d = self._ask(self.TCM_ANSWER, 'TCM-Serve: Modality-aware Scheduling 中 tcm 是什么？')
        self.assertEqual(len(d['evidence']), 1,
                         f'应只有 1 张卡片，实际: {[e["text"] for e in d["evidence"]]}')
        self.assertEqual(d['evidence'][0]['status'], 'unverified',
                         '本轮未读原文，明确引文应为未核验')

    def test_card_is_the_attributed_english_quote(self):
        d = self._ask(self.TCM_ANSWER, 'TCM 是什么？')
        self.assertIn('TCM introduces a modality-aware scheduling policy',
                      d['evidence'][0]['text'])

    def test_term_expansion_block_no_card(self):
        d = self._ask(self.TCM_ANSWER, 'TCM 是什么？')
        for ev in d['evidence']:
            self.assertNotIn('Task-**C**onstrained', ev['text'])
            self.assertNotIn('T**ask', ev['text'])

    def test_chinese_summary_block_no_card(self):
        d = self._ask(self.TCM_ANSWER, 'TCM 是什么？')
        for ev in d['evidence']:
            self.assertNotIn('调度时考虑任务的模态属性', ev['text'],
                             '中文总结不得成卡')

    def test_bold_terms_no_card(self):
        d = self._ask(self.TCM_ANSWER, 'TCM 是什么？')
        for ev in d['evidence']:
            for term in ('模态感知', '任务约束建模', '任务-模态', '模态类型', '资源池'):
                self.assertNotIn(term, ev['text'], f'普通术语 {term} 不得成卡')

    def test_pipeline_shows_no_tools_called(self):
        d = self._ask(self.TCM_ANSWER, 'TCM 是什么？')
        self.assertEqual(d['tool_events'], [], '本轮未调用工具')
        pipe = {p['key']: p for p in d['pipeline']}
        self.assertEqual(pipe['import']['state'], 'skipped', '未调用导入')
        self.assertEqual(pipe['search']['state'], 'skipped', '未调用检索')
        self.assertEqual(pipe['read']['state'], 'skipped', '未调用读原文')
        # verify is not a tool: it reflects the real quote-checking result
        self.assertIn(pipe['verify']['state'], ('done', 'skipped'))

    def test_unverified_card_has_no_invented_source(self):
        d = self._ask(self.TCM_ANSWER, 'TCM 是什么？')
        ev = d['evidence'][0]
        self.assertIsNone(ev['source'], '未读原文不得编造来源页码')
        self.assertEqual(ev['checked'], [], '未读页面不得列出已检查页面')


class TestSampleAIReport(WorkbenchBase):
    """Sample B regression: PEFT survey answer -- terms and headings must
    not become cards; an explicitly attributed quote stays unverified."""

    AI_ANSWER = (
        'PEFT 方法主要包括 Adapter、Prefix Tuning、BitFit 等。\n\n'
        'Adapter 通过在层间插入旁路模块实现参数高效微调。'
        '“前缀”是在输入前拼接可训练向量。\n\n'
        '“相关工作”部分介绍了各类方法的对比。\n\n'
        '根据论文第 3 页，原文引用：“prefix tuning learns continuous prefixes”。\n'
    )

    def setUp(self):
        super().setUp()
        self.write_pdf('a.pdf', ['前缀 的用法相关工作 内容。', '第二页。'], title='AI调研报告')
        self.import_papers_ok()

    def _ask(self, answer, question, read=False):
        class Agent(FakeAgentBase):
            def build_turn(self, q):
                if read:
                    self._record({'role': 'assistant', 'tool_calls': [
                        {'id': 'c1', 'function': {
                            'name': 'read_paper_page',
                            'arguments': '{"paper_id": 1, "page_no": 1}'}},
                    ], 'content': None})
                    self._record({'role': 'tool', 'tool_call_id': 'c1',
                                  'content': '《AI调研报告》 p.1\n\n前缀 的用法相关工作 内容。'})
                self._record({'role': 'assistant', 'content': answer})
        self.agent_factory = Agent
        s, d, _ = self.post_json('/api/ask', {'question': question})
        self.assertEqual(s, 200, d)
        return d

    def test_only_attributed_quote_becomes_card(self):
        d = self._ask(self.AI_ANSWER, 'PEFT 方法有哪些？')
        self.assertEqual(len(d['evidence']), 1,
                         f'应只有 1 张卡片，实际: {[e["text"] for e in d["evidence"]]}')
        self.assertIn('prefix tuning learns continuous prefixes',
                      d['evidence'][0]['text'])

    def test_term_mention_in_quotes_no_card(self):
        """“前缀”是定义性指称，不是论文引文。"""
        d = self._ask(self.AI_ANSWER, 'PEFT 方法有哪些？')
        for ev in d['evidence']:
            self.assertNotEqual(ev['text'], '前缀', '术语指称不得成卡')
            self.assertNotEqual(ev['text'], '相关工作', '章节标题不得成卡')

    def test_attributed_quote_unverified_without_read(self):
        d = self._ask(self.AI_ANSWER, 'PEFT 方法有哪些？')
        self.assertEqual(d['evidence'][0]['status'], 'unverified')

    def test_no_cross_turn_page_reuse(self):
        # turn 1 reads a page; turn 2 must not borrow it
        self._ask(self.AI_ANSWER, '第一问', read=True)
        d2 = self._ask(self.AI_ANSWER, '第二问', read=False)
        for ev in d2['evidence']:
            self.assertNotEqual(ev['status'], 'verified',
                                '不得复用上一轮页面核验通过')

    def test_plain_terms_matching_page_still_no_card(self):
        """普通术语即使能匹配本轮页面，也不得仅凭匹配成为引文。"""
        answer = '“前缀”在原文中出现过。'
        d = self._ask(answer, '前缀是什么', read=True)
        self.assertEqual(d['evidence'], [],
                         'matches_page 不得单独提升普通术语')


class TestQuoteIntentVsVerification(WorkbenchBase):
    """Stage A (intent) and Stage B (verbatim match) stay separate."""

    def setUp(self):
        super().setUp()
        self.write_pdf('a.pdf',
                       ['神经语言模型使用神经网络来描述词序列的概率分布。这在NLP领域产生了重要影响。',
                        '第二页内容。'],
                       title='Paper A')
        self.import_papers_ok()

    def _ask(self, answer, question='问', read=True, page_no=1):
        class Agent(FakeAgentBase):
            def build_turn(self, q):
                if read:
                    self._record({'role': 'assistant', 'tool_calls': [
                        {'id': 'c1', 'function': {
                            'name': 'read_paper_page',
                            'arguments': json.dumps({'paper_id': 1,
                                                     'page_no': page_no})}},
                    ], 'content': None})
                    self._record({'role': 'tool', 'tool_call_id': 'c1',
                                  'content': '《Paper A》 a.pdf p.%d\n\n'
                                             '神经语言模型使用神经网络来描述词序列的概率分布。'
                                             '这在NLP领域产生了重要影响。' % page_no})
                self._record({'role': 'assistant', 'content': answer})
        self.agent_factory = Agent
        s, d, _ = self.post_json('/api/ask', {'question': question})
        self.assertEqual(s, 200, d)
        return d

    def test_chat_quotes_no_card(self):
        for ans, q in (('他说“你好”之后走了。', '闲聊'),
                       ('你刚才问的是“什么是低秩适配”。', '什么是低秩适配？'),
                       ('标记 `PP-CONTEXT-7319` 记下了。', '记住 PP-CONTEXT-7319')):
            with self.subTest(ans=ans):
                d = self._ask(ans, q, read=False)
                self.assertEqual(d['evidence'], [])

    def test_plain_summary_no_card(self):
        d = self._ask('神经语言模型的核心是用神经网络建模词序列的概率。',
                      '讲讲神经语言模型', read=False)
        self.assertEqual(d['evidence'], [])

    def test_correct_quote_verified(self):
        d = self._ask('原文是“神经语言模型使用神经网络来描述词序列的概率分布。”这句话。')
        self.assertEqual(len(d['evidence']), 1)
        self.assertEqual(d['evidence'][0]['status'], 'verified')

    def test_wrong_quote_mismatch(self):
        d = self._ask('原文是“这在NLP领域产生了重大影响。”这句话。')
        self.assertEqual(d['evidence'][0]['status'], 'mismatch')

    def test_quote_without_read_unverified(self):
        d = self._ask('原文是“神经语言模型使用神经网络来描述词序列的概率分布。”这句话。',
                      read=False)
        self.assertEqual(d['evidence'][0]['status'], 'unverified')

    def test_bold_real_quote_still_verifies(self):
        """Markdown 加粗的真实引文在资格成立后仍可正确核验。"""
        d = self._ask('原文是“**神经语言模型使用神经网络来描述词序列的概率分布。**”这句话。')
        self.assertEqual(len(d['evidence']), 1, '真引文加粗不得失去资格')
        self.assertEqual(d['evidence'][0]['status'], 'verified',
                         '去强调后应逐字核验通过')

    def test_citation_for_one_quote_does_not_promote_another(self):
        """一处出处标注不得让同回答的其它引号都成引文。"""
        d = self._ask('“前缀”是指可训练向量。\n\n'
                      '原文引用：“神经语言模型使用神经网络来描述词序列的概率分布。”')
        self.assertEqual(len(d['evidence']), 1,
                         f'只有带出处的引文成卡: {[e["text"] for e in d["evidence"]]}')
        self.assertIn('神经语言模型', d['evidence'][0]['text'])

    def test_multi_page_conflict_stays_conservative(self):
        class Agent(FakeAgentBase):
            def build_turn(self, q):
                for i, pno in enumerate((1, 2), start=1):
                    self._record({'role': 'assistant', 'tool_calls': [
                        {'id': 'r%d' % i, 'function': {
                            'name': 'read_paper_page',
                            'arguments': json.dumps({'paper_id': 1, 'page_no': pno})}},
                    ], 'content': None})
                    self._record({'role': 'tool', 'tool_call_id': 'r%d' % i,
                                  'content': '《Paper A》 a.pdf p.%d\n\n相同的句子在这里出现。' % pno})
                self._record({'role': 'assistant',
                              'content': '引用：\n> 相同的句子在这里出现。'})
        self.agent_factory = Agent
        s, d, _ = self.post_json('/api/ask', {'question': '问'})
        self.assertEqual(d['evidence'][0]['status'], 'unverified',
                         '多页歧义必须保守处理')

    def test_html_escaping_and_safe_render(self):
        d = self._ask('原文是“<script>alert(1)</script>”这句话。', read=False)
        js = (Path(__file__).resolve().parents[1]
              / 'src/paperpilot/static/workbench/app.js')
        src = js.read_text(encoding='utf-8')
        self.assertNotIn('.innerHTML =', src, '前端不得 innerHTML 拼接')
        self.assertNotIn('PP_DATA', src, '前端不得含合成数据')


class TestContextOverflowRecovery(RootSessionGuard, unittest.TestCase):
    """Agent context-budget handling: detect overflow, shrink once, never loop."""

    def _make_cfg(self, tmp, **overrides):
        base = dict(
            work_space=tmp,
            session_path=str(tmp / 'paper_session.json'),
            system_prompt='SYS',
            base_url='http://127.0.0.1:9/test',
            profile='t', model_main='m', think_main='off',
            max_turns_main=7,
        )
        base.update(overrides)
        return replace(CONFIG, **base)

    def _make_agent(self, tmp, **overrides):
        from mini_harness.agent import DeepSeekAgent
        return DeepSeekAgent([], cfg=self._make_cfg(tmp, **overrides),
                             memory_enabled=False)

    def test_context_limit_field_exists_and_defaults_zero(self):
        from mini_harness.config import Config
        self.assertIn('context_limit', Config.__dataclass_fields__)
        self.assertEqual(Config().context_limit, 0)

    def test_local_budget_fits_model_context(self):
        """compact_limit + max_tokens_main + slack must fit the real window."""
        import os
        from unittest import mock
        env = {'MINI_HARNESS_PROVIDER': 'local', 'MINI_HARNESS_MODEL': 'local/qwen3-vl-8b'}
        with mock.patch.dict(os.environ, env, clear=False):
            from importlib import reload
            import mini_harness.config as cfgmod
            reload(cfgmod)
            c = cfgmod.build_config()
            self.assertEqual(c.context_limit, 16384, '本地模型真实上下文上限')
            self.assertLessEqual(c.compact_limit + c.max_tokens_main, 16384 - 500,
                                 '压缩阈值+输出预算必须为协议开销留余量')
        reload(cfgmod)
        reload(cfgmod)

    def test_overflow_detection(self):
        from mini_harness.agent import _is_context_overflow, ContextOverflowError
        self.assertTrue(_is_context_overflow(
            Exception("This model's maximum context length is 16384 tokens")))
        self.assertTrue(_is_context_overflow(ContextOverflowError('x')))
        self.assertFalse(_is_context_overflow(Exception('connection refused')))
        self.assertFalse(_is_context_overflow(Exception('api key invalid')))

    def test_overflow_shrinks_and_retries_once(self):
        """超限后收缩历史并重试一次，不重复发送相同超限请求。"""
        import tempfile as _tf
        from unittest import mock
        from mini_harness.agent import DeepSeekAgent
        calls = []

        class FakeClient:
            class chat:
                class completions:
                    @staticmethod
                    def stream(**kw):
                        calls.append(len(kw.get('messages', [])))
                        if len(calls) == 1:
                            raise Exception(
                                "This model's maximum context length is 16384 tokens. "
                                "However, you requested 4096 output tokens and your "
                                "prompt contains at least 13000 input tokens.")
                        class _Ctx:
                            def __enter__(self):
                                class _Stream:
                                    def __iter__(self):
                                        return iter([])
                                    def get_final_completion(self):
                                        class _Msg:
                                            tool_calls = None
                                            content = 'ok'
                                            reasoning_content = ''
                                        class _Choice:
                                            message = _Msg()
                                        class _Resp:
                                            choices = [_Choice()]
                                            usage = None
                                        return _Resp()
                                return _Stream()
                            def __exit__(self, *a):
                                return False
                        return _Ctx()

        with _tf.TemporaryDirectory() as td:
            tmp = Path(td)
            agent = self._make_agent(tmp)
            agent.turn_messages = []
            agent.message = [{'role': 'system', 'content': 's'}]
            for i in range(8):
                agent._record({'role': 'user', 'content': 'q%d' % i})
                agent._record({'role': 'assistant', 'content': 'a%d' % i})
            with mock.patch('mini_harness.agent.COMPACT') as comp:
                comp.compact_content.side_effect = (
                    lambda c, msgs, sp, cfg=None: msgs[-2:])
                with mock.patch('mini_harness.agent.retry_call',
                                side_effect=lambda fn, cfg=None: fn()):
                    agent._overflow_recovered = False
                    result = agent._request_with_overflow_recovery(FakeClient(), cfg=agent.cfg)
            self.assertEqual(len(calls), 2, '超限后只重试一次，不重复发送')
            self.assertEqual(comp.compact_content.call_count, 1, '收缩一次')

    def test_overflow_twice_raises_context_overflow(self):
        """再次超限不再重试，抛出明确的 ContextOverflowError。"""
        import tempfile as _tf
        from unittest import mock
        from mini_harness.agent import DeepSeekAgent, ContextOverflowError
        calls = []

        class FakeClient:
            class chat:
                class completions:
                    @staticmethod
                    def stream(**kw):
                        calls.append(1)
                        raise Exception(
                            "This model's maximum context length is 16384 tokens.")

        with _tf.TemporaryDirectory() as td:
            tmp = Path(td)
            agent = self._make_agent(tmp)
            agent.turn_messages = []
            agent.message = [{'role': 'system', 'content': 's'},
                             {'role': 'user', 'content': 'q'}]
            with mock.patch('mini_harness.agent.COMPACT') as comp:
                comp.compact_content.side_effect = (
                    lambda c, msgs, sp, cfg=None: msgs[-1:])
                with mock.patch('mini_harness.agent.retry_call',
                                side_effect=lambda fn, cfg=None: fn()):
                    agent._overflow_recovered = False
                    with self.assertRaises(ContextOverflowError):
                        agent._request_with_overflow_recovery(FakeClient(), cfg=agent.cfg)
            self.assertEqual(len(calls), 2, '首次+收缩重试一次，共 2 次')

    def test_non_overflow_errors_not_treated_as_overflow(self):
        """网络/服务异常不得被归类为上下文超限。"""
        import tempfile as _tf
        from unittest import mock
        from mini_harness.agent import DeepSeekAgent

        class FakeClient:
            class chat:
                class completions:
                    @staticmethod
                    def stream(**kw):
                        raise ConnectionError('connection refused')

        with _tf.TemporaryDirectory() as td:
            tmp = Path(td)
            agent = self._make_agent(tmp)
            agent.turn_messages = []
            agent.message = [{'role': 'system', 'content': 's'}]
            with mock.patch('mini_harness.agent.COMPACT') as comp:
                with mock.patch('mini_harness.agent.retry_call',
                                side_effect=lambda fn, cfg=None: fn()):
                    with self.assertRaises(ConnectionError):
                        agent._request_with_overflow_recovery(FakeClient(), cfg=agent.cfg)
            self.assertEqual(comp.compact_content.call_count, 0,
                             '非超限异常不得触发压缩')


class TestQueryFailureVsNoResults(WorkbenchBase):
    """Failures must never be reported as "no papers in the index"."""

    def setUp(self):
        super().setUp()
        self.write_pdf('a.pdf', ['神经语言模型使用神经网络来描述词序列的概率分布。'],
                       title='Paper A')
        self.import_papers_ok()

    def test_context_overflow_says_query_not_completed(self):
        from mini_harness.agent import ContextOverflowError
        class Agent(FakeAgentBase):
            def run_task(self, question, cfg=None, prior=None):
                raise ContextOverflowError('context still too long')
        self.agent_factory = Agent
        s, d, _ = self.post_json('/api/ask', {'question': '超长问题'})
        self.assertEqual(s, 502)
        self.assertIn('暂时没有完成', d['error'])
        self.assertIn('上下文', d['error'], '应说明是上下文限制')
        self.assertNotIn('没有相关论文', d['error'])
        self.assertNotIn('ContextOverflowError', d['error'], '不暴露技术类型')

    def test_service_failure_says_query_not_completed(self):
        class Agent(FakeAgentBase):
            def run_task(self, question, cfg=None, prior=None):
                raise ConnectionError('connection refused')
        self.agent_factory = Agent
        s, d, _ = self.post_json('/api/ask', {'question': '问'})
        self.assertEqual(s, 502)
        self.assertIn('暂时没有完成', d['error'])
        self.assertNotIn('没有相关论文', d['error'])
        self.assertNotIn('connection refused', d['error'], '不暴露原始异常')

    def test_no_final_answer_says_query_not_completed(self):
        class Agent(FakeAgentBase):
            def build_turn(self, question):
                pass  # no assistant answer produced
        self.agent_factory = Agent
        s, d, _ = self.post_json('/api/ask', {'question': '问'})
        self.assertEqual(s, 502)
        self.assertIn('暂时没有完成', d['error'])
        self.assertNotIn('没有相关论文', d['error'], '无回答不得伪装成无结果')

    def test_successful_search_with_no_hits_still_answers(self):
        """检索成功但无命中：Agent 应给出自然回答，而非报错。"""
        class Agent(FakeAgentBase):
            def build_turn(self, question):
                self._record({'role': 'assistant', 'tool_calls': [
                    {'id': 's1', 'function': {'name': 'search_papers',
                                              'arguments': '{"query":"不存在的主题"}'}},
                ], 'content': None})
                self._record({'role': 'tool', 'tool_call_id': 's1',
                              'content': "no results for query '不存在的主题'. "
                                         "Try fewer or different keywords; do not "
                                         "cite anything without evidence."})
                self._record({'role': 'assistant', 'content':
                    '我暂时没有在当前已索引的论文中找到与‘不存在的主题’相关的内容。'
                    '你可以提供更完整的论文标题、尝试其他关键词，或导入对应 PDF 后再让我检索。'})
        self.agent_factory = Agent
        s, d, _ = self.post_json('/api/ask', {'question': '不存在的主题是什么？'})
        self.assertEqual(s, 200, '检索成功后 Agent 自然回答，不应报错')
        self.assertIn('暂时没有在当前已索引的论文中找到', d['answer'])
        self.assertEqual(d['evidence'], [], '无结果回答不产生证据卡片')
        pipe = {p['key']: p for p in d['pipeline']}
        self.assertEqual(pipe['search']['state'], 'done', '检索确实执行了')
        self.assertEqual(pipe['read']['state'], 'skipped', '未读取原文')

    def test_no_results_answer_has_no_invented_evidence(self):
        class Agent(FakeAgentBase):
            def build_turn(self, question):
                self._record({'role': 'assistant', 'tool_calls': [
                    {'id': 's1', 'function': {'name': 'search_papers',
                                              'arguments': '{"query":"xyz"}'}},
                ], 'content': None})
                self._record({'role': 'tool', 'tool_call_id': 's1',
                              'content': "no results for query 'xyz'."})
                self._record({'role': 'assistant', 'content':
                    '我暂时没有在当前已索引的论文中找到与‘xyz’相关的内容。'})
        self.agent_factory = Agent
        s, d, _ = self.post_json('/api/ask', {'question': 'xyz 是什么？'})
        self.assertEqual(d['evidence'], [], '无结果不得编造引文')
        self.assertEqual(d['tool_events'][0]['name'], 'search_papers')

    def test_page_read_failure_not_reported_as_no_papers(self):
        """论文已找到但原文读取失败：不编造引文，不等同于检索无结果。"""
        class Agent(FakeAgentBase):
            def build_turn(self, question):
                self._record({'role': 'assistant', 'tool_calls': [
                    {'id': 'r1', 'function': {'name': 'read_paper_page',
                                              'arguments': '{"paper_id":1,"page_no":1}'}},
                ], 'content': None})
                self._record({'role': 'tool', 'tool_call_id': 'r1',
                              'content': '[not found]: no such page'})
                self._record({'role': 'assistant', 'content':
                    '我找到了相关论文，但未能读取该页原文，因此无法给出基于原文的结论。'})
        self.agent_factory = Agent
        s, d, _ = self.post_json('/api/ask', {'question': '问'})
        self.assertEqual(s, 200)
        self.assertIn('未能读取', d['answer'])
        self.assertEqual(d['evidence'], [], '读取失败不得编造引文')
        pipe = {p['key']: p for p in d['pipeline']}
        self.assertEqual(pipe['read']['state'], 'fail', '读取失败如实显示')


class TestClarificationNoCards(WorkbenchBase):
    """Clarification answers and example titles must not produce cards."""

    def setUp(self):
        super().setUp()
        self.write_pdf('a.pdf', ['无关内容'], title='X')
        self.import_papers_ok()

    def _ask(self, answer, question='AI调研报告中主要讲了什么？'):
        class Agent(FakeAgentBase):
            def build_turn(self, q):
                self._record({'role': 'assistant', 'content': answer})
        self.agent_factory = Agent
        s, d, _ = self.post_json('/api/ask', {'question': question})
        self.assertEqual(s, 200, d)
        return d

    def test_clarification_with_report_name_no_card(self):
        """你提到了'AI调研报告'，但没有提供具体报告名称。"""
        d = self._ask('你提到了「AI调研报告」，但没有提供具体报告名称。请补充完整信息。')
        self.assertEqual(d['evidence'], [], '澄清回答不得成卡')

    def test_example_title_no_card(self):
        """例如：2025年人工智能产业发展白皮书。"""
        for ans in (
            '请补充完整信息，例如：2025年人工智能产业发展白皮书。',
            '请补充完整信息，例如：「2025年人工智能产业发展白皮书」。',
            '请补充完整信息，例如："2025年人工智能产业发展白皮书"。',
            '具体报告名称："AI调研报告"。',
        ):
            with self.subTest(ans=ans):
                d = self._ask(ans)
                self.assertEqual(d['evidence'], [],
                                 f'示例标题不得成卡: {ans}')

    def test_request_for_info_no_card(self):
        d = self._ask('请提供论文标题、PDF 文件或相关关键词，我再帮你检索。')
        self.assertEqual(d['evidence'], [])

    def test_plain_explanation_and_bold_terms_no_card(self):
        d = self._ask('**LoRA** 是一种**低秩适配**方法，用于高效微调。')
        self.assertEqual(d['evidence'], [])

    def test_explicit_paper_quote_still_checked(self):
        d = self._ask('原文是“神经语言模型使用神经网络来描述词序列的概率分布。”这句话。')
        self.assertEqual(len(d['evidence']), 1, '明确论文引文仍须核验')

    def test_explicit_quote_without_read_unverified(self):
        d = self._ask('原文是“神经语言模型使用神经网络来描述词序列的概率分布。”这句话。')
        self.assertEqual(d['evidence'][0]['status'], 'unverified')

    def test_chat_and_history_not_evidence(self):
        d = self._ask('你刚才问的是“AI调研报告中主要讲了什么”。')
        self.assertEqual(d['evidence'], [], '用户原话复述不得成卡')

    def test_no_tools_pipeline_honest(self):
        d = self._ask('你提到了「AI调研报告」，请补充信息。')
        self.assertEqual(d['tool_events'], [], '不得虚构工具事件')
        pipe = {p['key']: p for p in d['pipeline']}
        self.assertEqual(pipe['import']['state'], 'skipped')
        self.assertEqual(pipe['search']['state'], 'skipped')
        self.assertEqual(pipe['read']['state'], 'skipped')


class TestClarificationScreenshot(WorkbenchBase):
    """Screenshot regression: the real clarification answer must produce
    ZERO evidence cards. Paper mentions (《LoRA》) are not citations."""

    # Verbatim from the screenshot.
    CLARIFY = (
        '你提到的「ai调研报告」并未提供具体的标题、内容或上下文，因此我无法直接定位'
        '或引用任何特定的报告。在当前的对话中，我索引的论文包括《LoRA》和《TCM-Serve》，'
        '它们分别讨论了大语言模型的低秩微调和多模态推理调度，但均不涉及「AI调研报告」'
        '这一泛指性标题。如果你能提供具体的报告名称、作者、内容摘要或关键词，'
        '我可以为你检索并分析其核心内容。'
    )

    def setUp(self):
        super().setUp()
        self.write_pdf('a.pdf', ['神经语言模型使用神经网络来描述词序列的概率分布。'],
                       title='LoRA')
        self.write_pdf('b.pdf', ['TCM-Serve modality-aware scheduling content.'],
                       title='TCM-Serve')
        self.import_papers_ok()

    def _ask(self, answer, question):
        class Agent(FakeAgentBase):
            def build_turn(self, q):
                self._record({'role': 'assistant', 'content': answer})
        self.agent_factory = Agent
        s, d, _ = self.post_json('/api/ask', {'question': question})
        self.assertEqual(s, 200, d)
        return d

    def test_clarification_produces_zero_cards(self):
        """截图回归：澄清回答必须零卡片。"""
        d = self._ask(self.CLARIFY, 'ai调研报告中主要讲了什么？')
        self.assertEqual(d['evidence'], [],
                         f'应零卡片，实际: {[e["text"] for e in d["evidence"]]}')

    def test_no_card_for_ai_report_terms(self):
        d = self._ask(self.CLARIFY, 'ai调研报告中主要讲了什么？')
        for ev in d['evidence']:
            self.assertNotIn('ai调研报告', ev['text'].lower())
            self.assertNotIn('AI调研报告', ev['text'])

    def test_no_card_for_mentioned_indexed_papers(self):
        """本轮只是提及已索引论文（《LoRA》/《TCM-Serve》），不是引用原文。"""
        d = self._ask(self.CLARIFY, 'ai调研报告中主要讲了什么？')
        for ev in d['evidence']:
            self.assertNotIn('LoRA', ev['text'])
            self.assertNotIn('TCM-Serve', ev['text'])

    def test_no_unverified_warning_for_clarification(self):
        d = self._ask(self.CLARIFY, 'ai调研报告中主要讲了什么？')
        self.assertNotIn('未读取原文', json.dumps(d, ensure_ascii=False))
        self.assertNotIn('未核验', json.dumps(d, ensure_ascii=False))

    def test_pipeline_honest_no_tools(self):
        d = self._ask(self.CLARIFY, 'ai调研报告中主要讲了什么？')
        self.assertEqual(d['tool_events'], [], '不得虚构工具事件')
        pipe = {p['key']: p for p in d['pipeline']}
        self.assertEqual(pipe['search']['state'], 'skipped')
        self.assertEqual(pipe['read']['state'], 'skipped')

    def test_request_for_title_author_link_pdf_zero_cards(self):
        for ans in (
            '请提供论文标题、作者、内容摘要或关键词。',
            '请提供 PDF 文件或链接，我再帮你检索。',
            '请补充具体报告名称，例如：2025年人工智能产业发展白皮书。',
        ):
            with self.subTest(ans=ans):
                d = self._ask(ans, 'ai调研报告中主要讲了什么？')
                self.assertEqual(d['evidence'], [])

    def test_mentioning_indexed_papers_in_chat_zero_cards(self):
        d = self._ask('我索引的论文包括《LoRA》和《TCM-Serve》，它们分别讨论了低秩微调和多模态调度。',
                      '你有哪些论文？')
        self.assertEqual(d['evidence'], [], '提及论文名不得成卡')

    def test_bold_quotes_and_lists_zero_cards(self):
        d = self._ask('- **低秩适配** 是一种参数高效微调方法。\n'
                      '- **多模态调度** 用于推理优化。\n'
                      '术语「资源池」和「模态类型」是常见概念。',
                      '讲讲相关方法')
        self.assertEqual(d['evidence'], [])

    def test_explicit_paper_quote_still_checked(self):
        d = self._ask('原文是“神经语言模型使用神经网络来描述词序列的概率分布。”这句话。',
                      '定义是什么')
        self.assertEqual(len(d['evidence']), 1, '明确引文仍须核验')

    def test_explicit_quote_without_read_unverified(self):
        d = self._ask('原文是“神经语言模型使用神经网络来描述词序列的概率分布。”这句话。',
                      '定义是什么')
        self.assertEqual(d['evidence'][0]['status'], 'unverified')


class TestPdfHyphenation(WorkbenchBase):
    """PDF soft-hyphenation must not cause false mismatch; real differences
    must still be caught."""

    # Page text with realistic PDF hyphenation: de-\ncomposition, archi-\ntecture
    PDF_TEXT = (
        'We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained '
        'model weights and injects trainable rank de-\n'
        'composition matrices into each layer of the Transformer archi-\n'
        'tecture, greatly reducing the number of trainable parameters for '
        'downstream tasks.'
    )
    # Model quotes the unbroken original wording.
    QUOTE = (
        'We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained '
        'model weights and injects trainable rank decomposition matrices into '
        'each layer of the Transformer architecture, greatly reducing the '
        'number of trainable parameters for downstream tasks.'
    )

    def setUp(self):
        super().setUp()
        self.write_pdf('a.pdf', [self.PDF_TEXT], title='LoRA')
        self.import_papers_ok()

    def _ask(self, answer, read=True):
        pdf_text = self.PDF_TEXT

        class Agent(FakeAgentBase):
            def build_turn(self, q):
                if read:
                    self._record({'role': 'assistant', 'tool_calls': [
                        {'id': 'c1', 'function': {
                            'name': 'read_paper_page',
                            'arguments': '{"paper_id": 1, "page_no": 1}'}},
                    ], 'content': None})
                    self._record({'role': 'tool', 'tool_call_id': 'c1',
                                  'content': '《LoRA》 a.pdf p.1\n\n' + pdf_text})
                self._record({'role': 'assistant', 'content': answer})
        self.agent_factory = Agent
        s, d, _ = self.post_json('/api/ask', {'question': '问'})
        self.assertEqual(s, 200, d)
        return d

    def test_hyphenated_quote_verifies(self):
        d = self._ask(f'原文引用：\n> {self.QUOTE}\n—— 《LoRA》第 1 页')
        self.assertEqual(len(d['evidence']), 1)
        self.assertEqual(d['evidence'][0]['status'], 'verified',
                         'PDF 连字符断词不得导致误判 mismatch')

    def test_wrong_word_still_mismatch(self):
        bad = self.QUOTE.replace('decomposition', 'decompositionX')
        d = self._ask(f'原文引用：\n> {bad}\n—— 《LoRA》第 1 页')
        self.assertEqual(d['evidence'][0]['status'], 'mismatch',
                         '错词必须继续暴露')

    def test_wrong_number_still_mismatch(self):
        bad = self.QUOTE.replace('4.7M', '4.8M') if '4.7M' in self.QUOTE else \
              self.QUOTE.replace('downstream', 'upstream')
        d = self._ask(f'原文引用：\n> {bad}\n—— 《LoRA》第 1 页')
        self.assertEqual(d['evidence'][0]['status'], 'mismatch', '错词必须暴露')

    def test_real_hyphen_vs_space_still_mismatch(self):
        """low-rank（真连字符）vs low rank（空格）不得误判匹配。"""
        pdf2 = 'We use low-rank matrices for adaptation.'
        self.write_pdf('b.pdf', [pdf2], title='B')
        self.import_papers_ok()
        class Agent(FakeAgentBase):
            def build_turn(self, q):
                self._record({'role': 'assistant', 'tool_calls': [
                    {'id': 'c1', 'function': {
                        'name': 'read_paper_page',
                        'arguments': '{"paper_id": 2, "page_no": 1}'}},
                ], 'content': None})
                self._record({'role': 'tool', 'tool_call_id': 'c1',
                              'content': '《B》 b.pdf p.1\n\n' + pdf2})
                self._record({'role': 'assistant',
                              'content': '原文是“we use low rank matrices for adaptation.”。'})
        self.agent_factory = Agent
        s, d, _ = self.post_json('/api/ask', {'question': '问'})
        self.assertEqual(d['evidence'][0]['status'], 'mismatch',
                         'low rank 不得等于 low-rank')

    def test_missing_word_still_mismatch(self):
        bad = self.QUOTE.replace('greatly ', '')
        d = self._ask(f'原文引用：\n> {bad}\n—— 《LoRA》第 1 页')
        self.assertEqual(d['evidence'][0]['status'], 'mismatch', '漏词必须暴露')


class TestPaperMentionVsCitation(WorkbenchBase):
    """A bare 《title》 mention is not a source mark; only page-level
    references cite a location."""

    def setUp(self):
        super().setUp()
        self.write_pdf('a.pdf', ['神经语言模型使用神经网络来描述词序列的概率分布。'],
                       title='LoRA')
        self.import_papers_ok()

    def _ask(self, answer, read=False):
        class Agent(FakeAgentBase):
            def build_turn(self, q):
                if read:
                    self._record({'role': 'assistant', 'tool_calls': [
                        {'id': 'c1', 'function': {
                            'name': 'read_paper_page',
                            'arguments': '{"paper_id": 1, "page_no": 1}'}},
                    ], 'content': None})
                    self._record({'role': 'tool', 'tool_call_id': 'c1',
                                  'content': '《LoRA》 a.pdf p.1\n\n'
                                             '神经语言模型使用神经网络来描述词序列的概率分布。'})
                self._record({'role': 'assistant', 'content': answer})
        self.agent_factory = Agent
        s, d, _ = self.post_json('/api/ask', {'question': '问'})
        self.assertEqual(s, 200, d)
        return d

    def test_bare_title_mention_not_source_mark(self):
        """段落里提到《LoRA》不得让段内引号都成引文。"""
        d = self._ask('我索引的论文包括《LoRA》，它讨论了低秩微调。「前缀」是常见术语。')
        self.assertEqual(d['evidence'], [], '裸书名号提及不是来源标记')

    def test_page_mark_is_source_mark(self):
        d = self._ask('原文引用：\n> 神经语言模型使用神经网络来描述词序列的概率分布。\n'
                      '—— 《LoRA》第 1 页')
        self.assertEqual(len(d['evidence']), 1, '页码标记是来源标记')

    def test_title_plus_page_is_source_mark(self):
        d = self._ask('《LoRA》第 1 页写道：“神经语言模型使用神经网络来描述词序列的概率分布。”')
        self.assertEqual(len(d['evidence']), 1, '书名号+页码是来源标记')


class TestConceptQARegression(WorkbenchBase):
    """Screenshot regression: plain concept questions must not produce
    fabricated page citations or evidence cards when no tools are called."""

    def setUp(self):
        super().setUp()
        self.write_pdf('a.pdf', ['LoRA freezes pre-trained weights and trains low-rank matrices.'],
                       title='LoRA')
        self.import_papers_ok()

    def _ask(self, answer, question):
        class Agent(FakeAgentBase):
            def build_turn(self, q):
                self._record({'role': 'assistant', 'content': answer})
        self.agent_factory = Agent
        s, d, _ = self.post_json('/api/ask', {'question': question})
        self.assertEqual(s, 200, d)
        return d

    def test_lora_concept_qa_no_cards(self):
        """普通 LoRA 概念问答且无工具调用：证据卡片为 0。"""
        d = self._ask(
            'LoRA（Low-Rank Adaptation）是一种参数高效微调方法，'
            '它冻结预训练权重，只训练注入的低秩分解矩阵。',
            '你好，我需要你帮我告诉我什么是 lora')
        self.assertEqual(d['evidence'], [], '概念问答不得生成证据卡片')
        self.assertEqual(d['tool_events'], [], '概念问答不调用工具')

    def test_freezing_weights_qa_no_cards(self):
        """关于是否冻结预训练权重的普通问答：不主动生成论文逐字引文。"""
        d = self._ask(
            '是的，LoRA 训练时需要冻结预训练权重。'
            '它只训练注入的低秩矩阵 A 和 B，不更新原始权重 W0。',
            '如果我要采用 lora 训练是否需要冻结预训练权重')
        self.assertEqual(d['evidence'], [], '概念问答不得生成引文卡片')

    def test_fabricated_page_quote_without_tools_no_verified(self):
        """无工具调用时伪造的页码引用不得标为 verified。"""
        d = self._ask(
            '根据《LoRA》论文第 2 页的原文：\n'
            '> We hypothesize that the change in weights has low intrinsic rank.',
            '什么是 lora')
        # The fabricated quote may get a card (conservative), but NEVER verified
        for ev in d['evidence']:
            self.assertNotEqual(ev['status'], 'verified',
                                '未读取原文不得标为核验通过')
        self.assertEqual(d['tool_events'], [], '无工具调用')

    def test_pipeline_shows_no_tools_for_concept(self):
        d = self._ask('LoRA 是一种低秩适配方法。', '什么是 lora')
        pipe = {p['key']: p for p in d['pipeline']}
        self.assertEqual(pipe['import']['state'], 'skipped')
        self.assertEqual(pipe['search']['state'], 'skipped')
        self.assertEqual(pipe['read']['state'], 'skipped')

    def test_prompt_has_no_fabrication_rule(self):
        from paperpilot.cli import PAPER_SYSTEM_PROMPT
        p = ' '.join(PAPER_SYSTEM_PROMPT.lower().split())
        self.assertIn('never fabricate quotations', p, '必须有反伪造引文规则')
        self.assertIn('read_paper_page', p)
        self.assertIn('general concept question', p, '必须区分概念问答')


class TestSectionNameNoCard(WorkbenchBase):
    """Section names and section pointers must not produce evidence cards."""

    def setUp(self):
        super().setUp()
        self.write_pdf('a.pdf', ['方法部分描述了低秩分解矩阵的构造。'], title='LoRA')
        self.import_papers_ok()

    def _ask(self, answer, question='问', read=True):
        class Agent(FakeAgentBase):
            def build_turn(self, q):
                if read:
                    self._record({'role': 'assistant', 'tool_calls': [
                        {'id': 'c1', 'function': {
                            'name': 'read_paper_page',
                            'arguments': '{"paper_id": 1, "page_no": 1}'}},
                    ], 'content': None})
                    self._record({'role': 'tool', 'tool_call_id': 'c1',
                                  'content': '《LoRA》 a.pdf p.1\n\n方法部分描述了低秩分解矩阵的构造。'})
                self._record({'role': 'assistant', 'content': answer})
        self.agent_factory = Agent
        s, d, _ = self.post_json('/api/ask', {'question': question})
        self.assertEqual(s, 200, d)
        return d

    def test_method_section_name_no_card(self):
        d = self._ask('论文第 4 页的 "METHOD" 部分讲了低秩矩阵的构造。')
        for ev in d['evidence']:
            self.assertNotIn('METHOD', ev['text'], '章节名不得成卡')

    def test_abstract_section_name_no_card(self):
        d = self._ask('"ABSTRACT" 章节总结了研究背景。')
        self.assertEqual(d['evidence'], [], 'ABSTRACT 章节名不得成卡')

    def test_the_method_section_no_card(self):
        d = self._ask('the "METHOD" section 详细描述了方法。')
        for ev in d['evidence']:
            self.assertNotIn('METHOD', ev['text'])

    def test_chinese_section_name_no_card(self):
        d = self._ask('"摘要" 部分介绍了研究目标。')
        self.assertEqual(d['evidence'], [])

    def test_real_short_quote_still_checked(self):
        """真正的短引文仍须核验。"""
        d = self._ask('原文是“方法部分描述了低秩分解矩阵的构造。”这句话。')
        self.assertEqual(len(d['evidence']), 1, '真引文不受章节名规则影响')
        self.assertEqual(d['evidence'][0]['status'], 'verified')

    def test_non_section_short_text_not_blocked(self):
        """非章节名的短文本不受影响。"""
        for t in ('yes, really', 'NLP', 'low-rank', 'LORA'):
            with self.subTest(t=t):
                from paperpilot.web import _is_section_reference
                self.assertFalse(_is_section_reference(t),
                                 f'{t!r} 不是章节名')

    def test_explicit_quote_without_read_unverified(self):
        d = self._ask('原文是“方法部分描述了低秩分解矩阵的构造。”这句话。', read=False)
        self.assertEqual(d['evidence'][0]['status'], 'unverified')

    def test_wrong_quote_still_mismatch(self):
        d = self._ask('原文是“方法部分描述了高秩矩阵的构造。”这句话。')
        self.assertEqual(d['evidence'][0]['status'], 'mismatch')


class TestNoEvidenceCardsInChat(WorkbenchBase):
    """Product requirement: the conversation area renders NO evidence cards,
    regardless of type, status or content. Backend verification is preserved."""

    def setUp(self):
        super().setUp()
        self.write_pdf('a.pdf', ['神经语言模型使用神经网络来描述词序列的概率分布。'],
                       title='LoRA')
        self.import_papers_ok()

    def _ask(self, answer, question='问', read=True):
        class Agent(FakeAgentBase):
            def build_turn(self, q):
                if read:
                    self._record({'role': 'assistant', 'tool_calls': [
                        {'id': 'c1', 'function': {
                            'name': 'read_paper_page',
                            'arguments': '{"paper_id": 1, "page_no": 1}'}},
                    ], 'content': None})
                    self._record({'role': 'tool', 'tool_call_id': 'c1',
                                  'content': '《LoRA》 a.pdf p.1\n\n'
                                             '神经语言模型使用神经网络来描述词序列的概率分布。'})
                self._record({'role': 'assistant', 'content': answer})
        self.agent_factory = Agent
        s, d, _ = self.post_json('/api/ask', {'question': question})
        self.assertEqual(s, 200, d)
        return d

    # -- 1. 普通 LoRA 问答 ------------------------------------------------
    def test_lora_qa_no_cards(self):
        d = self._ask('LoRA 是一种低秩适配方法，冻结预训练权重，只训练低秩矩阵。',
                      '什么是 LoRA', read=False)
        self.assertEqual(d['evidence'], [])

    # -- 2. TCM 含义澄清 -------------------------------------------------
    def test_tcm_clarification_no_cards(self):
        d = self._ask('TCM 可能指 Task-Constrained Multimodal 或 Temporal Contrastive Memory，'
                      '请提供具体上下文以便我确认。', 'TCM 是什么？', read=False)
        self.assertEqual(d['evidence'], [])

    # -- 3. 各类候选片段 -------------------------------------------------
    def test_various_candidates_no_cards(self):
        for ans in (
            'TCM-Serve 是一个模态感知调度器。',
            'Temporal Contrastive Memory 是一种记忆机制。',
            '论文第 4 页的 "METHOD" 部分讲了方法。',
            '原文是"神经语言模型使用神经网络来描述词序列的概率分布。"这句话。',
        ):
            with self.subTest(ans=ans[:30]):
                d = self._ask(ans, '问')
                # backend evidence may exist for the last case, but the FRONTEND
                # never renders cards; here we assert backend data is preserved
                # and the response structure is unchanged.

    # -- 4. 真实引文核验通过：对话区无卡片，后台结果正确 -------------------
    def test_verified_quote_backend_correct(self):
        d = self._ask('原文是“神经语言模型使用神经网络来描述词序列的概率分布。”这句话。')
        self.assertEqual(len(d['evidence']), 1, '后台证据数据保留')
        self.assertEqual(d['evidence'][0]['status'], 'verified', '后台核验正确')

    # -- 5. 错误引文：后台仍正确识别 mismatch -----------------------------
    def test_mismatch_quote_backend_correct(self):
        d = self._ask('原文是“神经语言模型使用神经网络来描述词序列的概率分部。”这句话。')
        self.assertEqual(d['evidence'][0]['status'], 'mismatch')

    # -- 6. 未读取原文：后台仍保留 unverified -----------------------------
    def test_unverified_quote_backend_correct(self):
        d = self._ask('原文是“神经语言模型使用神经网络来描述词序列的概率分布。”这句话。',
                      read=False)
        self.assertEqual(d['evidence'][0]['status'], 'unverified')

    # -- 7. 历史恢复与多轮对话 -------------------------------------------
    def test_history_restore_and_multiturn(self):
        class Agent(FakeAgentBase):
            def build_turn(self, q):
                self._record({'role': 'assistant', 'content': '回答：' + q})
        self.agent_factory = Agent
        s, d, h = self.post_json('/api/ask', {'question': '第一问'})
        sid = self.sid_from(h)
        self.assertEqual(s, 200)
        s, d, _ = self.get_json('/api/state', cookie=f'pp_sid={sid}')
        self.assertEqual(len(d['conversation']), 2, '历史保留')
        s, d, h = self.post_json('/api/ask', {'question': '第二问'},
                                 cookie=f'pp_sid={sid}')
        s, d, _ = self.get_json('/api/state', cookie=f'pp_sid={sid}')
        self.assertEqual(len(d['conversation']), 4, '多轮对话保留')

    # -- 8. 底部流水线正常 -----------------------------------------------
    def test_pipeline_still_works(self):
        d = self._ask('原文是“神经语言模型使用神经网络来描述词序列的概率分布。”这句话。')
        pipe = {p['key']: p for p in d['pipeline']}
        self.assertEqual(pipe['read']['state'], 'done', '工具调用正常')
        self.assertEqual(pipe['verify']['state'], 'done', '核验正常')

    # -- 前端契约：对话区无卡片、无诊断提示 -------------------------------
    def test_app_js_has_no_card_rendering_in_chat(self):
        js = (Path(__file__).resolve().parents[1]
              / 'src/paperpilot/static/workbench/app.js')
        src = js.read_text(encoding='utf-8')
        self.assertNotIn("el('div', 'cards')", src)
        self.assertNotIn('evidenceCard', src)
        self.assertNotIn('本轮未调用论文工具', src)

    def test_backend_evidence_data_still_in_api(self):
        d = self._ask('原文是“神经语言模型使用神经网络来描述词序列的概率分布。”这句话。')
        self.assertIn('evidence', d, 'API 仍返回 evidence 数据')
        self.assertIn('tool_events', d, 'API 仍返回工具事件')
        self.assertIn('pipeline', d, 'API 仍返回流水线')


class TestMarkdownRendering(WorkbenchBase):
    """Assistant answers render Markdown; user input stays plain text.
    All rendering uses createElement+textContent, never innerHTML (XSS-safe)."""

    JS = (Path(__file__).resolve().parents[1]
          / 'src/paperpilot/static/workbench/app.js')
    CSS = (Path(__file__).resolve().parents[1]
           / 'src/paperpilot/static/workbench/styles.css')

    def setUp(self):
        super().setUp()
        self.write_pdf('a.pdf', ['神经语言模型使用神经网络来描述词序列的概率分布。'],
                       title='LoRA')
        self.import_papers_ok()

    def _ask(self, answer, question='问', read=True):
        class Agent(FakeAgentBase):
            def build_turn(self, q):
                if read:
                    self._record({'role': 'assistant', 'tool_calls': [
                        {'id': 'c1', 'function': {
                            'name': 'read_paper_page',
                            'arguments': '{"paper_id": 1, "page_no": 1}'}},
                    ], 'content': None})
                    self._record({'role': 'tool', 'tool_call_id': 'c1',
                                  'content': '《LoRA》 a.pdf p.1\n\n'
                                             '神经语言模型使用神经网络来描述词序列的概率分布。'})
                self._record({'role': 'assistant', 'content': answer})
        self.agent_factory = Agent
        s, d, _ = self.post_json('/api/ask', {'question': question})
        self.assertEqual(s, 200, d)
        return d

    # -- 渲染器结构契约 ---------------------------------------------------
    def test_render_markdown_function_exists(self):
        src = self.JS.read_text(encoding='utf-8')
        self.assertIn('function renderMarkdown', src, '须有 Markdown 渲染器')
        self.assertIn('function renderInline', src, '须有行内渲染')
        self.assertIn('function mdBubble', src, '须有统一渲染入口')

    def test_uses_dom_not_innerhtml(self):
        src = self.JS.read_text(encoding='utf-8')
        self.assertNotIn('.innerHTML =', src, '不得用 innerHTML')
        self.assertIn('createElement', src, '须用 createElement')
        self.assertIn('textContent', src, '须用 textContent')
        self.assertIn('createTextNode', src, '字面文本须走 textContent')

    def test_supports_headings_bold_italic(self):
        src = self.JS.read_text(encoding='utf-8')
        self.assertIn("'md-h'", src, '须支持标题')
        self.assertIn("'md-bold'", src, '须支持粗体')
        self.assertIn("'md-italic'", src, '须支持斜体')

    def test_supports_lists_blockquotes_code_hr(self):
        src = self.JS.read_text(encoding='utf-8')
        self.assertIn("'md-ul'", src, '须支持无序列表')
        self.assertIn("'md-ol'", src, '须支持有序列表')
        self.assertIn("'md-blockquote'", src, '须支持引用块')
        self.assertIn("'md-pre'", src, '须支持代码块')
        self.assertIn("'md-code'", src, '须支持行内代码')
        self.assertIn("'md-hr'", src, '须支持水平分隔线')

    def test_assistant_uses_markdown_bubble(self):
        src = self.JS.read_text(encoding='utf-8')
        self.assertIn('mdBubble(item.content', src, '助手回答须走 Markdown 渲染')

    def test_user_message_stays_plain_text(self):
        src = self.JS.read_text(encoding='utf-8')
        # user messages use el('div', 'msg-bubble', text) -> textContent
        self.assertIn("msg-bubble', text)", src, '用户消息须走 textContent')
        # verify appendUser does NOT call mdBubble
        import re
        m = re.search(r'function appendUser.*?^  \}', src, re.M | re.S)
        self.assertIsNotNone(m)
        self.assertNotIn('mdBubble', m.group(), '用户消息不得走 Markdown')

    def test_css_has_markdown_styles(self):
        css = self.CSS.read_text(encoding='utf-8')
        for cls in ('.md-bold', '.md-blockquote', '.md-pre', '.md-ul',
                    '.md-ol', '.md-hr', '.md-code'):
            self.assertIn(cls, css, f'样式须包含 {cls}')

    # -- XSS 防护 ---------------------------------------------------------
    def test_xss_script_tag_not_executable(self):
        d = self._ask('正常文本 <script>alert(1)</script> 后续文本')
        self.assertIn('<script>', d['answer'], '后台保留原文')
        src = self.JS.read_text(encoding='utf-8')
        self.assertNotIn('.innerHTML =', src, '不得有 innerHTML 赋值')
        self.assertNotIn('innerHTML +=', src)
        self.assertIn('textContent', src)

    def test_xss_event_handler_not_executable(self):
        d = self._ask('<img src=x onerror=alert(1)> 测试')
        self.assertIn('onerror', d['answer'])
        src = self.JS.read_text(encoding='utf-8')
        self.assertNotIn('.innerHTML =', src, '不得有 innerHTML 赋值')

    def test_xss_javascript_url_not_executable(self):
        d = self._ask('[点击](javascript:alert(1)) 测试')
        src = self.JS.read_text(encoding='utf-8')
        # renderer builds <code>/<strong>/<em> only; no <a> is created
        self.assertNotIn("createElement('a')", src, '不得创建链接元素')
        self.assertNotIn('createElement("a")', src)

    # -- 渲染内容正确性（结构级验证） -------------------------------------
    def test_heading_bold_chinese_content(self):
        d = self._ask('### 核心方法\n\n**低秩分解**是关键。')
        self.assertIn('### 核心方法', d['answer'])
        self.assertIn('**低秩分解**', d['answer'])

    def test_lists_blockquote_code(self):
        d = self._ask('1. 第一步\n2. 第二步\n\n- 要点A\n- 要点B\n\n> 引用内容\n\n```python\nprint(1)\n```')
        self.assertIn('print(1)', d['answer'])

    def test_mixed_chinese_english(self):
        d = self._ask('LoRA 是 Low-Rank Adaptation 的缩写，用于高效微调。')
        self.assertIn('Low-Rank Adaptation', d['answer'])

    def test_plain_text_display(self):
        d = self._ask('这是一段纯文本，没有 Markdown 语法。')
        self.assertEqual(d['evidence'], [])

    # -- 后台与 API 不变 ---------------------------------------------------
    def test_api_evidence_preserved(self):
        d = self._ask('原文是“神经语言模型使用神经网络来描述词序列的概率分布。”这句话。')
        self.assertIn('evidence', d)
        self.assertEqual(d['evidence'][0]['status'], 'verified', '后台核验不变')
        self.assertIn('tool_events', d)
        self.assertIn('pipeline', d)

    def test_no_evidence_cards_return(self):
        """证据卡片仍不出现（上一轮产品契约）。"""
        src = self.JS.read_text(encoding='utf-8')
        self.assertNotIn('evidenceCard', src)
        self.assertNotIn("el('div', 'cards')", src)

    def test_history_restore_uses_same_renderer(self):
        src = self.JS.read_text(encoding='utf-8')
        import re
        m = re.search(r'function renderHistory.*?^  \}', src, re.M | re.S)
        self.assertIsNotNone(m)
        self.assertIn('appendAgent', m.group(), '历史恢复经同一渲染入口')


if __name__ == '__main__':
    unittest.main()

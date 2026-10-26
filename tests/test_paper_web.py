"""Offline tests for the minimal PaperPilot web UI (no model calls, no real data)."""

import logging
import re
import tempfile
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import replace
from http.client import HTTPResponse
from pathlib import Path

from mini_harness.config import CONFIG

from paperpilot import web
from paperpilot.store import default_db_path
from paperpilot.tools import ImportPapersInput, import_papers

from tests._support import RootSessionGuard
from tests.test_paper_pdf_import import make_pdf


class WebTestBase(RootSessionGuard, unittest.TestCase):
    """Isolated workspace + local HTTP server; never touches the project sandbox."""

    def setUp(self):
        super().setUp()
        logging.getLogger('pypdf').setLevel(logging.CRITICAL)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.ws = self.tmp / 'ws'
        (self.ws / 'papers').mkdir(parents=True)
        self.cfg = replace(CONFIG, work_space=self.ws)
        self.server = web.make_server('127.0.0.1', 0, cfg=self.cfg)  # ephemeral port
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def write_pdf(self, name, pages, title=None):
        (self.ws / 'papers' / name).write_bytes(make_pdf(pages, title=title))

    def request(self, path, data=None):
        url = f'http://127.0.0.1:{self.port}{path}'
        if data is None:
            req = urllib.request.Request(url, method='GET')
        else:
            body = urllib.parse.urlencode(data).encode()
            req = urllib.request.Request(url, data=body, method='POST')
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, resp.read().decode('utf-8')
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode('utf-8')


class TestIndex(WebTestBase):
    def test_get_root_returns_html(self):
        status, body = self.request('/')
        self.assertEqual(status, 200)
        # GET / now serves the real workbench (contract change: the old form
        # page was replaced by the workbench UI, which talks to the JSON API).
        self.assertIn('<title>PaperPilot 研究工作台</title>', body)
        self.assertIn('id="chat-scroll"', body)
        self.assertIn('id="paper-list"', body)
        self.assertIn('id="ask-form"', body)
        self.assertIn('id="btn-import"', body)
        self.assertIn('id="btn-reset"', body)
        self.assertNotIn('data.js', body, '合成数据不得进入正式页面')

    def test_unknown_path_is_404(self):
        status, body = self.request('/nope')
        self.assertEqual(status, 404)
        self.assertIn('未找到', body)


class TestImport(WebTestBase):
    def test_import_success_reports_counts(self):
        self.write_pdf('a.pdf', ['hello paper body'], title='Hello Paper')
        status, body = self.request('/import', {'paper_dir': 'papers'})
        self.assertEqual(status, 200)
        self.assertIn('导入完成', body)
        self.assertIn('Hello Paper', body)
        self.assertIn('imported', body)

    def test_import_missing_directory_is_rejected(self):
        status, body = self.request('/import', {'paper_dir': 'nonexistent'})
        self.assertEqual(status, 400)
        self.assertIn('导入被拒绝', body)

    def test_import_rejects_absolute_path(self):
        status, body = self.request('/import', {'paper_dir': '/etc'})
        self.assertEqual(status, 400)
        self.assertIn('绝对路径', body)

    def test_import_rejects_traversal(self):
        status, body = self.request('/import', {'paper_dir': '../outside'})
        self.assertEqual(status, 400)
        self.assertIn('..', body)

    def test_import_rejects_empty_dir(self):
        status, body = self.request('/import', {'paper_dir': '   '})
        self.assertEqual(status, 400)
        self.assertIn('不能为空', body)

    def test_import_shows_failed_files(self):
        bad = self.ws / 'papers' / 'bad.pdf'
        bad.write_bytes(b'%PDF-1.4 broken')
        self.write_pdf('ok.pdf', ['good body'], title='Good Paper')
        status, body = self.request('/import', {'paper_dir': 'papers'})
        self.assertEqual(status, 200)
        self.assertIn('failed', body)
        self.assertIn('Good Paper', body)


class TestAsk(WebTestBase):
    def test_ask_extracts_final_assistant_answer(self):
        class FakeAgent:
            def __init__(self, tools, cfg=None, memory_enabled=True):
                self.message = [
                    {'role': 'user', 'content': 'q'},
                    {'role': 'assistant', 'tool_calls': [
                        {'function': {'name': 'search_papers', 'arguments': '{"query":"x"}'}}
                    ], 'content': None},
                    {'role': 'tool', 'tool_call_id': '1', 'content': 'no results'},
                    {'role': 'assistant', 'content': '这是最终回答。'},
                ]
                self.turn_messages = list(self.message)
            def run_task(self, question, cfg=None, prior=None):
                class R:
                    outcome = 'completed'
                return R()

        orig = web.DeepSeekAgent
        web.DeepSeekAgent = FakeAgent
        self.addCleanup(lambda: setattr(web, 'DeepSeekAgent', orig))

        status, body = self.request('/ask', {'question': '测试问题'})
        self.assertEqual(status, 200)
        self.assertIn('这是最终回答。', body)
        self.assertIn('search_papers', body, 'tool trace should be shown')

    def test_ask_rejects_empty_question(self):
        status, body = self.request('/ask', {'question': '   '})
        self.assertEqual(status, 400)
        self.assertIn('问题为空', body)

    def test_ask_reports_model_failure(self):
        class BoomAgent:
            def __init__(self, tools, cfg=None, memory_enabled=True):
                self.message = []
                self.turn_messages = []
            def run_task(self, question, cfg=None, prior=None):
                raise RuntimeError('model unreachable')

        orig = web.DeepSeekAgent
        web.DeepSeekAgent = BoomAgent
        self.addCleanup(lambda: setattr(web, 'DeepSeekAgent', orig))

        status, body = self.request('/ask', {'question': 'q'})
        self.assertEqual(status, 502)
        # Contract change (task 三): user-facing text names the failure
        # category without leaking the raw exception.
        self.assertIn('查询未完成', body)
        self.assertNotIn('model unreachable', body, '不得向用户暴露原始异常')

    def test_ask_reports_missing_answer(self):
        class EmptyAgent:
            def __init__(self, tools, cfg=None, memory_enabled=True):
                self.message = [{'role': 'user', 'content': 'q'}]
                self.turn_messages = list(self.message)
            def run_task(self, question, cfg=None, prior=None):
                class R:
                    outcome = 'error'
                return R()

        orig = web.DeepSeekAgent
        web.DeepSeekAgent = EmptyAgent
        self.addCleanup(lambda: setattr(web, 'DeepSeekAgent', orig))

        status, body = self.request('/ask', {'question': 'q'})
        self.assertEqual(status, 502)
        # Contract change (task 二/三): a query that produced no answer must
        # NOT be reported as "no papers found" -- it is a failed query.
        self.assertIn('查询未完成', body)
        self.assertNotIn('没有相关论文', body, '失败不得伪装成无结果')

    def test_ask_tolerates_attribute_style_messages(self):
        class ObjAgent:
            def __init__(self, tools, cfg=None, memory_enabled=True):
                class Msg:
                    role = 'assistant'
                    tool_calls = None
                    content = '对象形态回答'
                self.message = [Msg()]
                self.turn_messages = list(self.message)
            def run_task(self, question, cfg=None, prior=None):
                class R:
                    outcome = 'completed'
                return R()

        orig = web.DeepSeekAgent
        web.DeepSeekAgent = ObjAgent
        self.addCleanup(lambda: setattr(web, 'DeepSeekAgent', orig))

        status, body = self.request('/ask', {'question': 'q'})
        self.assertEqual(status, 200)
        self.assertIn('对象形态回答', body)


class TestSecurity(WebTestBase):
    def test_malformed_request_is_rejected(self):
        url = f'http://127.0.0.1:{self.port}/import'
        req = urllib.request.Request(url, data=b'paper_dir=../../etc', method='POST',
                                     headers={'Content-Type': 'application/x-www-form-urlencoded'})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                status, body = resp.status, resp.read().decode()
        except urllib.error.HTTPError as e:
            status, body = e.code, e.read().decode()
        self.assertEqual(status, 400, 'traversal must be rejected')
        self.assertIn('..', body)
        self.assertNotIn('/etc/passwd', body)

    def test_empty_body_rejected(self):
        url = f'http://127.0.0.1:{self.port}/ask'
        req = urllib.request.Request(url, data=b'', method='POST',
                                     headers={'Content-Type': 'application/x-www-form-urlencoded'})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                status, body = resp.status, resp.read().decode()
        except urllib.error.HTTPError as e:
            status, body = e.code, e.read().decode()
        self.assertEqual(status, 400)
        self.assertIn('请求体为空', body)

    def test_unsupported_content_type_rejected(self):
        url = f'http://127.0.0.1:{self.port}/ask'
        req = urllib.request.Request(url, data=b'{"question":"x"}', method='POST',
                                     headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                status, body = resp.status, resp.read().decode()
        except urllib.error.HTTPError as e:
            status, body = e.code, e.read().decode()
        self.assertEqual(status, 400)
        self.assertIn('不支持的内容类型', body)

    def test_oversized_body_rejected(self):
        url = f'http://127.0.0.1:{self.port}/ask'
        big = 'question=' + 'a' * (web.MAX_BODY_BYTES + 100)
        req = urllib.request.Request(url, data=big.encode(), method='POST',
                                     headers={'Content-Type': 'application/x-www-form-urlencoded'})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                status, body = resp.status, resp.read().decode()
        except urllib.error.HTTPError as e:
            status, body = e.code, e.read().decode()
        self.assertEqual(status, 400)
        self.assertIn('请求体过大', body)

    def test_html_is_escaped(self):
        self.write_pdf('x.pdf', ['正文 <script>alert(1)</script> 内容'], title='T<script>')
        status, body = self.request('/import', {'paper_dir': 'papers'})
        self.assertEqual(status, 200)
        self.assertNotIn('<script>alert(1)</script>', body)
        self.assertIn('&lt;script&gt;', body)

    def test_bind_is_localhost_only(self):
        self.assertEqual(web.DEFAULT_HOST, '127.0.0.1')
        self.assertEqual(self.server.server_address[0], '127.0.0.1')

    def test_default_db_stays_in_workspace(self):
        db = default_db_path(self.cfg)
        self.assertTrue(str(db).startswith(str(self.ws)), db)
        self.assertIn('sandbox', str(db))


class TestHelpers(WebTestBase):
    def test_safe_paper_dir_accepts_relative(self):
        value, err = web._safe_paper_dir('papers', cfg=self.cfg)
        self.assertIsNone(err)
        self.assertEqual(value, 'papers')

    def test_safe_paper_dir_rejects_home_expansion(self):
        value, err = web._safe_paper_dir('~/.ssh', cfg=self.cfg)
        self.assertIsNone(value)
        self.assertIsNotNone(err)

    def test_extract_answer_and_trace_handles_empty(self):
        class A:
            message = []
        answer, trace = web._extract_answer_and_trace(A())
        self.assertEqual(answer, '')
        self.assertEqual(trace, [])


class _QuoteHelpers(WebTestBase):
    """Shared fixtures for quotation tests (no tests of its own)."""

    PAGE = '神经语言模型使用神经网络来描述词序列的概率分布。这在NLP领域产生了重要影响。'

    def _agent_with(self, answer, page_body=PAGE, paper_id=1, page_no=3, read=True):
        class A:
            pass
        a = A()
        msgs = []
        if read:
            msgs.append({'role': 'assistant', 'tool_calls': [
                {'id': 'c1', 'function': {'name': 'read_paper_page',
                                          'arguments': f'{{"paper_id": {paper_id}, "page_no": {page_no}}}'}}
            ], 'content': None})
            msgs.append({'role': 'tool', 'tool_call_id': 'c1',
                         'content': f'《AI调研报告》 input/a.pdf p.{page_no}\n\n{page_body}'})
        msgs.append({'role': 'assistant', 'content': answer})
        a.message = msgs
        return a

    def _verify(self, answer, **kw):
        agent = self._agent_with(answer, **kw)
        pages = web._collect_read_pages(agent)
        return web._verify_quotes(answer, pages), pages


class TestQuoteVerification(_QuoteHelpers):
    """Verbatim quotation checks: deterministic, conservative, honest."""

    def test_exact_quote_passes(self):
        answer = '定义如下：\n> 神经语言模型使用神经网络来描述词序列的概率分布。'
        checks, pages = self._verify(answer)
        self.assertEqual(len(checks), 1)
        self.assertEqual(checks[0]['status'], 'verified')
        self.assertEqual(checks[0]['source'], (1, 3))

    def test_one_word_change_fails(self):
        # 重要 -> 重大：一字之差必须判不通过
        answer = '结论：\n> 这在NLP领域产生了重大影响。'
        checks, _ = self._verify(answer)
        self.assertEqual(checks[0]['status'], 'mismatch')
        self.assertNotEqual(checks[0]['status'], 'verified')

    def test_absent_quote_fails(self):
        answer = '引述：\n> 这句话在原文中完全不存在。'
        checks, _ = self._verify(answer)
        self.assertEqual(checks[0]['status'], 'mismatch')

    def test_pdf_line_breaks_and_nfkc_tolerated(self):
        body = '神经语⾔模型使用神经网络来描述词序列的概\n率分布。'
        answer = '原文：\n> 神经语⾔模型使用神经网络来描述词序列的概率分布。'
        checks, _ = self._verify(answer, page_body=body)
        self.assertEqual(checks[0]['status'], 'verified', '换行/兼容汉字不应误报')

    def test_no_case_folding_or_hyphen_stripping(self):
        body = 'The model uses low-rank matrices.'
        a1 = 'See:\n> The model uses low-rank matrices.'
        a2 = 'See:\n> The model uses LOW-RANK matrices.'
        a3 = 'See:\n> The model uses lowrank matrices.'
        c1, _ = self._verify(a1, page_body=body)
        c2, _ = self._verify(a2, page_body=body)
        c3, _ = self._verify(a3, page_body=body)
        self.assertEqual(c1[0]['status'], 'verified')
        self.assertEqual(c2[0]['status'], 'mismatch', '大小写归一是被禁止的')
        self.assertEqual(c3[0]['status'], 'mismatch', '删除连字符掩盖差异是被禁止的')

    def test_without_read_page_not_verified(self):
        answer = '引述：\n> 神经语言模型使用神经网络来描述词序列的概率分布。'
        agent = self._agent_with(answer, read=False)
        pages = web._collect_read_pages(agent)
        checks = web._verify_quotes(answer, pages)
        self.assertEqual(checks[0]['status'], 'unverified')
        # wording updated per the source-display contract (section 4.4)
        self.assertIn('未读取原文，无法逐字核验', checks[0]['note'])

    def test_failed_or_clipped_tool_result_not_verified(self):
        for content in ('[not found]: no paper with paper_id=9',
                        '《X》 a.pdf p.1\n\nbody\n.....clipped at 80000 of 90000 chars',
                        ''):
            class A:
                message = [
                    {'role': 'assistant', 'tool_calls': [
                        {'id': 'c1', 'function': {'name': 'read_paper_page',
                                                  'arguments': '{"paper_id": 1, "page_no": 1}'}}],
                     'content': None},
                    {'role': 'tool', 'tool_call_id': 'c1', 'content': content},
                    {'role': 'assistant', 'content': '引述：\n> 神经语言模型'},
                ]
            pages = web._collect_read_pages(A())
            checks = web._verify_quotes('引述：\n> 神经语言模型', pages)
            self.assertEqual(checks[0]['status'], 'unverified', content[:30])

    def test_multi_page_ambiguity_is_not_verified(self):
        class A:
            message = [
                {'role': 'assistant', 'tool_calls': [
                    {'id': 'c1', 'function': {'name': 'read_paper_page', 'arguments': '{"paper_id":1,"page_no":1}'}},
                    {'id': 'c2', 'function': {'name': 'read_paper_page', 'arguments': '{"paper_id":1,"page_no":2}'}},
                ], 'content': None},
                {'role': 'tool', 'tool_call_id': 'c1', 'content': '《A》 a.pdf p.1\n\n相同的句子在这里出现。'},
                {'role': 'tool', 'tool_call_id': 'c2', 'content': '《A》 a.pdf p.2\n\n相同的句子在这里出现。'},
                {'role': 'assistant', 'content': '引用：\n> 相同的句子在这里出现。'},
            ]
        pages = web._collect_read_pages(A())
        self.assertEqual(len(pages), 2)
        checks = web._verify_quotes('引用：\n> 相同的句子在这里出现。', pages)
        self.assertEqual(checks[0]['status'], 'unverified')
        self.assertIn('多个已读页面', checks[0]['note'])

    def test_explicit_source_disambiguates_multi_page(self):
        answer = ('来自 p.2 的引述：\n> 相同的句子在这里出现。')
        class A:
            message = [
                {'role': 'assistant', 'tool_calls': [
                    {'id': 'c1', 'function': {'name': 'read_paper_page', 'arguments': '{"paper_id":1,"page_no":1}'}},
                    {'id': 'c2', 'function': {'name': 'read_paper_page', 'arguments': '{"paper_id":1,"page_no":2}'}},
                ], 'content': None},
                {'role': 'tool', 'tool_call_id': 'c1', 'content': '《A》 a.pdf p.1\n\n相同的句子在这里出现。'},
                {'role': 'tool', 'tool_call_id': 'c2', 'content': '《A》 a.pdf p.2\n\n相同的句子在这里出现。'},
                {'role': 'assistant', 'content': answer},
            ]
        pages = web._collect_read_pages(A())
        checks = web._verify_quotes(answer, pages)
        self.assertEqual(checks[0]['status'], 'verified')
        self.assertEqual(checks[0]['source'], (1, 2))

    def test_source_annotation_conflict_fails(self):
        answer = '来自 p.7 的引述：\n> 神经语言模型使用神经网络来描述词序列的概率分布。'
        checks, _ = self._verify(answer)  # 实际读的是 p.3
        self.assertEqual(checks[0]['status'], 'mismatch')
        self.assertIn('来源标注', checks[0]['note'])

    def test_failed_check_shows_real_page_text(self):
        answer = '结论：\n> 这在NLP领域产生了重大影响。'
        checks, pages = self._verify(answer)
        self.assertEqual(checks[0]['status'], 'mismatch')
        # 展示的原文必须来自真实工具返回，不得由模型改写
        self.assertIn((1, 3), pages)
        self.assertIn('重要影响', pages[(1, 3)])
        html_out = web.render_ok('回答', answer, [], checks, pages)
        self.assertIn('未通过逐字核验', html_out)
        self.assertIn('重要影响', html_out, '真实原文应可核对')

    def test_paired_quotes_are_verified(self):
        answer = '原文是“神经语言模型使用神经网络来描述词序列的概率分布。”这句话。'
        checks, _ = self._verify(answer)
        self.assertEqual(len(checks), 1)
        self.assertEqual(checks[0]['status'], 'verified')

    def test_apostrophe_is_not_a_quote_boundary(self):
        answer = "The model's design is described in the paper."
        checks, _ = self._verify(answer)
        self.assertEqual(checks, [], '撇号不得当作引文边界')

    def test_unquoted_prose_is_not_claimed_verified(self):
        answer = '这篇论文说神经语言模型使用神经网络。（转述，非引文）'
        checks, _ = self._verify(answer)
        self.assertEqual(checks, [], '普通转述不得宣称逐字核验')

    def test_dynamic_content_is_escaped(self):
        answer = '引述：\n> <script>alert(1)</script>'
        checks, pages = self._verify(answer, page_body='<script>alert(1)</script> 原文')
        html_out = web.render_ok('回答', answer, [], checks, pages)
        self.assertNotIn('<script>alert(1)</script>', html_out)
        self.assertIn('&lt;script&gt;', html_out)


class TestWhitespaceNormalization(_QuoteHelpers):
    """Word-internal spaces must survive; only PDF hard wraps are dropped."""

    def test_word_space_preserved_and_matches(self):
        body = 'we use low rank matrices here.'
        answer = '原文：\n> we use low rank matrices here.'
        checks, _ = self._verify(answer, page_body=body)
        self.assertEqual(checks[0]['status'], 'verified')

    def test_low_rank_never_equals_lowrank(self):
        body = 'we use low rank matrices here.'
        answer = '原文：\n> we use lowrank matrices here.'
        checks, _ = self._verify(answer, page_body=body)
        self.assertEqual(checks[0]['status'], 'mismatch',
                         '删除词间空格会掩盖真实差异，禁止')

    def test_a_b_never_equals_ab(self):
        self.assertEqual(web._normalize_quote('a b'), 'a b')
        self.assertEqual(web._normalize_quote('ab'), 'ab')
        self.assertNotEqual(web._normalize_quote('a b'), web._normalize_quote('ab'))
        checks, _ = self._verify('原文：\n> ab', page_body='a b text')
        self.assertEqual(checks[0]['status'], 'mismatch')

    def test_pdf_hard_wrap_still_verifies(self):
        # PDF 折行可落在中文词内（概\n率）或词边界（low \n rank）
        body = '神经语言模型使用神经网络来描述词序列的概\n率分布。'
        answer = '原文：\n> 神经语言模型使用神经网络来描述词序列的概率分布。'
        checks, _ = self._verify(answer, page_body=body)
        self.assertEqual(checks[0]['status'], 'verified', '中文词内折行应可核验')

    def test_pdf_wrap_with_spaces_still_verifies(self):
        body = 'we use low \n rank matrices here.'
        answer = '原文：\n> we use low rank matrices here.'
        checks, _ = self._verify(answer, page_body=body)
        self.assertEqual(checks[0]['status'], 'verified')

    def test_important_word_difference_still_detected(self):
        # 保留原有关键断言：重要 vs 重大
        answer = '结论：\n> 这在NLP领域产生了重大影响。'
        checks, _ = self._verify(answer)
        self.assertEqual(checks[0]['status'], 'mismatch')

    def test_cjk_wrap_joins_word(self):
        # 中文词内 PDF 折行（概\n率）应可与「概率」匹配
        body = '词序列的概\n率分布。'
        answer = '原文：\n> 词序列的概率分布。'
        checks, _ = self._verify(answer, page_body=body)
        self.assertEqual(checks[0]['status'], 'verified')

    def test_latin_wrap_keeps_word_boundary(self):
        # 英文词间折行（low\nrank）绝不能粘成 lowrank
        self.assertEqual(web._normalize_quote('low\nrank'), 'low rank')
        self.assertNotEqual(web._normalize_quote('low\nrank'), web._normalize_quote('lowrank'))
        body = 'we use low\nrank matrices here.'
        answer = '原文：\n> we use lowrank matrices here.'
        checks, _ = self._verify(answer, page_body=body)
        self.assertEqual(checks[0]['status'], 'mismatch', 'low\\nrank 不得匹配 lowrank')

    def test_latin_wrap_matches_spaced_form(self):
        # 英文词间折行规范化为单空格后，应与「low rank」匹配
        body = 'we use low\nrank matrices here.'
        answer = '原文：\n> we use low rank matrices here.'
        checks, _ = self._verify(answer, page_body=body)
        self.assertEqual(checks[0]['status'], 'verified')


class TestNestedQuotes(_QuoteHelpers):
    """Nested or unbalanced quote marks must not be silently 'verified'."""

    def test_nested_inner_quote_not_truncated_and_verified(self):
        # 外层 “...「inner」...” 不得只截到第一个右引号就判通过
        body = '文中说：“他讲了「内层引文」这部分内容”然后结束。'
        answer = '引用：\n> 文中说：“他讲了「内层引文」这部分内容”然后结束。'
        checks, _ = self._verify(answer, page_body=body)
        self.assertEqual(checks[0]['status'], 'verified', '整体引文应按原文核验')

    def test_nested_pairs_flagged_unreliable(self):
        # 外层引号内含同类左引号 → 保守判未核验
        answer = '他说“外层开始“内层”结束了”'
        checks, _ = self._verify(answer, page_body='随便原文')
        self.assertEqual(checks[0]['status'], 'unverified')
        self.assertIn('嵌套', checks[0]['note'])

    def test_outer_and_inner_not_double_counted(self):
        body = 'x “outer inner” y'
        answer = 'A：“outer inner” B：“inner” C'
        checks, _ = self._verify(answer, page_body=body)
        # 内层 "inner" 是外层子串 → 不得重复计为独立引文
        texts = [web._normalize_quote(c['text']) for c in checks]
        self.assertEqual(len(texts), len(set(texts)))
        self.assertEqual(len(checks), 1, f'应只有一条独立引文: {checks}')

    def test_unbalanced_quote_is_unverified(self):
        checks, _ = self._verify('他说“没有右引号', page_body='原文')
        self.assertEqual(checks, [], '未闭合引号不产生可核验引文')


class TestSourceAttribution(_QuoteHelpers):
    """paper_id/page_no annotations must agree with what was actually read."""

    def test_paper_id_conflict_fails(self):
        answer = '来自 paper_id=2 p.3 的引述：\n> 神经语言模型使用神经网络来描述词序列的概率分布。'
        checks, _ = self._verify(answer)  # 实际读的是 paper_id=1, page_no=3
        self.assertEqual(checks[0]['status'], 'mismatch')
        self.assertIn('来源标注', checks[0]['note'])

    def test_paper_id_agreement_passes(self):
        answer = '来自 paper_id=1 p.3 的引述：\n> 神经语言模型使用神经网络来描述词序列的概率分布。'
        checks, _ = self._verify(answer)
        self.assertEqual(checks[0]['status'], 'verified')
        self.assertEqual(checks[0]['source'], (1, 3))

    def test_page_and_paper_conflict_fails(self):
        answer = '来自 paper_id=1 p.9 的引述：\n> 神经语言模型使用神经网络来描述词序列的概率分布。'
        checks, _ = self._verify(answer)
        self.assertEqual(checks[0]['status'], 'mismatch')


class TestEvidenceCards(_QuoteHelpers):
    """P2: structured evidence cards (M1-M5, no character-level highlight)."""

    def _card_html(self, answer, **kw):
        agent = self._agent_with(answer, **kw)
        pages = web._collect_read_pages(agent)
        checks = web._verify_quotes(answer, pages)
        return web.render_ok('回答', answer, [], checks, pages), checks

    def test_three_statuses_render_distinct_badges(self):
        cases = {
            'verified': '引述：\n> 神经语言模型使用神经网络来描述词序列的概率分布。',
            'mismatch': '引述：\n> 这在NLP领域产生了重大影响。',
            'unverified': '引述：\n> 任意内容',
        }
        for expected, answer in cases.items():
            with self.subTest(status=expected):
                html, checks = self._card_html(answer, read=(expected != 'unverified'))
                self.assertEqual(checks[0]['status'], expected)
                self.assertIn('class="card ', html)
                label = {'verified': '逐字核验通过', 'mismatch': '未通过逐字核验',
                         'unverified': '未核验'}[expected]
                self.assertIn(label, html)

    def test_card_shows_paper_id_and_page(self):
        html, checks = self._card_html('引述：\n> 神经语言模型使用神经网络来描述词序列的概率分布。')
        self.assertEqual(checks[0]['source'], (1, 3))
        self.assertIn('paper_id=1', html)
        self.assertIn('p.3', html)

    def test_source_missing_is_not_invented(self):
        html, checks = self._card_html('引述：\n> 这在NLP领域产生了重大影响。')
        self.assertIsNone(checks[0]['source'])
        # Source-display contract (section 4): checked pages shown honestly,
        # no invented paper id / page, no internal implementation jargon.
        self.assertIn('已检查页面', html)
        self.assertNotIn('不编造', html)

    def test_anchor_and_jump_link_are_consistent(self):
        html, _ = self._card_html('引述：\n> 神经语言模型使用神经网络来描述词序列的概率分布。')
        self.assertIn('href="#src-1-3"', html)
        self.assertIn('id="src-1-3"', html)
        self.assertIn('跳转到原文区域', html)

    def test_anchor_locates_page_region_not_chars(self):
        html, _ = self._card_html('引述：\n> 神经语言模型使用神经网络来描述词序列的概率分布。')
        self.assertIn('未做字符级高亮', html, 'M6 暂缓：不得声称字符级定位')

    def test_unverified_never_shows_passed(self):
        html, _ = self._card_html('引述：\n> 任意内容', read=False)
        self.assertNotIn('逐字核验通过', html)
        self.assertIn('未核验', html)

    def test_mismatch_shows_real_original_text(self):
        html, _ = self._card_html('引述：\n> 这在NLP领域产生了重大影响。')
        self.assertIn('未通过逐字核验', html)
        self.assertIn('重要影响', html, '展示的是工具真实原文')

    def test_nested_quote_keeps_conservative_status(self):
        html, checks = self._card_html('他说“外层开始“内层”结束了”')
        self.assertEqual(checks[0]['status'], 'unverified')
        self.assertIn('未核验', html)
        self.assertNotIn('逐字核验通过', html)

    def test_unbalanced_quote_produces_no_verified_card(self):
        html, checks = self._card_html('他说“没有右引号')
        self.assertEqual(checks, [])
        self.assertNotIn('逐字核验通过', html)

    def test_quote_and_original_are_html_escaped(self):
        html, _ = self._card_html('引述：\n> <script>alert(1)</script>',
                                 page_body='<script>alert(1)</script> 原文')
        self.assertNotIn('<script>alert(1)</script>', html)
        self.assertIn('&lt;script&gt;', html)

    def test_no_read_page_shows_honest_note(self):
        html, _ = self._card_html('引述：\n> 神经语言模型使用神经网络来描述词序列的概率分布。', read=False)
        self.assertIn('未读取原文，无法逐字核验', html)
        self.assertNotIn('逐字核验通过', html)

    def test_same_page_quotes_share_unique_anchor(self):
        """两条引文来自同一页：跳转链接均指向唯一且存在的原文区域。"""
        answer = ('引述A：\n> 神经语言模型使用神经网络来描述词序列的概率分布。\n\n'
                  '引述B：\n> 这在NLP领域产生了重要影响。')
        html, checks = self._card_html(answer)
        self.assertEqual(len(checks), 2)
        self.assertEqual(checks[0]['source'], (1, 3))
        self.assertEqual(checks[1]['source'], (1, 3))
        # 共享原文区域：src-1-3 只出现一次作为 id
        self.assertEqual(html.count('id="src-1-3"'), 1, '原文区域 id 必须唯一')
        # 两个跳转链接都指向它
        self.assertEqual(html.count('href="#src-1-3"'), 2, '同页引文共享跳转目标')
        # 跳转目标确实存在
        self.assertIn('id="src-1-3"', html)

    def test_all_jump_targets_exist_and_are_unique(self):
        """通用断言：每个 href 都有对应唯一 id，且全页 id 不重复。"""
        answer = ('引述A：\n> 神经语言模型使用神经网络来描述词序列的概率分布。\n\n'
                  '引述B：\n> 这在NLP领域产生了重要影响。\n\n'
                  '引述C：\n> 神经语言模型使用神经网络来描述词序列的概率分布。')
        html, checks = self._card_html(answer)
        ids = re.findall(r'id="([^"]+)"', html)
        hrefs = re.findall(r'href="#([^"]+)"', html)
        self.assertEqual(len(ids), len(set(ids)), f'HTML id 必须唯一: {ids}')
        for target in hrefs:
            self.assertIn(target, ids, f'跳转目标 {target} 不存在')
        self.assertGreaterEqual(len(checks), 2)

    def test_card_itself_carries_no_duplicate_id(self):
        """卡片容器不得携带 id，避免同页多卡片重复。"""
        answer = ('引述A：\n> 神经语言模型使用神经网络来描述词序列的概率分布。\n\n'
                  '引述B：\n> 这在NLP领域产生了重要影响。')
        html, _ = self._card_html(answer)
        self.assertEqual(html.count('id="src-1-3"'), 1, '卡片不得复制原文区 id')
        self.assertEqual(html.count('class="card '), 2)


if __name__ == '__main__':
    unittest.main()

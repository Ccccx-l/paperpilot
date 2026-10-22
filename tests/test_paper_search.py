"""Tests for paperpilot.search: ranking, evidence snippets, query safety."""

import logging
import tempfile
import unittest

from dataclasses import replace
from pathlib import Path

from mini_harness.config import CONFIG

from paperpilot.pdf_import import PaperDocument, PageText
from paperpilot.search import (
    BODY_WEIGHT,
    TITLE_WEIGHT,
    build_match_query,
    build_snippet,
    evidence_terms,
    search,
)
from paperpilot.store import PaperStore


def make_doc(title, pages, sha='sha', source_path='papers/x.pdf'):
    return PaperDocument(
        title=title,
        source_path=source_path,
        sha256=sha,
        pages=tuple(PageText(page_no=i + 1, text=t) for i, t in enumerate(pages)),
    )


class SearchTestBase(unittest.TestCase):
    def setUp(self):
        logging.getLogger('pypdf').setLevel(logging.CRITICAL)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.ws = self.tmp / 'ws'
        self.papers_dir = self.ws / 'papers'
        self.papers_dir.mkdir(parents=True)
        self.cfg = replace(CONFIG, work_space=self.ws)
        self.store = PaperStore(self.tmp / 'index.db')
        self.addCleanup(self.store.close)

        self._add('quantum.pdf', '量子计算导论', ['本文研究量子计算的基本原理。'])
        self._add('image.pdf', '图像处理', ['量子在图像处理中的应用较少。'])
        self._add('db.pdf', '数据库系统', ['完全无关的文本内容。'])
        self._add('empty_body.pdf', '量子专题', [''])
        self._add('mixed.pdf', '混合文本', ['GPT-4模型与fts5检索。'])

    def _add(self, name, title, pages):
        # 中文/混合内容走结构化导入（合成 PDF 简单字体无法承载 CJK 字面串）；
        # PDF 往返与路径策略由 store 测试覆盖。
        source_path = f'papers/{name}'
        self.store.import_document(
            make_doc(title, pages, sha=f'sha-{name}', source_path=source_path),
            source_path=source_path,
        )


class TestQueries(SearchTestBase):
    def test_chinese_two_char_query(self):
        results = search(self.store, '量子')
        self.assertEqual({r.title for r in results}, {'量子计算导论', '图像处理', '量子专题'})

    def test_chinese_single_char_query(self):
        results = search(self.store, '量')
        self.assertTrue(results)
        self.assertEqual({r.title for r in results}, {'量子计算导论', '图像处理', '量子专题'})

    def test_chinese_multi_char_query(self):
        results = search(self.store, '量子计算')
        self.assertEqual([r.title for r in results], ['量子计算导论'])

    def test_english_query(self):
        results = search(self.store, 'fts5')
        self.assertEqual([r.title for r in results], ['混合文本'])

    def test_mixed_text_query(self):
        results = search(self.store, 'GPT-4')
        self.assertEqual([r.title for r in results], ['混合文本'])

    def test_no_results_query(self):
        self.assertEqual(search(self.store, '不存在的词汇xyz'), [])

    def test_empty_query_returns_empty(self):
        self.assertEqual(search(self.store, ''), [])

    def test_punctuation_only_query_returns_empty(self):
        self.assertEqual(search(self.store, '，。！？ *** ---'), [])

    def test_top_k_limits_results(self):
        results = search(self.store, '量子', top_k=2)
        self.assertEqual(len(results), 2)


class TestMaliciousQueries(SearchTestBase):
    def test_fts_operators_are_neutralized(self):
        for bad in (
            '"',
            '""""',
            'NEAR(a b, 3)',
            'a OR b',
            'title:量子',
            '量子*',
            '^量子',
            'foo" OR "bar',
            'x OR (y NEAR z)',
            '*',
            '()',
        ):
            with self.subTest(query=bad):
                results = search(self.store, bad)  # must never raise
                self.assertIsInstance(results, list)

    def test_injection_cannot_widen_results(self):
        harmless = search(self.store, '量子')
        injected = search(self.store, '量子" OR "图像')
        self.assertLessEqual({r.title for r in injected}, {r.title for r in harmless} | {'图像处理'})
        self.assertNotIn('数据库系统', {r.title for r in injected})

    def test_match_query_is_fully_quoted(self):
        expr = build_match_query('foo bar')
        self.assertEqual(expr, '"foo" "bar"')
        self.assertEqual(build_match_query('，。'), '')
        for token_expr in build_match_query('a* NEAR').split(' '):
            self.assertTrue(token_expr.startswith('"') and token_expr.endswith('"'))


class TestRanking(SearchTestBase):
    def test_title_weight_beats_body_match(self):
        results = search(self.store, '量子')
        titles = [r.title for r in results]
        self.assertIn('图像处理', titles)
        # 标题权重 8.0 > 正文权重 1.0：标题命中必须全部排在仅正文命中之前
        self.assertLess(titles.index('量子专题'), titles.index('图像处理'))
        self.assertLess(titles.index('量子计算导论'), titles.index('图像处理'))
        # 两个标题命中之间的次序由 BM25 长度归一化决定（短标题更密），
        # 这里只锁定 bm25 距离语义：分数越小越相关
        scores = [r.score for r in results]
        self.assertEqual(scores, sorted(scores))

    def test_weights_are_explicit(self):
        self.assertEqual(TITLE_WEIGHT, 8.0)
        self.assertEqual(BODY_WEIGHT, 1.0)
        self.assertGreater(TITLE_WEIGHT, BODY_WEIGHT)

    def test_result_fields_are_complete(self):
        results = search(self.store, '量子计算')
        row = results[0]
        self.assertIsInstance(row.paper_id, int)
        self.assertEqual(row.title, '量子计算导论')
        self.assertEqual(row.page_no, 1)
        self.assertIsInstance(row.score, float)


class TestEvidenceSnippets(SearchTestBase):
    def test_snippet_is_original_text_not_segmented(self):
        results = search(self.store, '量子计算')
        snippet = results[0].snippet
        self.assertIn('量子计算', snippet, 'evidence must contain the raw run')
        self.assertNotIn('量 子', snippet, 'segmented text must not appear as evidence')
        self.assertIn('[量子计算]', snippet)

    def test_snippet_marks_real_positions_only(self):
        snippet = build_snippet('前文研究量子计算方法后文', evidence_terms('量子计算'))
        self.assertIn('研究[量子计算]方法', snippet)
        self.assertEqual(build_snippet('没有命中内容', evidence_terms('量子')), '')

    def test_title_only_hit_has_empty_snippet(self):
        results = search(self.store, '量子专题')
        self.assertEqual(results[0].title, '量子专题')
        self.assertEqual(results[0].snippet, '', 'no body match -> no fabricated snippet')

    def test_snippet_preserves_original_characters(self):
        body = 'GPT-4模型与fts5检索。'
        snippet = build_snippet(body, evidence_terms('fts5'))
        self.assertIn('[fts5]', snippet)
        self.assertIn('GPT-4模型', snippet.replace('…', ''))

    def test_evidence_terms_are_original_substrings(self):
        terms = evidence_terms('量子计算')
        self.assertIn('量子', terms)
        self.assertIn('量子计算', terms)
        self.assertTrue(all(' ' not in t for t in terms))

    def test_page_id_maps_back_to_original_page(self):
        results = search(self.store, '量子计算')
        page = self.store.page_by_id(results[0].page_id)
        self.assertEqual(page['body'], '本文研究量子计算的基本原理。')
        self.assertEqual(page['page_no'], results[0].page_no)


class TestCompatHanEvidence(unittest.TestCase):
    """Compatibility characters in raw bodies: real hits, real highlights."""

    COMPAT_BODY = '本⽂调研⼤语⾔模型的发展'

    def setUp(self):
        logging.getLogger('pypdf').setLevel(logging.CRITICAL)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.store = PaperStore(self.tmp / 'index.db')
        self.addCleanup(self.store.close)
        self.store.import_document(
            make_doc('兼容汉字论文', [self.COMPAT_BODY], sha='compat',
                     source_path='papers/c.pdf'),
            source_path='papers/c.pdf',
        )
        self.store.import_document(
            make_doc('连字论文', ['ﬁne-tuning methods'], sha='lig',
                     source_path='papers/f.pdf'),
            source_path='papers/f.pdf',
        )

    def test_normal_query_hits_compat_index_text(self):
        results = search(self.store, '大语言模型')
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, '兼容汉字论文')
        self.assertEqual(results[0].page_no, 1)

    def test_snippet_marks_real_spans_in_raw_compat_body(self):
        result = search(self.store, '大语言模型')[0]
        self.assertIn('本⽂调研[⼤语⾔模型]', result.snippet, 'highlight must cover the raw span')
        self.assertIn('⼤', result.snippet, 'raw compatibility characters must be preserved')
        self.assertEqual(result.status, 'highlighted')
        page = self.store.page_by_id(result.page_id)
        self.assertEqual(page['body'], self.COMPAT_BODY, 'pages.body stays raw')

    def test_length_changing_normalization_never_fakes_highlights(self):
        results = search(self.store, 'fine')
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.title, '连字论文')
        self.assertEqual(result.snippet, '', 'no proven positions -> no highlight')
        self.assertEqual(result.status, 'body_matched_no_highlight')

    def test_evidence_terms_normalized(self):
        self.assertEqual(evidence_terms('⼤语⾔模型'), evidence_terms('大语言模型'))
        self.assertIn('大语言模型', evidence_terms('⼤语⾔模型'))
        self.assertNotIn('⼤', evidence_terms('⼤语⾔模型'))


if __name__ == '__main__':
    unittest.main()

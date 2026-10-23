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
    MIN_LOOSE_TERM_HITS,
    TITLE_WEIGHT,
    build_loose_match,
    build_match_query,
    build_snippet,
    discriminative_terms,
    evidence_terms,
    min_term_hits,
    search,
    term_coverage,
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


class TestLooseTerms(unittest.TestCase):
    """Loose-pass vocabulary: distinctive only, never bare CJK unigrams."""

    def test_cjk_run_yields_bigrams_not_unigrams(self):
        terms = discriminative_terms('统计语言模型')
        self.assertEqual(terms, ['统计', '计语', '语言', '言模', '模型'])
        self.assertNotIn('统', terms)
        self.assertNotIn('语', terms)

    def test_latin_words_are_whole_runs(self):
        self.assertEqual(
            discriminative_terms('low-rank weight updates'),
            ['low', 'rank', 'weight', 'updates'],
        )

    def test_single_char_cjk_query_keeps_its_unigram(self):
        self.assertEqual(discriminative_terms('量'), ['量'])

    def test_duplicates_removed_order_stable(self):
        self.assertEqual(discriminative_terms('语言模型 语言'), ['语言', '言模', '模型'])

    def test_threshold_is_two_for_multi_term_queries(self):
        self.assertEqual(MIN_LOOSE_TERM_HITS, 2)
        self.assertEqual(min_term_hits(0), 0)
        self.assertEqual(min_term_hits(1), 1)
        for count in (2, 3, 5, 8):
            self.assertEqual(min_term_hits(count), 2)

    def test_coverage_is_token_exact_not_substring(self):
        self.assertEqual(term_coverage({'weights'}, ['weight']), 0)
        self.assertEqual(term_coverage({'weight'}, ['weights']), 0)
        self.assertEqual(term_coverage({'weight', 'weights'}, ['weight', 'weights']), 2)
        # casefolded membership, same as FTS5 unicode61 folding
        self.assertEqual(term_coverage({'transformer'}, ['Transformer']), 1)

    def test_loose_match_quotes_every_term(self):
        self.assertEqual(build_loose_match(['alpha', 'beta']), '"alpha" OR "beta"')
        self.assertEqual(build_loose_match([]), '')
        for piece in build_loose_match(['a* NEAR', '"x"']).split(' OR '):
            self.assertTrue(piece.startswith('"') and piece.endswith('"'))


class TestLooseRetrieval(SearchTestBase):
    """Loose candidates rescue AND misses and compete on score, not arrival order."""

    def setUp(self):
        super().setUp()
        # 严格 AND 会因跨词边界 bigram '计语' 缺失而零命中；
        # 页面实际说的是「统计学习方法的语言模型」。
        self._add('slm.pdf', '统计方法', ['本文采用基于统计学习方法的语言模型建模。'])
        self._add('slm_exact.pdf', '语言模型', ['统计语言模型是常见基线模型。'])
        self._add('noise.pdf', '噪声页', ['这里只提到语言两个字而已。'])
        self._add('one_word.pdf', '单词页', ['quantum appears once here.'])

    def test_loose_rescues_and_zero_hit_query(self):
        results = search(self.store, '统计语言模型')
        titles = [r.title for r in results]
        self.assertIn('统计方法', titles, 'wording variant must be recalled')
        self.assertEqual(
            search(self.store, '统计语言模型', top_k=1)[0].title, '语言模型',
            'exact phrase page still ranks first',
        )

    def test_loose_fires_even_when_strict_already_hits(self):
        # 严格路径已能命中「语言模型」页，但不能因此放弃「统计方法」页
        strict_only = build_match_query('统计语言模型')
        self.assertTrue(strict_only)
        titles = [r.title for r in search(self.store, '统计语言模型')]
        self.assertIn('语言模型', titles)
        self.assertIn('统计方法', titles)

    def test_partial_coverage_below_threshold_is_refused(self):
        # 「噪声页」只覆盖 bigram「语言」= 1 < 2
        results = search(self.store, '统计语言模型')
        self.assertNotIn('噪声页', [r.title for r in results])

    def test_loose_page_can_enter_a_full_top_k(self):
        # 严格结果已填满 top_k=1，宽松页仍不能被钉死在末尾之外
        results = search(self.store, '统计语言模型', top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, '语言模型')
        # top_k=2 时第二席由分数决定，「统计方法」必须在候选池中参与竞争
        two = search(self.store, '统计语言模型', top_k=2)
        self.assertEqual([r.title for r in two], ['语言模型', '统计方法'])

    def test_negative_multi_term_query_stays_empty(self):
        results = search(self.store, 'quantum computing applications')
        self.assertEqual(results, [], 'coverage 1/3 must not pass min_hits=2')

    def test_single_term_query_still_retrieves(self):
        results = search(self.store, 'quantum')
        self.assertEqual([r.title for r in results], ['单词页'])

    def test_merged_results_keep_evidence_status(self):
        rescued = [r for r in search(self.store, '统计语言模型') if r.title == '统计方法'][0]
        self.assertEqual(rescued.status, 'highlighted')
        self.assertIn('[统计', rescued.snippet.replace('[统计学习', '[统计'))
        self.assertTrue(rescued.snippet, 'rescued page must still carry 原文 evidence')

    def test_scores_are_sorted_ascending(self):
        scores = [r.score for r in search(self.store, '统计语言模型')]
        self.assertEqual(scores, sorted(scores))


if __name__ == '__main__':
    unittest.main()

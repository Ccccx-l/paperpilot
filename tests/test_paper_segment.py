"""Tests for paperpilot.segment: index and query must share one rule."""

import unittest

from paperpilot.segment import segment, segment_tokens


class TestSegmentBasics(unittest.TestCase):
    def test_empty_string(self):
        self.assertEqual(segment(''), '')
        self.assertEqual(segment_tokens(''), [])

    def test_whitespace_only(self):
        self.assertEqual(segment('   \n\t '), '')

    def test_english(self):
        self.assertEqual(segment('Hello World'), 'Hello World')

    def test_numbers(self):
        self.assertEqual(segment('2024'), '2024')
        self.assertEqual(segment('3.14'), '3 14')

    def test_chinese_unigram_and_bigram(self):
        self.assertEqual(
            segment('研究量子计算'),
            '研 究 量 子 计 算 研究 究量 量子 子计 计算',
        )

    def test_single_chinese_char(self):
        self.assertEqual(segment('量'), '量')

    def test_two_chinese_chars(self):
        self.assertEqual(segment('量子'), '量 子 量子')

    def test_mixed_chinese_english_digits(self):
        self.assertEqual(segment('GPT-4模型'), 'GPT 4 模 型 模型')

    def test_punctuation_is_separator(self):
        self.assertEqual(segment('你好，世界！'), '你 好 你好 世 界 世界')

    def test_punctuation_only(self):
        self.assertEqual(segment('，。！？'), '')
        self.assertEqual(segment('...---___'), '')

    def test_ascii_underscores_split(self):
        self.assertEqual(segment('state_of'), 'state of')


class TestSegmentSymmetry(unittest.TestCase):
    """Query tokens must be a subset of the tokens of the matched text."""

    def test_two_char_query_subset(self):
        query_tokens = set(segment_tokens('量子'))
        doc_tokens = set(segment_tokens('研究量子计算方法'))
        self.assertTrue(query_tokens <= doc_tokens)

    def test_single_char_query_subset(self):
        self.assertTrue(set(segment_tokens('量')) <= set(segment_tokens('研究量子计算方法')))

    def test_unrelated_query_not_subset(self):
        self.assertFalse(set(segment_tokens('检索')) <= set(segment_tokens('研究量子计算方法')))

    def test_same_rule_for_index_and_query(self):
        for text in ('研究量子计算', 'GPT-4模型', 'Hello 世界', '2024年'):
            self.assertEqual(segment(text), ' '.join(segment_tokens(text)))

    def test_deterministic(self):
        self.assertEqual(segment('量子计算'), segment('量子计算'))


class TestSegmentTokenShape(unittest.TestCase):
    def test_no_punctuation_leaks_into_tokens(self):
        for token in segment_tokens('你好，世界！(test) 2024。'):
            self.assertFalse(any(ch in token for ch in '，！（）。,()!'))

    def test_unicode_ranges(self):
        self.assertEqual(segment('かな'), 'か な かな')
        self.assertEqual(segment('한국'), '한 국 한국')


class TestSegmentCjkBlockPunctuation(unittest.TestCase):
    """Punctuation inside CJK code blocks is a separator, not a token."""

    def test_katakana_middle_dot_is_separator(self):
        self.assertEqual(segment('オ・ト'), 'オ ト')

    def test_double_hyphen_is_separator(self):
        self.assertEqual(segment('ア゠イ'), 'ア イ')

    def test_prolonged_sound_mark_stays_in_run(self):
        self.assertEqual(segment('ラーメン'), 'ラ ー メ ン ラー ーメ メン')

    def test_cjk_punct_only_input_is_empty(self):
        self.assertEqual(segment('・゠'), '')

    def test_cjk_punct_splits_runs_consistently(self):
        # 查询端与索引端同一规则：被标点分开的两段各自成 run
        self.assertEqual(segment_tokens('中・心'), ['中', '心'])
        self.assertTrue(set(segment_tokens('中心')) <= set(segment_tokens('中心方法')))


class TestSegmentNFKC(unittest.TestCase):
    """NFKC folds compatibility characters before tokenizing (index == query)."""

    def test_compat_han_equals_normal(self):
        self.assertEqual(segment('⼤语⾔模型'), segment('大语言模型'))
        self.assertEqual(segment('⼤语⾔模型'), '大 语 言 模 型 大语 语言 言模 模型')

    def test_compat_query_tokens_subset_of_compat_doc(self):
        self.assertTrue(set(segment_tokens('⼤语')) <= set(segment_tokens('⼤语⾔模型')))

    def test_fullwidth_normalized(self):
        self.assertEqual(segment('ＡＢＣ１２３'), 'ABC123')

    def test_ligature_normalized(self):
        self.assertEqual(segment('ﬁne'), 'fine')

    def test_single_compat_char(self):
        self.assertEqual(segment('⼤'), '大')

    def test_existing_behaviors_unchanged(self):
        self.assertEqual(segment('研究量子计算'), '研 究 量 子 计 算 研究 究量 量子 子计 计算')
        self.assertEqual(segment('Hello World'), 'Hello World')
        self.assertEqual(segment('3.14'), '3 14')
        self.assertEqual(segment('GPT-4模型'), 'GPT 4 模 型 模型')
        self.assertEqual(segment('你好，世界！'), '你 好 你好 世 界 世界')
        self.assertEqual(segment(''), '')


if __name__ == '__main__':
    unittest.main()

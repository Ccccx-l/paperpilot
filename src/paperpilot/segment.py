"""Search-normalized text segmentation for CJK-aware FTS5 indexing.

Index and query must go through exactly the same function, so that the
tokens of a query are always a subset of the tokens of the text it matches.

Normalization: before tokenizing, text is passed through
``unicodedata.normalize("NFKC", text)``. NFKC folds compatibility variants
(e.g. '⼤' -> '大', 'ＡＢＣ' -> 'ABC', 'ﬁne' -> 'fine') so that the index and
the query agree on one canonical form. This normalization serves ONLY the
search index and query tokens. It must never be used to rewrite the raw PDF
page text stored in ``pages.body``, which stays exactly as extracted.

Rules (applied after NFKC):
- A run of CJK token characters (Han / Hiragana / Katakana / Hangul letters,
  numbers and marks) emits every character (unigram) followed by every
  adjacent pair (bigram).
- A run of non-CJK alphanumerics emits the run as one token, unchanged.
- Everything else is a separator and is dropped: whitespace, punctuation,
  symbols, and the punctuation/symbols that happen to sit inside CJK code
  blocks (e.g. '・' U+30FB, '゠' U+30A0). The prolonged sound mark 'ー'
  (U+30FC, category Lm) is a letter and stays in the CJK run.

Examples:
    '研究量子计算'   -> '研 究 量 子 计 算 研究 究量 量子 子计 计算'
    '量子'           -> '量 子 量子'
    'GPT-4模型'      -> 'GPT 4 模 型 模型'
    'オ・ト'         -> 'オ ト'
    '⼤语⾔模型'     -> '大 语 言 模 型 大语 语言 言模 模型'
    ''               -> ''

FTS5 query assembly (quoting / escaping) is intentionally NOT done here.
The search layer must quote each token before building a MATCH expression;
never concatenate raw user text into an FTS5 query.
"""

import unicodedata

# Version of the indexing rules implemented here. Databases whose stored
# segment_version is lower get their FTS index rebuilt on open; higher
# versions are refused. Bump this whenever tokenization changes.
SEGMENT_VERSION = 2

_CJK_RANGES = (
    (0x3040, 0x30FF),    # Hiragana + Katakana
    (0x3400, 0x4DBF),    # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
    (0xAC00, 0xD7AF),    # Hangul Syllables
    (0x20000, 0x2FA1F),  # CJK Unified Ideographs Extension B+
)


def is_cjk(char: str) -> bool:
    """True for CJK *token* characters: letters, numbers and marks.

    Punctuation, symbols and unassigned code points inside CJK blocks
    (e.g. '・', '゠') are NOT token characters and act as separators.
    """
    code = ord(char)
    if not any(low <= code <= high for low, high in _CJK_RANGES):
        return False
    return unicodedata.category(char)[0] in 'LMN'


def segment_tokens(text: str) -> list[str]:
    """Split text into search tokens. Used for both indexing and querying."""
    text = unicodedata.normalize("NFKC", text)
    tokens: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        char = text[i]
        if is_cjk(char):
            j = i
            while j < n and is_cjk(text[j]):
                j += 1
            run = text[i:j]
            tokens.extend(run)
            tokens.extend(run[k:k + 2] for k in range(len(run) - 1))
            i = j
        elif char.isalnum():
            j = i
            while j < n and text[j].isalnum() and not is_cjk(text[j]):
                j += 1
            tokens.append(text[i:j])
            i = j
        else:
            i += 1
    return tokens


def segment(text: str) -> str:
    """Space-joined segment_tokens(); identical rule for index and query."""
    return ' '.join(segment_tokens(text))

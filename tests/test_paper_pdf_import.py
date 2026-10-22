"""Tests for paperpilot.pdf_import: structured extraction, no side effects.

All PDFs are tiny synthetic files built in a temporary directory; no real
paper is committed to the repository.
"""

import hashlib
import logging
import tempfile
import unittest

from pathlib import Path

from paperpilot.pdf_import import (
    MAX_PDF_PAGES,
    PaperDocument,
    PaperImportError,
    PageText,
    extract_paper,
)


def make_pdf(page_texts: list[str], title: str | None = None) -> bytes:
    """Build a minimal valid PDF with one content stream per page."""
    count = len(page_texts)
    font_id = 3
    page_ids = [4 + 2 * i for i in range(count)]
    content_ids = [5 + 2 * i for i in range(count)]
    objects: list[tuple[int, bytes]] = [
        (1, b'<< /Type /Catalog /Pages 2 0 R >>'),
        (2, f'<< /Type /Pages /Kids [{" ".join(f"{n} 0 R" for n in page_ids)}] /Count {count} >>'.encode()),
        (3, b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>'),
    ]
    for i, text in enumerate(page_texts):
        objects.append((
            page_ids[i],
            f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] '
            f'/Contents {content_ids[i]} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>'.encode(),
        ))
        escaped = text.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
        stream = f'BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET'.encode()
        objects.append((
            content_ids[i],
            b'<< /Length ' + str(len(stream)).encode() + b' >>\nstream\n' + stream + b'\nendstream',
        ))

    info_id = None
    if title is not None:
        info_id = max(num for num, _ in objects) + 1
        escaped = title.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
        objects.append((info_id, f'<< /Title ({escaped}) >>'.encode()))

    out = bytearray(b'%PDF-1.4\n')
    offsets = {}
    for num, body in objects:
        offsets[num] = len(out)
        out += f'{num} 0 obj\n'.encode() + body + b'\nendobj\n'

    xref_pos = len(out)
    size = max(offsets) + 1
    out += f'xref\n0 {size}\n'.encode()
    out += b'0000000000 65535 f \n'
    for num in range(1, size):
        out += f'{offsets.get(num, 0):010d} 00000 n \n'.encode()
    trailer = f'trailer\n<< /Size {size} /Root 1 0 R'
    if info_id is not None:
        trailer += f' /Info {info_id} 0 R'
    out += (trailer + ' >>\n').encode()
    out += f'startxref\n{xref_pos}\n%%EOF\n'.encode()
    return bytes(out)


class TestPdfExtraction(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def write_pdf(self, name, page_texts, title=None) -> Path:
        target = self.tmp / name
        target.write_bytes(make_pdf(page_texts, title=title))
        return target

    def test_single_page_extraction(self):
        path = self.write_pdf('one.pdf', ['Hello PDF'])
        doc = extract_paper(path)
        self.assertIsInstance(doc, PaperDocument)
        self.assertEqual(doc.page_count, 1)
        self.assertEqual(doc.pages[0], PageText(page_no=1, text='Hello PDF'))

    def test_page_numbers_start_at_one_and_map(self):
        path = self.write_pdf('three.pdf', ['page-one-text', 'page-two-text', 'page-three-text'])
        doc = extract_paper(path)
        self.assertEqual([p.page_no for p in doc.pages], [1, 2, 3])
        self.assertEqual([p.text for p in doc.pages], ['page-one-text', 'page-two-text', 'page-three-text'])

    def test_title_from_metadata(self):
        path = self.write_pdf('meta.pdf', ['body'], title='My Great Paper')
        self.assertEqual(extract_paper(path).title, 'My Great Paper')

    def test_title_falls_back_to_filename(self):
        path = self.write_pdf('fallback-name.pdf', ['body'])
        self.assertEqual(extract_paper(path).title, 'fallback-name')

    def test_blank_title_falls_back_to_filename(self):
        path = self.write_pdf('blank-title.pdf', ['body'], title='   ')
        self.assertEqual(extract_paper(path).title, 'blank-title')

    def test_sha256_matches_file_bytes(self):
        path = self.write_pdf('hash.pdf', ['body'], title='T')
        doc = extract_paper(path)
        self.assertEqual(doc.sha256, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_blank_page_is_kept(self):
        path = self.write_pdf('blank.pdf', ['first', '', 'third'])
        doc = extract_paper(path)
        self.assertEqual(doc.page_count, 3)
        self.assertEqual(doc.pages[1], PageText(page_no=2, text=''))
        self.assertEqual(doc.pages[0].text, 'first')
        self.assertEqual(doc.pages[2].text, 'third')

    def test_extraction_does_not_modify_source(self):
        path = self.write_pdf('read-only.pdf', ['body'], title='T')
        before = path.read_bytes()
        extract_paper(path)
        self.assertEqual(path.read_bytes(), before)

    def test_source_path_recorded(self):
        path = self.write_pdf('src.pdf', ['body'])
        self.assertEqual(extract_paper(path).source_path, str(path))


class TestPdfErrors(unittest.TestCase):
    def setUp(self):
        logging.getLogger('pypdf').setLevel(logging.CRITICAL)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_corrupt_pdf_raises(self):
        path = self.tmp / 'corrupt.pdf'
        path.write_bytes(b'%PDF-1.4 this is not a real pdf body')
        with self.assertRaises(PaperImportError) as ctx:
            extract_paper(path)
        self.assertIn('corrupt or unreadable PDF', str(ctx.exception))

    def test_non_pdf_bytes_raise(self):
        path = self.tmp / 'notes.pdf'
        path.write_text('just some text', encoding='utf-8')
        with self.assertRaises(PaperImportError):
            extract_paper(path)

    def test_oversize_pdf_raises(self):
        path = self.tmp / 'big.pdf'
        path.write_bytes(make_pdf(['x' * 100]))
        with self.assertRaises(PaperImportError) as ctx:
            extract_paper(path, max_bytes=10)
        self.assertIn('oversize PDF', str(ctx.exception))

    def test_zero_pages_raises(self):
        path = self.tmp / 'empty.pdf'
        path.write_bytes(make_pdf([]))
        with self.assertRaises(PaperImportError) as ctx:
            extract_paper(path)
        self.assertIn('abnormal page count', str(ctx.exception))

    def test_too_many_pages_raises(self):
        path = self.tmp / 'many.pdf'
        path.write_bytes(make_pdf(['a', 'b', 'c']))
        with self.assertRaises(PaperImportError) as ctx:
            extract_paper(path, max_pages=2)
        self.assertIn('exceeds limit 2', str(ctx.exception))

    def test_missing_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            extract_paper(self.tmp / 'no-such-file.pdf')

    def test_failed_parse_produces_no_document(self):
        path = self.tmp / 'half.pdf'
        path.write_bytes(b'%PDF-1.4 broken')
        result = None
        try:
            result = extract_paper(path)
        except PaperImportError:
            pass
        self.assertIsNone(result)

    def test_default_limits_are_documented_constants(self):
        self.assertGreater(MAX_PDF_PAGES, 0)


class TestPdfEncryptedAndPageFailure(unittest.TestCase):
    def setUp(self):
        logging.getLogger('pypdf').setLevel(logging.CRITICAL)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_encrypted_pdf_raises(self):
        from pypdf import PdfWriter

        path = self.tmp / 'encrypted.pdf'
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        writer.encrypt('secret')
        with path.open('wb') as handle:
            writer.write(handle)
        with self.assertRaises(PaperImportError) as ctx:
            extract_paper(path)
        self.assertIn('encrypted PDF is not supported', str(ctx.exception))

    def test_page_extract_failure_raises_with_page_number(self):
        path = self.tmp / 'page-fail.pdf'
        path.write_bytes(make_pdf(['ok-page', 'boom-page', 'later-page']))

        from unittest import mock
        from pypdf._page import PageObject

        original = PageObject.extract_text
        calls = {'n': 0}

        def flaky(self_page, *args, **kwargs):
            calls['n'] += 1
            if calls['n'] == 2:
                raise RuntimeError('synthetic extract failure')
            return original(self_page, *args, **kwargs)

        with mock.patch.object(PageObject, 'extract_text', flaky):
            with self.assertRaises(PaperImportError) as ctx:
                extract_paper(path)
        self.assertIn('failed to extract text on page 2', str(ctx.exception))

    def test_page_extract_failure_returns_no_partial_document(self):
        path = self.tmp / 'page-fail2.pdf'
        path.write_bytes(make_pdf(['one', 'two']))

        from unittest import mock
        from pypdf._page import PageObject

        original = PageObject.extract_text
        calls = {'n': 0}

        def flaky(self_page, *args, **kwargs):
            calls['n'] += 1
            if calls['n'] == 2:
                raise RuntimeError('synthetic extract failure')
            return original(self_page, *args, **kwargs)

        result = None
        with mock.patch.object(PageObject, 'extract_text', flaky):
            try:
                result = extract_paper(path)
            except PaperImportError:
                pass
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()

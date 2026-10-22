"""PaperPilot: a research-paper retrieval agent built on mini-harness.

Data-model decision for the upcoming store layer (recorded here because
store.py is intentionally NOT part of this batch):

Import-record identity is ``source_path`` (work_space-relative). ``sha256``
is a change-detection signal only and carries NO unique constraint.

- same path + same sha256    -> idempotent skip
- same path + changed sha256 -> update that record's pages and index inside
  one transaction (store.py, later batch)
- different path + same content -> allowed as separate records in v1
  (no cross-path deduplication)
- PDF parse failure          -> no partial paper record may be created

The v1 tool surface is exactly three tools: import_papers, search_papers,
read_paper_page. Long-term research memory is out of scope for v1.
"""

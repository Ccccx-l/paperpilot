"""Shared helpers for the phase-1 config-propagation regression tests.

Every test runs inside a temporary directory and must leave the project root
session.json byte-identical. `fingerprint` hashes the file (never printing its
content) so the guard can prove nothing wrote there.
"""

import hashlib

from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
ROOT_SESSION = ROOT / 'session.json'


def fingerprint(path: Path) -> tuple | None:
    if not path.is_file():
        return None
    stat = path.stat()
    return (stat.st_size, stat.st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())


def make_call(name: str, arguments: str, call_id: str = 'call-1') -> SimpleNamespace:
    """Minimal stand-in for an OpenAI tool_call object."""
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class OpenAISpy:
    """Fake OpenAI client factory.

    The api_key argument is accepted and dropped immediately so a real key from
    the environment can never be stored on the instance, appear in mock
    reprs, or leak into test output. Only base_url/max_retries are kept.
    """

    instances: list = []

    def __init__(self, api_key=None, base_url=None, max_retries=None):
        self.base_url = base_url
        self.max_retries = max_retries
        OpenAISpy.instances.append(self)


class RootSessionGuard:
    """Mixin: fail any test that modifies the project root session.json."""

    def setUp(self):
        super().setUp()
        self._root_session_before = fingerprint(ROOT_SESSION)

    def tearDown(self):
        self.assertEqual(
            fingerprint(ROOT_SESSION),
            self._root_session_before,
            'project root session.json must not be read-modified or overwritten by tests',
        )
        super().tearDown()

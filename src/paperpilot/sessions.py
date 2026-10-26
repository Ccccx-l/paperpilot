"""Browser-session store for the PaperPilot web UI.

Each browser session gets its own directory under
``<work_space>/sandbox/paperpilot/web_sessions/<sid>/`` holding:

- ``ui.json``   -- the UI conversation history (what the workbench renders)
- ``agent.json``-- the Agent message history (no system message), fed back as
                   ``prior`` to ``DeepSeekAgent.run_task()``

Design constraints (multi-turn web workbench):

- Session ids are server-generated ``secrets.token_urlsafe`` values, carried in
  an HttpOnly cookie. Clients cannot pick or forge another session's id.
- Per-session locks serialize ask/reset for one session without serializing
  unrelated sessions.
- Writes are atomic (tmp file + replace) so a crash never leaves a half-written
  history. A corrupt session file is treated as empty and does not affect any
  other session.
- Nothing here touches ``sandbox/paperpilot/session.json`` (the CLI session)
  or ``mini_harness_history.jsonl`` (the shared compaction log).
"""

from __future__ import annotations

import json
import re
import secrets
import threading

from pathlib import Path

# Server-generated ids only: url-safe base64 of 32 bytes -> 43 chars.
_SID_RE = re.compile(r'^[A-Za-z0-9_-]{16,64}$')

UI_FILE = 'ui.json'
AGENT_FILE = 'agent.json'

# Cap on how many complete turns of history are kept per session. Truncation
# always drops whole turns so an assistant tool_calls message is never
# separated from its role='tool' results.
MAX_TURNS = 8


def new_session_id() -> str:
    """A fresh unguessable session id."""
    return secrets.token_urlsafe(32)


def is_valid_session_id(sid: object) -> bool:
    return isinstance(sid, str) and bool(_SID_RE.match(sid))


class Session:
    """One browser session's histories plus its own lock."""

    def __init__(self, sid: str, root: Path):
        self.sid = sid
        self.root = Path(root)
        self.lock = threading.RLock()
        self.ui_history: list = []
        self.agent_history: list = []
        self._load()

    # ---- persistence -------------------------------------------------

    def _load(self) -> None:
        self.ui_history = _read_json_list(self.root / UI_FILE)
        self.agent_history = _read_json_list(self.root / AGENT_FILE)

    def _persist(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(self.root / UI_FILE, self.ui_history)
        _atomic_write_json(self.root / AGENT_FILE, self.agent_history)

    # ---- mutations (call under self.lock) ----------------------------

    def append_turn(self, ui_items: list, agent_messages: list) -> None:
        """Record one finished turn and truncate to MAX_TURNS whole turns."""
        self.ui_history.extend(ui_items)
        self.agent_history.extend(agent_messages)
        self.agent_history = trim_to_turns(self.agent_history, MAX_TURNS)
        self.ui_history = trim_ui_to_turns(self.ui_history, MAX_TURNS)
        self._persist()

    def reset(self) -> None:
        self.ui_history = []
        self.agent_history = []
        self._persist()


class SessionStore:
    """Directory-backed store of Session objects with per-session locks."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self._guard = threading.Lock()
        self._sessions: dict[str, Session] = {}

    def session_dir(self, sid: str) -> Path:
        return self.base_dir / sid

    def get(self, sid: str) -> Session | None:
        if not is_valid_session_id(sid):
            return None
        with self._guard:
            sess = self._sessions.get(sid)
            if sess is None:
                d = self.session_dir(sid)
                if not d.is_dir():
                    return None
                sess = Session(sid, d)
                self._sessions[sid] = sess
            return sess

    def create(self, sid: str | None = None) -> Session:
        """Create a brand-new session (sid generated if not given)."""
        if sid is None:
            sid = new_session_id()
        if not is_valid_session_id(sid):
            raise ValueError('invalid session id')
        with self._guard:
            existing = self._sessions.get(sid)
            if existing is not None:
                return existing
            sess = Session(sid, self.session_dir(sid))
            self._sessions[sid] = sess
            return sess


# ---- helpers -----------------------------------------------------------

def trim_to_turns(messages: list, max_turns: int) -> list:
    """Keep only the last ``max_turns`` complete turns of an agent history.

    A turn starts at a ``role='user'`` message and runs until the next one.
    Cutting at user boundaries keeps every assistant tool_calls message with
    its role='tool' results, which the OpenAI-compatible API requires.
    """
    if max_turns <= 0:
        return list(messages)
    user_idx = [i for i, m in enumerate(messages)
                if isinstance(m, dict) and m.get('role') == 'user']
    if len(user_idx) <= max_turns:
        return list(messages)
    cut = user_idx[len(user_idx) - max_turns]
    return list(messages[cut:])


def trim_ui_to_turns(items: list, max_turns: int) -> list:
    """Same whole-turn trim for the UI history (user messages start turns)."""
    if max_turns <= 0:
        return list(items)
    user_idx = [i for i, m in enumerate(items)
                if isinstance(m, dict) and m.get('role') == 'user']
    if len(user_idx) <= max_turns:
        return list(items)
    cut = user_idx[len(user_idx) - max_turns]
    return list(items[cut:])


def _read_json_list(path: Path) -> list:
    """Read a JSON list; a missing or corrupt file is an empty list."""
    try:
        if not path.is_file():
            return []
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def _atomic_write_json(path: Path, payload: list) -> None:
    """Write JSON atomically (tmp + replace) so readers never see partial data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                       encoding='utf-8')
        tmp.replace(path)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise

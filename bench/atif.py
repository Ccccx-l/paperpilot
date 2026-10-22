import json
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from harbor.models.trajectories.agent import Agent
from harbor.models.trajectories.final_metrics import FinalMetrics
from harbor.models.trajectories.observation import Observation
from harbor.models.trajectories.observation_result import ObservationResult
from harbor.models.trajectories.step import Step
from harbor.models.trajectories.tool_call import ToolCall
from harbor.models.trajectories.trajectory import Trajectory

SESSION = "mini_harness_session.json"
HISTORY = "mini_harness_history.jsonl"
TELE = "mini_harness_tele.json"
LOG = "mini_harness.txt"
SUMMARY_PREFIX = "Here is the summary of the history conversation:"
MODEL_FALLBACK = "deepseek-v4-flash"
COMPACTION_EXTRA = {"context_management": {"type": "compaction", "boundary": "replace"}}
CTX_RE = re.compile(
    r"\[ctx\]:\s*(\d+)\s*/\s*\d+\s*tokens(?:,\s*out\s*(\d+))?(\s*\[TRUNCATED\])?"
)


def git_version(repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        v = out.stdout.strip()
        return v if out.returncode == 0 and v else "unknown"
    except Exception:
        return "unknown"


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def _load_history(agent_dir: Path) -> tuple[list, list]:
    (events, notes) = ([], [])
    path = agent_dir / HISTORY
    if not path.exists():
        return (events, notes)
    dropped = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            dropped += 1
            continue
        if isinstance(ev, dict) and isinstance(ev.get("removed"), list):
            events.append(ev)
        else:
            dropped += 1
    if dropped:
        notes.append(f"dropped {dropped} malformed history line(s)")
    return (events, notes)


def _splice(events: list, session: list) -> tuple[list, dict, list]:
    notes = []
    events = list(events)
    if events:
        last = events[-1].get("removed") or []
        if last and session[1 : 1 + len(last)] == last:
            events.pop()
            notes.append("dropped uncommitted trailing history event")
    stream: list = []
    summary_ts: dict[int, object] = {}
    for j, ev in enumerate(events):
        if j >= 1:
            summary_ts[len(stream)] = events[j - 1].get("ts")
        stream.extend(ev["removed"])
    if events:
        summary_ts[len(stream)] = events[-1].get("ts")
    stream.extend(session[1:])
    return (stream, summary_ts, notes)


def _iso(ts) -> str | None:
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def _parse_call(tc: dict) -> ToolCall:
    fn = tc.get("function") or {}
    raw = fn.get("arguments", "")
    args = raw if isinstance(raw, dict) else None
    if args is None and isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            args = parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            args = None
    if args is None:
        args = {"raw": raw}
    return ToolCall(
        tool_call_id=str(tc.get("id") or uuid.uuid4().hex[:8]),
        function_name=str(fn.get("name") or "unknown"),
        arguments=args,
    )


def _fold(stream: list, summary_ts: dict) -> tuple[list, list]:
    steps: list[Step] = []
    notes: list[str] = []
    orphans = mismatches = 0
    i = 0
    while i < len(stream):
        m = stream[i]
        if not isinstance(m, dict):
            i += 1
            continue
        role = m.get("role")
        content = m.get("content")
        text = (
            content
            if isinstance(content, str)
            else ""
            if content is None
            else str(content)
        )
        looks_summary = role == "assistant" and text.startswith(SUMMARY_PREFIX)
        if i in summary_ts and (not looks_summary):
            mismatches += 1
        if looks_summary:
            steps.append(
                Step(
                    step_id=len(steps) + 1,
                    timestamp=_iso(summary_ts.get(i)),
                    source="system",
                    message=text,
                    extra=dict(COMPACTION_EXTRA),
                )
            )
        elif role == "user":
            steps.append(Step(step_id=len(steps) + 1, source="user", message=text))
        elif role == "assistant" and m.get("tool_calls"):
            calls = [_parse_call(tc) for tc in m["tool_calls"] if isinstance(tc, dict)]
            ids = {c.tool_call_id for c in calls}
            results = []
            while (
                i + 1 < len(stream)
                and isinstance(stream[i + 1], dict)
                and (stream[i + 1].get("role") == "tool")
            ):
                i += 1
                t = stream[i]
                cid = t.get("tool_call_id")
                results.append(
                    ObservationResult(
                        source_call_id=cid if cid in ids else None,
                        content=str(t.get("content") or ""),
                    )
                )
            steps.append(
                Step(
                    step_id=len(steps) + 1,
                    source="agent",
                    message=text if text else "[tool call]",
                    tool_calls=calls or None,
                    observation=Observation(results=results) if results else None,
                )
            )
        elif role == "assistant":
            steps.append(Step(step_id=len(steps) + 1, source="agent", message=text))
        elif role == "tool":
            orphans += 1
            prev = steps[-1] if steps else None
            if prev is not None and prev.source == "agent":
                res = ObservationResult(
                    source_call_id=None, content=str(m.get("content") or "")
                )
                if prev.observation is None:
                    prev.observation = Observation(results=[res])
                else:
                    prev.observation.results.append(res)
        else:
            notes.append(f"skipped unexpected role {role!r} in stream")
        i += 1
    if orphans:
        notes.append(
            f"{orphans} orphan tool message(s) reattached without source_call_id or dropped"
        )
    if mismatches:
        notes.append(
            f"{mismatches} positional summary mark(s) without the summary prefix; rendered as normal step(s)"
        )
    return (steps, notes)


def _tokens(agent_dir: Path, tele) -> tuple[int | None, int | None, str | None]:
    if isinstance(tele, dict) and isinstance(tele.get("prompt_total"), int):
        return (tele.get("prompt_total"), tele.get("completion_total"), "telemetry")
    log = agent_dir / LOG
    if log.exists():
        hits = CTX_RE.findall(log.read_text(encoding="utf-8", errors="replace"))
        if hits:
            n_in = sum((int(p) for (p, _, _) in hits))
            n_out = sum((int(o) for (_, o, _) in hits if o))
            return (n_in, n_out, "reconstructed")
    return (None, None, None)


def build(
    agent_dir: Path, version: str = "unknown", model_name: str | None = None
) -> Trajectory | None:
    agent_dir = Path(agent_dir)
    session = _load_json(agent_dir / SESSION)
    if not isinstance(session, list) or len(session) < 2:
        return None
    (events, notes) = _load_history(agent_dir)
    (stream, summary_ts, n2) = _splice(events, session)
    (steps, n3) = _fold(stream, summary_ts)
    notes += n2 + n3
    if not steps:
        return None
    tele = _load_json(agent_dir / TELE)
    if not isinstance(tele, dict):
        tele = None
    (n_in, n_out, src) = _tokens(agent_dir, tele)
    return Trajectory(
        session_id=agent_dir.parent.name,
        agent=Agent(
            name="mini-harness",
            version=version,
            model_name=(tele or {}).get("model") or model_name or MODEL_FALLBACK,
        ),
        steps=steps,
        final_metrics=FinalMetrics(
            total_prompt_tokens=n_in,
            total_completion_tokens=n_out,
            total_steps=len(steps),
            extra={"token_source": src} if src else None,
        ),
        notes="; ".join(notes) if notes else None,
        extra={"outcome": tele.get("outcome")}
        if tele and tele.get("outcome")
        else None,
    )


def write_trajectory(agent_dir: Path, traj: Trajectory) -> Path:
    out = Path(agent_dir) / "trajectory.json"
    out.write_text(
        json.dumps(traj.to_json_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent.parent
SUBSET = "swe-bench_verified"
SPLIT = "test"
ADAPTER = "bench.sb_cli_adapter:SBCLIPredictionAgent"
INSTANCE_PATTERN = r"[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-\d+"


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    instance_id: str = Field(pattern=f"^{INSTANCE_PATTERN}$")
    model_name_or_path: str = Field(min_length=1)
    model_patch: str


def write_json(path: Path, data, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as f:
        temporary = Path(f.name)
        try:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        if overwrite:
            temporary.replace(path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def instance_id_for_trial(trial: Path) -> str:
    config = trial / "config.json"
    if config.exists():
        task = json.loads(config.read_text())["task"]
        candidate = task.get("name") or Path(task.get("path", "")).name
    else:
        candidate = trial.name.rsplit("__", 1)[0]
    if not re.fullmatch(INSTANCE_PATTERN, candidate):
        raise ValueError(f"Cannot identify a SWE-bench instance in {trial}")
    return candidate


def validate_predictions(rows: list) -> list[dict]:
    predictions = [Prediction.model_validate(row).model_dump() for row in rows]
    if not predictions:
        raise ValueError("No predictions found")
    ids = [p["instance_id"] for p in predictions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate instance IDs; export a single attempt per task")
    if len({p["model_name_or_path"] for p in predictions}) != 1:
        raise ValueError("All predictions must use the same model")
    return sorted(predictions, key=lambda p: p["instance_id"])


def load_predictions(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        rows = json.loads(text)
        if isinstance(rows, dict):
            rows = [{**value, "instance_id": key} for key, value in rows.items()]
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list):
        raise ValueError("Predictions must be a JSON list, mapping, or JSONL")
    return validate_predictions(rows)


def export_predictions(job_dir: Path, output: Path) -> list[dict]:
    result = json.loads((job_dir / "result.json").read_text())
    if not result.get("finished_at"):
        raise ValueError("The Harbor job has not finished")
    trials = sorted(
        p for p in job_dir.iterdir() if p.is_dir() and (p / "config.json").is_file()
    )
    lock = json.loads((job_dir / "lock.json").read_text())
    if not trials or len(trials) != len(lock["trials"]):
        raise ValueError("Incomplete trial directories; refusing a partial export")
    rows = []
    missing = []
    for trial in trials:
        path = trial / "agent" / "prediction.json"
        if not path.exists():
            missing.append(trial.name)
            continue
        row = Prediction.model_validate_json(path.read_text()).model_dump()
        if row["instance_id"] != instance_id_for_trial(trial):
            raise ValueError(f"Prediction instance does not match {trial.name}")
        rows.append(row)
    if missing:
        raise ValueError(
            f"Missing predictions for {len(missing)} trial(s): {', '.join(missing[:5])}. Use {ADAPTER} when generating patches."
        )
    predictions = validate_predictions(rows)
    write_json(output, predictions)
    print(f"Exported {len(predictions)} predictions to {output}", flush=True)
    return predictions


def positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def job_config(args) -> dict:
    dataset = {"name": "swebench-verified"}
    if args.instance_id:
        for instance in args.instance_id:
            if not re.fullmatch(INSTANCE_PATTERN, instance):
                raise ValueError(f"Invalid instance ID: {instance}")
        dataset["task_names"] = args.instance_id
    if not args.all:
        dataset["n_tasks"] = args.n_tasks or len(args.instance_id or []) or 1
    kwargs = {
        key: getattr(args, key)
        for key in ("provider", "base_url", "api_key_env")
        if getattr(args, key) is not None
    }
    return {
        "job_name": args.job_dir.name,
        "jobs_dir": str(args.job_dir.parent),
        "n_attempts": 1,
        "n_concurrent_trials": args.concurrency,
        "retry": {"max_retries": 0},
        "environment": {"type": args.environment, "delete": True},
        "verifier": {"disable": True},
        "agents": [
            {"import_path": ADAPTER, "model_name": args.model, "kwargs": kwargs}
        ],
        "datasets": [dataset],
    }


def execute(command: list[str], dry_run: bool = False) -> None:
    print(shlex.join(command), flush=True)
    if dry_run:
        return
    if not shutil.which(command[0]):
        raise RuntimeError(
            f"{command[0]} is not installed; install bench/requirements-sb-cli.txt"
        )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(ROOT), env.get("PYTHONPATH")])
    )
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def generate(args) -> None:
    config = job_config(args)
    config_path = args.job_dir.parent / f"{args.job_dir.name}.harbor.json"
    if args.dry_run:
        print(json.dumps(config, indent=2))
    else:
        if not shutil.which("harbor"):
            raise RuntimeError(
                "harbor is not installed; install bench/requirements-sb-cli.txt"
            )
        if args.job_dir.exists():
            raise FileExistsError(f"Choose a new job directory: {args.job_dir}")
        if config_path.exists():
            if json.loads(config_path.read_text()) != config:
                raise FileExistsError(config_path)
        else:
            write_json(config_path, config)
    execute(["harbor", "run", "--config", str(config_path)], args.dry_run)


def submit(
    predictions: Path, run_id: str, report_dir: Path, dry_run: bool = False
) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id) or run_id in {
        "PARENT",
        "STEM",
    }:
        raise ValueError(
            "Use an explicit run ID containing letters, numbers, dots, underscores or hyphens"
        )
    rows = load_predictions(predictions)
    if not dry_run and not os.environ.get("SWEBENCH_API_KEY"):
        raise RuntimeError("Set SWEBENCH_API_KEY to a verified sb-cli API key")
    print(f"Submitting {len(rows)} predictions to {SUBSET}/{SPLIT}", flush=True)
    execute(
        [
            "sb-cli",
            "submit",
            SUBSET,
            SPLIT,
            "--predictions_path",
            str(predictions),
            "--run_id",
            run_id,
            "--output_dir",
            str(report_dir),
            "--verify_submission",
            "1",
            "--wait_for_evaluation",
            "1",
            "--gen_report",
            "1",
            "--overwrite",
            "1",
        ],
        dry_run,
    )
    if not dry_run:
        report_path = report_dir / f"{SUBSET}__{SPLIT}__{run_id}.json"
        report = json.loads(report_path.read_text())
        if (
            report.get("submitted_instances") != len(rows)
            or report.get("pending_instances") != 0
        ):
            raise RuntimeError(f"Incomplete cloud evaluation; inspect {report_path}")
        if report.get("error_instances", 0):
            raise RuntimeError(
                f"Cloud evaluation reported errors; inspect {report_path}"
            )


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Generate SWE-bench Verified patches with Harbor and evaluate them with sb-cli."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "run"):
        command = commands.add_parser(
            name,
            help="Generate patches"
            if name == "generate"
            else "Generate, export, and submit patches",
        )
        command.add_argument("--job-dir", type=Path, required=True)
        command.add_argument(
            "--model", default=os.environ.get("MINI_HARNESS_MODEL", "openai/gpt-4.1")
        )
        command.add_argument("--provider", choices=("openai", "deepseek"))
        command.add_argument("--base-url")
        command.add_argument("--api-key-env")
        command.add_argument(
            "--environment",
            choices=("modal", "docker"),
            default="modal",
            help="Agent execution backend (default: modal); sb-cli performs cloud scoring",
        )
        command.add_argument("--concurrency", type=positive, default=1)
        count = command.add_mutually_exclusive_group()
        count.add_argument(
            "--n-tasks", type=positive, help="Number of tasks to generate (default: 1)"
        )
        count.add_argument(
            "--all",
            action="store_true",
            help="Generate all 500 Verified tasks; otherwise defaults to one task",
        )
        command.add_argument("--instance-id", action="append")
        command.add_argument("--dry-run", action="store_true")
        if name == "run":
            command.add_argument("--run-id", required=True)
    command = commands.add_parser(
        "export", help="Collect one prediction per Harbor trial"
    )
    command.add_argument("--job-dir", type=Path, required=True)
    command.add_argument("--output", type=Path)
    command = commands.add_parser(
        "submit", help="Submit existing predictions to sb-cli cloud scoring"
    )
    command.add_argument("--predictions", type=Path, required=True)
    command.add_argument("--run-id", required=True)
    command.add_argument("--report-dir", type=Path)
    command.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        if hasattr(args, "job_dir"):
            args.job_dir = args.job_dir.expanduser().resolve()
        if args.command in {"run", "submit"}:
            if not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id
            ) or args.run_id in {"PARENT", "STEM"}:
                raise ValueError(
                    "Use an explicit run ID containing letters, numbers, dots, underscores or hyphens"
                )
        if args.command == "run" and not args.dry_run:
            if not os.environ.get("SWEBENCH_API_KEY"):
                raise RuntimeError(
                    "Set SWEBENCH_API_KEY before starting the cloud pipeline"
                )
            if not shutil.which("sb-cli"):
                raise RuntimeError(
                    "sb-cli is not installed; install bench/requirements-sb-cli.txt"
                )
        if args.command in {"generate", "run"}:
            generate(args)
        if args.command == "export":
            export_predictions(
                args.job_dir,
                (args.output or args.job_dir / "predictions.json").resolve(),
            )
        if args.command == "run":
            output = args.job_dir / "predictions.json"
            if args.dry_run:
                print(f"Export to {output}, then submit to sb-cli as {args.run_id}")
            else:
                export_predictions(args.job_dir, output)
                submit(output, args.run_id, args.job_dir / "sb-cli-reports")
        if args.command == "submit":
            predictions = args.predictions.expanduser().resolve()
            submit(
                predictions,
                args.run_id,
                (args.report_dir or predictions.parent / "sb-cli-reports").resolve(),
                args.dry_run,
            )
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"sb-cli pipeline: {error}\n")


if __name__ == "__main__":
    main()

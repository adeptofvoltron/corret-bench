"""Entry point: Intrinsic vs. Grounded LLM self-correction experiment."""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rich.console import Console

from config import Config, _resolve_python, make_client
from dataset import DEFAULT_EXEC_PYTHON, exclusions_path, load_tasks
from experiment import LLM, Experiment, execute, print_summary, save_results, summarize


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", help="override OPENAI_MODEL")
    p.add_argument("--max-samples", type=int, help="override MAX_SAMPLES")
    p.add_argument("--dataset", help="mbpp | bigcodebench-hard | builtin | jsonl:<path> (override DATASET)")
    p.add_argument("--rounds", type=int, help="max correction rounds per arm (override MAX_CORRECTION_ROUNDS)")
    p.add_argument("--concurrency", type=int, help="parallel tasks (override CONCURRENCY)")
    p.add_argument("--exec-python", help="interpreter used to run generated code (override EXEC_PYTHON)")
    p.add_argument("--validate-runs", type=int, default=2,
                   help="with --validate-dataset: run each reference solution N times; "
                        "a task that fails in any run is excluded (catches unstable tests)")
    p.add_argument("--validate-dataset", action="store_true",
                   help="run reference solutions against the tests (no API calls), write the list of "
                        "tasks that fail in this environment to data/excluded-<dataset>.json and exit")
    return p.parse_args()


def validate_dataset(tasks, cfg: Config, console: Console, runs: int = 2) -> int:
    """Reference solutions that fail here point at the environment (missing
    library, version drift, network), not at the model - exclude those tasks."""

    def check(t):
        for run in range(1, runs + 1):
            for kind, tests in (("public", t.public_tests), ("held-out", t.unit_tests)):
                r = execute(t, t.ground_truth, tests, cfg)
                if not r.passed:
                    where = kind if run == 1 else f"{kind} (run {run}, passed earlier: unstable)"
                    return t.id, where, r.feedback()
        return None

    with_gt = [t for t in tasks if t.ground_truth]
    with ThreadPoolExecutor(max_workers=max(1, cfg.concurrency)) as pool:
        failures = [f for f in pool.map(check, with_gt) if f]

    # A missing library means the interpreter is wrong, not that the task is broken:
    # do not let it overwrite a good exclusion list.
    missing = sorted({m for _, _, fb in failures for m in re.findall(r"No module named '([^']+)'", fb)})
    if missing:
        console.print(
            f"[red]Interpreter {cfg.exec_python} is missing {len(missing)} module(s): {', '.join(missing)}.[/]\n"
            "Fix the environment (see requirements-bigcodebench.txt) or pass --exec-python. "
            "Exclusion list NOT written."
        )
        return 1
    for task_id, kind, feedback in failures:
        last = feedback.strip().splitlines()[-1] if feedback.strip() else ""
        console.print(f"[red]{task_id}[/]: reference fails {kind} tests: {last}")

    excl_file = exclusions_path(cfg.dataset, cfg.data_dir)
    excl_file.parent.mkdir(parents=True, exist_ok=True)
    excl_file.write_text(json.dumps(sorted(f[0] for f in failures), indent=2), encoding="utf-8")
    console.print(f"Validated {len(with_gt)} task(s) x {runs} run(s), {len(failures)} failure(s) -> {excl_file}")
    return 0


def main() -> int:
    args = parse_args()
    console = Console()
    cfg = Config.from_env()
    overrides = {
        "model": args.model,
        "max_samples": args.max_samples,
        "dataset": args.dataset,
        "max_correction_rounds": args.rounds,
        "concurrency": args.concurrency,
        "exec_python": _resolve_python(args.exec_python) if args.exec_python else None,
    }
    cfg = dataclasses.replace(cfg, **{k: v for k, v in overrides.items() if v is not None})
    if cfg.exec_python is None:
        default = DEFAULT_EXEC_PYTHON.get(cfg.dataset)
        if default and not (resolved := Path(_resolve_python(default))).exists():
            console.print(f"[red]{cfg.dataset} needs its own interpreter at {resolved} (not found).[/] "
                          "Create it with the commands in requirements-bigcodebench.txt or pass --exec-python.")
            return 1
        cfg = dataclasses.replace(cfg, exec_python=_resolve_python(default) if default else sys.executable)

    if args.validate_dataset:
        # Always validate the whole dataset so the exclusion list is complete.
        tasks = load_tasks(cfg.dataset, cfg.data_dir, 10**9, apply_exclusions=False)
    else:
        tasks = load_tasks(cfg.dataset, cfg.data_dir, cfg.max_samples)
    console.print(f"Loaded {len(tasks)} task(s) from '{cfg.dataset}'. Executing code with {cfg.exec_python}")
    if args.validate_dataset:
        return validate_dataset(tasks, cfg, console, max(1, args.validate_runs))

    experiment = Experiment(cfg, LLM(make_client(), cfg.model, cfg.temperature), console)
    results = experiment.run(tasks)
    summary = summarize(results, cfg)
    save_results(results, summary, cfg)
    print_summary(summary, console)
    console.print(f"Saved {cfg.results_dir / 'raw_results.json'} and {cfg.results_dir / 'summary_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

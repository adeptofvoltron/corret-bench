"""Entry point: Intrinsic vs. Grounded LLM self-correction experiment."""

from __future__ import annotations

import argparse
import dataclasses
import sys

from rich.console import Console

from config import Config, make_client
from dataset import load_tasks
from evaluator import run_tests
from experiment import LLM, Experiment, print_summary, save_results, summarize


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", help="override OPENAI_MODEL")
    p.add_argument("--max-samples", type=int, help="override MAX_SAMPLES")
    p.add_argument("--dataset", help="mbpp | builtin | jsonl:<path> (override DATASET)")
    p.add_argument("--rounds", type=int, help="max correction rounds per arm (override MAX_CORRECTION_ROUNDS)")
    p.add_argument("--concurrency", type=int, help="parallel tasks (override CONCURRENCY)")
    p.add_argument("--validate-dataset", action="store_true",
                   help="run ground-truth solutions against the tests (no API calls) and exit")
    return p.parse_args()


def validate_dataset(tasks, cfg: Config, console: Console) -> int:
    bad = 0
    for t in tasks:
        if not t.ground_truth:
            continue
        for kind, tests in (("public", t.public_tests), ("held-out", t.unit_tests)):
            r = run_tests(t.ground_truth, tests, t.setup_code, cfg.exec_timeout_s)
            if not r.passed:
                bad += 1
                console.print(f"[red]{t.id}: ground truth fails {kind} tests[/]\n{r.feedback()}")
    console.print(f"Validated {len(tasks)} tasks, {bad} failure(s).")
    return 1 if bad else 0


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
    }
    cfg = dataclasses.replace(cfg, **{k: v for k, v in overrides.items() if v is not None})

    tasks = load_tasks(cfg.dataset, cfg.data_dir, cfg.max_samples)
    console.print(f"Loaded {len(tasks)} task(s) from '{cfg.dataset}'.")
    if args.validate_dataset:
        return validate_dataset(tasks, cfg, console)

    experiment = Experiment(cfg, LLM(make_client(), cfg.model, cfg.temperature), console)
    results = experiment.run(tasks)
    summary = summarize(results, cfg)
    save_results(results, summary, cfg)
    print_summary(summary, console)
    console.print(f"Saved {cfg.results_dir / 'raw_results.json'} and {cfg.results_dir / 'summary_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

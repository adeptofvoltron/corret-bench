"""Three experimental arms and the statistics over them.

A  Baseline (single pass)  - one generation.
B  Intrinsic SC            - the model re-reads its own answer, text only.
                             Stops when the model returns unchanged code
                             (its own decision) or after N rounds.
C  Grounded SC             - the answer is executed against the PUBLIC tests
                             only; on failure stderr/traceback is fed back.
                             Stops when the public tests pass or after N rounds.

Oracle-leakage guard: no correction loop ever reads `task.unit_tests` or
`task.ground_truth`. Held-out tests are used only by `_score()`, whose result
is recorded for analysis and never feeds back into a stopping decision.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone

import openai
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from config import Config
from dataset import Task
from evaluator import extract_code, run_tests

SYSTEM_PROMPT = "You are an expert Python programmer."
FORMAT_INSTRUCTION = (
    "Return the complete solution in a single ```python code block. "
    "Do not include test code or example usage."
)
INTRINSIC_PROMPT = (
    "Review your output carefully. If you find any logic or syntax errors, output the "
    "corrected version. If it's correct, return it as is."
)
GROUNDED_PROMPT = "Your code was executed and failed with the following error:\n```\n{error}\n```\nFix the code."

TRANSITIONS = ("C->C", "C->I", "I->C", "I->I")


class LLM:
    def __init__(self, client: openai.OpenAI, model: str, temperature: float | None):
        self.client = client
        self.model = model
        self.temperature = temperature

    def complete(self, messages: list[dict]) -> tuple[str, dict]:
        kwargs = {"model": self.model, "messages": messages}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except openai.BadRequestError as exc:
            # Reasoning models (o-series, gpt-5) reject a custom temperature.
            # Check this request's kwargs, not self.temperature: another thread may
            # have already cleared it after the same error.
            if "temperature" not in str(exc) or "temperature" not in kwargs:
                raise
            self.temperature = None
            kwargs.pop("temperature")
            resp = self.client.chat.completions.create(**kwargs)
        usage = resp.usage.model_dump() if resp.usage else {}
        return resp.choices[0].message.content or "", usage


def _task_message(task: Task) -> str:
    parts = [task.prompt]
    if task.show_public_tests and task.public_tests.strip():
        parts.append(f"Your code should pass this test:\n```python\n{task.public_tests}\n```")
    parts.append(FORMAT_INSTRUCTION)
    return "\n\n".join(parts)


def execute(task: Task, code: str, tests: str, cfg: Config):
    return run_tests(code, tests, task.setup_code, task.exec_timeout_s or cfg.exec_timeout_s,
                     python=cfg.exec_python, memory_mb=task.exec_memory_mb or cfg.exec_memory_mb)


def _normalize(code: str) -> str:
    return "\n".join(line.rstrip() for line in code.strip().splitlines())


class Experiment:
    def __init__(self, cfg: Config, llm: LLM, console: Console | None = None):
        self.cfg = cfg
        self.llm = llm
        self.console = console or Console()
        # (task, code, tests) -> ExecResult. Unchanged code is very common (e.g. intrinsic
        # "return it as is"), and BigCodeBench suites can take minutes to run.
        self._exec_cache: dict[tuple[str, str, str], object] = {}
        self._cache_lock = threading.Lock()

    def _exec(self, task: Task, code: str, tests: str):
        key = (task.id, _normalize(code), tests)
        with self._cache_lock:
            if key in self._exec_cache:
                return self._exec_cache[key]
        result = execute(task, code, tests, self.cfg)
        with self._cache_lock:
            self._exec_cache[key] = result
        return result

    # -- scoring (held-out tests; analysis only) ---------------------------------
    def _score(self, task: Task, code: str) -> dict:
        r = self._exec(task, code, task.unit_tests)
        return {"correct": r.passed, "exec": r.to_dict()}

    # -- arms --------------------------------------------------------------------
    def _baseline(self, task: Task) -> tuple[list[dict], dict]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _task_message(task)},
        ]
        text, usage = self.llm.complete(messages)
        code = extract_code(text)
        messages.append({"role": "assistant", "content": text})
        return messages, {"response": text, "code": code, "usage": usage, **self._score(task, code)}

    def _intrinsic(self, task: Task, history: list[dict], code: str) -> dict:
        messages = list(history)
        rounds = []
        stop_reason = "max_rounds"
        for _ in range(self.cfg.max_correction_rounds):
            messages.append({"role": "user", "content": f"{INTRINSIC_PROMPT}\n\n{FORMAT_INSTRUCTION}"})
            text, usage = self.llm.complete(messages)
            messages.append({"role": "assistant", "content": text})
            # No code block (e.g. "The code is correct.") means the model kept its answer.
            new_code = extract_code(text) if "```" in text else code
            changed = _normalize(new_code) != _normalize(code)
            rounds.append({"response": text, "code": new_code, "changed": changed, "usage": usage,
                           "hidden_correct": self._score(task, new_code)["correct"]})
            code = new_code
            if not changed:
                stop_reason = "model_kept_answer"
                break
        return {"rounds": rounds, "stop_reason": stop_reason, "final_code": code, **self._score(task, code)}

    def _grounded(self, task: Task, history: list[dict], code: str) -> dict:
        messages = list(history)
        rounds = []
        stop_reason = "max_rounds"
        for _ in range(self.cfg.max_correction_rounds):
            check = self._exec(task, code, task.public_tests)
            if check.passed:
                stop_reason = "public_tests_passed"
                break
            feedback = check.feedback()
            messages.append({"role": "user",
                             "content": f"{GROUNDED_PROMPT.format(error=feedback)}\n\n{FORMAT_INSTRUCTION}"})
            text, usage = self.llm.complete(messages)
            messages.append({"role": "assistant", "content": text})
            code = extract_code(text)
            rounds.append({"feedback": feedback, "response": text, "code": code, "usage": usage,
                           "hidden_correct": self._score(task, code)["correct"]})
        final_public = self._exec(task, code, task.public_tests)
        return {
            "rounds": rounds,
            "stop_reason": stop_reason,
            "feedback_triggered": bool(rounds),
            "final_code": code,
            "final_public_passed": final_public.passed,
            **self._score(task, code),
        }

    def run_task(self, task: Task) -> dict:
        record = {"id": task.id, "prompt": task.prompt, "public_tests": task.public_tests}
        started = time.monotonic()
        try:
            history, baseline = self._baseline(task)
            record["baseline"] = baseline
            record["intrinsic"] = self._intrinsic(task, history, baseline["code"])
            record["grounded"] = self._grounded(task, history, baseline["code"])
        except openai.OpenAIError as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["elapsed_s"] = round(time.monotonic() - started, 2)
        return record

    def run(self, tasks: list[Task]) -> list[dict]:
        results: list[dict] = []
        columns = (TextColumn("[bold]{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn())
        with Progress(*columns, console=self.console) as progress, \
                ThreadPoolExecutor(max_workers=max(1, self.cfg.concurrency)) as pool:
            bar = progress.add_task(f"{self.cfg.model}", total=len(tasks))
            futures = {pool.submit(self.run_task, t): t for t in tasks}
            for fut in as_completed(futures):
                rec = fut.result()
                results.append(rec)
                if "error" in rec:
                    progress.console.print(f"[red]{rec['id']}: {rec['error']}")
                progress.advance(bar)
        order = {t.id: i for i, t in enumerate(tasks)}
        return sorted(results, key=lambda r: order[r["id"]])


# -- statistics ----------------------------------------------------------------

def _pct(num: int, den: int) -> float | None:
    return round(100.0 * num / den, 2) if den else None


def _transition(before: bool, after: bool) -> str:
    return f"{'C' if before else 'I'}->{'C' if after else 'I'}"


def summarize(results: list[dict], cfg: Config) -> dict:
    ok = [r for r in results if "error" not in r]
    n = len(ok)
    n_base_correct = sum(r["baseline"]["correct"] for r in ok)
    summary = {
        "meta": {
            "model": cfg.model,
            "dataset": cfg.dataset,
            "max_correction_rounds": cfg.max_correction_rounds,
            "temperature": cfg.temperature,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_tasks": len(results),
            "n_evaluated": n,
            "n_errors": len(results) - n,
        },
        "accuracy": {"baseline": _pct(n_base_correct, n)},
        "variants": {},
    }

    for arm in ("intrinsic", "grounded"):
        counts = dict.fromkeys(TRANSITIONS, 0)
        for r in ok:
            counts[_transition(r["baseline"]["correct"], r[arm]["correct"])] += 1
        n_after = counts["C->C"] + counts["I->C"]
        variant = {
            "transitions": counts,
            "transitions_pct": {k: _pct(v, n) for k, v in counts.items()},
            "accuracy": _pct(n_after, n),
            "net_accuracy_change_pp": _pct(counts["I->C"] - counts["C->I"], n),
            "destroyed_pct_of_all": _pct(counts["C->I"], n),
            "repaired_pct_of_all": _pct(counts["I->C"], n),
            "destruction_rate_pct": _pct(counts["C->I"], n_base_correct),
            "repair_rate_pct": _pct(counts["I->C"], n - n_base_correct),
            "avg_rounds": round(sum(len(r[arm]["rounds"]) for r in ok) / n, 3) if n else None,
            "stop_reasons": {},
        }
        for r in ok:
            reason = r[arm]["stop_reason"]
            variant["stop_reasons"][reason] = variant["stop_reasons"].get(reason, 0) + 1
        if arm == "grounded":
            variant["feedback_triggered"] = sum(r[arm]["feedback_triggered"] for r in ok)
            # Public tests passed but held-out tests fail: the grounded signal was too weak.
            variant["accepted_but_hidden_fail"] = sum(
                r[arm]["final_public_passed"] and not r[arm]["correct"] for r in ok
            )
        summary["accuracy"][arm] = variant["accuracy"]
        summary["variants"][arm] = variant

    tokens = {"prompt_tokens": 0, "completion_tokens": 0}
    for r in ok:
        usages = [r["baseline"]["usage"]] + [x["usage"] for a in ("intrinsic", "grounded") for x in r[a]["rounds"]]
        for u in usages:
            for k in tokens:
                tokens[k] += u.get(k) or 0
    summary["tokens"] = tokens
    return summary


def save_results(results: list[dict], summary: dict, cfg: Config) -> None:
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    raw = {"config": {k: str(v) for k, v in asdict(cfg).items()}, "results": results}
    (cfg.results_dir / "raw_results.json").write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    (cfg.results_dir / "summary_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def print_summary(summary: dict, console: Console) -> None:
    meta = summary["meta"]
    console.print(
        f"\n[bold]Model:[/] {meta['model']}   [bold]Dataset:[/] {meta['dataset']}   "
        f"[bold]Tasks:[/] {meta['n_evaluated']}/{meta['n_tasks']}   "
        f"[bold]Correction rounds:[/] {meta['max_correction_rounds']}"
    )

    def fmt(v, suffix="%"):
        return "-" if v is None else f"{v:.2f}{suffix}"

    acc = Table(title="Accuracy (held-out tests)")
    acc.add_column("Variant")
    acc.add_column("Accuracy", justify="right")
    acc.add_column("Net change", justify="right")
    acc.add_row("A  Baseline (single pass)", fmt(summary["accuracy"]["baseline"]), "")
    for arm, label in (("intrinsic", "B  Intrinsic SC"), ("grounded", "C  Grounded SC")):
        v = summary["variants"][arm]
        net = v["net_accuracy_change_pp"]
        color = "green" if (net or 0) > 0 else "red" if (net or 0) < 0 else "white"
        acc.add_row(label, fmt(v["accuracy"]), f"[{color}]{'+' if (net or 0) > 0 else ''}{fmt(net, ' pp')}")
    console.print(acc)

    trans = Table(title="State transitions (baseline -> after correction)")
    trans.add_column("Variant")
    for col in ("C->C kept", "C->I destroyed", "I->C repaired", "I->I unfixed", "Destruction rate", "Repair rate"):
        trans.add_column(col, justify="right")
    for arm, label in (("intrinsic", "Intrinsic"), ("grounded", "Grounded")):
        v = summary["variants"][arm]
        c, p = v["transitions"], v["transitions_pct"]
        trans.add_row(
            label,
            *(f"{c[k]} ({fmt(p[k])})" for k in TRANSITIONS),
            fmt(v["destruction_rate_pct"]),
            fmt(v["repair_rate_pct"]),
        )
    console.print(trans)

    g = summary["variants"]["grounded"]
    console.print(
        f"Grounded: feedback triggered on {g['feedback_triggered']} task(s); "
        f"{g['accepted_but_hidden_fail']} accepted on public tests but failed held-out tests.\n"
        f"Tokens: {summary['tokens']['prompt_tokens']} prompt / {summary['tokens']['completion_tokens']} completion."
    )

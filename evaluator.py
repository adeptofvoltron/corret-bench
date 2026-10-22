"""Execute model-generated Python against assertion tests in a separate process.

Isolation measures (best effort, NOT a security sandbox - run untrusted
benchmarks inside a container/VM if that matters to you):
  * fresh interpreter in isolated mode (`python -I`), throwaway temp dir as cwd
  * minimal environment (the OpenAI key and other secrets are NOT inherited)
  * wall-clock timeout + CPU / memory / file-size rlimits, whole process group killed
  * stdin closed, solution executed with __name__ != "__main__"
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

SENTINEL = "__ALL_TESTS_PASSED_7f3a__"
MAX_FEEDBACK_CHARS = 3000

_RUNNER = f'''
import sys, traceback
ns = {{"__name__": "solution"}}

def _run(path):
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    exec(compile(src, path, "exec"), ns)

try:
    _run("setup.py")
    _run("solution.py")
    _run("tests.py")
except BaseException as exc:
    te = traceback.TracebackException.from_exception(exc)
    te.stack = traceback.StackSummary.from_list(
        [f for f in te.stack if not f.filename.endswith("_runner.py")]
    )
    sys.stderr.write("".join(te.format()))
    sys.exit(1)
sys.stdout.write("\\n{SENTINEL}\\n")
'''

_CODE_BLOCK = re.compile(r"```(?:python|py|python3)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    """Return the longest fenced code block, or the raw text if there is none."""
    blocks = _CODE_BLOCK.findall(text or "")
    if blocks:
        return max(blocks, key=len).strip()
    return (text or "").strip()


@dataclass
class ExecResult:
    passed: bool
    exit_code: int | None
    timed_out: bool
    stdout: str
    stderr: str

    def feedback(self) -> str:
        """Error text suitable for sending back to the model."""
        if self.timed_out:
            return "Execution timed out (possible infinite loop or far too slow)."
        err = self.stderr.strip() or f"Process exited with code {self.exit_code} and no error output."
        if len(err) > MAX_FEEDBACK_CHARS:
            err = "...(truncated)...\n" + err[-MAX_FEEDBACK_CHARS:]
        return err

    def to_dict(self) -> dict:
        d = asdict(self)
        d["stdout"] = d["stdout"][-1000:]
        d["stderr"] = d["stderr"][-MAX_FEEDBACK_CHARS:]
        return d


def _limits(cpu_s: int):
    def apply():
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 1))
        mem = 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        fsize = 10 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))

    return apply


def run_tests(solution: str, tests: str, setup_code: str = "", timeout_s: float = 10.0) -> ExecResult:
    with tempfile.TemporaryDirectory(prefix="sc_eval_") as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "_runner.py").write_text(_RUNNER, encoding="utf-8")
        (tmp_path / "setup.py").write_text(setup_code, encoding="utf-8")
        (tmp_path / "solution.py").write_text(solution, encoding="utf-8")
        (tmp_path / "tests.py").write_text(tests, encoding="utf-8")

        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONHASHSEED": "0", "LANG": "C.UTF-8"}
        preexec = _limits(int(timeout_s) + 1) if sys.platform != "win32" else None

        proc = subprocess.Popen(
            [sys.executable, "-I", "_runner.py"],
            cwd=tmp,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            preexec_fn=preexec,
            start_new_session=True,
        )
        try:
            out, err = proc.communicate(timeout=timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, AttributeError):
                proc.kill()
            out, err = proc.communicate()
            timed_out = True

    passed = not timed_out and proc.returncode == 0 and SENTINEL in out
    return ExecResult(
        passed=passed,
        exit_code=None if timed_out else proc.returncode,
        timed_out=timed_out,
        stdout=out.replace(SENTINEL, "").strip(),
        stderr=err,
    )

"""Task loaders.

Every task separates tests into two disjoint roles to avoid oracle leakage:

* ``public_tests``  - shown to the model in the prompt and the ONLY tests the
                      grounded-feedback loop is allowed to execute.
* ``unit_tests``    - held-out tests used exclusively for final scoring
                      (they include the public ones). Never seen by any
                      correction loop.
"""

from __future__ import annotations

import ast
import json
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

MBPP_URL = (
    "https://raw.githubusercontent.com/google-research/google-research/"
    "master/mbpp/sanitized-mbpp.json"
)
# Official MBPP test split.
MBPP_TEST_IDS = range(11, 511)

BCB_HARD_DATASET = "bigcode/bigcodebench-hard"
BCB_HARD_VERSION = "v0.1.4"
BCB_ROWS_URL = "https://datasets-server.huggingface.co/rows"
# Same limits as the official BigCodeBench evaluator: some tasks legitimately run
# for minutes (psutil/socket timing) and TensorFlow reserves a lot of address space.
BCB_TIMEOUT_S = 240.0
BCB_MEMORY_MB = 30 * 1024

# Tasks whose reference solution passes or fails depending on machine load. Validation
# can miss them, so they are always excluded, on top of data/excluded-<dataset>.json.
KNOWN_UNSTABLE = {
    "bigcodebench-hard": {
        "BigCodeBench/1040": "socket server timing test; fails under parallel load",
    },
}

# Interpreter used when EXEC_PYTHON / --exec-python is not given (relative to the project root).
DEFAULT_EXEC_PYTHON = {"bigcodebench-hard": ".venv-bcb/bin/python"}


@dataclass
class Task:
    id: str
    prompt: str
    public_tests: str
    unit_tests: str
    ground_truth: str | None = None
    setup_code: str = ""
    # False when public_tests is not human-readable (e.g. a whole unittest module
    # that also contains the held-out cases), so it must not go into the prompt.
    show_public_tests: bool = True
    exec_timeout_s: float | None = None
    exec_memory_mb: int | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _load_mbpp(data_dir: Path, limit: int) -> list[Task]:
    path = data_dir / "sanitized-mbpp.json"
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MBPP (sanitized) -> {path}")
        urllib.request.urlretrieve(MBPP_URL, path)

    records = json.loads(path.read_text(encoding="utf-8"))
    records = sorted(
        (r for r in records if r["task_id"] in MBPP_TEST_IDS), key=lambda r: r["task_id"]
    )

    tasks = []
    for r in records[:limit]:
        tests = r["test_list"]
        setup = "\n".join(r.get("test_imports") or [])
        tasks.append(
            Task(
                id=f"mbpp/{r['task_id']}",
                prompt=r["prompt"],
                public_tests=tests[0],
                unit_tests="\n".join(tests),
                ground_truth=r["code"],
                setup_code=setup,
            )
        )
    return tasks


_UNITTEST_RUNNER = """

import sys as _sys, unittest as _unittest
_suite = _unittest.TestSuite({cls}(_name) for _name in {names!r})
_result = _unittest.TextTestRunner(stream=_sys.stderr, verbosity=1).run(_suite)
if not _result.wasSuccessful():
    raise AssertionError(f"{{len(_result.failures) + len(_result.errors)}} of {{_result.testsRun}} test(s) failed")
"""


def _unittest_methods(test_code: str) -> tuple[str, list[str]]:
    """Name of the TestCase class and its test methods, in source order."""
    for node in ast.parse(test_code).body:
        if isinstance(node, ast.ClassDef) and any(
            getattr(b, "attr", getattr(b, "id", "")) == "TestCase" for b in node.bases
        ):
            names = [m.name for m in node.body if isinstance(m, ast.FunctionDef) and m.name.startswith("test")]
            return node.name, names
    raise ValueError("no unittest.TestCase class found")


def _fetch_bcb_hard(path: Path) -> None:
    rows, offset, total = [], 0, None
    while total is None or offset < total:
        query = urllib.parse.urlencode({"dataset": BCB_HARD_DATASET, "config": "default",
                                        "split": BCB_HARD_VERSION, "offset": offset, "length": 100})
        with urllib.request.urlopen(f"{BCB_ROWS_URL}?{query}", timeout=60) as resp:
            page = json.load(resp)
        total = page["num_rows_total"]
        rows.extend(r["row"] for r in page["rows"])
        offset += len(page["rows"])
        if not page["rows"]:
            break
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


def _load_bigcodebench_hard(data_dir: Path, limit: int) -> list[Task]:
    path = data_dir / f"bigcodebench-hard-{BCB_HARD_VERSION}.json"
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading BigCodeBench-Hard {BCB_HARD_VERSION} -> {path}")
        _fetch_bcb_hard(path)

    tasks = []
    for r in json.loads(path.read_text(encoding="utf-8"))[:limit]:
        cls, names = _unittest_methods(r["test"])
        tasks.append(
            Task(
                id=r["task_id"],
                prompt=r["instruct_prompt"],
                # Public = first test method only; held-out = the whole TestCase.
                public_tests=r["test"] + _UNITTEST_RUNNER.format(cls=cls, names=names[:1]),
                unit_tests=r["test"] + _UNITTEST_RUNNER.format(cls=cls, names=names),
                ground_truth=r["code_prompt"] + r["canonical_solution"],
                show_public_tests=False,
                exec_timeout_s=BCB_TIMEOUT_S,
                exec_memory_mb=BCB_MEMORY_MB,
                meta={"public_test": names[0], "n_tests": len(names), "libs": r["libs"]},
            )
        )
    return tasks


def _load_jsonl(path: Path, limit: int) -> list[Task]:
    """Custom tasks, one JSON object per line.

    Required keys: id, prompt, unit_tests. Optional: public_tests,
    ground_truth, setup_code. Without public_tests the grounded loop can only
    report crashes/syntax errors (the solution is run without any asserts).
    """
    tasks = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            tasks.append(
                Task(
                    id=str(r["id"]),
                    prompt=r["prompt"],
                    public_tests=r.get("public_tests", ""),
                    unit_tests=r["unit_tests"],
                    ground_truth=r.get("ground_truth"),
                    setup_code=r.get("setup_code", ""),
                )
            )
            if len(tasks) >= limit:
                break
    return tasks


BUILTIN_TASKS = [
    Task(
        id="builtin/median",
        prompt="Write a function `median(xs)` that returns the median of a non-empty list of numbers. For an even number of elements return the mean of the two middle values.",
        public_tests="assert median([3, 1, 2]) == 2",
        unit_tests="assert median([3, 1, 2]) == 2\nassert median([4, 1, 3, 2]) == 2.5\nassert median([5]) == 5\nassert median([-1, -5, 10, 0]) == -0.5",
        ground_truth="def median(xs):\n    s = sorted(xs)\n    n = len(s)\n    m = n // 2\n    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2",
    ),
    Task(
        id="builtin/roman",
        prompt="Write a function `to_roman(n)` that converts an integer 1 <= n <= 3999 to a Roman numeral string using subtractive notation.",
        public_tests="assert to_roman(4) == 'IV'",
        unit_tests="assert to_roman(4) == 'IV'\nassert to_roman(1994) == 'MCMXCIV'\nassert to_roman(3999) == 'MMMCMXCIX'\nassert to_roman(40) == 'XL'",
        ground_truth="def to_roman(n):\n    vals = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),(50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]\n    out = []\n    for v, s in vals:\n        while n >= v:\n            out.append(s)\n            n -= v\n    return ''.join(out)",
    ),
    Task(
        id="builtin/merge_intervals",
        prompt="Write a function `merge_intervals(intervals)` that takes a list of [start, end] pairs and returns a list of merged, non-overlapping intervals sorted by start. Intervals that touch (end == next start) must be merged.",
        public_tests="assert merge_intervals([[1, 3], [2, 6], [8, 10]]) == [[1, 6], [8, 10]]",
        unit_tests="assert merge_intervals([[1, 3], [2, 6], [8, 10]]) == [[1, 6], [8, 10]]\nassert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]\nassert merge_intervals([]) == []\nassert merge_intervals([[5, 6], [1, 2]]) == [[1, 2], [5, 6]]\nassert merge_intervals([[1, 10], [2, 3]]) == [[1, 10]]",
        ground_truth="def merge_intervals(intervals):\n    out = []\n    for s, e in sorted(intervals):\n        if out and s <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], e)\n        else:\n            out.append([s, e])\n    return out",
    ),
    Task(
        id="builtin/is_valid_parens",
        prompt="Write a function `is_valid(s)` that returns True if the string of brackets '()[]{}' is balanced and correctly nested, otherwise False.",
        public_tests="assert is_valid('()[]{}') is True",
        unit_tests="assert is_valid('()[]{}') is True\nassert is_valid('(]') is False\nassert is_valid('([)]') is False\nassert is_valid('') is True\nassert is_valid('((') is False\nassert is_valid(')(') is False",
        ground_truth="def is_valid(s):\n    pairs = {')': '(', ']': '[', '}': '{'}\n    stack = []\n    for c in s:\n        if c in pairs:\n            if not stack or stack.pop() != pairs[c]:\n                return False\n        else:\n            stack.append(c)\n    return not stack",
    ),
    Task(
        id="builtin/run_length",
        prompt="Write a function `rle(s)` that run-length encodes a string, e.g. 'aaabcc' -> 'a3b1c2'.",
        public_tests="assert rle('aaabcc') == 'a3b1c2'",
        unit_tests="assert rle('aaabcc') == 'a3b1c2'\nassert rle('') == ''\nassert rle('a') == 'a1'\nassert rle('aaaaaaaaaaaa') == 'a12'\nassert rle('abab') == 'a1b1a1b1'",
        ground_truth="def rle(s):\n    out = []\n    i = 0\n    while i < len(s):\n        j = i\n        while j < len(s) and s[j] == s[i]:\n            j += 1\n        out.append(f'{s[i]}{j - i}')\n        i = j\n    return ''.join(out)",
    ),
    Task(
        id="builtin/kth_largest",
        prompt="Write a function `kth_largest(nums, k)` that returns the k-th largest element (1-indexed, counting duplicates) of a list.",
        public_tests="assert kth_largest([3, 2, 1, 5, 6, 4], 2) == 5",
        unit_tests="assert kth_largest([3, 2, 1, 5, 6, 4], 2) == 5\nassert kth_largest([3, 2, 3, 1, 2, 4, 5, 5, 6], 4) == 4\nassert kth_largest([1], 1) == 1\nassert kth_largest([2, 2, 2], 3) == 2",
        ground_truth="def kth_largest(nums, k):\n    return sorted(nums, reverse=True)[k - 1]",
    ),
    Task(
        id="builtin/word_freq",
        prompt="Write a function `top_words(text, n)` that returns the n most frequent words (case-insensitive, words are runs of letters a-z) as a list of (word, count) tuples, sorted by count descending and then alphabetically.",
        public_tests="assert top_words('a b a', 1) == [('a', 2)]",
        unit_tests="assert top_words('a b a', 1) == [('a', 2)]\nassert top_words('The cat. the DOG, the cat!', 2) == [('the', 3), ('cat', 2)]\nassert top_words('b a', 2) == [('a', 1), ('b', 1)]\nassert top_words('', 3) == []\nassert top_words(\"it's\", 2) == [('it', 1), ('s', 1)]",
        ground_truth="import re\nfrom collections import Counter\n\ndef top_words(text, n):\n    c = Counter(re.findall(r'[a-z]+', text.lower()))\n    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:n]",
    ),
    Task(
        id="builtin/flatten",
        prompt="Write a function `flatten(x)` that flattens arbitrarily nested lists and tuples into a flat list, preserving order. Strings must not be split into characters.",
        public_tests="assert flatten([1, [2, [3, 4]], 5]) == [1, 2, 3, 4, 5]",
        unit_tests="assert flatten([1, [2, [3, 4]], 5]) == [1, 2, 3, 4, 5]\nassert flatten([]) == []\nassert flatten([('a', ['bc']), [[[]]]]) == ['a', 'bc']\nassert flatten([[1], (2, (3,))]) == [1, 2, 3]",
        ground_truth="def flatten(x):\n    out = []\n    for item in x:\n        if isinstance(item, (list, tuple)):\n            out.extend(flatten(item))\n        else:\n            out.append(item)\n    return out",
    ),
]


def exclusions_path(dataset: str, data_dir: Path) -> Path:
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in dataset)
    return data_dir / f"excluded-{slug}.json"


def load_tasks(dataset: str, data_dir: Path, limit: int, apply_exclusions: bool = True) -> list[Task]:
    """`dataset` is one of: ``mbpp``, ``bigcodebench-hard``, ``builtin``, ``jsonl:<path>``.

    Tasks listed in ``data/excluded-<dataset>.json`` (written by
    ``main.py --validate-dataset``: reference solution fails in this execution
    environment) and in ``KNOWN_UNSTABLE`` are skipped before ``limit`` is applied.
    """
    loaders = {
        "mbpp": lambda n: _load_mbpp(data_dir, n),
        "bigcodebench-hard": lambda n: _load_bigcodebench_hard(data_dir, n),
        "builtin": lambda n: BUILTIN_TASKS[:n],
    }
    if dataset in loaders:
        load = loaders[dataset]
    elif dataset.startswith("jsonl:"):
        load = lambda n: _load_jsonl(Path(dataset.split(":", 1)[1]), n)  # noqa: E731
    else:
        raise ValueError(f"Unknown DATASET={dataset!r} (expected mbpp | bigcodebench-hard | builtin | jsonl:<path>)")

    if not apply_exclusions:
        return load(limit)
    excluded = set(KNOWN_UNSTABLE.get(dataset, {}))
    excl_file = exclusions_path(dataset, data_dir)
    if excl_file.exists():
        excluded |= set(json.loads(excl_file.read_text(encoding="utf-8")))
    if not excluded:
        return load(limit)
    tasks = [t for t in load(10**9) if t.id not in excluded]
    print(f"Skipping {len(excluded)} excluded task(s): {', '.join(sorted(excluded))}")
    return tasks[:limit]

# corret-bench

Intrinsic vs. grounded self-correction of LLM-generated code.

Can a current model fix its own code when it is only asked to "review" it, or does it still need a signal from
the environment? Each task is solved once (baseline), then corrected two ways from that same answer:

| Variant | What the model gets in the correction round | Stops when |
| --- | --- | --- |
| A Baseline | Task description, one answer | - |
| B Intrinsic SC | "Review your output carefully. If you find any logic or syntax errors, output the corrected version. If it's correct, return it as is." | Model returns the same code, or round limit |
| C Grounded SC | Error output from running the **public** test | Public test passes, or round limit |

Scoring always uses the **held-out** test suite, which no correction loop ever sees. Results report accuracy and
the state transitions baseline -> corrected: C->C kept, C->I destroyed, I->C repaired, I->I unfixed.

## Results

`gpt-6-sol`, temperature 0, 1 correction round, first 50 tasks of each dataset.

| Dataset | Baseline | Intrinsic SC | Grounded SC | Repaired (I->C) | Destroyed (C->I) |
| --- | --- | --- | --- | --- | --- |
| MBPP (sanitized, test split) | 96% | 96% | 96% | 0 / 0 | 0 / 0 |
| BigCodeBench-Hard v0.1.4 | 34% | 34% (0.00 pp) | 48% (+14.00 pp) | intrinsic 0, grounded 7 | 0 / 0 |

- **Intrinsic correction changes nothing.** On BigCodeBench-Hard the model kept its answer in 48 of 50 tasks,
  including 33 that were wrong. The zero net change comes from no attempts to fix, not from repairs and
  regressions cancelling out.
- **Grounded correction helps.** 7 of 33 wrong answers repaired, none broken (McNemar exact p ≈ 0.016).
- **The grounded signal is weak.** The public test is a single test from each task's suite, so 21 of the 26 tasks
  still wrong after grounded correction pass it.
- **MBPP is saturated.** At 96% baseline there is nothing left to correct; the 2 remaining failures pass the
  public test, so feedback never triggers.

This matches the earlier finding that LLMs cannot self-correct without external feedback
(Huang et al., 2023, *Large Language Models Cannot Self-Correct Reasoning Yet*). The symptom differs: here the
model does not break correct answers, it just keeps every answer.

Caveats: n = 50 per dataset (95% CI on accuracy roughly ±13 pp), first 50 tasks rather than a random sample, one
model, one round.

Raw data: `results/<dataset>/raw_results.json` (every prompt, response, code and execution log) and
`results/<dataset>/summary_report.json`.

## Running

Requires Python 3.10+ and an OpenAI API key.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env
python main.py --dataset mbpp --max-samples 50
```

To keep the key out of `.env`, inject it from 1Password instead: create `.env.op` (git-ignored) with
`OPENAI_API_KEY=op://<vault>/<item>/<field>` and run `op run --env-file=.env.op -- python main.py`.

BigCodeBench-Hard runs generated code in a separate Python 3.10 environment with the benchmark's pinned
libraries (setup commands at the top of `requirements-bigcodebench.txt`):

```bash
RESULTS_DIR=results/bcb-hard python main.py --dataset bigcodebench-hard --max-samples 50
```

Datasets are downloaded to `data/` on first use. `--validate-dataset` runs the reference solutions (no API calls)
and writes tasks that fail in your environment to `data/excluded-<dataset>.json`; those are skipped in later runs.
Other options: `python main.py --help` and `.env.example`.

> **Warning:** generated code runs as a normal subprocess with only time and memory limits. It has network and
> filesystem access. Run it in a container or VM.

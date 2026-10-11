# Laptop validation

Measured on the Dell 14 Plus, Ryzen AI 7 350 / Radeon 860M (`gfx1152`).
Benchmark completed at 2026-10-11T18:34:02Z (UTC).

- Ubuntu 24.04.4, kernel 7.0.0-38-generic.
- Isolated PyTorch 2.12.0 + ROCm 7.14.1; no system driver/kernel changes.
- NLLB-200 distilled 600M, pinned revision, float32, batch size 4, four CPU threads.
- Actual GPU matrix multiplication and translation smoke tests passed.
- All three mixed-batch versus individual smoke outputs matched exactly.
- Real checkpoint recovery passed without rewriting committed rounds.
- Two launches reproduced all 96 benchmark translation outputs exactly.
- 53 CPU unit/workflow tests passed, including from a standalone copy without the parent project.
- The parent project's 50 tests passed; its execution code/configuration are unchanged.

The final benchmark collected **96 translations in 215.7 seconds**
(**0.445 translations/second**): 16 FLORES sentences over three
translation rounds, with diagnostic English at every round. There were no failed,
truncated or time-limited benchmark outputs. Smoke/loading time is separate.

| Workload | Translations | Short-benchmark extrapolation | Planning estimate |
|---|---:|---:|---:|
| Local: 50 sentences, 30 rounds, 50-language pool | 1,950 | 1.22 hours | 2.47 hours |
| Original main: 1,000 sentences, 100 rounds | 123,000 | 76.8 hours | 153.5 hours |

The planning estimate applies a 2× margin plus loading/packaging allowance. It is not
a measured full-run duration or a guaranteed bound. Larger batches, differing proportions
of diagnostic translations, text-length drift, and thermal throttling affect extrapolation.
The original main workload is unsuitable for the chosen four-hour laptop budget under
these tested settings. The smaller local study passes the estimate gate and has a hard
four-hour inference watchdog; it is not scientifically equivalent to the original main.

**Only smoke tests and short benchmarks have run. No long experiment is running.**

From the parent project directory, the explicit command for the smaller study is:

```bash
bash laptop/run_local.sh run rocm
```

The default `bash laptop/run_local.sh` remains benchmark-only. Plug the laptop into
AC power and keep it awake. Checkpointing preserves progress if the run stops early.
CPU fallback code is unit-tested, but actual CPU translation throughput was not benchmarked.

Raw data, exact environment metadata, per-file checksums and reproduction files are in:

- `.runtime/data/laptop_translation_small_v1_rocm/`
- `.runtime/archives/laptop_translation_small_v1_rocm_complete.zip`
- `.runtime/archives/laptop_translation_small_v1_rocm_complete.zip.sha256`

The verified archive contains 81 files; its completion scope is the
**benchmark**, not a completed main experiment. Runtime files are excluded from Git.

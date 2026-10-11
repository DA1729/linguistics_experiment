# Local multilingual translation on the Dell laptop

An independent copy of the experiment for the Ryzen AI 7 350 / Radeon 860M laptop.
It has no imports or runtime dependency on the parent project and can be copied into
its own repository. The parent remains the RunPod version. No nested Git repository
is created automatically.

See [VALIDATION.md](VALIDATION.md) for the measured Radeon results and the selected
study's approximately 1.2-hour estimate (2.5 hours with the planning margin).

```bash
bash run_local.sh
```

The default command installs an isolated environment, validates the Radeon with a
small compute test, loads the pinned NLLB model, and runs a **benchmark only**. It
never starts the main experiment automatically. Initial downloads are several GB;
later launches use the cache. Setup is bounded separately; the benchmark allows at
most ten minutes of execution. The laptop must be plugged in.

AMD packages are pinned to PyTorch 2.12.0 + ROCm 7.14.1 with `gfx1152` support,
following [AMD's installation instructions](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html).
The script installs no kernel, driver, system package or global Python package.
All environments, model files and outputs live under `.runtime/` and are excluded
from Git. If the AMD compute check fails in auto mode, a CPU benchmark is used and
clearly identified. Choose an explicit backend to prohibit fallback:

```bash
bash run_local.sh benchmark rocm
bash run_local.sh benchmark cpu
bash run_local.sh setup rocm
bash run_local.sh test
```

## Four-hour budget

After a successful benchmark, inspect the printed estimates. They cover the
configured local experiment and the original Pod pilot/main workloads. The planning
estimate doubles measured translation time and adds loading/packaging allowance.
Failures, truncation, or an incomplete/undersized benchmark block the run gate.
This is a conservative extrapolation, not a guarantee: sentence length, thermal
throttling and batching can change over a long trajectory.

Only when the configured workload passes that gate, start it explicitly:

```bash
bash run_local.sh run rocm
```

A run cannot be configured above **four hours**. A watchdog stops inference within
that window; packaging may take additional time. Unplugging AC power, low available
RAM, or a temperature at/above 90°C requests a stop between batches. Configurable
memory reserve defaults to 1 GiB. Sensor values are best-effort hardware readings.
No sleep prevention or background scheduling is installed; keep the lid open and
avoid suspend during a run. Four CPU worker threads leave capacity for other work.

The default local study uses **50 sentences, 50 languages, 30 rounds**, with English
observations at 0, 1, 2, 3 and multiples of 5. This is a smaller study than the Pod
main experiment, not an equivalent replacement for it. The benchmark uses 16 corpus
sentences over three rounds with observations every round. Edit `config/experiment.toml`
before benchmarking if a different local study is needed. Both schedules and text
are collected in full; the scientific protocol is otherwise the same.

## Results and recovery

```text
.runtime/
  venv_rocm/ or venv_cpu/       isolated packages
  cache/                      model/download caches
  probes/                     device correctness test results
  logs/                       setup output
  data/laptop_translation_small_v1_<backend>/
    corpus/                   original text and provenance
    runs/smoke_001/            real translation and recovery tests
    runs/pilot_001/            short timing benchmark, all raw records
    runs/main_001/             local experiment, if explicitly requested
    runtime_estimate.json      measurements and run/no-run decision
    environment.json          model files, package versions, device and precision
    reproduction/             code/config snapshot
  archives/                   verified portable ZIP and SHA-256 sidecar
```

Rerun the same command to resume its incomplete phase. Complete batches are retained;
completed rounds are verified against checksums and checkpoint state. Diagnostic
English never enters active trajectories. Empty/failed trajectories terminate, raw
outputs and errors remain available, and the archive distinguishes incomplete work.
A fresh invocation has a fresh wall-clock deadline, but the whole configured workload
must still pass the four-hour planning gate; changing settings requires a new benchmark.
For a changed model, precision, code or scientific configuration, select a new
`experiment.name`. CPU and ROCm have separate data directories and cannot silently
resume one another's trajectories.

```bash
bash run_local.sh package rocm
python3 -m src.package_data /absolute/path/to/results.zip
```

Use `LAPTOP_ROOT=/absolute/path` to relocate environments and data, or pass a custom
configuration as the third script argument. Source files, requirements and tests are
self-contained in this directory. Unit tests require only Python 3.11+ and download
nothing. Actual device/inference measurements are written by the benchmark.

## Scientific qualifications

Greedy decoding uses the same NLLB revision, explicit source tokens, target-grouped
mixed-source batching, and exact-text FLORES corpus selection as the Pod project.
Default precision is float32. CPU and GPU rounding can yield different exact strings;
recorded hardware/precision matter for this convergence experiment. Neither a successful
decoding nor a repeated string demonstrates semantic correctness or convergence.
The benchmark performs no statistical or semantic analysis.

The model is **CC-BY-NC 4.0** and FLORES is **CC-BY-SA 4.0**. See
[ATTRIBUTION.md](ATTRIBUTION.md). The independent code copy preserves full trajectory,
observation, failure, schedule, and checkpoint formats from the parent project.

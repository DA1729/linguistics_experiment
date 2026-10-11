# Repeated multilingual translation

A data-collection experiment testing whether initially distinct English sentences,
translated along independent randomized language paths, tend toward exact textual
convergence. Convergence is **not assumed**. FLORES sentences are distinct strings,
not necessarily semantically independent. No statistical or semantic analysis runs
on the GPU.

Each sentence traverses shuffled permutations of the language pool, with no immediate
repeat across permutation boundaries. English observations translate copies of current
states; they never replace a trajectory. All raw outputs, token IDs, warnings,
schedules, and source sentences are retained.

## Run on RunPod

1. Create an empty GitHub repository in your browser (without a README). In this
   project's local terminal, replace `YOUR_NAME` and run:
   ```bash
   git init
   git add .
   git commit -m "Add multilingual translation experiment"
   git branch -M main
   git remote add origin https://github.com/YOUR_NAME/linguistic_big.git
   git push -u origin main
   ```
2. Create a **GPU Pod**, preferably one RTX A5000 (24 GB). Select a RunPod
   PyTorch/CUDA template with **Python 3.11–3.13**, PyTorch **2.6 or newer**, and
   JupyterLab. Reserve at least **25 GB persistent storage** at `/workspace` for
   weights, environment, and results. Use a network volume if data must survive
   Pod deletion; verify the mount before starting. A container disk is ephemeral.
   See [RunPod storage](https://docs.runpod.io/pods/storage/types).
3. Open the Pod's JupyterLab terminal:
   ```bash
   cd /workspace
   git clone https://github.com/YOUR_NAME/linguistic_big.git
   cd linguistic_big
   bash run_pod.sh
   ```
   For a private repository, use GitHub's normal authenticated clone procedure;
   never put credentials in the configuration or this repository.
4. Watch the terminal or `tail -f
   /workspace/linguistic_big_runtime/data/multilingual_translation_v1/logs/execution.log`.
   Keep the terminal open. For a browser-independent run:
   ```bash
   nohup bash run_pod.sh > /workspace/translation_launcher.log 2>&1 &
   ```
5. Rerun **the same command** after interruption. Completed rounds are verified and
   reused; finished batches in an unfinished round are reused too.
6. In JupyterLab, browse `/workspace/linguistic_big_runtime/archives/`, right-click
   the ZIP, and download it and its `.sha256` sidecar. The terminal prints the exact
   location. Download before deleting storage.
7. **Stop the Pod in RunPod when finished.** Ending Python or this script does not
   stop GPU billing. Storage charges may continue while a Pod is stopped.

The script creates its environment, preserves a working CUDA PyTorch installation,
installs pinned inference packages, downloads the pinned model/corpus, runs GPU smoke
checks, runs the pilot, prints measured main-runtime/cost estimates, and collects the
main experiment. No local model download or CUDA setup is needed.

```bash
bash run_pod.sh smoke       # GPU integration checks and tiny resumable experiment
bash run_pod.sh pilot       # smoke then pilot
bash run_pod.sh main        # smoke, pilot (reused if complete), estimate, main
bash run_pod.sh resume      # same workflow as the default; never creates a new run
bash run_pod.sh package     # package existing data; no GPU or third-party libraries
bash run_pod.sh all /absolute/path/custom.toml
```

Set `LINGUISTIC_ROOT=/your/persistent/path` to override the storage root. Without
`/workspace`, the default is this repository's `.runtime/`; verify durability yourself.
Inference uses one visible GPU. First installation needs internet; cached reruns avoid
reinstalling matching packages and downloading unchanged weights. An older/unusable
PyTorch gets a pinned CUDA 12.4 fallback, requiring a compatible NVIDIA driver.
Unsupported Python/template environments fail clearly before running the experiment.

## Configure

Edit `config/experiment.toml` **before the first run**. Defaults:

| Phase | Sentences | Languages | Rounds | Repetitions | English observations |
|---|---:|---:|---:|---:|---|
| Smoke | 4 | 50 | 2 | 1 | Every round |
| Pilot | 100 | 50 | 30 | 1 | Every round |
| Main | 1,000 | 50 | 100 | 1 | 0, 1, 2, 3, then multiples of 5 |

`sentences`, `rounds`, `repetitions`, `observe_every`, and `observation_rounds` are
per-phase settings. Round zero and the final round are always observed; explicit
rounds and the regular interval are combined. Set `observe_every = 0` for explicit
rounds only. The smoke recovery check always runs once; keep `smoke.repetitions = 1`.

`experiment.language_count` selects a prefix of the documented diverse 50-code pool
in `src/languages.py`. `experiment.languages` overrides that count with an explicit
list of at least two unique NLLB codes excluding English. All codes are checked
against the installed tokenizer. Sentence IDs are stable hashes of exact source text;
each repetition gets independently derived schedule seeds and the **same corpus**.
Pilot/smoke use prefixes of the main shuffled corpus. The corpus selection always
contains the maximum sentence count requested across the three phases.

`dataset.local_file` accepts a UTF-8 file with one sentence per line; relative paths
are relative to the configuration file. Set `dataset.local_license` appropriately.
Spaces and Unicode are preserved; only line delimiters, empty lines, and exact
duplicates are removed. The default accessible, pinned FLORES mirror has 1,012 lines;
requests beyond the unique count fail explicitly. There is no synthetic fallback.

`model.batch_size`, `precision`, `max_input_tokens`, `max_new_tokens`, and
`max_batch_seconds` control inference. Defaults use float32, greedy decoding, batch
size 32, 512 input tokens, and 256 generated tokens. CUDA OOM halves batch size;
an OOM at size one stops safely. Input truncation and output limits are recorded.
A decoding time limit without EOS marks an output failed and preserves its raw text.
Empty/failed trajectories terminate; subsequent entries are marked skipped. Other
suspicious outputs remain in trajectories with flags. Repetition/Unicode heuristics
do not reliably detect incorrect languages, unintelligibility, or generic fallbacks.

The 18-hour limit covers one invocation, including setup; rerunning grants a new
window. Normal stopping occurs between batches. A hard watchdog allows up to 120
additional seconds before killing a stuck inference process, then attempts packaging.
Set `runtime.gpu_hourly_usd` to your actual Pod price; **$0.30/hour is an illustrative
assumption, not a quote**. Pilot-based estimates include diagnostics but are rough:
length drift and different target-group occupancy affect throughput. Setup, idle time,
storage, packaging, and other allocated time also cost money. Neither the deadline
nor estimates guarantee a $16 billing cap. Check your RunPod balance and stop the Pod.

For changed scientific settings, seeds, model revision, or code, use a **new
`experiment.name`**. Resume rejects mismatches and corrupted data. Runtime limit,
progress frequency, and assumed price may change without starting a new experiment.

## Data and recovery

```text
linguistic_big_runtime/
  venv/                       Python environment (not archived)
  cache/huggingface/           model cache (not archived)
  logs/bootstrap.log          setup output
  data/<experiment_name>/
    corpus/                   exact selected corpus, original file, provenance
    reproduction/             code, config, requirements, README, attribution
    environment.json          exact model/tokenizer revisions, packages, GPU, decoding
    language_pool.json        readable names and NLLB identifiers
    runs/<phase>_<repetition>/
      schedules.jsonl         independent full paths and per-sentence seeds
      metadata.json           immutable run identity
      status.json             completion and operational counters
      rounds/round_000000/    observations, translations, checkpoint, SHA-256 manifest
      rounds/pending_000001/  durable partial journal, if interrupted
    smoke_*                   real mixed/individual probes and recovery checks
    logs/execution.log        progress, warnings, failures
    workflow_status.json      completion for the requested workflow
    runtime_estimate.json     measured pilot extrapolation
  archives/<experiment_name>_<complete|incomplete>.zip
```

Every round directory has `translations.jsonl` and `observations.jsonl`. Rows include
experiment/run/sentence IDs, schedule seed, round, source and target codes, exact input
and output, status, and warnings. Observations additionally include original/current
sentences and `english_translation`. `status = ok` means a nonempty completed decoding,
not linguistic correctness. `model_invoked = false` identifies round-zero copies and
skipped terminated trajectories. Do not treat failures, skipped/null values, or empty
strings as successful convergence. Token-limit outputs can still be `ok`, with flags.

Batches append to an fsynced journal. After all translations **and observations** finish,
the program writes the checkpoint and checksums, then atomically renames the round
directory. Committed directories are authoritative; the status file is descriptive.
Resume verifies checksums, labels, IDs, schedule/request digests, observations, and
checkpoint states. Torn final journal bytes are quarantined in `.torn_*` files; complete
rows are reused. Only inference that had not reached a durable journal write may need
repeating. `journal.jsonl` is recovery evidence duplicating the materialized records:
use **translations.jsonl and observations.jsonl in committed rounds** for standard
analysis; examine pending journals separately for incomplete work.

A filesystem lock prevents concurrent writers. A ZIP contains all experiment data,
partial evidence, logs, reproduction files, and a per-file SHA-256 manifest, verified
by both ZIP CRC and SHA-256. It excludes caches, environments, and weights. Packaging
does not remove resumable state. An incomplete ZIP retains completed data. A hard
machine shutdown cannot produce an archive at that instant: rerun `package` or resume
once storage is accessible. Do not manually edit state files.

On your CPU computer, from this project directory:
```bash
python3 -m src.package_data /path/to/downloaded.zip
python3 -m unittest discover -s tests -v
```

Unit tests use lightweight doubles and need only Python 3.11+. They never download a
model. GPU smoke tests check actual source/target tokens, direct non-English translation,
mixed versus individual outputs, disk records, and resume without changing committed
rounds. The comparison is retained even when floating-point batching changes a greedy
output; a difference is logged. Exact bitwise reproduction across devices, library
versions, padding/batch sizes, OOM retries, or timing cutoffs is **not guaranteed**.
Pinned software, complete token records, and recorded environment/batch sizes make
these effects inspectable. No hidden English pivot is used.

## Research scope and attribution

NLLB is a research translation model with model-specific errors and biases; repeated
translation can amplify them. A loss of meaning does not imply convergence. GPU work
here collects data; later CPU analysis is a separate project.

The [NLLB model](https://huggingface.co/facebook/nllb-200-distilled-600M) is
**CC-BY-NC 4.0 (noncommercial)**. The upstream FLORES corpus is **CC-BY-SA 4.0**;
retain its attribution and license when sharing. See [ATTRIBUTION.md](ATTRIBUTION.md)
for sources and the mirror's differing license label. Translation token handling follows
[Hugging Face's NLLB documentation](https://huggingface.co/docs/transformers/model_doc/nllb),
using the nonlegacy source-prefix convention and a forced destination token.

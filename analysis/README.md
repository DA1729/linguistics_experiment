# CPU analysis of repeated multilingual translation

Run from the repository root:

```bash
bash analyze_local.sh
```

The default analyzes the immutable `data/multilingual_translation_v1_complete.zip`, audits the main/pilot/smoke runs, and writes derived results under `reports/translation_v1/`. It performs **no translation inference**, uses no GPU or paid service, and limits numerical libraries and the embedding model to four CPU threads. The main scientific estimates use `main_001` only.

Python 3.11–3.13 is recommended. The launcher creates `.analysis_runtime/venv` using system packages when available and installs the pinned analysis dependencies there. An existing PyTorch is used on CPU; if absent, the launcher installs the CPU wheel. No system package or driver is changed. The report records the actual versions, source hashes, archive hash and analysis settings. Exact reproducibility across different CPUs/library builds is not guaranteed at the last floating-point bit.

The first semantic run downloads the small public MiniLM model pinned in `config/analysis.toml` (approximately 90 MB of weights). It never downloads NLLB. Model files and embeddings stay in ignored `.analysis_runtime/`; the scientific tables and figures are publishable. All original text stays unchanged. The loader reads the ZIP directly; journal copies are checked but excluded from translation totals.

## Stages and reruns

```bash
bash analyze_local.sh --skip-semantic
bash analyze_local.sh --stage audit --refresh-audit
bash analyze_local.sh --stage exact
bash analyze_local.sh --stage temporal
bash analyze_local.sh --stage content
bash analyze_local.sh --stage semantic
bash analyze_local.sh --stage language
bash analyze_local.sh --stage robustness
bash analyze_local.sh --stage figures
bash analyze_local.sh --stage report
```

Individual stages reuse a hash-matched reconstruction; figures/report require prior metric tables. The full run is the recommended entry point. `--config PATH` accepts another TOML configuration, though the narrative report is designed for translation v1's fixed protocol. Do not point outputs or caches at the original archive. If you change the archive or experiment, review narrative assumptions, not just numerical tables.

Embedding inference is cached in atomic chunks of 512 distinct texts with a content/model/settings/package fingerprint. A complete cache has its own checksum. Interrupted work resumes at chunk boundaries. The archive and loader hashes invalidate reconstructed-data caches. Caches are trusted local files; do not load a reconstruction pickle obtained from another person. Removing `.analysis_runtime/cache` forces recomputation. An active-run file lock prevents competing writers.

`--skip-semantic` marks semantic results as skipped; old semantic tables from an earlier run are not used for current figures/report conclusions. If a public model download fails, exact, lexical, temporal and language analyses remain available and the semantic reason is recorded. To retry, rerun the full command with internet access. The bundled HTML embeds all report figures and has no rendering dependency on a network connection; its supplementary table links still refer to repository files.

## Units and definitions

- `round_level_metrics.csv`: observation round × normalization. Only observed rounds appear. Raw Unicode equality is primary; all collector-accepted, nonempty outputs are included.
- `exact_output_clusters.csv`: observation round × exact output label, with frequency, rank and collision-group membership. Singleton membership IDs are omitted to reduce duplication; source observations remain in the ZIP.
- `active_state_metrics.csv`: all 101 checkpoints; identity is `(language, exact_text)`, and inactive states are excluded.
- `projection_collision_metrics.csv`: English-equal unordered pairs split into same/different active states. Undefined collision shares are blank, not zero.
- `collision_persistence.csv`: adjacent **observed** endpoints on common valid IDs, with actual round gap, null expectation, 999-permutation interval and Holm-adjusted p-value. The null reassigns membership preserving each endpoint's group sizes.
- `collision_pair_recurrence.csv`: unique pairs ever equal, their observed equality rounds and whether a later collision followed observed separation.
- `group_transitions.csv`: observed interval × output-to-output transition touching a nonsingleton group, with trajectory count.
- `semantic_round_metrics.csv`: observation round; fixed 50,000-pair sample quantiles, exact all-pair mean, original cosine, geometry and truncation counts.
- `semantic_trajectory_metrics.csv`: observation round × trajectory. Cosine is an embedding proxy, not a factual fidelity score.
- `semantic_pair_sample.csv`: the fixed sampled unordered trajectory pairs, not independent experimental replicates.
- `semantic_graph_metrics.csv`: observation round × threshold. Connected components permit chaining and do not define semantic equivalence.
- `semantic_robustness.csv`: observation round × filter, using the remaining population's embeddings.
- `trajectory_bootstrap.csv`: 500 whole-trajectory bootstrap replicates summarized by round. Repeated measures are preserved; intervals describe corpus sensitivity.
- `tfidf_round_metrics.csv`: one fixed retrospective TF-IDF representation fitted to all distinct texts. Includes all-pair mean and original cosine; no inference model is required.
- `lexical_trajectory_metrics.csv`: observation round × trajectory; characters, Unicode regex words/tokens, ratios and content-marker retention.
- `lexical_round_metrics.csv`: observation round; averages, quantiles, corpus vocabulary and explicit eligible denominators for retention proxies.
- `translation_failure_metrics.csv`: round × record kind. Recorded failures, recorded warnings and heuristic flags remain separate. Warning sets overlap.
- `language_statistics.csv`: language × source/target role; translation and trajectory counts, lengths and flags.
- `language_transitions.csv`: source-target pair; all observed transitions and sample sizes, with descriptive quality statistics.
- `language_semantic_associations.csv`: current observation language. Cosines are also centered within observation rounds; no causal interpretation.
- `language_interval_associations.csv`: preceding-interval language exposure. Multi-step changes are associated with every encountered language and cannot be assigned to one of them.
- `candidate_attractors.csv`: exact strings shared by at least two trajectories at some observation; a screening label, not a mathematical claim.
- `robustness_metrics.csv`: round × filter/subsample metric. Subsets are fixed across rounds; smaller samples naturally suppress rare collisions.
- `archive_inventory.csv`, `record_status_counts.csv`, `data_quality_summary.csv`: audit units are explicit in the columns.

Blank numeric CSV entries are undefined/not applicable, never imputed observations. All logarithms are natural. Labels contain real text and may include commas/newlines; use a CSV parser rather than splitting on commas. No table estimates channel mutual information.

## Heuristics

Recorded `ok` is preserved. Additional flags mark empty/replacement-character outputs, at least 12 words with unique-word proportion ≤ 0.25, a trigram appearing in at least 40% of at least 10 trigram positions, more than 1,000 characters, or output length below 20% of an input of at least 40 characters. English uses Unicode regex words; multilingual active-output repetition uses whitespace tokens and is script-dependent. A generic sentence is not automatically a failure.

Numeric retention compares sets of canonical decimal tokens (comma thousands separators removed); it does not recognize spelled-out numbers. Date proxies use English month names and years 1500–2099, with possible ambiguity such as “may”. Negation markers ignore scope. Named-entity and target-language identification are not performed. These limitations are explicit in the report.

## Tests

```bash
.analysis_runtime/venv/bin/python -m unittest discover -s tests -v
```

Synthetic tests cover known convergence partitions, irregular observation schedules, missing/invalid data, projection collisions, permutation expectations, cosine geometry, archive loading, continuity validation and deterministic output generation. The executed real-data consistency checks and visual review are recorded in `reports/translation_v1/validation.md`.

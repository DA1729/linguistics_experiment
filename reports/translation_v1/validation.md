# Validation record

The full local analysis ran the synthetic analysis tests and existing collector tests. See [test results](test_results.json) and [full output](test_results.txt). No GPU or translation inference was used.

The archive audit validated canonical records, continuity, schedules, all checksums and observation-state separation. [Consistency checks](consistency_checks.json) reconcile generated counts, projection partitions, cosine identities, figure files and report links. [Output manifest](output_manifest.json) lists final derived-file hashes. These checks do not establish human semantic correctness.

## Executed results

- **68 tests passed:** 18 synthetic analysis tests and 50 collector tests.
- **248 real-data consistency checks passed.**
- **34 CSV tables reproduced byte-for-byte** across complete executions; see [rerun verification](rerun_verification.json).
- A clean non-embedding pipeline run passed **216 checks**, with semantic results explicitly marked skipped.
- **22 PNG/PDF figure families** were generated and visually reviewed. All 22 PNGs were inspected in a contact sheet; figures 12 and 22 were additionally viewed at full size. [Visual review and file hashes](visual_review.json).
- The complete cached analysis took approximately **20 seconds** and peaked at approximately **1.08 GiB RSS** in the measured final run; first-time model download/embedding preparation is additional. See [execution record](last_execution.json).
- The archive checksum remained unchanged before and after analysis. No original data, experiment-generation source, Git history or remote repository was modified by the analysis.

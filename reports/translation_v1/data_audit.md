# Data audit: multilingual translation v1

The original ZIP was opened read-only. Its sidecar SHA-256, ZIP CRCs, all 717 manifest-listed file hashes and every nested round manifest passed. The ZIP contains 718 members including its integrity manifest. SHA-256: `47c3b3c552a24e8da9dcfedd0a308fa95595feae73a3c3e76d3b19fedc81e3ab`.

## Actual counts

| run_id | trajectories | checkpoints | translation_records | translation_valid | observation_records | observation_valid | model_invoked |
|---|---|---|---|---|---|---|---|
| main_001 | 1000 | 101 | 100000 | 100000 | 24000 | 24000 | 123000 |
| pilot_001 | 100 | 31 | 3000 | 3000 | 3100 | 3100 | 6000 |
| smoke_001 | 4 | 3 | 8 | 8 | 12 | 12 | 16 |

The main run contains 100,000 accepted active-trajectory translation records and 24,000 accepted English observations. Of the latter, 1,000 are initial copies, leaving 23,000 diagnostic model calls. Thus the main model-call count is 123,000; the canonical record count is 124,000. Journal entries duplicate canonical records for recovery and are **not** counted as additional translations. Pilot and smoke results are audited separately and are not pooled into main scientific estimates.

The actual main English observation rounds are:

`[0, 1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]`

There are **24 observation rounds**, **101 main checkpoints**, and **0 pending files**. All 1,000 final trajectories are active. No duplicate IDs, duplicate canonical records, missing expected observations, missing rounds, invalid status values, malformed JSON records, or empty outputs were found. Explicit failed/skipped records: zero.

## Reconstruction and integrity checks

All three run histories were reconstructed from the source corpus. Every translation's input, original sentence, source/target language, status, request digest and record digest was validated. Seeded schedules were reproduced and checked against archived schedules, including complete 50-language permutations and no adjacent repeat across permutation boundaries. Checkpoints matched reconstructed active states at every round; cumulative model-call counters matched. English observation inputs matched the current active state, and subsequent active translations continued from that state, establishing that diagnostic outputs did not replace it in this archive. Journals matched canonical record IDs and digests; journal file hashes were separately verified.

**Discrepancies: 0.** Machine-readable findings: [data_quality_summary.json](data_quality_summary.json), [data_discrepancies.json](data_discrepancies.json), [archive inventory](tables/archive_inventory.csv), and [status counts](tables/record_status_counts.csv).

## Provenance and model

The corpus is 1,000 distinct exact English strings selected from FLORES-200 devtest, public mirror `DGME/FLORES-200`, revision `8a8758a392e429432f6e95fcac1445550a4d3ccd`, selection seed 1729. Source checksum `612e9fbe87997617c0fa8fa8929654a4f49b728d96738112c2b86ef6a1d78d88` matched. The archived source has 1,012 lines; exact deduplication and seeded shuffling selected the main corpus. Upstream attribution: NLLB Team et al. (2022), *No Language Left Behind: Scaling Human-Centered Machine Translation*. The archive records CC-BY-SA-4.0 from upstream and notes that the mirror card lists a different license; retain upstream attribution and consult the upstream terms when redistributing derived texts.

NLLB model `facebook/nllb-200-distilled-600M`, pinned revision `f8d333a098d19b4fd9a8b18f94170487ad3f821d`, greedy decoding, float32, Transformers 4.57.6, PyTorch 2.8.0+cu128, NVIDIA RTX A5000. Every model-file hash and generation setting remains available in the archive's `environment.json`. The analysis downloads no translation model and performs no translation inference.

## Language pool

| code | name |
|---|---|
| fra_Latn | French |
| deu_Latn | German |
| arb_Arab | Modern Standard Arabic |
| jpn_Jpan | Japanese |
| hin_Deva | Hindi |
| spa_Latn | Spanish |
| rus_Cyrl | Russian |
| zho_Hans | Chinese (Simplified) |
| swh_Latn | Swahili |
| tam_Taml | Tamil |
| kor_Hang | Korean |
| tur_Latn | Turkish |
| ben_Beng | Bengali |
| ita_Latn | Italian |
| ell_Grek | Greek |
| heb_Hebr | Hebrew |
| tha_Thai | Thai |
| vie_Latn | Vietnamese |
| ind_Latn | Indonesian |
| fin_Latn | Finnish |
| pol_Latn | Polish |
| ukr_Cyrl | Ukrainian |
| nld_Latn | Dutch |
| por_Latn | Portuguese |
| pes_Arab | Western Persian |
| urd_Arab | Urdu |
| tel_Telu | Telugu |
| mal_Mlym | Malayalam |
| kan_Knda | Kannada |
| mar_Deva | Marathi |
| guj_Gujr | Gujarati |
| pan_Guru | Eastern Panjabi |
| npi_Deva | Nepali |
| sin_Sinh | Sinhala |
| mya_Mymr | Burmese |
| khm_Khmr | Khmer |
| lao_Laoo | Lao |
| bod_Tibt | Tibetan |
| kat_Geor | Georgian |
| hye_Armn | Armenian |
| amh_Ethi | Amharic |
| hau_Latn | Hausa |
| yor_Latn | Yoruba |
| zul_Latn | Zulu |
| som_Latn | Somali |
| ron_Latn | Romanian |
| hun_Latn | Hungarian |
| swe_Latn | Swedish |
| ces_Latn | Czech |
| kaz_Cyrl | Kazakh |

## Quality limitations

The main run has 6,511 recorded generation-token-limit warnings and 6,740 repetition warnings across active translations and English observations. These warning categories overlap. They remain `ok` records under the original collection rules. “Valid” in this report means structurally valid and collector-accepted, **not independently verified as meaningful or faithful translation**.

No input truncation or explicit translation failures were recorded. Additional analysis flags are separate from recorded warnings. We did not independently validate target-language correctness, run named-entity recognition, or adjudicate multilingual translation quality. Broad whitespace/character heuristics vary by script; language comparisons require particular caution. See [quality metrics](tables/translation_failure_metrics.csv) and the full report for sensitivity analyses.

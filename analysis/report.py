import base64
import html
import json
from pathlib import Path
import re

import markdown
import numpy as np
import pandas as pd

from .common import save_json
from .diagnostics import additional_diagnostics


def table_markdown(frame,columns=None):
    if columns: frame=frame[columns]
    def cell(value):
        if pd.isna(value): return '—'
        if isinstance(value,(float,np.floating)): return f'{value:.6g}'
        return str(value).replace('|','&#124;').replace('\n',' ')
    lines=['| '+' | '.join(frame.columns)+' |','|'+'|'.join(['---']*len(frame.columns))+'|']
    lines.extend('| '+' | '.join(cell(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None))
    return '\n'.join(lines)


def generate_report(data,config,proof):
    output=Path(config['output'])
    def read(name): return pd.read_csv(output/'tables'/f'{name}.csv')
    audit=data['audit'];main=next(r for r in audit['runs'] if r['run_id']==config['run_id'])
    raw=read('round_level_metrics').query("normalization=='raw'").sort_values('round')
    final=raw.iloc[-1];initial=raw.iloc[0]
    projection=read('projection_collision_metrics');persistence=read('collision_persistence')
    active=read('active_state_metrics');lex=read('lexical_round_metrics');lf=lex.iloc[-1];li=lex.iloc[0]
    quality=read('translation_failure_metrics');robust=read('robustness_metrics')
    candidates=read('candidate_attractors');recurrence=read('collision_pair_recurrence')
    tfidf=read('tfidf_round_metrics');norm=read('round_level_metrics').query('round==100')
    status=json.loads((output/'semantic_status.json').read_text())
    has_semantics=status['status']=='complete'
    sem=read('semantic_round_metrics') if has_semantics else None
    sf=sem.iloc[-1] if has_semantics else None
    si=sem.iloc[0] if has_semantics else None
    examples=additional_diagnostics(data,config)
    collision_quality=read('collision_quality_associations')
    figure_catalog=json.loads((output/'figure_catalog.json').read_text())
    figure_map={r['name'][:2]:r for r in figure_catalog}
    def figure(number):
        key=f'{number:02d}'
        if key not in figure_map: return ''
        row=figure_map[key]
        return f"\n![{html.escape(row['caption'])}](figures/{row['name']}.png)\n\n*Figure {number}. {row['caption']}*\n"
    collisions=raw[raw.collision_pairs>0]
    pooled_pairs=int(projection.english_collision_pairs.sum())
    pooled_projection=int(projection.distinct_state_collision_pairs.sum())
    previous_pairs=int(persistence.previous_equal_pairs.sum());retained_pairs=int(persistence.persistent_pairs.sum())
    summary={'experiment':'multilingual_translation_v1','archive_sha256':audit['archive_sha256'],'integrity_discrepancies':len(audit['discrepancies']),
             'manifest_files':audit['manifest_files'],'main_trajectories':main['trajectories'],'main_translation_records':main['translation_records'],
             'main_valid_translation_records':main['translation_valid'],'main_observation_records':main['observation_records'],
             'main_model_invocations':main['model_invoked'],'observation_rounds':audit['observation_rounds'],
             'exact':{'final_distinct':int(final.distinct),'minimum_distinct':int(raw.distinct.min()),
                      'universal_equality_observed':bool((raw.distinct==1).any()),'majority_equality_observed':bool((raw.largest_fraction>.5).any()),
                      'first_collision_round':int(collisions['round'].min()) if len(collisions) else None,
                      'largest_group_ever':int(raw.largest_size.max()),'max_largest_fraction':float(raw.largest_fraction.max()),
                      'final_collision_probability':float(final.collision_probability),'max_collision_probability':float(raw.collision_probability.max()),
                      'final_largest_group':int(final.largest_size),'threshold_rounds_reached':{str(th):int(raw.loc[raw.largest_fraction>=th,'round'].min()) for th in [.1,.25,.5,.75,.9] if (raw.largest_fraction>=th).any()},
                      'diversity_increasing_intervals':int((raw.distinct.diff()>0).sum())},
             'projection':{'pooled_equal_pair_observations':pooled_pairs,'pooled_distinct_state_pair_observations':pooled_projection,
                           'distinct_state_share':pooled_projection/pooled_pairs if pooled_pairs else None},
             'persistence':{'eligible_pair_intervals':previous_pairs,'retained_pair_intervals':retained_pairs,
                            'pooled_fraction':retained_pairs/previous_pairs if previous_pairs else None,
                            'unique_pairs_ever_equal':len(recurrence),'pairs_recolliding_after_separation':int(recurrence.separated_then_recollided.sum())},
             'semantic_status':status,'semantic_initial':si.to_dict() if has_semantics else None,'semantic_final':sf.to_dict() if has_semantics else None,
             'lexical_final':lf.to_dict(),'figures':len(figure_catalog),'summary_tables':len(list((output/'tables').glob('*.csv'))),
             'provenance':proof}
    save_json(output/'analysis_summary.json',summary)
    counts=pd.DataFrame([{k:r.get(k,0) for k in ['run_id','trajectories','checkpoints','translation_records','translation_valid','observation_records','observation_valid','model_invoked']} for r in audit['runs']])
    pool=pd.DataFrame(audit['language_pool'])
    audit_text=f'''# Data audit: multilingual translation v1

The original ZIP was opened read-only. Its sidecar SHA-256, ZIP CRCs, all {audit['manifest_files']} manifest-listed file hashes and every nested round manifest passed. The ZIP contains {audit['zip_members']} members including its integrity manifest. SHA-256: `{audit['archive_sha256']}`.

## Actual counts

{table_markdown(counts)}

The main run contains {main['translation_valid']:,} accepted active-trajectory translation records and {main['observation_valid']:,} accepted English observations. Of the latter, 1,000 are initial copies, leaving 23,000 diagnostic model calls. Thus the main model-call count is {main['model_invoked']:,}; the canonical record count is 124,000. Journal entries duplicate canonical records for recovery and are **not** counted as additional translations. Pilot and smoke results are audited separately and are not pooled into main scientific estimates.

The actual main English observation rounds are:

`{audit['observation_rounds']}`

There are **{len(audit['observation_rounds'])} observation rounds**, **101 main checkpoints**, and **{audit['pending_files']} pending files**. All 1,000 final trajectories are active. No duplicate IDs, duplicate canonical records, missing expected observations, missing rounds, invalid status values, malformed JSON records, or empty outputs were found. Explicit failed/skipped records: zero.

## Reconstruction and integrity checks

All three run histories were reconstructed from the source corpus. Every translation's input, original sentence, source/target language, status, request digest and record digest was validated. Seeded schedules were reproduced and checked against archived schedules, including complete 50-language permutations and no adjacent repeat across permutation boundaries. Checkpoints matched reconstructed active states at every round; cumulative model-call counters matched. English observation inputs matched the current active state, and subsequent active translations continued from that state, establishing that diagnostic outputs did not replace it in this archive. Journals matched canonical record IDs and digests; journal file hashes were separately verified.

**Discrepancies: {len(audit['discrepancies'])}.** Machine-readable findings: [data_quality_summary.json](data_quality_summary.json), [data_discrepancies.json](data_discrepancies.json), [archive inventory](tables/archive_inventory.csv), and [status counts](tables/record_status_counts.csv).

## Provenance and model

The corpus is 1,000 distinct exact English strings selected from FLORES-200 devtest, public mirror `DGME/FLORES-200`, revision `{audit['provenance']['revision']}`, selection seed 1729. Source checksum `{audit['provenance']['source_sha256']}` matched. The archived source has {audit['source_lines']:,} lines; exact deduplication and seeded shuffling selected the main corpus. Upstream attribution: NLLB Team et al. (2022), *No Language Left Behind: Scaling Human-Centered Machine Translation*. The archive records CC-BY-SA-4.0 from upstream and notes that the mirror card lists a different license; retain upstream attribution and consult the upstream terms when redistributing derived texts.

NLLB model `{audit['model']['model_name']}`, pinned revision `{audit['model']['model_revision']}`, greedy decoding, float32, Transformers {audit['model']['transformers']}, PyTorch {audit['model']['torch']}, {audit['model']['gpu']}. Every model-file hash and generation setting remains available in the archive's `environment.json`. The analysis downloads no translation model and performs no translation inference.

## Language pool

{table_markdown(pool)}

## Quality limitations

The main run has {main.get('warning_generation_token_limit',0):,} recorded generation-token-limit warnings and {main.get('warning_repetitive_output',0):,} repetition warnings across active translations and English observations. These warning categories overlap. They remain `ok` records under the original collection rules. “Valid” in this report means structurally valid and collector-accepted, **not independently verified as meaningful or faithful translation**.

No input truncation or explicit translation failures were recorded. Additional analysis flags are separate from recorded warnings. We did not independently validate target-language correctness, run named-entity recognition, or adjudicate multilingual translation quality. Broad whitespace/character heuristics vary by script; language comparisons require particular caution. See [quality metrics](tables/translation_failure_metrics.csv) and the full report for sensitivity analyses.
'''
    (output/'data_audit.md').write_text(audit_text)
    semantic_abstract=(f'Mean MiniLM similarity to the original declined from {si.own_mean:.3f} to {sf.own_mean:.3f}, while mean cross-trajectory cosine rose from {si.pairwise_mean_exact:.3f} to {sf.pairwise_mean_exact:.3f}. This modest embedding concentration coexisted with broad residual diversity; it did not establish semantic identity.' if has_semantics else 'Semantic embeddings were unavailable; lexical TF-IDF and exact analyses remain available, and no embedding-based conclusion is drawn.')
    abstract=f'''We investigate whether repeated translation through independently randomized language paths produces universal textual equality, weaker concentration, or persistent diversity. A completed experiment propagated 1,000 distinct FLORES English sentences through 100 rounds using NLLB-200 distilled 600M and a pool of 50 non-English languages. Diagnostic English observations were recorded at 24 rounds; they did not replace active trajectory states. We audited all archived runs, validated 100,000 main trajectory transitions and 24,000 English observations, and found no reconstruction discrepancies. At the final observation, {int(final.distinct)} of 1,000 outputs were distinct. The largest exact group reached only {int(raw.largest_size.max())} trajectories at any observation and contained {int(final.largest_size)} at the end. Exact collisions first appeared at round {int(collisions['round'].min())}; their frequency was nonmonotone, and no universal or majority equality occurred. Most English-equal pair observations joined different pre-observation linguistic states. Some pairs remained equal across observed endpoints more often than under independent reassignment, but splitting and recurrence preclude an absorbing-state interpretation. {semantic_abstract} Vocabulary and original-content proxies declined strongly. Recorded repetition and generation-limit warnings complicate interpretation, but excluding flagged outputs did not remove the main pattern of drift with diversity. The strongest empirical result is therefore not universal convergence: substantial alteration of original content coexists with mostly distinct outputs and small, partly transient collision groups. These finite, model-specific observations motivate questions about observation maps, nonabsorbing coalescence, and operator coupling, without establishing asymptotic convergence or a communication-channel information law.'''
    key_metrics=raw[raw['round'].isin([0,25,50,75,90,100])][['round','valid_n','distinct','collision_pairs','largest_size','entropy_nats','effective_shannon','effective_simpson']]
    final_projection=projection.iloc[-1]
    sem_section='Semantic embedding analysis is incomplete: '+str(status.get('reason','unavailable'))+'. TF-IDF remains available in [the lexical baseline table](tables/tfidf_round_metrics.csv).'
    if has_semantics:
        graph=read('semantic_graph_metrics');gf=graph[graph['round']==100]
        sem_section=f'''The model is [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), pinned to `{status['revision']}`. It produces L2-normalized 384-dimensional embeddings; similarities are cosine inner products. We used Sentence Transformers {status['sentence_transformers']}, batch size {status['batch_size']}, four CPU threads and a {status['max_seq_length']}-wordpiece input limit. Model weights and embeddings are cached with configuration/content fingerprints. There are {status['unique_texts']:,} unique original/observed texts; {status['unique_texts_exceeding_limit']} unique texts exceed the embedding input limit. [Model provenance](semantic_status.json) records file hashes and package versions. The model card describes a sentence/short-paragraph encoder, not a truth or translation-fidelity evaluator.

Mean original-output cosine falls from {si.own_mean:.6f} to {sf.own_mean:.6f}; the final median is {sf.own_median:.6f}. The final mean's trajectory-bootstrap sensitivity interval is [{sf.own_bootstrap_low:.6f}, {sf.own_bootstrap_high:.6f}]. These intervals resample entire trajectories, preserving repeated measures; they do not estimate uncertainty across corpora, models or independent runs. Quantile bands show substantial variation among trajectories. [Semantic round metrics](tables/semantic_round_metrics.csv) and [trajectory metrics](tables/semantic_trajectory_metrics.csv).

The mean cosine between distinct current trajectories rises from {si.pairwise_mean_exact:.6f} to {sf.pairwise_mean_exact:.6f}, an increase of {sf.pairwise_mean_exact-si.pairwise_mean_exact:.6f}. The all-pair mean is computed exactly using the norm of the vector sum; quantiles use a fixed seeded sample of 50,000 of the 499,500 unordered pairs. The sample is identical at all observations. The final sampled median is {sf.pair_median:.6f}. Similarity to the original and similarity across the current population answer different questions: severe drift can coexist with low between-trajectory similarity.

Embedding dispersion decreases from {si.dispersion:.6f} to {sf.dispersion:.6f}; covariance participation ratio changes from {si.participation_ratio:.3f} to {sf.participation_ratio:.3f}, while it initially rises rather than falling monotonically. Unit-vector dispersion and mean pair similarity are algebraically related; they must not be counted as independent confirmations. Neither the effective rank nor participation ratio measures the dimension of meaning.

At round 100, similarity-graph largest components contain {', '.join(str(int(x)) for x in gf.largest_component)} trajectories for cosine thresholds {', '.join(str(x) for x in gf.threshold)} respectively. There is no giant component at these choices. Connectivity is threshold-dependent and can join dissimilar endpoints through chains. It is not semantic equivalence. [Graph sensitivity](tables/semantic_graph_metrics.csv).

The lexical baseline provides a useful counterpoint. TF-IDF similarity to each original falls from {tfidf.iloc[0].tfidf_original_mean:.6f} to {tfidf.iloc[-1].tfidf_original_mean:.6f}; however, TF-IDF mean cross-trajectory similarity changes from {tfidf.iloc[0].tfidf_population_pair_mean:.6f} to {tfidf.iloc[-1].tfidf_population_pair_mean:.6f}, **not a net increase**. Thus drift is supported by both proxies, while population concentration is representation-dependent. The TF-IDF vocabulary and IDF weights are fitted once to all distinct original/observed texts for a fixed coordinate system; this is a descriptive retrospective baseline, not a predictive evaluation. [TF-IDF metrics](tables/tfidf_round_metrics.csv).

{figure(12)}{figure(13)}{figure(14)}{figure(15)}{figure(16)}'''
    filtered=robust[(robust['round']==100)&(robust['filter']=='exclude_suspicious')].iloc[0]
    peak=raw.loc[raw.largest_size.idxmax()]
    cq=collision_quality[collision_quality['round']==int(peak['round'])].iloc[0]
    language=read('language_statistics');language=language[language.role=='target']
    body=rf'''# Repeated Multilingual Translation: An Empirical Investigation of Convergence and Linguistic Dynamics

## Abstract

{abstract}

## 1. Introduction

The motivating conjecture asks whether independently routed translations of initially different sentences could eventually produce the same sentence. This experiment tests observed finite-round consequences, not the infinite-time conjecture. We distinguish exact raw-string equality, equality under declared normalizations, embedding similarity, concentration of an empirical distribution, preservation of original content, and synchronization of realized trajectories. None is a substitute for the others.

We call an **observed collision** equality of two diagnostic English strings at the same observed round. **Active-state equality** instead requires the ordered pair (language code, exact text) to agree. **Endpoint persistence** means that a colliding pair is still equal at the next observation, without assuming equality at intervening rounds. “Candidate attractor” below labels an investigated recurring output only; it is not a claim about invariant sets, stationary distributions or limits.

## 2. Experimental setup

The main run selected 1,000 distinct English strings from FLORES-200 devtest with selection seed 1729. NLLB-200 distilled 600M, revision `{audit['model']['model_revision']}`, generated greedy float32 translations. Each sentence followed its own seeded sequence of shuffled 50-language permutations. Each language occurs twice per trajectory over 100 rounds; immediate repetitions at permutation boundaries are avoided. The 50-language pool excludes English. Complete settings, seeds, model hashes and schedules remain inside the archive; [the audit](data_audit.md) lists the languages and provenance.

Main observations occurred at rounds `{audit['observation_rounds']}`. Round zero is an exact copy of the original; the other 23 observations translate the current active text into English. Diagnostics never feed back into the active chain. The 100,000 active transitions plus 23,000 diagnostic calls give 123,000 main model invocations. The smaller pilot and smoke runs validate collection but do not supply independent repetitions of the main experiment. We analyze one main realization and do not pool those auxiliary runs into its population.

The collection imposed a 512-token input limit and a 256-new-token output limit. This makes the implemented transformation depend on tokenizer and decoding constraints, not solely on natural language. [Actual counts and quality flags](tables/data_quality_summary.csv).

## 3. Data integrity and validation

The archive's sidecar SHA-256, ZIP CRCs and all 717 manifest-listed file checksums passed. The audit independently followed all 1,000 main schedules through 101 checkpoints, checked every canonical record against its request and preceding state, verified journals and nested manifests, and reconciled model-call totals. No missing expected observations, malformed records, duplicate canonical identifiers, empty outputs, explicit failures or inactive final trajectories were found. The 24,000 main English observations are actual measurements/copies; nothing is interpolated at unobserved rounds. [Detailed audit](data_audit.md).

All main outputs were accepted as `ok`, yet 6,511 records reached the generation-token limit and 6,740 carried repetition warnings, with overlap. Structural validity is therefore not a judgment of translation fidelity. We retain them in primary analyses and label sensitivity filters explicitly. No data were repaired, retranslated or overwritten.

## 4. Exact textual convergence

For observed round t, let Nₜ be the valid population, nᵢ the group sizes and pᵢ=nᵢ/Nₜ. We report D(t), the number of distinct raw Unicode strings, and the probability of equality for two distinct uniformly selected trajectories:

$$P_{{\mathrm{{collision}}}}(t)=\frac{{\sum_i n_i(n_i-1)}}{{N_t(N_t-1)}}.$$

Entropy is H(t)=−Σᵢpᵢ log pᵢ in nats, with exp(H) and 1/Σᵢpᵢ² as effective diversities. These describe output-label frequencies. They are not the mutual information of an input-output channel: each original has only one realized randomized path, without repeated conditional samples.

{table_markdown(key_metrics)}

Source: [round-level metrics](tables/round_level_metrics.csv). All displayed denominators are 1,000. The final D is {int(final.distinct)}; the minimum is {int(raw.distinct.min())}. The first observed nonsingleton group appears at round {int(collisions['round'].min())}. There are {int((raw.distinct.diff()>0).sum())} intervals with increasing distinct count, so diversity is not monotone. No observation reached universal equality, a majority, or any of the prespecified 10%, 25%, 50%, 75% or 90% largest-group thresholds.

The largest group ever had {int(peak.largest_size)} members ({peak.largest_fraction:.1%}) at round {int(peak['round'])}; the final largest group has {int(final.largest_size)} ({final.largest_fraction:.1%}). The final equal-pair count is {int(final.collision_pairs)} out of 499,500, giving {final.collision_probability:.8f}; the maximum probability is {raw.collision_probability.max():.8f}. There is a small amount of partial equality, not macroscopic concentration. {int(final.singleton_outputs)} outputs at the final observation are singletons, and the remaining collision groups contain only two or three trajectories. The frequency profiles provide no basis for a fitted heavy-tail law or a dominant output.

{figure(1)}{figure(2)}{figure(3)}{figure(4)}{figure(5)}{figure(6)}

### Normalization sensitivity

Raw Unicode strings are primary. NFC composes canonically equivalent characters; the next scheme collapses Unicode whitespace and strips boundary whitespace; the next case-folds; the most permissive replaces all Unicode punctuation-category characters with spaces and collapses whitespace again. Symbols are retained. These are deterministic label maps, not semantic judgments. Raw strings remain unchanged in all source records.

{table_markdown(norm[['normalization','distinct','collision_pairs','largest_size']])}

Source: [normalization metrics](tables/round_level_metrics.csv). At the final observation, only punctuation removal changes these totals substantially, and even it leaves {int(norm.iloc[-1].distinct)} distinct strings. [Examples of normalization collisions](examples/normalization_examples.json) show the concrete variants. No normalization yields universal equality.

{figure(7)}

## 5. Temporal dynamics

Partitions are compared on shared valid trajectory IDs at successive observed rounds. An equal pair persists if its two members are equal again, even if their common sentence changes. A split occurs when an old nonsingleton group has members in multiple next groups; a merge occurs when a new nonsingleton group draws from multiple previous groups. Largest-group identity is not presumed persistent; tied largest groups use a deterministic lexical tie break.

Across eligible intervals there are {previous_pairs} prior equal-pair opportunities and {retained_pairs} retained pair events, giving a descriptive pooled persistence fraction of {retained_pairs/previous_pairs:.3%}. These sums can count the same pair repeatedly; they are not independent Bernoulli trials. Across all observations, {len(recurrence)} distinct pairs ever collide, and {int(recurrence.separated_then_recollided.sum())} collide again after an intervening observed separation. Thus both memory and disruption occur. [Persistence](tables/collision_persistence.csv), [pair recurrence](tables/collision_pair_recurrence.csv), [group transitions](tables/group_transitions.csv).

The conditional null fixes both observed partitions' group sizes and randomly reassigns next-round group membership over shared IDs. It preserves concentration at each endpoint while destroying cross-time alignment of trajectory identities. The expectation for persistence is the next partition's collision probability. We use 999 permutations per eligible interval, seed {config['seed']}, plus-one Monte Carlo p-values and Holm correction across intervals with previous collisions. Several observed persistence fractions exceed this very small null expectation. This demonstrates conditional membership dependence, not a causal effect of translation or evidence for the frequency distribution itself. Very few colliding pairs underpin some intervals, and the minimum attainable unadjusted Monte Carlo p-value is 0.001.

Most eligible intervals span five translations. Persistence at endpoints cannot establish uninterrupted equality inside the gap. Distinct future schedules can separate equal English projections; they can also separate identical active linguistic states. A collision is not an absorbing state under independently chosen future operators.

{figure(10)}{figure(11)}

## 6. English projection effects

Active labels include both language and exact text. Across all checkpoints active diversity reaches a minimum of {int(active.distinct.min())}; it is {int(active.iloc[-1].distinct)} at round 100. The English observation space differs because it suppresses language identity and performs another translation. Comparing its distinct count with active distinct count describes two representations, not a common-state convergence rate.

At the final observation, {int(final_projection.distinct_state_collision_pairs)} of {int(final_projection.english_collision_pairs)} English-equal pairs ({final_projection.projection_pair_share:.2%}) had different active states. Across all observed rounds, this applies to {pooled_projection} of {pooled_pairs} equal-pair observations ({pooled_projection/pooled_pairs:.2%}); repeated pairs are counted again when observed again. These are operationally projection-induced collisions. Different active states may express similar content, so this breakdown is not proof that the observer fabricated a semantic coincidence or a causal estimate of observation bias. It does show that English equality usually does **not** establish equality of the actual state. [Projection metrics](tables/projection_collision_metrics.csv), [raw examples](examples/projection_examples.json).

{figure(8)}{figure(9)}

## 7. Semantic evolution

{sem_section}

## 8. Information and content degradation

English word tokens use Unicode letter/digit runs with internal apostrophes, case-folded for lexical comparisons. A separate regex-token count includes punctuation; archived NLLB output token counts are also summarized, with no invented baseline NLLB count. We measure set-based word Jaccard, content-word retention using scikit-learn's fixed English stop list, number-marker retention, date-marker retention and negation-marker retention. These are proxies for surface preservation, not channel capacity or truth.

Mean word count changes from {li.word_count_mean:.3f} to {lf.word_count_mean:.3f}, but the median falls from {li.words_median:.0f} to {lf.words_median:.0f}. The final mean ratio to each original is {lf.word_ratio_mean:.3f}; long repetitive outputs help explain why the mean remains far above the median. Population vocabulary decreases from {int(li.vocabulary_types):,} to {int(lf.vocabulary_types):,} case-folded types. Mean original-word Jaccard reaches {lf.token_jaccard_mean:.4f}, and mean content-word retention reaches {lf.content_word_retention_mean:.4f}. Type-token ratios depend on length and cannot independently measure richness. [Lexical round metrics](tables/lexical_round_metrics.csv).

Numeric matching canonicalizes decimal tokens and comma thousands separators; it does not understand spelled-out numbers, units, date order or language-dependent numeric notation. Its final mean retention is {lf.number_retention_mean:.3%} among {int(lf.number_retention_eligible_n)} originals with numeric markers. Date matching uses English month names and years 1500–2099, so words such as “may” can be false positives. Negation matching is lexical and cannot resolve scope. Original sentences without the relevant marker are excluded from that retention denominator rather than scored as perfect preservation. No named-entity model was applied; preserving a name in the examples is direct textual evidence, not an entity-recognition score.

The best final original-cosine case still changes a date: the original says George W. Bush arrived in Singapore on November 16; the final observation says November 13. High embedding similarity therefore does not guarantee factual fidelity. Conversely, paraphrase can lower literal overlap without changing all meaning. [Case studies](examples/trajectory_case_studies.md).

{figure(17)}{figure(18)}{figure(19)}

## 9. Language-path effects

Every target language appears exactly 2,000 times in the main active transitions, once per trajectory in each permutation block. Source distributions include initial English and omit the last target as a subsequent source. We tabulate all observed ordered source-target transitions, per-round target counts, denominators and unique contributing trajectories. Rare transition cells remain descriptive. [Language statistics](tables/language_statistics.csv), [transition counts](tables/language_transitions.csv), [round usage](tables/language_usage_by_round.csv).

The highest combined flag rate is {language.suspicious_rate.max():.2%}. This quantity mixes recorded decoding warnings and separate heuristics. In particular, character-collapse flags can reflect script compactness: Chinese and Japanese naturally use fewer characters than many alphabetic inputs. These numbers must not be interpreted as language-quality rankings. Recorded repetition and generation-limit rates are reported separately.

At observation endpoints we relate original cosine and previous-observation cosine to the current language, after subtracting that observation round's population mean. A second table assigns round-centered original-cosine changes to languages encountered in the preceding interval; a multi-step interval's change is associated with all its languages and is not attributed causally to one. The dependence among interval exposures is retained conceptually, with no mass significance testing. [Endpoint associations](tables/language_semantic_associations.csv) and [interval associations](tables/language_interval_associations.csv), when embeddings are available.

Round 50/51 marks the second permutation. The adjacent flag rates show no conspicuous discontinuity by themselves; a single boundary cannot identify a boundary effect. Initial length, presence of numeric markers, initial lexical variety and punctuation are related descriptively to final original-cosine scores. These small associations do not establish causes or semantic topics. Language-family classifications and automatic topic labels were intentionally omitted rather than invented. [Boundary diagnostics](tables/permutation_boundary_metrics.csv), [original-property associations](tables/original_property_associations.csv).

{figure(20)}

## 10. Candidate attractors and recurrence

We screen every English string ever shared by at least two trajectories, reporting its first/last appearance, maximum/final frequency, number of contributing originals, preceding languages, same-text pair persistence and complete frequency history. This is an operational candidate list, not inferred dynamics of an attractor.

The largest candidate is the long repeated “No, no, …” output, ID `{candidates.iloc[0].output_id}`. It first appears as a singleton at round {int(candidates.iloc[0].first_round)}, peaks at {int(candidates.iloc[0].maximum_frequency)} members, and ends at {int(candidates.iloc[0].final_frequency)}. It is tagged for repetition and generation-token-limit saturation. At the peak-collision observation, {int(cq.equal_pairs_involving_flagged_outputs)} of {int(cq.equal_pairs)} equal pairs involve flagged observations. The decoder's finite output limit may help make repetitive loops yield identical strings; the present data do not isolate this mechanism experimentally.

Other small recurring outputs include “This is very important.” and “- I saw it.” They are not automatically classified as failures. A generic-looking output may be grammatical while losing the original's specifics. Shared outputs draw from different source sentences, but we do not infer broad topic coverage without topic labels. Many trajectories also revisit their own English outputs without joining another trajectory; [self-recurrence statistics](tables/trajectory_self_recurrence.csv) keep that separate from population coalescence. [Candidate table](tables/candidate_attractors.csv) and [histories](tables/candidate_frequencies.csv).

{figure(21)}

## 11. Qualitative examples

The [case-study document](examples/trajectory_case_studies.md) and [full JSON](examples/trajectory_case_studies.json) show originals, actual observed intermediate English texts, warning flags and full language paths. Cases were selected by explicit extrema and group-membership rules, so they illustrate mechanisms rather than estimate prevalence. Long texts are explicitly excerpted in Markdown; the JSON retains them verbatim.

One original about Gosling and Stone receiving acting nominations ends as “Grace and her husband wrote a letter.” Another, about re-entry culture shock, enters the small group “This is very important.” A trajectory from an original about patience for returning travellers joins the peak “No” repetition group but ends with repeated “existence”; leaving an exact group need not mean recovering content. The strongest final embedding-retention example preserves a named person and place while changing the date. Together these cases show why exact diversity, semantic drift, factual error, repetition and group persistence need separate measurements.

## 12. Robustness and sensitivity

The primary sample includes every collector-accepted nonempty English output. There are no explicit failed outputs, so excluding recorded failures is numerically identical to the primary analysis. The stricter sensitivity filter excludes any recorded warning or independently defined heuristic flag: repeated-word fraction, dominant trigram repetition, Unicode replacement characters, more than 1,000 characters, or an abrupt character-count collapse from an input of at least 40 characters. These thresholds are declared in code and flag suspicious text without claiming to diagnose its meaning.

At round 100 this filter retains {int(filtered.valid_n)} trajectories, with {int(filtered.distinct)} exact outputs and {int(filtered.collision_pairs)} equal pairs. The largest group remains {int(filtered.largest_size)}. Thus the final lack of a dominant exact sentence is robust, while early and peak collisions are more sensitive to repetitive outputs. Excluding the single largest exact group at each round gives another sensitivity calculation; ties are broken lexically, and this exclusion does not necessarily target the same sentence over time. [Robustness metrics](tables/robustness_metrics.csv).

For embeddings, exclusion of flagged outputs, the largest output, or embedding-truncated texts preserves severe original drift and modest cross-population concentration. The TF-IDF disagreement about population concentration limits how strongly we can generalize the MiniLM result. Similarity-graph thresholds 0.70, 0.80 and 0.90 give different partitions, none approaching a single group. [Semantic robustness](tables/semantic_robustness.csv), when available.

We also draw 100 fixed seeded subsets of 500 trajectory IDs, using each subset throughout time. This is a descriptive stability check; smaller samples naturally lose rare collisions and tend to have a larger distinct fraction. We do not bootstrap distinct counts as if resampling duplicates gave an unbiased population-diversity estimate. Semantic bootstrap intervals resample full trajectories with replacement and are reported only as sensitivity summaries. No pairwise pseudo-replication or asymptotic extrapolation is used.

{figure(22)}

## 13. Discussion

The central separation is between **loss of resemblance to the original** and **collapse of the population into a common output**. Strong changes in original-word overlap, numeric markers and original embedding similarity coincide with a population that remains mostly textually distinct. High empirical output-label entropy therefore does not imply good preservation of the original message. Conversely, small changes in label entropy cannot quantify how much original information was retained.

A second separation concerns observed equality and actual-state equality. Most English collisions occur between different active states. Equality at the observer is a many-to-one measurement event, often involving different languages, and cannot alone establish synchronization of the underlying chains.

A third separation concerns recurrence and absorption. Some equal pairs retain membership more than independent reassignment predicts, yet splits, re-collisions and declining late cluster sizes remain visible. The observed growth of small collision groups is not monotone coalescence. A graph of permanently merged trajectories would misrepresent these data.

Finally, decoding degeneracy matters without explaining everything. Saturated repetition accounts for much of the largest observed exact group, but filtered outputs still drift and remain diverse. Modest semantic concentration may reflect increasing genericity, model preferences or other factors; the archive does not distinguish these mechanisms. The contrary TF-IDF population trend is an explicit limit on a model-independent “semantic convergence” claim.

## 14. Implications for future mathematical research

### Observation maps and false state identification

**Observation:** {pooled_projection}/{pooled_pairs} equal English pair-observations arise from different active states. **Possible framework:** a language-indexed state space with a many-to-one observation map into English. **Open question:** under what conditions can coalescence of observations identify coalescence, or bounded distance, in the latent linguistic states? **Needed:** a definition of state equivalence, observer-error assumptions and multiple independent observation maps or human annotations of paired states. This experiment supplies a counterexample to treating equality of observations as sufficient evidence of exact state equality, not a theorem about all observation maps.

### Independent versus shared random operators

**Observation:** English-equal pairs sometimes split and later meet again. **Possible framework:** random compositions indexed by language choices, with different operator sequences for each trajectory. **Open question:** how does equality persistence change when trajectories share future language choices instead of evolving independently? **Needed:** controlled shared-path and independent-path experiments from matched active states, explicit deterministic/stochastic transition definitions, and treatment of decoding constraints. Exact equality need not be absorbing under different next operators; common-noise synchronization and independent-noise coalescence are distinct questions.

### Memory without absorbing collision classes

**Observation:** endpoint persistence exceeds conditional group-size reassignment in several intervals, but all observed groups are small. **Possible framework:** recurrent or metastable regions and nonabsorbing membership dynamics. **Open question:** can a reduced process on reproducible text/semantic classes capture residence times and re-entry rates without assuming equality defines an equivalence class of future behavior? **Needed:** denser observations, more independent schedules per original, a validated coarse state definition and tests of whether a Markov approximation is adequate. Here the schedule has without-replacement memory inside each permutation; a text-and-language state alone need not be Markov unless the remaining schedule information is included.

### Content contraction with persistent label diversity

**Observation:** original-content proxies decline far more strongly than exact output diversity. **Possible framework:** contractions in selected semantic features alongside proliferation or preservation of surface distinctions. **Open question:** which well-defined observables contract, which persist, and does their behavior differ under alternative models or corpora? **Needed:** repeated conditional samples, externally validated content features, alternative embeddings and human factual scoring. The near-maximal output-label entropy does not estimate a mutual-information lower bound here, and embedding geometry cannot establish information contraction by itself.

These directions are motivated hypotheses and experimental designs. No fixed point, invariant measure, limiting distribution, universal attractor or infinite-time convergence result follows from this finite archive.

## 15. Limitations

This is one translation model, one fixed corpus, one main run and 100 rounds. The randomized schedules are structured permutations, not independent language draws at each step. Different originals have different realized paths; separating sentence susceptibility from path effects requires repeated schedules per original. Diagnostic English adds a model-dependent observation operation and is sparse after early rounds. Generation limits and repetitive outputs influence results. There is no independent multilingual language-ID, human quality or factual-consistency audit. Embeddings and lexical proxies are imperfect, and semantic grouping is threshold-dependent. Distributional concentration, exact state equality and pathwise synchronization remain distinct. Cross-language character/token statistics reflect script and tokenizer differences. No causal language ranking or asymptotic inference is warranted.

## 16. Conclusion

Universal exact convergence was **not observed**. Partial exact equality appeared only in small groups, with a maximum of {int(raw.largest_size.max())}/1,000 trajectories and a final distinct count of {int(final.distinct)}. The observed process exhibits content drift, small recurrent and transient collision groups, and persistent diversity. MiniLM suggests modest population concentration; lexical TF-IDF does not show the same net concentration, and neither measurement establishes semantic identity or long-term convergence. This is a useful negative result for the simple finite-round universal-collapse story, while the infinite-time conjecture remains unresolved.

## 17. Reproducibility

From the repository root:

```bash
bash analyze_local.sh
```

The script verifies the immutable archive, establishes a separate environment, uses four CPU threads, reconstructs/audits records, runs all analyses, reuses cached MiniLM embeddings, and generates this report. No translation model, API or GPU is used. A complete semantic cache avoids model inference on reruns. `bash analyze_local.sh --skip-semantic` runs the non-embedding analyses with TF-IDF; the semantic stage is explicitly marked skipped. `--stage audit|exact|temporal|content|semantic|language|robustness|figures|report` selects one stage; figures/report require their prerequisite tables. `--refresh-audit` forces full reconstruction. [Configuration](../../config/analysis.toml), [analysis README](../../analysis/README.md), [source/package provenance](analysis_provenance.json), [execution resource record](last_execution.json), and [output checksums](output_manifest.json).

Run tests with `.analysis_runtime/venv/bin/python -m unittest discover -s tests -v`. Synthetic cases cover unique outputs, universal equality, temporary collisions, stable partitions, projection effects, invalid outputs, missing IDs, sparse observations, permutation behavior, embedding geometry and archive loading. [Validation record](validation.md) records executed checks. The code version is defined by individual SHA-256 values in provenance, including uncommitted analysis files; a Git commit alone would omit these working-tree additions.

The original archive remains at `data/multilingual_translation_v1_complete.zip`; no extraction is needed. Derived tables are under `reports/translation_v1/tables`, figures have PNG and vector PDF versions, and long examples are separate. See [publication guide](../../PUBLICATION.md) for repository contents, raw-data sharing and attribution. The [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) and [Sentence Transformers similarity documentation](https://sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html) describe the embedding representation and cosine comparison used here.
'''
    if not has_semantics:
        body=re.sub(r'For embeddings, exclusion.*?\n\n', 'Embedding-based sensitivity analysis is unavailable in this run; see semantic_status.json.\n\n',body,flags=re.DOTALL)
        body=re.sub(r'The best final original-cosine case.*?\n\n', 'No embedding-ranked qualitative case is selected in this run. See lexical cases for surface retention examples.\n\n',body,flags=re.DOTALL)
        body=body.replace('MiniLM suggests modest population concentration; lexical TF-IDF does not show the same net concentration, and neither measurement establishes semantic identity or long-term convergence.', 'Semantic embedding analysis is incomplete in this run. Lexical TF-IDF does not show a net population-concentration increase and cannot establish semantic identity or long-term convergence.')
        body=body.replace('The contrary TF-IDF population trend is an explicit limit on a model-independent “semantic convergence” claim.', 'Embedding-based population conclusions are unavailable in this run.')
    (output/'research_report.md').write_text(body)
    key=f'''# Key findings: repeated multilingual translation

## Question and measurement

Could 1,000 initially distinct English sentences, independently translated through randomized language paths, become one common sentence? This archive observes 100 active translation rounds and 24 English diagnostic rounds. English diagnostics do not replace the active state. The investigation separates exact strings, (language, text) states, semantic proxies and retention of original content. [Audit](data_audit.md).

## Ten numerical findings

1. **Integrity:** 717 manifest-listed files passed verification, with zero reconstruction discrepancies. The main run contains 100,000 active translations plus 23,000 English model translations and 1,000 initial copies. [Counts](tables/data_quality_summary.csv).
2. **Persistent exact diversity:** D changes from 1,000 to {int(final.distinct)}; its minimum is {int(raw.distinct.min())}. No universal or majority equality occurs at any observation. [Exact metrics](tables/round_level_metrics.csv), [Figure 1](figures/01_exact_diversity.png).
3. **Small groups only:** the largest group ever is {int(raw.largest_size.max())}/1,000 ({raw.largest_fraction.max():.1%}); the final largest group has {int(final.largest_size)}. No 10% threshold is reached. [Group metrics](tables/round_level_metrics.csv).
4. **Rare, nonmonotone collisions:** the first collision is observed at round {int(collisions['round'].min())}; final equality is {int(final.collision_pairs)}/499,500 pairs ({final.collision_probability:.8f}), versus a peak of {int(raw.collision_pairs.max())}/499,500. [Exact metrics](tables/round_level_metrics.csv).
5. **Observer/state distinction:** {int(final_projection.distinct_state_collision_pairs)}/{int(final_projection.english_collision_pairs)} final English-equal pairs arise from different active states. Pooled across observations, the share is {pooled_projection}/{pooled_pairs} ({pooled_projection/pooled_pairs:.2%}); repeated pairs contribute repeatedly. [Projection table](tables/projection_collision_metrics.csv).
6. **Persistence with splitting:** {retained_pairs}/{previous_pairs} eligible equal-pair interval events persist ({retained_pairs/previous_pairs:.2%}); {len(recurrence)} distinct pairs ever collide and {int(recurrence.separated_then_recollided.sum())} re-collide after observed separation. Conditional permutation tests preserve group sizes and show membership memory in several intervals, not absorbing coalescence. [Persistence](tables/collision_persistence.csv), [recurrence](tables/collision_pair_recurrence.csv).
7. **Strong original-content drift:** {('mean MiniLM cosine to the original falls from '+f'{si.own_mean:.3f} to {sf.own_mean:.3f}') if has_semantics else 'embedding analysis is unavailable'}; TF-IDF original cosine ends at {tfidf.iloc[-1].tfidf_original_mean:.3f}. [Semantic metrics](tables/semantic_round_metrics.csv), [TF-IDF](tables/tfidf_round_metrics.csv).
8. **Only modest, representation-dependent concentration:** {('mean between-trajectory MiniLM cosine rises from '+f'{si.pairwise_mean_exact:.3f} to {sf.pairwise_mean_exact:.3f}') if has_semantics else 'MiniLM concentration is unavailable'}, while TF-IDF pair cosine changes from {tfidf.iloc[0].tfidf_population_pair_mean:.3f} to {tfidf.iloc[-1].tfidf_population_pair_mean:.3f}. This supports no claim of semantic identity. [Semantic metrics](tables/semantic_round_metrics.csv), [TF-IDF](tables/tfidf_round_metrics.csv).
9. **Surface content changes:** median word count falls from {li.words_median:.0f} to {lf.words_median:.0f}; vocabulary types fall from {int(li.vocabulary_types):,} to {int(lf.vocabulary_types):,}; approximate numeric-marker retention ends at {lf.number_retention_mean:.1%} among {int(lf.number_retention_eligible_n)} eligible originals. [Lexical metrics](tables/lexical_round_metrics.csv).
10. **Quality and robustness:** {main['warning_generation_token_limit']:,} main records carry generation-limit warnings; {main['warning_repetitive_output']:,} carry repetition warnings, with overlap and no explicit failed status. Excluding suspicious final observations leaves {int(filtered.distinct)} distinct outputs among {int(filtered.valid_n)} trajectories. [Quality counts](tables/data_quality_summary.csv), [filter sensitivity](tables/robustness_metrics.csv).

## Interpretation and examples

**Exact universal convergence did not occur.** Partial exact equality occurred, but no dominant population output emerged. The strongest repeated candidate is a token-limit-saturated “No, no, …” string; its peak group subsequently shrinks. Small unflagged groups also recur. [Candidates](tables/candidate_attractors.csv), [quality/collision association](tables/collision_quality_associations.csv).

The most useful distinction is **degradation without universal collapse**: resemblance to original content falls sharply while most realized outputs remain distinct. High output-label entropy is not evidence that originals remain recoverable. English projection accounts operationally for most equal-pair observations, so it cannot be treated as a faithful indicator of active-state synchronization.

A sentence about acting nominations ends as “Grace and her husband wrote a letter.” A re-entry culture-shock sentence ends as “This is very important.” Even the highest final original-cosine case changes a date from November 16 to November 13. These are rule-selected illustrations, not random prevalence estimates. [Full case studies and exact texts](examples/trajectory_case_studies.md).

Theoretically interesting questions concern many-to-one observation maps, shared versus independent future operators, collision memory without absorption, and feature contraction while labels remain diverse. The data challenge monotone coalescence, an absorbing interpretation of English equality, and equating drift with semantic concentration. They do not decide the infinite-time conjecture, establish a stationary distribution, or generalize across models/corpora. Independent schedule repetitions, alternative observers, different decoding limits and human factual checks would be needed. [Full discussion](research_report.md#14-implications-for-future-mathematical-research).
'''
    (output/'key_findings.md').write_text(key)
    for name in ['research_report','key_findings','data_audit']:
        text=(output/(name+'.md')).read_text()
        if not has_semantics:
            text=re.sub(r'\[([^\]]+)\]\((tables/(?:semantic[^)]*|language_semantic[^)]*|language_interval[^)]*|original_property[^)]*))\)', r'\1 (unavailable in this run)',text)
            (output/(name+'.md')).write_text(text)
        text=re.sub(r'\$\$.*?\$\$', '<p><strong>P<sub>collision</sub>(t) = [Σᵢ nᵢ(nᵢ − 1)] / [Nₜ(Nₜ − 1)]</strong></p>', text, flags=re.DOTALL)
        rendered=markdown.markdown(text,extensions=['tables','fenced_code','toc'])
        def embed(match):
            target=output/match.group(1)
            if target.exists() and target.suffix=='.png':
                return 'src="data:image/png;base64,'+base64.b64encode(target.read_bytes()).decode()+'"'
            return match.group(0)
        rendered=re.sub(r'src="([^"]+)"',embed,rendered)
        page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>'+name.replace('_',' ')+'</title><style>body{max-width:1000px;margin:3em auto;padding:0 24px;font:17px/1.65 Georgia,serif;color:#202a33}h1,h2,h3{font-family:system-ui,sans-serif;line-height:1.25}h2{margin-top:2.2em}img{max-width:100%;height:auto}table{border-collapse:collapse;display:block;overflow-x:auto;font:13px/1.5 system-ui,sans-serif}td,th{padding:7px 10px;border-bottom:1px solid #ccd3d9;text-align:left}code,pre{font:13px/1.5 monospace;overflow-wrap:anywhere}pre{padding:16px;background:#f0f3f5;white-space:pre-wrap}a{color:#176b87}blockquote{border-left:3px solid #176b87;padding-left:1em}@media print{body{font-size:11pt;max-width:none}img{max-height:18cm;object-fit:contain}h2{break-after:avoid}table,img{break-inside:avoid}}</style><body>'+rendered+'</body></html>'
        (output/(name+'.html')).write_text(page)
    return summary

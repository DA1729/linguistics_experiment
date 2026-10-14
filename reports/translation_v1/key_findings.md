# Key findings: repeated multilingual translation

## Question and measurement

Could 1,000 initially distinct English sentences, independently translated through randomized language paths, become one common sentence? This archive observes 100 active translation rounds and 24 English diagnostic rounds. English diagnostics do not replace the active state. The investigation separates exact strings, (language, text) states, semantic proxies and retention of original content. [Audit](data_audit.md).

## Ten numerical findings

1. **Integrity:** 717 manifest-listed files passed verification, with zero reconstruction discrepancies. The main run contains 100,000 active translations plus 23,000 English model translations and 1,000 initial copies. [Counts](tables/data_quality_summary.csv).
2. **Persistent exact diversity:** D changes from 1,000 to 985; its minimum is 984. No universal or majority equality occurs at any observation. [Exact metrics](tables/round_level_metrics.csv), [Figure 1](figures/01_exact_diversity.png).
3. **Small groups only:** the largest group ever is 8/1,000 (0.8%); the final largest group has 3. No 10% threshold is reached. [Group metrics](tables/round_level_metrics.csv).
4. **Rare, nonmonotone collisions:** the first collision is observed at round 30; final equality is 18/499,500 pairs (0.00003604), versus a peak of 40/499,500. [Exact metrics](tables/round_level_metrics.csv).
5. **Observer/state distinction:** 17/18 final English-equal pairs arise from different active states. Pooled across observations, the share is 172/181 (95.03%); repeated pairs contribute repeatedly. [Projection table](tables/projection_collision_metrics.csv).
6. **Persistence with splitting:** 66/163 eligible equal-pair interval events persist (40.49%); 95 distinct pairs ever collide and 18 re-collide after observed separation. Conditional permutation tests preserve group sizes and show membership memory in several intervals, not absorbing coalescence. [Persistence](tables/collision_persistence.csv), [recurrence](tables/collision_pair_recurrence.csv).
7. **Strong original-content drift:** mean MiniLM cosine to the original falls from 1.000 to 0.127; TF-IDF original cosine ends at 0.047. [Semantic metrics](tables/semantic_round_metrics.csv), [TF-IDF](tables/tfidf_round_metrics.csv).
8. **Only modest, representation-dependent concentration:** mean between-trajectory MiniLM cosine rises from 0.035 to 0.102, while TF-IDF pair cosine changes from 0.023 to 0.023. This supports no claim of semantic identity. [Semantic metrics](tables/semantic_round_metrics.csv), [TF-IDF](tables/tfidf_round_metrics.csv).
9. **Surface content changes:** median word count falls from 21 to 5; vocabulary types fall from 5,649 to 1,254; approximate numeric-marker retention ends at 18.1% among 173 eligible originals. [Lexical metrics](tables/lexical_round_metrics.csv).
10. **Quality and robustness:** 6,511 main records carry generation-limit warnings; 6,740 carry repetition warnings, with overlap and no explicit failed status. Excluding suspicious final observations leaves 923 distinct outputs among 936 trajectories. [Quality counts](tables/data_quality_summary.csv), [filter sensitivity](tables/robustness_metrics.csv).

## Interpretation and examples

**Exact universal convergence did not occur.** Partial exact equality occurred, but no dominant population output emerged. The strongest repeated candidate is a token-limit-saturated “No, no, …” string; its peak group subsequently shrinks. Small unflagged groups also recur. [Candidates](tables/candidate_attractors.csv), [quality/collision association](tables/collision_quality_associations.csv).

The most useful distinction is **degradation without universal collapse**: resemblance to original content falls sharply while most realized outputs remain distinct. High output-label entropy is not evidence that originals remain recoverable. English projection accounts operationally for most equal-pair observations, so it cannot be treated as a faithful indicator of active-state synchronization.

A sentence about acting nominations ends as “Grace and her husband wrote a letter.” A re-entry culture-shock sentence ends as “This is very important.” Even the highest final original-cosine case changes a date from November 16 to November 13. These are rule-selected illustrations, not random prevalence estimates. [Full case studies and exact texts](examples/trajectory_case_studies.md).

Theoretically interesting questions concern many-to-one observation maps, shared versus independent future operators, collision memory without absorption, and feature contraction while labels remain diverse. The data challenge monotone coalescence, an absorbing interpretation of English equality, and equating drift with semantic concentration. They do not decide the infinite-time conjecture, establish a stationary distribution, or generalize across models/corpora. Independent schedule repetitions, alternative observers, different decoding limits and human factual checks would be needed. [Full discussion](research_report.md#14-implications-for-future-mathematical-research).

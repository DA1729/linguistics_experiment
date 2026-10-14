from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter as percent_formatter, MaxNLocator as max_n_locator
import numpy as np
import pandas as pd

from .common import save_json


colors=['#176B87','#D26035','#72953B','#75558C','#B79027']


def generate_figures(config):
    output=Path(config['output'])
    def read(name): return pd.read_csv(output/'tables'/f'{name}.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':13,'axes.labelsize':11,
                         'axes.spines.top':False,'axes.spines.right':False,'axes.prop_cycle':plt.cycler(color=colors),
                         'savefig.dpi':170,'pdf.fonttype':42,'svg.fonttype':'none','figure.constrained_layout.use':True})
    catalog=[]
    def axes(title,ylabel,width=8):
        fig,ax=plt.subplots(figsize=(width,4.4))
        ax.set(title=title,xlabel='Translation round',ylabel=ylabel)
        ax.grid(axis='y',alpha=.18)
        return fig,ax
    def save(fig,name,caption):
        fig.savefig(output/'figures'/f'{name}.png')
        fig.savefig(output/'figures'/f'{name}.pdf')
        plt.close(fig)
        catalog.append({'name':name,'caption':caption})
    def line(ax,frame,column,label=None,**kwargs):
        ax.plot(frame['round'],frame[column],marker='o',markersize=3,linewidth=1.6,label=label,**kwargs)
    metrics=read('round_level_metrics');raw=metrics[metrics.normalization=='raw']
    fig,ax=axes('English outputs remain mostly distinct','Distinct exact outputs')
    line(ax,raw,'distinct');ax.set_ylim(min(raw.distinct)-3,1002)
    save(fig,'01_exact_diversity','Exact raw-string diversity at the 24 observed rounds. The vertical axis is restricted to make small changes visible; 1,000 trajectories were observed each time.')
    fig,ax=axes('Exact collisions are rare','Equal pairs per million distinct pairs')
    ax.plot(raw['round'],raw.collision_probability*1e6,'o-',markersize=3)
    save(fig,'02_collision_probability','Unordered equal-output pairs divided by N(N−1)/2, scaled to one million pairs. Pair comparisons are not independent experimental replicates.')
    fig,ax=axes('Largest exact-output groups','Trajectories in group')
    for col,label in [('largest_size','Largest'),('second_size','Second'),('third_size','Third')]: line(ax,raw,col,label)
    ax.yaxis.set_major_locator(max_n_locator(integer=True));ax.legend()
    save(fig,'03_largest_groups','Largest three group sizes at each observation. Ranks can refer to different sentences and members at different rounds; ties use lexical string order.')
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    line(axs[0],raw,'entropy_nats');axs[0].set(title='Empirical label entropy',xlabel='Round',ylabel='H (nats)')
    for col,label in [('effective_shannon','exp(H)'),('effective_simpson','Inverse Simpson'),('distinct','Distinct')]: line(axs[1],raw,col,label)
    axs[1].set(title='Effective output diversity',xlabel='Round',ylabel='Effective number of outputs');axs[1].legend()
    save(fig,'04_entropy_diversity','Entropy of empirical English output labels and Hill diversity numbers. These are not estimates of retained information about the original sentence.')
    clusters=read('exact_output_clusters')
    fig,ax=axes('Rank-frequency profiles at selected observations','Output frequency')
    for t in [0,25,50,75,100]:
        g=clusters[clusters['round']==t];ax.plot(g['rank'],g['size'],label=f'Round {t}')
    ax.set(xlabel='Output rank (log scale)',xscale='log');ax.legend()
    save(fig,'05_rank_frequency','Empirical rank-frequency profiles, dominated by singleton outputs. A logarithmic rank axis exposes the few repeated outputs; no power-law model is fitted.')
    hist=read('cluster_size_histogram')
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    line(axs[0],raw,'singleton_outputs');axs[0].set(title='Singleton output groups',xlabel='Round',ylabel='Number of groups')
    for size in sorted(hist.cluster_size.unique()):
        if size==1: continue
        g=hist[hist.cluster_size==size].set_index('round').reindex(raw['round'],fill_value=0).reset_index()
        axs[1].plot(g['round'],g.number_of_clusters,'o-',markersize=3,label=f'Size {size}')
    axs[1].set(title='Collision-group size distribution',xlabel='Round',ylabel='Number of groups');axs[1].legend(fontsize=8,ncol=2)
    save(fig,'06_cluster_size_distribution','Singleton counts and counts of each nonsingleton group size, separated to avoid hiding the small collision groups.')
    fig,ax=axes('Normalization sensitivity','Distinct English outputs')
    for scheme,g in metrics.groupby('normalization',sort=False): line(ax,g,'distinct',scheme)
    ax.legend(ncol=2)
    save(fig,'07_normalization','Raw Unicode, NFC, collapsed whitespace, case folding, and punctuation-to-space comparisons. Normalization-induced equality is not semantic equivalence.')
    active=read('active_state_metrics')
    fig,ax=axes('Active states and diagnostic English','Distinct labels')
    line(ax,active,'distinct','Active (language, exact text)');line(ax,raw,'distinct','English exact text');ax.legend()
    save(fig,'08_active_vs_english','Active linguistic states use both language and raw text and are available at all 101 checkpoints. English observations use a different representation space at only the displayed observation rounds.')
    projection=read('projection_collision_metrics')
    fig,ax=axes('Where English equality arises','Equal English pairs')
    ax.stackplot(projection['round'],projection.same_state_collision_pairs,projection.distinct_state_collision_pairs,labels=['Identical active states','Distinct active states'],colors=colors[:2],alpha=.8)
    ax.scatter(projection['round'],projection.english_collision_pairs,s=15,color='#333333');ax.legend()
    save(fig,'09_projection_collisions','English-equal pairs split into identical and distinct pre-observation states. This is an operational partition, not a causal estimate of diagnostic translation effects.')
    persistence=read('collision_persistence')
    fig,ax=axes('Do equal pairs stay equal at the next observation?','Fraction of previous equal pairs')
    ax.plot(persistence.to_round,persistence.persistence,'o-',label='Observed',markersize=4)
    ax.plot(persistence.to_round,persistence.null_expected_persistence,'--',label='Independent reassignment expectation')
    ax.fill_between(persistence.to_round,persistence.null_q025,persistence.null_q975,alpha=.2,label='Permutation 95% interval')
    ax.yaxis.set_major_formatter(percent_formatter(1));ax.legend(fontsize=8)
    save(fig,'10_collision_persistence','Pair persistence across successive observed endpoints. Intervals are 1, 1, 1, 2, then 5 rounds; undefined values when no previous collisions exist are left blank. Shading is a conditional null interval, not a confidence interval.')
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for col,label in [('split_groups','Splitting groups'),('merge_groups','Merging groups')]: axs[0].plot(persistence.to_round,persistence[col],'o-',markersize=3,label=label)
    axs[0].set(title='Changes in output partitions',xlabel='Next observation round',ylabel='Groups');axs[0].legend()
    matrix=np.array([persistence.dominant_exits,persistence.dominant_overlap,persistence.dominant_entries])
    im=axs[1].imshow(matrix,aspect='auto',cmap='Blues');axs[1].set_yticks(range(3),['Exit','Overlap','Entry'])
    axs[1].set_xticks(range(0,len(persistence),3),persistence.to_round.iloc[::3]);axs[1].set(title='Largest-group membership turnover',xlabel='Next observation round')
    fig.colorbar(im,ax=axs[1],label='Trajectories')
    save(fig,'11_partition_dynamics','Merge/split counts and turnover between the largest groups at successive observations. Tied largest groups are selected lexicographically; the heatmap does not imply a single persistent dominant sentence.')
    semantic_path=output/'tables/semantic_round_metrics.csv'
    semantic_status=output/'semantic_status.json'
    import json
    complete=semantic_path.exists() and semantic_status.exists() and json.loads(semantic_status.read_text())['status']=='complete'
    if complete:
        sem=read('semantic_round_metrics')
        fig,ax=axes('Similarity to each original sentence','MiniLM cosine similarity')
        line(ax,sem,'own_mean','Mean');line(ax,sem,'own_median','Median')
        ax.fill_between(sem['round'],sem.own_q10,sem.own_q90,alpha=.16,label='Trajectory 10th–90th percentiles')
        ax.fill_between(sem['round'],sem.own_bootstrap_low,sem.own_bootstrap_high,alpha=.5,label='Mean bootstrap sensitivity interval')
        ax.set_ylim(-.05,1.03);ax.legend(fontsize=8)
        save(fig,'12_semantic_drift','Normalized MiniLM cosine to the corresponding original. Whole trajectories are resampled with the same 500 draws across rounds; bootstrap intervals describe corpus sensitivity, not uncertainty across models or independent experiments.')
        fig,ax=axes('Population pairwise semantic similarity','MiniLM cosine similarity')
        line(ax,sem,'pairwise_mean_exact','All-pair mean');line(ax,sem,'pair_median','Fixed-pair sample median')
        ax.fill_between(sem['round'],sem.pair_q10,sem.pair_q90,alpha=.2,label='Fixed-pair 10th–90th percentiles');ax.legend(fontsize=8)
        save(fig,'13_semantic_pair_distribution','Pair-similarity summaries use the same 50,000 sampled distinct trajectory pairs at every observation. The mean is also calculated exactly over all pairs; pair quantiles are descriptive and not independent replicates.')
        fig,axs=plt.subplots(1,2,figsize=(10,4))
        line(axs[0],sem,'centroid_norm');axs[0].set(title='Centroid concentration',xlabel='Round',ylabel='Norm of mean unit embedding')
        line(axs[1],sem,'dispersion');axs[1].set(title='Embedding dispersion',xlabel='Round',ylabel='Mean squared distance to centroid')
        save(fig,'14_semantic_geometry','For unit vectors, dispersion equals one minus squared centroid norm. These are related geometric descriptions, not independent evidence of semantic convergence.')
        fig,axs=plt.subplots(1,2,figsize=(10,4))
        for col,label in [('participation_ratio','Participation ratio'),('effective_rank','Entropy effective rank')]: line(axs[0],sem,col,label)
        axs[0].set(title='Effective covariance dimensionality',xlabel='Round',ylabel='Effective dimensions');axs[0].legend(fontsize=8)
        for col,label in [('pc1_variance_share','First PC'),('pc10_variance_share','First 10 PCs')]: line(axs[1],sem,col,label)
        axs[1].set(title='PCA variance concentration',xlabel='Round',ylabel='Variance fraction');axs[1].legend()
        save(fig,'15_embedding_dimension','Eigenvalues of centered embedding covariance determine participation ratio, effective rank and PCA variance shares. These are finite-sample properties of this embedding model, not dimensions of language.')
        fig,axs=plt.subplots(1,2,figsize=(10,4))
        axs[0].plot(raw.collision_probability*1e6,sem.pairwise_mean_exact,'o-',markersize=4)
        axs[0].set(xlabel='Exact equal pairs per million',ylabel='Mean pair cosine',title='Exact and semantic concentration')
        graph=read('semantic_graph_metrics')
        for threshold,g in graph.groupby('threshold'): line(axs[1],g,'largest_component_fraction',f'Cosine ≥ {threshold}')
        axs[1].set(title='Threshold-graph sensitivity',xlabel='Round',ylabel='Largest component fraction');axs[1].legend(fontsize=8)
        save(fig,'16_semantic_grouping','Exact collision rates compared with semantic similarity, and largest similarity-graph components at three thresholds. Graph connectivity can chain dissimilar sentences and is not an equivalence relation on meaning.')
    lex=read('lexical_round_metrics')
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    line(axs[0],lex,'word_count_mean','Mean');line(axs[0],lex,'words_median','Median')
    axs[0].fill_between(lex['round'],lex.words_q10,lex.words_q90,alpha=.2);axs[0].set(title='English word counts',xlabel='Round',ylabel='Regex word count');axs[0].legend()
    line(axs[1],lex,'word_ratio_mean','Mean');line(axs[1],lex,'word_ratio_median','Median');axs[1].set(title='Length relative to each original',xlabel='Round',ylabel='Word-count ratio');axs[1].legend()
    save(fig,'17_sentence_lengths','Word counts and length ratios; shaded region gives trajectory 10th–90th percentiles. Repeated outputs can increase lengths, so shortening is not presumed.')
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    line(axs[0],lex,'vocabulary_types');axs[0].set(title='Population vocabulary',xlabel='Round',ylabel='Distinct case-folded word types')
    for col,label in [('token_jaccard_mean','Original-word Jaccard'),('number_retention_mean','Number retention'),('content_word_retention_mean','Content-word retention'),('date_marker_retention_mean','Date-marker retention')]: line(axs[1],lex,col,label)
    axs[1].set(title='Lexical content proxies',xlabel='Round',ylabel='Mean proportion');axs[1].legend(fontsize=7)
    save(fig,'18_lexical_retention','Vocabulary and approximate content retention. Number/date averages include only originals containing those markers; dates use English month names and years, and content words exclude a fixed stop list.')
    fail=read('translation_failure_metrics');g=fail[fail.kind=='observations']
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for col,label in [('generation_token_limit','Token limit'),('repetitive_output','Recorded repetition'),('suspicious','Any recorded/heuristic flag')]: line(axs[0],g,col,label)
    axs[0].set(title='English observations with warnings',xlabel='Round',ylabel='Records');axs[0].legend(fontsize=8)
    g=fail[fail.kind=='translations']
    for col,label in [('generation_token_limit','Token limit'),('repetitive_output','Recorded repetition')]: line(axs[1],g,col,label)
    axs[1].set(title='Active translation warnings',xlabel='Round',ylabel='Records');axs[1].legend(fontsize=8)
    save(fig,'19_quality_flags','Recorded warnings and separately defined heuristic flags. Warning sets overlap. The collector accepted these outputs with status ok; they are not retrospectively relabeled as explicit failures.')
    lang=read('language_statistics');g=lang[lang.role=='target'].sort_values('suspicious_rate').tail(15)
    fig,ax=plt.subplots(figsize=(8,5.5));ax.barh(g.language,g.suspicious_rate,color=colors[0]);ax.xaxis.set_major_formatter(percent_formatter(1));ax.set(title='Target languages associated with flagged outputs',xlabel='Flagged fraction of translations (2,000 per target)')
    save(fig,'20_language_associations','The 15 target languages with highest descriptive flag rates. Input histories and scripts differ, and this is not a causal ranking of language quality.')
    candidates=read('candidate_attractors');history=read('candidate_frequencies')
    fig,ax=axes('Recurrent exact outputs: small candidate groups','Trajectories with this exact output')
    for row in candidates.head(5).itertuples():
        g=history[history.output_id==row.output_id];line(ax,g,'frequency',row.output_id[:8])
    ax.legend(title='Output ID prefix',fontsize=8)
    save(fig,'21_candidate_outputs','Frequency histories for the five outputs with highest peak membership, tied by total contributing trajectories. Exact texts and membership evidence are in candidate_attractors.csv; none is asserted to be an attractor.')
    robust=read('robustness_metrics')
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for label,g in robust.groupby('filter'):
        if label in ['all_recorded_successful','exclude_suspicious','exclude_largest_output']:
            axs[0].plot(g['round'],g.collision_probability*1e6,'o-',markersize=3,label=label.replace('_',' '))
    axs[0].set(title='Exact concentration sensitivity',xlabel='Round',ylabel='Equal pairs per million');axs[0].legend(fontsize=7)
    if complete:
        sr=read('semantic_robustness')
        line(axs[1],sem,'pairwise_mean_exact','All successful')
        for label,g in sr.groupby('filter'): line(axs[1],g,'pairwise_mean_exact',label.replace('_',' '))
        axs[1].set(title='Semantic concentration sensitivity',xlabel='Round',ylabel='Mean pair cosine');axs[1].legend(fontsize=7)
    else:
        base=read('tfidf_round_metrics');line(axs[1],base,'tfidf_original_mean');axs[1].set(title='TF-IDF baseline',xlabel='Round',ylabel='Original cosine')
    save(fig,'22_filtering_robustness','Primary metrics alongside flagged-output and largest-group exclusions. Filtering changes the population and is a sensitivity analysis, not a correction to the observed experiment.')
    save_json(output/'figure_catalog.json',catalog)
    return catalog

from collections import Counter
from decimal import Decimal, InvalidOperation
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as english_stop_words

from .common import save_table, valid_rows
from .exact import distribution_metrics


word_pattern = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
number_pattern = re.compile(r'(?<!\w)[+-]?\d+(?:,\d{3})*(?:\.\d+)?(?!\w)')
negations = {'no','not','never','neither','nor','nobody','nothing','nowhere','without','cannot'}


def words(text):
    return word_pattern.findall(text.casefold())


def numbers(text):
    found=set()
    for token in number_pattern.findall(text):
        try:
            found.add(str(Decimal(token.replace(',','')).normalize()))
        except InvalidOperation:
            pass
    return found


def date_markers(text):
    return set(re.findall(r'\b(?:1[5-9]\d{2}|20\d{2}|january|february|march|april|may|june|july|august|september|october|november|december)\b',text.casefold()))


def retention(before, after):
    return len(before&after)/len(before) if before else np.nan


def lexical_measures(original, output):
    before,after=words(original),words(output)
    a,b=set(before),set(after)
    content_a=a-set(english_stop_words)
    return {'characters':len(output),'word_count':len(after),'regex_token_count':len(re.findall(r'\w+|[^\w\s]',output)),
            'character_ratio':len(output)/len(original) if original else np.nan,
            'word_ratio':len(after)/len(before) if before else np.nan,
            'within_sentence_ttr':len(b)/len(after) if after else np.nan,
            'repeated_word_fraction':1-len(b)/len(after) if after else np.nan,
            'token_jaccard':len(a&b)/len(a|b) if a|b else np.nan,
            'content_word_retention':retention(content_a,b),
            'number_retention':retention(numbers(original),numbers(output)),
            'date_marker_retention':retention(date_markers(original),date_markers(output)),
            'negation_retention':retention({w for w in a if w in negations or w.endswith("n't")},{w for w in b if w in negations or w.endswith("n't")})}


def heuristic_flags(text, original='', english=True):
    if not isinstance(text,str):
        return ['missing_output']
    flags=[]
    w=words(text) if english else text.split()
    if not text.strip(): flags.append('empty_output')
    if '\ufffd' in text: flags.append('replacement_character')
    if len(w)>=12 and len(set(w))/len(w)<=.25: flags.append('repeated_words')
    grams=[tuple(w[i:i+3]) for i in range(max(0,len(w)-2))]
    if len(grams)>=10 and max(Counter(grams).values(),default=0)>=len(grams)*.4: flags.append('repeated_phrase')
    if len(text)>1000: flags.append('over_1000_characters')
    if original and len(original)>=40 and len(text)/len(original)<.2: flags.append('character_collapse')
    return flags


def annotate(data):
    for name in ['observations','translations']:
        frame=data[name]
        frame['heuristic_flags']=['|'.join(heuristic_flags(text,before,name=='observations')) for text,before in zip(frame.output_text,frame.input_text)]
        frame['suspicious']=(frame.warnings!='') | (frame.heuristic_flags!='')
    return data


def content_analysis(data,config):
    output=config['output']
    obs=valid_rows(data['observations'])
    originals=dict(zip(data['corpus'].sentence_id,data['corpus'].text))
    individual=[]
    for r in obs.itertuples():
        individual.append({'round':r.round,'sentence_id':r.sentence_id,**lexical_measures(originals[r.sentence_id],r.output_text)})
    detail=pd.DataFrame(individual)
    save_table(output,'lexical_trajectory_metrics',detail)
    summaries=[]
    for t,g in obs.groupby('round'):
        d=detail[detail['round']==t]
        tokens=[w for text in g.output_text for w in words(text)]
        summaries.append({'round':t,'valid_n':len(g),'vocabulary_types':len(set(tokens)),'word_tokens':len(tokens),
                          'corpus_ttr':len(set(tokens))/len(tokens) if tokens else np.nan,
                          **{col+'_mean':d[col].mean() for col in d.columns if col not in ['round','sentence_id']},
                          **{col+'_eligible_n':int(d[col].notna().sum()) for col in ['number_retention','date_marker_retention','negation_retention']},
                          'words_q10':d.word_count.quantile(.1),'words_median':d.word_count.median(),'words_q90':d.word_count.quantile(.9),
                          'word_ratio_median':d.word_ratio.median(),'recorded_nllb_output_tokens_mean':g.output_tokens.mean()})
    save_table(output,'lexical_round_metrics',summaries)
    failures=[]
    for name in ['observations','translations']:
        for t,g in data[name].groupby('round'):
            row={'round':t,'kind':name,'records':len(g),'ok':int((g.status=='ok').sum()),'failed':int((g.status=='failed').sum()),'skipped':int((g.status=='skipped').sum()),
                 'suspicious':int(g.suspicious.sum()),'empty':sum(isinstance(x,str) and not x.strip() for x in g.output_text)}
            for label in ['input_truncated','generation_token_limit','repetitive_output','generation_time_limit']:
                row[label]=int(g.warnings.str.split('|').map(lambda x:label in x).sum())
            for label in ['repeated_words','repeated_phrase','over_1000_characters','character_collapse']:
                row['heuristic_'+label]=int(g.heuristic_flags.str.split('|').map(lambda x:label in x).sum())
            failures.append(row)
    save_table(output,'translation_failure_metrics',failures)
    return detail


def language_analysis(data,config,semantic=None,lexical=None):
    output=config['output']
    trans=data['translations'].copy()
    trans['input_characters']=trans.input_text.map(lambda x:len(x) if isinstance(x,str) else 0)
    trans['output_characters']=trans.output_text.map(lambda x:len(x) if isinstance(x,str) else 0)
    trans['character_ratio']=trans.output_characters/trans.input_characters.replace(0,np.nan)
    trans['failed']=(trans.status!='ok').astype(int)
    trans['generation_token_limit']=trans.warnings.str.contains('generation_token_limit').astype(int)
    trans['repetition']=trans.warnings.str.contains('repetitive_output').astype(int)
    rows=[]
    for role in ['source','target']:
        for lang,g in trans.groupby(role+'_language'):
            rows.append({'role':role,'language':lang,'translations':len(g),'trajectories':g.sentence_id.nunique(),
                         'input_characters_mean':g.input_characters.mean(),'output_characters_mean':g.output_characters.mean(),'character_ratio_median':g.character_ratio.median(),
                         'failure_rate':g.failed.mean(),'suspicious_rate':g.suspicious.mean(),'generation_token_limit_rate':g.generation_token_limit.mean(),'recorded_repetition_rate':g.repetition.mean()})
    save_table(output,'language_statistics',rows)
    transitions=trans.groupby(['source_language','target_language']).agg(translations=('sentence_id','size'),trajectories=('sentence_id','nunique'),
                 character_ratio_median=('character_ratio','median'),failure_rate=('failed','mean'),suspicious_rate=('suspicious','mean')).reset_index()
    save_table(output,'language_transitions',transitions)
    save_table(output,'language_usage_by_round',trans.groupby(['round','target_language']).size().reset_index(name='translations'))
    save_table(output,'permutation_boundary_metrics',trans.groupby('round').agg(translations=('sentence_id','size'),suspicious_rate=('suspicious','mean'),
                 generation_token_limit_rate=('generation_token_limit','mean'),character_ratio_median=('character_ratio','median')).reset_index())
    if semantic is None:
        return
    sem=semantic[semantic['round']>0].copy()
    sem['round_centered_original_cosine']=sem.original_cosine-sem.groupby('round').original_cosine.transform('mean')
    sem['round_centered_previous_cosine']=sem.previous_observation_cosine-sem.groupby('round').previous_observation_cosine.transform('mean')
    save_table(output,'language_semantic_associations',sem.groupby('source_language').agg(observations=('sentence_id','size'),trajectories=('sentence_id','nunique'),
        original_cosine_mean=('original_cosine','mean'),round_centered_original_cosine=('round_centered_original_cosine','mean'),
        previous_observation_cosine=('previous_observation_cosine','mean'),round_centered_previous_cosine=('round_centered_previous_cosine','mean')).reset_index())
    schedule_map={r['sentence_id']:r['languages'] for r in data['schedules']}
    exposures=[]
    rounds=sorted(semantic['round'].unique())
    for t,u in zip(rounds[:-1],rounds[1:]):
        left=semantic[semantic['round']==t].set_index('sentence_id').original_cosine
        right=semantic[semantic['round']==u].set_index('sentence_id').original_cosine
        difference=(right-left).dropna()
        centered=difference-difference.mean()
        for sid,value in centered.items():
            for language in schedule_map[sid][t:u]:
                exposures.append({'from_round':t,'to_round':u,'language':language,'sentence_id':sid,'centered_cosine_change':value})
    exposure=pd.DataFrame(exposures)
    save_table(output,'language_interval_associations',exposure.groupby('language').agg(exposures=('sentence_id','size'),trajectories=('sentence_id','nunique'),
        mean_interval_centered_cosine_change=('centered_cosine_change','mean')).reset_index())
    final=semantic[semantic['round']==max(rounds)].set_index('sentence_id')
    properties=[]
    original=data['corpus'].set_index('sentence_id').text
    for sid,text in original.items():
        if sid in final.index:
            w=words(text)
            properties.append({'sentence_id':sid,'initial_words':len(w),'has_number':int(bool(numbers(text))),
                'initial_ttr':len(set(w))/len(w) if w else np.nan,'punctuation_count':len(re.findall(r'[^\w\s]',text)),
                'final_original_cosine':final.loc[sid,'original_cosine']})
    prop=pd.DataFrame(properties)
    save_table(output,'original_sentence_properties',prop)
    associations=[]
    for col in ['initial_words','has_number','initial_ttr','punctuation_count']:
        associations.append({'property':col,'trajectories':len(prop),'spearman_with_final_original_cosine':spearmanr(prop[col],prop.final_original_cosine).statistic})
    save_table(output,'original_property_associations',associations)


def robustness_analysis(data,config):
    output=config['output']
    obs=data['observations']
    ids=sorted(data['corpus'].sentence_id)
    rng=np.random.default_rng(config['seed'])
    samples=[set(rng.choice(ids,min(len(ids),config['subsample_size']),replace=False)) for _ in range(config['subsample_repetitions'])]
    rows=[]
    for t,all_rows in obs.groupby('round'):
        valid=valid_rows(all_rows)
        largest=sorted(valid.output_text.value_counts().items(),key=lambda x:(-x[1],x[0]))[0][0] if len(valid) else None
        filters={'all_recorded_successful':valid,'exclude_recorded_failures':valid,
                 'exclude_suspicious':valid[~valid.suspicious], 'exclude_largest_output':valid[valid.output_text!=largest]}
        for label,g in filters.items():
            rows.append({'round':t,'filter':label,**distribution_metrics(g.output_text)})
        sampled=[distribution_metrics(valid[valid.sentence_id.isin(sample)].output_text) for sample in samples]
        for metric in ['distinct_fraction','collision_probability','largest_fraction']:
            vals=[r[metric] for r in sampled]
            rows.append({'round':t,'filter':'trajectory_subsample_'+metric,'valid_n':min(len(ids),config['subsample_size']),
                         'subsample_q025':np.nanquantile(vals,.025),'subsample_median':np.nanmedian(vals),'subsample_q975':np.nanquantile(vals,.975)})
    return save_table(output,'robustness_metrics',rows)

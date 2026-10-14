from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import save_json, save_table, text_id, valid_rows
from .content import numbers


def additional_diagnostics(data,config):
    output=Path(config['output'])
    obs=valid_rows(data['observations'])
    originals=dict(zip(data['corpus'].sentence_id,data['corpus'].text))
    temporal=[]
    for sid,g in obs.groupby('sentence_id'):
        values=g.sort_values('round').output_text.tolist()
        temporal.append({'sentence_id':sid,'observed_rounds':len(values),'unique_english_outputs':len(set(values)),
                         'equal_adjacent_observed_endpoints':sum(a==b for a,b in zip(values,values[1:])),
                         'most_frequent_self_output_count':max(Counter(values).values())})
    save_table(output,'trajectory_self_recurrence',temporal)
    collision_flags=[]
    for t,g in obs.groupby('round'):
        total=flagged=0
        for text,members in g.groupby('output_text'):
            n=len(members);clean=int((~members.suspicious).sum())
            total+=n*(n-1)//2
            flagged+=n*(n-1)//2-clean*(clean-1)//2
        collision_flags.append({'round':t,'equal_pairs':total,'equal_pairs_involving_flagged_outputs':flagged,
                                'flagged_pair_share':flagged/total if total else np.nan})
    save_table(output,'collision_quality_associations',collision_flags)
    candidates=pd.read_csv(output/'tables/candidate_attractors.csv')
    semantic_path=output/'tables/semantic_trajectory_metrics.csv'
    semantic_status=output/'semantic_status.json'
    semantic_available=semantic_path.exists() and semantic_status.exists() and json.loads(semantic_status.read_text())['status']=='complete'
    cases=[]
    if semantic_available:
        semantic=pd.read_csv(semantic_path)
        final=semantic[semantic['round']==obs['round'].max()]
        cases.extend([('highest_final_original_cosine',final.sort_values(['original_cosine','sentence_id'],ascending=[False,True]).iloc[0].sentence_id),
                      ('lowest_final_original_cosine',final.sort_values(['original_cosine','sentence_id']).iloc[0].sentence_id)])
        early=semantic[semantic['round']==5]
        cases.append(('rapid_drift_by_round_5',early.sort_values(['original_cosine','sentence_id']).iloc[0].sentence_id))
    top=candidates.iloc[0]
    members=obs[obs.output_text==top.text]
    peak_round=members.groupby('round').size().idxmax()
    peak=members[members['round']==peak_round]
    final_map=obs[obs['round']==obs['round'].max()].set_index('sentence_id').output_text
    departing=[sid for sid in peak.sentence_id if final_map.get(sid)!=top.text]
    if departing: cases.append(('leaves_peak_repetition_group',sorted(departing)[0]))
    ordinary=candidates[candidates.recorded_warnings.isna()].head(1)
    if len(ordinary): cases.append(('joins_small_unflagged_cluster',obs[obs.output_text==ordinary.iloc[0].text].sort_values(['round','sentence_id']).iloc[0].sentence_id))
    numbered=[sid for sid,text in originals.items() if numbers(text)]
    if semantic_available:
        numbered_final=final[final.sentence_id.isin(numbered)]
        cases.append(('number_bearing_original',numbered_final.sort_values('original_cosine',ascending=False).iloc[0].sentence_id))
    cases.append(('representative_first_corpus_sentence',data['corpus'].iloc[0].sentence_id))
    seen=set();examples=[]
    for label,sid in cases:
        if sid in seen: continue
        seen.add(sid)
        g=obs[obs.sentence_id==sid].sort_values('round')
        examples.append({'selection':label,'sentence_id':sid,'original':originals[sid],
                         'full_language_schedule':next(x['languages'] for x in data['schedules'] if x['sentence_id']==sid),
                         'observations':[{'round':int(r.round),'language':r.source_language,'active_text':data['states'][r.round][sid][1],
                                          'english':r.output_text,'recorded_warnings':r.warnings,'heuristic_flags':r.heuristic_flags} for r in g.itertuples()]})
    save_json(output/'examples/trajectory_case_studies.json',examples)
    def excerpt(text):
        return text if len(text)<=300 else text[:300]+f' … [display excerpt; {len(text)} characters in full record]'
    lines=['# Qualitative trajectory case studies','', 'Cases were selected by the explicit rules below, not as a representative random sample. English texts are diagnostic projections. Full unabridged observations and active-language texts are in [the companion JSON](trajectory_case_studies.json). No new translation was performed.','']
    for case in examples:
        lines.extend([f"## {case['selection'].replace('_',' ')}",'',f"Trajectory `{case['sentence_id']}`.",'', '**Original:** '+case['original'],'', '| Round | Active language | Diagnostic English | Recorded warnings |','|---:|---|---|---|'])
        for r in case['observations']:
            if r['round'] not in [0,1,3,5,10,25,50,75,90,100]: continue
            text=excerpt(r['english']).replace('|','&#124;').replace('\n','<br>')
            lines.append(f"| {r['round']} | {r['language']} | {text} | {r['recorded_warnings'] or 'none'} |")
        lines.append('')
    (output/'examples/trajectory_case_studies.md').write_text('\n'.join(lines)+'\n')
    return examples

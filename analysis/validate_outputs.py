import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .common import file_hash, save_json


def validate_outputs(config):
    output=Path(config['output'])
    checks=[]
    def check(condition,label):
        if not condition: raise ValueError('Output consistency failure: '+label)
        checks.append(label)
    def read(name): return pd.read_csv(output/'tables'/f'{name}.csv')
    audit=json.loads((output/'data_quality_summary.json').read_text())
    summary=json.loads((output/'analysis_summary.json').read_text())
    check(file_hash(config['archive'])==audit['archive_sha256'],'Original archive hash unchanged')
    check(not audit['discrepancies'],'No reconstruction discrepancies')
    raw=read('round_level_metrics').query("normalization=='raw'")
    clusters=read('exact_output_clusters')
    for t,g in clusters.groupby('round'):
        row=raw[raw['round']==t].iloc[0]
        check(int(g['size'].sum())==int(row.valid_n),f'Round {t}: group memberships sum to valid population')
        check(len(g)==int(row.distinct),f'Round {t}: group count matches distinct count')
        pairs=int((g['size']*(g['size']-1)).sum()//2)
        check(pairs==int(row.collision_pairs),f'Round {t}: cluster sizes reproduce collision pairs')
        check(np.isclose(row.collision_probability,pairs/(row.valid_n*(row.valid_n-1)/2)),f'Round {t}: collision denominator')
    projection=read('projection_collision_metrics')
    check(np.array_equal(projection.english_collision_pairs,raw.collision_pairs),'Projection counts match raw equality counts')
    check(np.array_equal(projection.same_state_collision_pairs+projection.distinct_state_collision_pairs,projection.english_collision_pairs),'Projection partition is exhaustive')
    temporal=read('collision_persistence')
    check((temporal.persistent_pairs<=temporal.previous_equal_pairs).all(),'Persistent pairs bounded by previous equal pairs')
    check((temporal.persistent_pairs<=temporal.next_equal_pairs).all(),'Persistent pairs bounded by following equal pairs')
    check((temporal.to_round-temporal.from_round==temporal.round_gap).all(),'Observed interval labels correct')
    check(raw['round'].tolist()==audit['observation_rounds'],'No invented English observation rounds')
    check(summary['exact']['final_distinct']==int(raw.iloc[-1].distinct),'Summary matches final exact table')
    status=json.loads((output/'semantic_status.json').read_text())
    if status['status']=='complete':
        semantic=read('semantic_round_metrics');detail=read('semantic_trajectory_metrics')
        check(np.allclose(detail.groupby('round').original_cosine.mean(),semantic.own_mean,atol=1e-7),'Semantic trajectory means match round means')
        check(np.allclose(semantic.dispersion,1-semantic.centroid_norm**2,atol=2e-6),'Unit-embedding dispersion identity')
        n=semantic.valid_n
        check(np.allclose(semantic.pairwise_mean_exact,(n*semantic.centroid_norm**2-1)/(n-1),atol=2e-6),'All-pair cosine matches vector-sum identity')
        check(((detail.original_cosine>=-1.000001)&(detail.original_cosine<=1.000001)).all(),'Cosine range valid')
        check(len(detail)==len(raw)*1000,'Semantic records match all valid observations in this archive')
        graph=read('semantic_graph_metrics')
        check((graph.largest_component<=graph.valid_n).all(),'Graph components bounded by population')
    catalog=json.loads((output/'figure_catalog.json').read_text())
    for row in catalog:
        for suffix in ['png','pdf']:
            check((output/'figures'/f"{row['name']}.{suffix}").stat().st_size>1000,f"Figure exists: {row['name']}.{suffix}")
    for name in ['research_report.md','key_findings.md','data_audit.md','examples/trajectory_case_studies.md']:
        path=output/name;text=path.read_text()
        check(not any(ord(c)<32 and c not in '\n\t\r' for c in text),name+': no malformed control characters')
        for link in re.findall(r'\]\(([^)]+)\)',text):
            if link.startswith(('https://','http://','#')): continue
            target=link.split('#')[0]
            check((path.parent/target).exists(),name+': link '+target)
    check(summary['figures']==len(catalog),'Summary figure count matches catalog')
    check(summary['summary_tables']==len(list((output/'tables').glob('*.csv'))),'Summary table count matches generated tables')
    save_json(output/'consistency_checks.json',{'checks_passed':len(checks),'checks':checks})
    return checks

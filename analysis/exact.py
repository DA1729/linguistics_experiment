from collections import Counter, defaultdict
import itertools
import json
import unicodedata

import numpy as np
import pandas as pd

from .common import pair_count, save_json, save_table, text_id, valid_rows


schemes = ['raw', 'nfc', 'whitespace', 'casefold', 'punctuation']


def normalize(text, scheme):
    if scheme == 'raw':
        return text
    value = unicodedata.normalize('NFC', text)
    if scheme == 'nfc':
        return value
    value = ' '.join(value.split())
    if scheme == 'whitespace':
        return value
    value = value.casefold()
    if scheme == 'casefold':
        return value
    if scheme != 'punctuation':
        raise ValueError(scheme)
    return ' '.join(''.join(' ' if unicodedata.category(c).startswith('P') else c for c in value).split())


def distribution_metrics(labels):
    counts = sorted(Counter(labels).values(), reverse=True)
    n = sum(counts)
    p = np.array(counts, dtype=float) / n if n else np.array([])
    entropy = float(-np.sum(p * np.log(p))) if n else np.nan
    sizes = counts + [0, 0, 0]
    return {'valid_n': n, 'distinct': len(counts), 'distinct_fraction': len(counts)/n if n else np.nan,
            'collision_pairs': pair_count(counts), 'collision_probability': pair_count(counts)/(n*(n-1)/2) if n > 1 else np.nan,
            'largest_size': sizes[0], 'largest_fraction': sizes[0]/n if n else np.nan,
            'second_size': sizes[1], 'third_size': sizes[2], 'entropy_nats': entropy,
            'effective_shannon': float(np.exp(entropy)), 'effective_simpson': float(1/np.sum(p*p)) if n else np.nan,
            'singleton_outputs': sum(x == 1 for x in counts),
            **{f'clusters_ge_{k}': sum(x >= k for x in counts) for k in [2,5,10,50,100]}}


def projection_metrics(texts, states):
    groups = defaultdict(list)
    for sid, text in texts.items():
        groups[text].append(sid)
    total, same, involved, groups_n = 0, 0, 0, 0
    for members in groups.values():
        total += len(members)*(len(members)-1)//2
        active = Counter(states[sid][:2] for sid in members)
        same += pair_count(active.values())
        if len(members) > 1 and len(active) > 1:
            groups_n += 1
            involved += len(members)
    return {'english_collision_pairs': total, 'same_state_collision_pairs': same,
            'distinct_state_collision_pairs': total-same, 'projection_pair_share': (total-same)/total if total else np.nan,
            'projection_groups': groups_n, 'trajectories_in_projection_groups': involved}


def exact_analysis(data, config):
    output = config['output']
    obs = valid_rows(data['observations'])
    metrics, clusters, histogram, projection, norm_examples = [], [], [], [], []
    seen = set()
    projection_examples = []
    for t, group in obs.groupby('round', sort=True):
        for scheme in schemes:
            metrics.append({'round': t, 'normalization': scheme, **distribution_metrics(normalize(x, scheme) for x in group.output_text)})
        counts = group.output_text.value_counts()
        for rank, (text, size) in enumerate(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])), 1):
            members = group.loc[group.output_text == text, 'sentence_id'].tolist()
            clusters.append({'round': t, 'output_id': text_id(text), 'text': text, 'size': size, 'fraction': size/len(group), 'rank': rank,
                             'distinct_originals': len(members), 'seen_earlier': text in seen,
                             'sentence_ids': json.dumps(members) if size > 1 else ''})
            if size > 1:
                distinct = {data['states'][t][sid][:2] for sid in members}
                if len(distinct) > 1 and len(projection_examples) < 12:
                    projection_examples.append({'round': int(t), 'english': text, 'members': [{'sentence_id': sid, 'language': data['states'][t][sid][0], 'active_text': data['states'][t][sid][1]} for sid in members]})
        seen.update(counts.index)
        for size, count in Counter(counts.values).items():
            histogram.append({'round': t, 'cluster_size': int(size), 'number_of_clusters': count})
        projection.append({'round': t, 'valid_n': len(group), **projection_metrics(dict(zip(group.sentence_id, group.output_text)), data['states'][t])})
        for scheme in schemes[1:]:
            normalized = defaultdict(set)
            for text in counts.index:
                normalized[normalize(text, scheme)].add(text)
            for label, originals in normalized.items():
                if len(originals) > 1 and sum(x['scheme'] == scheme for x in norm_examples) < 6:
                    norm_examples.append({'round': int(t), 'scheme': scheme, 'normalized': label, 'raw_variants': sorted(originals)})
    frames = {}
    frames['round_level_metrics'] = save_table(output, 'round_level_metrics', metrics)
    frames['exact_output_clusters'] = save_table(output, 'exact_output_clusters', clusters)
    frames['cluster_size_histogram'] = save_table(output, 'cluster_size_histogram', histogram)
    frames['projection_collision_metrics'] = save_table(output, 'projection_collision_metrics', projection)
    frames['active_state_metrics'] = save_table(output, 'active_state_metrics', [{'round': t, **distribution_metrics(state[:2] for state in states.values() if state[2])} for t, states in data['states'].items()])
    save_json(f'{output}/examples/normalization_examples.json', norm_examples)
    save_json(f'{output}/examples/projection_examples.json', projection_examples)
    return frames


def transition_metrics(left, right):
    common = sorted(set(left) & set(right))
    a, b = [left[x] for x in common], [right[x] for x in common]
    ca, cb, joint = Counter(a), Counter(b), Counter(zip(a,b))
    previous, following, retained = pair_count(ca.values()), pair_count(cb.values()), pair_count(joint.values())
    total = len(common)*(len(common)-1)//2
    splits = sum(len({y for x,y in joint if x == label}) > 1 for label in ca if ca[label] >= 2)
    merges = sum(len({x for x,y in joint if y == label}) > 1 for label in cb if cb[label] >= 2)
    return {'common_n': len(common), 'previous_equal_pairs': previous, 'next_equal_pairs': following,
            'persistent_pairs': retained, 'new_equal_pairs': following-retained,
            'persistence': retained/previous if previous else np.nan,
            'new_collision_rate': (following-retained)/(total-previous) if total > previous else np.nan,
            'split_groups': splits, 'merge_groups': merges,
            'null_expected_persistence': following/total if total else np.nan}


def permutation_persistence(a, b, repeats, rng):
    a, b = np.asarray(a), np.asarray(b)
    denominator = pair_count(Counter(a).values())
    if denominator == 0:
        return np.full(repeats, np.nan)
    results = np.empty(repeats)
    width = int(b.max()) + 1
    for i in range(repeats):
        joint = a * width + rng.permutation(b)
        _, counts = np.unique(joint, return_counts=True)
        results[i] = pair_count(counts)/denominator
    return results


def temporal_analysis(data, config):
    output = config['output']
    obs = valid_rows(data['observations'])
    rounds = {t: dict(zip(g.sentence_id,g.output_text)) for t,g in obs.groupby('round')}
    rng = np.random.default_rng(config['seed'])
    rows, flows, pair_visits = [], [], defaultdict(list)
    candidates = defaultdict(lambda: defaultdict(set))
    for t, texts in rounds.items():
        groups = defaultdict(list)
        for sid, text in texts.items():
            groups[text].append(sid)
            candidates[text][t].add(sid)
        for text, members in groups.items():
            for pair in itertools.combinations(sorted(members), 2):
                pair_visits[pair].append(t)
    for t, u in zip(list(rounds)[:-1], list(rounds)[1:]):
        left, right = rounds[t], rounds[u]
        row = {'from_round': t, 'to_round': u, 'round_gap': u-t, **transition_metrics(left,right)}
        common = sorted(set(left) & set(right))
        a = pd.factorize([left[x] for x in common])[0]
        b = pd.factorize([right[x] for x in common])[0]
        null = permutation_persistence(a,b,config['permutations'],rng)
        if row['previous_equal_pairs']:
            row.update(null_q025=float(np.quantile(null,.025)), null_q975=float(np.quantile(null,.975)),
                       permutation_p=(1+sum(null >= row['persistence']))/(len(null)+1))
        else:
            row.update(null_q025=np.nan, null_q975=np.nan, permutation_p=np.nan)
        ca, cb = Counter(left.values()), Counter(right.values())
        top_left = sorted(ca, key=lambda x:(-ca[x],x))[0]
        top_right = sorted(cb, key=lambda x:(-cb[x],x))[0]
        old = {sid for sid,x in left.items() if x==top_left}
        new = {sid for sid,x in right.items() if x==top_right}
        row.update(dominant_same_text=top_left==top_right, dominant_overlap=len(old&new), dominant_exits=len(old-new), dominant_entries=len(new-old),
                   dominant_jaccard=len(old&new)/len(old|new), top_tie_count=sum(v==ca[top_left] for v in ca.values()))
        for (x,y), count in Counter((left[sid],right[sid]) for sid in common).items():
            if ca[x]>=2 or cb[y]>=2:
                flows.append({'from_round': t,'to_round':u,'from_output_id':text_id(x),'to_output_id':text_id(y),'trajectories':count,'from_size':ca[x],'to_size':cb[y]})
        rows.append(row)
    frame = pd.DataFrame(rows)
    p = frame.permutation_p.dropna()
    if len(p):
        order = p.sort_values().index
        adjusted = np.maximum.accumulate(p.loc[order].to_numpy() * np.arange(len(p),0,-1)).clip(0,1)
        frame.loc[order,'permutation_p_holm'] = adjusted
    save_table(output,'collision_persistence',frame)
    save_table(output,'group_transitions',flows,columns=['from_round','to_round','from_output_id','to_output_id','trajectories','from_size','to_size'])
    positions = {t:i for i,t in enumerate(rounds)}
    recurrence = []
    for (a,b), visits in pair_visits.items():
        recurrence.append({'sentence_id_a':a,'sentence_id_b':b,'equal_observations':len(visits),'first_round':min(visits),'last_round':max(visits),
                           'separated_then_recollided': any(positions[y]-positions[x]>1 for x,y in zip(visits,visits[1:])), 'rounds':json.dumps(visits)})
    save_table(output,'collision_pair_recurrence',recurrence,columns=['sentence_id_a','sentence_id_b','equal_observations','first_round','last_round','separated_then_recollided','rounds'])
    candidate_rows, histories = [], []
    final = max(rounds)
    for text, visits in candidates.items():
        maximum = max(map(len,visits.values()))
        if maximum < 2:
            continue
        same_pairs, old_pairs = 0,0
        for t,u in zip(list(rounds)[:-1],list(rounds)[1:]):
            a,b=visits.get(t,set()),visits.get(u,set())
            old_pairs+=len(a)*(len(a)-1)//2
            same_pairs+=len(a&b)*(len(a&b)-1)//2
        ids=set().union(*visits.values())
        relevant=obs[(obs.output_text==text)]
        candidate_rows.append({'output_id':text_id(text),'text':text,'first_round':min(visits),'last_round':max(visits),'observations_present':len(visits),
                               'maximum_frequency':maximum,'final_frequency':len(visits.get(final,set())), 'contributing_trajectories':len(ids),
                               'same_text_pair_persistence':same_pairs/old_pairs if old_pairs else np.nan,
                               'source_languages':json.dumps(relevant.source_language.value_counts().to_dict()),
                               'recorded_warnings': '|'.join(sorted(set(relevant.warnings)-{''}))})
        histories.extend({'round':t,'output_id':text_id(text),'frequency':len(visits.get(t,set()))} for t in rounds)
    save_table(output,'candidate_attractors', sorted(candidate_rows,key=lambda r:(-r['maximum_frequency'],-r['contributing_trajectories'],r['output_id'])))
    save_table(output,'candidate_frequencies',histories)
    return frame

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf_vectorizer

from .common import file_hash, save_json, save_table, valid_rows


def embedding_cache(texts, config):
    import torch
    from sentence_transformers import SentenceTransformer as sentence_transformer
    from huggingface_hub import snapshot_download
    torch.set_num_threads(config['cpu_threads'])
    torch.manual_seed(config['seed'])
    torch.use_deterministic_algorithms(True)
    settings = config['embedding']
    key = hashlib.sha256(json.dumps({'texts':texts,'settings':settings,'st':importlib.metadata.version('sentence-transformers'),'transformers':importlib.metadata.version('transformers'),'torch':torch.__version__},sort_keys=True).encode()).hexdigest()
    cache = Path(config['cache']) / ('embeddings_' + key)
    cache.mkdir(parents=True,exist_ok=True)
    meta_path = cache / 'metadata.json'
    if (cache/'complete.npy').exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if file_hash(cache/'complete.npy') == meta['embedding_sha256']:
            return np.load(cache/'complete.npy'),meta
    model_path = snapshot_download(settings['model'],revision=settings['revision'],allow_patterns=['*.json','*.txt','*.safetensors','1_Pooling/*'], max_workers=2)
    model = sentence_transformer(model_path,device='cpu',local_files_only=True)
    model.max_seq_length=settings['max_seq_length']
    arrays=[]
    for start in range(0,len(texts),512):
        target=cache/f'chunk_{start:06d}.npy'
        if target.exists():
            block=np.load(target)
            if block.shape != (min(512,len(texts)-start),model.get_sentence_embedding_dimension()) or not np.isfinite(block).all():
                raise ValueError('Invalid embedding cache chunk')
        else:
            block=model.encode(texts[start:start+512],batch_size=settings['batch_size'],normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False,device='cpu')
            with target.with_suffix('.tmp').open('wb') as stream:
                np.save(stream,block)
            target.with_suffix('.tmp').replace(target)
        arrays.append(block)
        print(f'CPU embeddings: {min(start+512,len(texts))}/{len(texts)} unique texts',flush=True)
    embeddings=np.concatenate(arrays)
    np.save(cache/'complete.npy',embeddings)
    lengths=[]
    for start in range(0,len(texts),512):
        encoded=model.tokenizer(texts[start:start+512],truncation=False,add_special_tokens=True)
        lengths.extend(map(len,encoded['input_ids']))
    meta={'model':settings['model'],'revision':settings['revision'],'device':'cpu','normalized':True,'metric':'cosine',
          'batch_size':settings['batch_size'],'max_seq_length':model.max_seq_length,'unique_texts':len(texts),
          'embedding_dimension':int(embeddings.shape[1]), 'sentence_transformers':importlib.metadata.version('sentence-transformers'),
          'torch':torch.__version__,'transformers':importlib.metadata.version('transformers'),
          'embedding_sha256':file_hash(cache/'complete.npy'), 'cache_key':key,
          'unique_texts_exceeding_limit':sum(x>model.max_seq_length for x in lengths),'token_lengths':lengths,
          'model_file_sha256':{str(p.relative_to(model_path)):file_hash(p) for p in Path(model_path).rglob('*') if p.is_file()}}
    save_json(meta_path,meta)
    return embeddings,meta


def geometry(vectors):
    vectors=np.asarray(vectors,dtype=np.float64)
    n=len(vectors)
    if not n:
        return {}
    center=vectors.mean(axis=0)
    centered=vectors-center
    variance=float(np.mean(np.sum(centered**2,axis=1)))
    values=np.maximum(np.linalg.eigvalsh(centered.T@centered/max(n-1,1)),0)
    total=values.sum()
    p=values[values>1e-12]/total if total>1e-12 else np.array([])
    return {'pairwise_mean_exact': float((np.sum(vectors.sum(axis=0)**2)-np.sum(vectors**2))/(n*(n-1))) if n>1 else np.nan,
            'centroid_norm':float(np.linalg.norm(center)),'dispersion':variance,
            'mean_centroid_distance':float(np.mean(np.linalg.norm(centered,axis=1))),
            'participation_ratio':float(total**2/np.sum(values**2)) if total>1e-12 else 0.,
            'effective_rank':float(np.exp(-np.sum(p*np.log(p)))) if len(p) else 0.,
            'pc1_variance_share':float(values[-1]/total) if total>1e-12 else np.nan,
            'pc10_variance_share':float(values[-10:].sum()/total) if total>1e-12 else np.nan}


def semantic_analysis(data,config,skip=False):
    output=config['output']
    obs=valid_rows(data['observations']).sort_values(['round','sentence_id']).copy()
    originals=dict(zip(data['corpus'].sentence_id,data['corpus'].text))
    texts=sorted(set(obs.output_text)|set(originals.values()))
    text_index={text:i for i,text in enumerate(texts)}
    print('Computing fixed-vocabulary TF-IDF baseline...',flush=True)
    tfidf=tfidf_vectorizer(lowercase=True,token_pattern=r'(?u)\b\w+\b',sublinear_tf=True,norm='l2').fit_transform(texts)
    lexical_rows=[]
    for t,g in obs.groupby('round'):
        a=tfidf[[text_index[x] for x in g.output_text]]
        b=tfidf[[text_index[originals[sid]] for sid in g.sentence_id]]
        own=np.asarray(a.multiply(b).sum(axis=1)).ravel()
        summed=np.asarray(a.sum(axis=0)).ravel()
        mean=(summed@summed-float(a.multiply(a).sum()))/(len(g)*(len(g)-1)) if len(g)>1 else np.nan
        lexical_rows.append({'round':t,'valid_n':len(g),'tfidf_original_mean':own.mean(),'tfidf_original_median':np.median(own),'tfidf_population_pair_mean':mean})
    save_table(output,'tfidf_round_metrics',lexical_rows)
    if skip:
        save_json(f'{output}/semantic_status.json',{'status':'skipped','reason':'Explicit --skip-semantic; TF-IDF baseline completed'})
        return None
    try:
        embeddings,metadata=embedding_cache(texts,config)
    except Exception as exc:
        save_json(f'{output}/semantic_status.json',{'status':'incomplete','reason':f'{type(exc).__name__}: {exc}'})
        print('Semantic embeddings unavailable:',exc,flush=True)
        return None
    save_json(f'{output}/semantic_status.json',{'status':'complete',**{k:v for k,v in metadata.items() if k!='token_lengths'}})
    token_lengths=dict(zip(texts,metadata['token_lengths']))
    ids=sorted(originals)
    sid_index={sid:i for i,sid in enumerate(ids)}
    n=len(ids)
    pairs=np.column_stack(np.triu_indices(n,1))
    rng=np.random.default_rng(config['seed'])
    pairs=pairs[rng.choice(len(pairs),min(config['pair_sample_size'],len(pairs)),replace=False)]
    save_table(output,'semantic_pair_sample',[{'sentence_id_a':ids[a],'sentence_id_b':ids[b]} for a,b in pairs])
    draws=rng.integers(0,n,size=(config['bootstrap_samples'],n))
    original_vectors=embeddings[[text_index[originals[sid]] for sid in ids]]
    summaries,trajectory_rows,graphs,robustness,bootstrap=[],[],[],[],[]
    for t,g in obs.groupby('round'):
        print('Semantic geometry round',t,flush=True)
        index=np.array([sid_index[x] for x in g.sentence_id])
        e=embeddings[[text_index[x] for x in g.output_text]]
        own=np.einsum('ij,ij->i',e,original_vectors[index]).clip(-1,1)
        step=np.full(len(g),np.nan)
        if summaries:
            previous_map=previous.set_index('sentence_id').output_text.to_dict()
            for j,(sid,text) in enumerate(zip(g.sentence_id,g.output_text)):
                if sid in previous_map:
                    step[j]=float(e[j]@embeddings[text_index[previous_map[sid]]])
        full=np.full((n,e.shape[1]),np.nan,dtype=np.float32)
        full[index]=e
        sims=np.einsum('ij,ij->i',full[pairs[:,0]],full[pairs[:,1]])
        sims=sims[np.isfinite(sims)].clip(-1,1)
        full_own=np.full(n,np.nan); full_own[index]=own
        boot=np.nanmean(full_own[draws],axis=1)
        row={'round':t,'valid_n':len(g),'own_mean':own.mean(),'own_median':np.median(own),'own_sd':own.std(ddof=1),
             'own_q10':np.quantile(own,.1),'own_q90':np.quantile(own,.9),'own_bootstrap_low':np.quantile(boot,.025),'own_bootstrap_high':np.quantile(boot,.975),
             'pair_sample_n':len(sims),'pair_median':np.median(sims),'pair_q10':np.quantile(sims,.1),'pair_q90':np.quantile(sims,.9),
             'pair_sample_mean':sims.mean(),'embedding_truncated_records':sum(token_lengths[x]>config['embedding']['max_seq_length'] for x in g.output_text),**geometry(e)}
        summaries.append(row)
        for j,r in enumerate(g.itertuples()):
            trajectory_rows.append({'round':t,'sentence_id':r.sentence_id,'source_language':r.source_language,'original_cosine':own[j],
                                    'previous_observation_cosine':step[j],'embedding_input_tokens':token_lengths[r.output_text],
                                    'embedding_truncated':token_lengths[r.output_text]>config['embedding']['max_seq_length']})
        matrix=e@e.T
        for threshold in config['embedding']['thresholds']:
            adjacency=matrix>=threshold
            np.fill_diagonal(adjacency,False)
            components,labels=connected_components(csr_matrix(adjacency),directed=False)
            sizes=np.bincount(labels)
            graphs.append({'round':t,'threshold':threshold,'valid_n':len(e),'components':components,'largest_component':sizes.max(),
                           'largest_component_fraction':sizes.max()/len(e),'edge_fraction':adjacency.sum()/(len(e)*(len(e)-1))})
        largest=sorted(g.output_text.value_counts().items(),key=lambda x:(-x[1],x[0]))[0][0]
        filters={'exclude_suspicious':~g.suspicious.to_numpy(), 'exclude_largest_output':(g.output_text!=largest).to_numpy(),
                 'exclude_embedding_truncation':np.array([token_lengths[x]<=config['embedding']['max_seq_length'] for x in g.output_text])}
        for label,mask in filters.items():
            robustness.append({'round':t,'filter':label,'valid_n':int(mask.sum()),'own_mean':own[mask].mean() if mask.any() else np.nan,**geometry(e[mask])})
        bootstrap.append({'round':t,'original_mean':own.mean(),'bootstrap_low':np.quantile(boot,.025),'bootstrap_high':np.quantile(boot,.975),'replicates':len(boot)})
        previous=g
    save_table(output,'semantic_round_metrics',summaries)
    trajectory=save_table(output,'semantic_trajectory_metrics',trajectory_rows)
    save_table(output,'semantic_graph_metrics',graphs)
    save_table(output,'semantic_robustness',robustness)
    save_table(output,'trajectory_bootstrap',bootstrap)
    return trajectory

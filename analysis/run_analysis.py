import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import platform
import resource
import subprocess
import sys
import re
import time
import tomllib

from .common import file_hash, save_json


def provenance(config):
    paths=list(Path('analysis').glob('*.py'))+list(Path('tests').glob('test_analysis*.py'))+[Path('config/analysis.toml'),Path('requirements_analysis.txt'),Path('analyze_local.sh')]
    packages={}
    for package in ['numpy','pandas','scipy','scikit-learn','matplotlib','sentence-transformers','transformers','torch','markdown']:
        try: packages[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: packages[package]=None
    git=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True)
    return {'configuration':config,'python':platform.python_version(),'platform':platform.platform(),'packages':packages,
            'git_head':git.stdout.strip(),'analysis_source_sha256':{str(p):file_hash(p) for p in paths},'device':'cpu','cpu_threads':config['cpu_threads']}


def main():
    parser=argparse.ArgumentParser(description='CPU-only analysis; original archive is read-only.')
    parser.add_argument('--config',type=Path,default=Path('config/analysis.toml'))
    parser.add_argument('--stage',choices=['all','audit','exact','temporal','content','semantic','language','robustness','figures','report'],default='all')
    parser.add_argument('--skip-semantic',action='store_true')
    parser.add_argument('--refresh-audit',action='store_true')
    args=parser.parse_args()
    config=tomllib.loads(args.config.read_text())
    start=time.time()
    for name in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']: os.environ[name]=''
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=config['cpu_threads'])
    from .load_data import load_and_audit
    from .content import annotate,content_analysis,language_analysis,robustness_analysis
    from .exact import exact_analysis,temporal_analysis
    from .semantic import semantic_analysis
    cache=Path(config['cache']);cache.mkdir(parents=True,exist_ok=True)
    output=Path(config['output'])
    for directory in ['figures','tables','examples']: (output/directory).mkdir(parents=True,exist_ok=True)
    proof=provenance(config)
    key={'archive':file_hash(config['archive']),'loader':file_hash('analysis/load_data.py'),
         'collector_validation':{str(p):file_hash(p) for p in [Path('src/experiment.py'),Path('src/storage.py'),Path('src/languages.py'),Path('src/package_data.py'),Path('src/dataset.py')]},'run_id':config['run_id']}
    stamp=cache/'audit_key.json';cache_path=cache/'audited_data.pkl'
    if not args.refresh_audit and stamp.exists() and cache_path.exists() and (output/'data_quality_summary.json').exists() and (output/'tables/data_quality_summary.csv').exists() and json.loads(stamp.read_text())==key:
        print('Reusing hash-matched audited reconstruction.',flush=True)
        with cache_path.open('rb') as stream: data=pickle.load(stream)
    else:
        data=load_and_audit(config)
        with cache_path.open('wb') as stream: pickle.dump(data,stream)
        save_json(stamp,key)
    data=annotate(data)
    if args.stage=='all':
        command=[sys.executable,'-m','unittest','discover','-s','tests','-v']
        test_run=subprocess.run(command,capture_output=True,text=True)
        test_output=test_run.stdout+test_run.stderr
        (output/'test_results.txt').write_text(test_output)
        count=re.search(r'Ran (\d+) tests',test_output)
        save_json(output/'test_results.json',{'command':command,'passed':test_run.returncode==0,'tests':int(count.group(1)) if count else None})
        if test_run.returncode: raise RuntimeError('Tests failed; see test_results.txt')
        (output/'validation.md').write_text('# Validation record\n\nThe full local analysis ran the synthetic analysis tests and existing collector tests. See [test results](test_results.json) and [full output](test_results.txt). No GPU or translation inference was used.\n\nThe archive audit validated canonical records, continuity, schedules, all checksums and observation-state separation. [Consistency checks](consistency_checks.json) reconcile generated counts, projection partitions, cosine identities, figure files and report links. [Output manifest](output_manifest.json) lists final derived-file hashes. These checks do not establish human semantic correctness.\n')
    save_json(output/'analysis_provenance.json',proof)
    def selected(name): return args.stage in ('all',name)
    if selected('exact'): exact_analysis(data,config)
    if selected('temporal'): temporal_analysis(data,config)
    lexical=content_analysis(data,config) if selected('content') else None
    semantic=semantic_analysis(data,config,args.skip_semantic) if selected('semantic') else None
    if selected('language'):
        if semantic is None and args.stage=='language' and (output/'tables/semantic_trajectory_metrics.csv').exists():
            import pandas as pd
            semantic=pd.read_csv(output/'tables/semantic_trajectory_metrics.csv')
        language_analysis(data,config,semantic,lexical)
    if selected('robustness'): robustness_analysis(data,config)
    if selected('figures'):
        from .figures import generate_figures
        generate_figures(config)
    if selected('report'):
        from .report import generate_report
        generate_report(data,config,proof)
    save_json(output/'last_execution.json',{'stage':args.stage,'skip_semantic':args.skip_semantic,'seconds':time.time()-start,
              'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    if args.stage=='all':
        required=['research_report.md','key_findings.md','data_audit.md','analysis_summary.json']
        required+=['tables/'+name+'.csv' for name in ['round_level_metrics','active_state_metrics','projection_collision_metrics','collision_persistence','lexical_round_metrics','language_statistics','candidate_attractors','translation_failure_metrics']]
        missing=[p for p in required if not (output/p).exists()]
        if missing: raise RuntimeError('Missing outputs: '+str(missing))
        files={str(p.relative_to(output)):file_hash(p) for p in output.rglob('*') if p.is_file() and p.name!='output_manifest.json'}
        save_json(output/'output_manifest.json',files)
        from .validate_outputs import validate_outputs
        save_json(output/'consistency_checks.json',{'status':'running'})
        checks=validate_outputs(config)
        print(f'Passed {len(checks)} real-data consistency checks.',flush=True)
        files={str(p.relative_to(output)):file_hash(p) for p in output.rglob('*') if p.is_file() and p.name!='output_manifest.json'}
        save_json(output/'output_manifest.json',files)
        if file_hash(config['archive'])!=key['archive']: raise RuntimeError('Original archive changed during analysis')
    print(f'Completed {args.stage}: {output}; {(time.time()-start)/60:.1f} minutes',flush=True)


if __name__=='__main__': main()

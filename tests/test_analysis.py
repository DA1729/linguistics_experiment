import json
import math
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd

from analysis.common import save_json, save_table, valid_rows
from analysis.content import heuristic_flags, lexical_measures, numbers
from analysis.exact import distribution_metrics, exact_analysis, normalize, permutation_persistence, projection_metrics, transition_metrics
from analysis.load_data import inspect_archive, read_rows
from analysis.semantic import geometry
from src.experiment import initial_state, make_requests, next_state, validate_record
from src.storage import digest


class analysis_tests(unittest.TestCase):
    def test_no_merges(self):
        result=distribution_metrics(['a','b','c'])
        self.assertEqual(result['distinct'],3)
        self.assertEqual(result['collision_probability'],0)
        self.assertAlmostEqual(result['entropy_nats'],math.log(3))
        self.assertAlmostEqual(result['effective_simpson'],3)

    def test_universal_equality(self):
        result=distribution_metrics(['a']*4)
        self.assertEqual(result['distinct'],1)
        self.assertEqual(result['collision_probability'],1)
        self.assertEqual(result['largest_fraction'],1)
        self.assertEqual(result['entropy_nats'],0)

    def test_multiple_clusters(self):
        result=distribution_metrics(['a','a','b','b'])
        self.assertAlmostEqual(result['collision_probability'],1/3)
        self.assertEqual(result['largest_size'],2)
        self.assertAlmostEqual(result['effective_shannon'],2)
        self.assertEqual(result['clusters_ge_2'],2)

    def test_empty_and_singleton(self):
        self.assertEqual(distribution_metrics([])['distinct'],0)
        self.assertTrue(math.isnan(distribution_metrics([])['entropy_nats']))
        self.assertTrue(math.isnan(distribution_metrics(['a'])['collision_probability']))

    def test_normalization(self):
        self.assertNotEqual(normalize('e\u0301','raw'),normalize('é','raw'))
        self.assertEqual(normalize('e\u0301','nfc'),'é')
        self.assertEqual(normalize(' A\tB\n','whitespace'),'A B')
        self.assertEqual(normalize('Straße','casefold'),'strasse')
        self.assertEqual(normalize('Hello, WORLD!','punctuation'),'hello world')
        self.assertNotEqual(normalize('A','whitespace'),normalize('a','whitespace'))

    def test_invalid_outputs_excluded(self):
        frame=pd.DataFrame({'status':['ok','failed','skipped','ok','ok'],'output_text':['a','a',None,'',' ']})
        self.assertEqual(valid_rows(frame).output_text.tolist(),['a'])

    def test_projection_requires_language(self):
        texts={'a':'same','b':'same','c':'same'}
        states={'a':('fr','x',True),'b':('de','x',True),'c':('fr','x',True)}
        result=projection_metrics(texts,states)
        self.assertEqual(result['english_collision_pairs'],3)
        self.assertEqual(result['same_state_collision_pairs'],1)
        self.assertAlmostEqual(result['projection_pair_share'],2/3)
        self.assertEqual(distribution_metrics(s[:2] for s in states.values())['distinct'],2)

    def test_temporary_merge_and_split(self):
        first={'a':'A','b':'B','c':'C'}
        middle={'a':'X','b':'X','c':'C'}
        final={'a':'A','b':'B','c':'C'}
        merged=transition_metrics(first,middle)
        self.assertEqual(merged['new_equal_pairs'],1)
        self.assertEqual(merged['merge_groups'],1)
        split=transition_metrics(middle,final)
        self.assertEqual(split['split_groups'],1)
        self.assertEqual(split['persistence'],0)

    def test_stable_partition_can_change_text(self):
        left={'a':'A','b':'A','c':'B','d':'B'}
        right={'a':'X','b':'X','c':'Y','d':'Y'}
        result=transition_metrics(left,right)
        self.assertEqual(result['persistence'],1)
        self.assertEqual(result['split_groups'],0)
        self.assertEqual(result['merge_groups'],0)

    def test_missing_trajectories_use_common_denominator(self):
        result=transition_metrics({'a':'x','b':'x','c':'z'},{'a':'x','c':'x'})
        self.assertEqual(result['common_n'],2)
        self.assertEqual(result['previous_equal_pairs'],0)
        self.assertTrue(math.isnan(result['persistence']))
        self.assertEqual(result['new_collision_rate'],1)

    def test_permutation_known_expectation_and_reproducibility(self):
        a=[0,0,1,1];b=[0,0,1,1]
        x=permutation_persistence(a,b,3000,np.random.default_rng(20))
        y=permutation_persistence(a,b,3000,np.random.default_rng(20))
        np.testing.assert_array_equal(x,y)
        self.assertAlmostEqual(x.mean(),1/3,delta=.04)
        x=permutation_persistence([0,0,0,0],b,20,np.random.default_rng(1))
        np.testing.assert_allclose(x,np.full(20,1/3))

    def test_semantic_aggregation(self):
        e=np.array([[1.,0.],[0.,1.]])
        result=geometry(e)
        self.assertAlmostEqual(result['pairwise_mean_exact'],0)
        self.assertAlmostEqual(result['dispersion'],.5)
        self.assertAlmostEqual(result['participation_ratio'],1)
        same=geometry(np.array([[1.,0.]]*3))
        self.assertAlmostEqual(same['pairwise_mean_exact'],1)
        self.assertAlmostEqual(same['dispersion'],0)
        self.assertEqual(same['effective_rank'],0)

    def test_numbers_and_retention_denominators(self):
        self.assertEqual(numbers('1,000.00 and 4.5'),numbers('1000 and 4.50'))
        row=lexical_measures('There were 10 boats.','There were 11 boats.')
        self.assertEqual(row['number_retention'],0)
        self.assertTrue(math.isnan(lexical_measures('No digits here.','Same here.')['number_retention']))

    def test_flags_separate_from_status(self):
        self.assertIn('repeated_words',heuristic_flags('no '*40))
        self.assertIn('character_collapse',heuristic_flags('yes','a'*100))
        self.assertEqual(heuristic_flags('This is a normal short sentence.'),[])

    def test_archive_loading_and_duplicate_members(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'test.zip'
            with zipfile.ZipFile(path,'w') as z:
                z.writestr('records.jsonl','{"a": 1}\n{"a": 2}\n')
            with zipfile.ZipFile(path) as z:
                self.assertEqual(read_rows(z,'records.jsonl'),[{'a':1},{'a':2}])
            self.assertEqual(inspect_archive(path)[0]['path'],'records.jsonl')
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with zipfile.ZipFile(path,'a') as z: z.writestr('records.jsonl','')
            with self.assertRaises(ValueError): inspect_archive(path)

    def test_sparse_schedule_outputs_and_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            rows=[]
            states={}
            for t in [0,3,10]:
                states[t]={'a':('fr','x',True),'b':('de','y',True),'c':('fr','z',False)}
                for sid,text,status in [('a','A' if t==0 else 'X','ok'),('b','B' if t==0 else 'X','ok'),('c','X','failed')]:
                    rows.append({'round':t,'sentence_id':sid,'output_text':text,'status':status})
            data={'observations':pd.DataFrame(rows),'states':states}
            frames=exact_analysis(data,{'output':directory})
            raw=frames['round_level_metrics'].query("normalization=='raw'")
            self.assertEqual(raw['round'].tolist(),[0,3,10])
            self.assertEqual(raw.valid_n.tolist(),[2,2,2])
            self.assertEqual(raw.distinct.tolist(),[2,1,1])
            projection=frames['projection_collision_metrics']
            self.assertEqual(projection.distinct_state_collision_pairs.tolist(),[0,1,1])
            self.assertTrue((Path(directory)/'tables/round_level_metrics.csv').exists())

    def test_state_reconstruction_and_observation_input_validation(self):
        corpus=[{'sentence_id':'a','text':'original'}]
        before=initial_state(corpus)
        after=next_state(before,[{'status':'ok','output_text':'bonjour','target_language':'fra_Latn'}])
        self.assertEqual(after[0]['text'],'bonjour')
        request=make_requests(after,corpus,{'a':['fra_Latn']},1,'observation','main_001',1,'test')[0]
        row={**request,'status':'ok','output_text':'hello','request_digest':digest(request)}
        row['record_digest']=digest(row)
        validate_record(row,request)
        self.assertEqual(after[0]['text'],'bonjour')
        row['input_text']='wrong';row['record_digest']=digest({k:v for k,v in row.items() if k!='record_digest'})
        with self.assertRaises(ValueError): validate_record(row,request)

    def test_json_output_has_no_nan_literals(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'a.json'
            save_json(path,{'a':np.nan,'b':np.int64(4)})
            self.assertEqual(json.loads(path.read_text()),{'a':None,'b':4})
            frame=save_table(directory,'small',[{'round':0,'count':2}])
            self.assertEqual(len(frame),1)

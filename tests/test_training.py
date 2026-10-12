import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from game_theory.training_bridge import StackelbergIDSController
from game_theory.ids_training import train_ids, metrics, WEAK_CLASSES
from tools.data_pipeline import preprocess_splits, prepare_features, hybrid_balance, resolve_data_paths


class TrainingTests(unittest.TestCase):
    names=['BenignTraffic','A','B']
    # Unit-test scenario only: costly targeted defence. All four actions are
    # evaluated by solve(); this is not claimed as measured calibration.
    reweight_scenario={'resource_cost':{'TARGETED_DEFENSE':1.0}}
    def test_game_evaluates_actions_and_consistent_matrices(self):
        c=StackelbergIDSController(self.names)
        f={'BenignTraffic':.9,'A':.2,'B':.6};d=c.step(f)
        self.assertEqual(len(d.all_action_results),4)
        self.assertEqual(d.recommended_defense,max(d.all_action_results,key=lambda r:r['defender_payoff']['utility'])['defense_action'])
        md,ma=c._utility_matrices(f)
        for i,action in enumerate(c.DEFENDER_ACTIONS):
            r=c.game.evaluate_action(action,f);j=c.attack_classes.index(r['attacker_best_response'])
            self.assertAlmostEqual(md[i,j],r['defender_payoff']['utility'])
            self.assertAlmostEqual(ma[i,j],r['attacker_utility'])

    def test_34_outputs_with_nonzero_benign_id(self):
        names=['Attack0','BenignTraffic']+[f'Attack{i}' for i in range(1,33)]
        controller=StackelbergIDSController(names,game_parameters=self.reweight_scenario)
        f={n:.85 for n in names};f['Attack0']=.2
        decision=controller.step(f)
        self.assertEqual(controller.class_to_id['BenignTraffic'],1)
        self.assertEqual(len(decision.p_att),33)
        self.assertEqual(len(decision.p_def),4)
        self.assertEqual(decision.U_def.shape,(4,))
        self.assertEqual(decision.U_att.shape,(33,))
        self.assertEqual(set(decision.current_class_weights),set(range(34)))
        self.assertEqual(decision.targeted_class,'Attack0')
        self.assertGreater(decision.current_class_weights[0],decision.current_class_weights[1])
        self.assertAlmostEqual(sum(decision.current_class_weights.values())/34,1)

    def test_severity_not_counted_twice(self):
        from game_theory.payoff import SecurityPayoff
        payoff=SecurityPayoff().defender_utility('A',2.,1.,dict(resource_cost=0.,latency_cost=0.,false_positive_cost=0.))
        self.assertEqual(payoff['security_benefit'],1.)
        c=StackelbergIDSController(self.names,severity={'A':2.,'B':.8})
        d=c.step({'A':.5,'B':.8});md,ma=c._utility_matrices(d.game_f1)
        for i,action in enumerate(c.DEFENDER_ACTIONS):
            result=c.game.evaluate_action(action,d.game_f1)
            j=c.attack_classes.index(result['attacker_best_response'])
            self.assertAlmostEqual(md[i,j],result['defender_payoff']['utility'])

    def test_bounds_normalization_deterioration(self):
        c=StackelbergIDSController(self.names,scale_step=.5,max_scale=1.3,game_parameters=self.reweight_scenario)
        f={'A':.2,'B':.2};d=c.step(f)
        self.assertEqual(d.recommended_defense,'CLASS_REWEIGHTING')
        self.assertTrue(d.targeted_class)
        self.assertAlmostEqual(np.mean(list(d.current_class_weights.values())),1)
        target=d.targeted_class;old_scale=c.class_scales[c.class_to_id[target]]
        f[target]=.01;d=c.step(f)
        self.assertTrue(d.deterioration)
        self.assertIsNone(d.applied_action)
        self.assertLess(c.class_scales[c.class_to_id[target]],old_scale)
        for _ in range(20): c.step({'A':.1,'B':.1})
        self.assertLessEqual(max(c.class_scales.values()),1.3)
        d=c.step({'A':.1,'B':.1})
        self.assertIsNone(d.applied_action)
        self.assertEqual(d.scale_change_reason,'scale_unchanged_at_bound_or_zero_weakness')

    def test_nonactuated_defence_and_input_validation(self):
        c=StackelbergIDSController(self.names)
        d=c.step({'A':1.,'B':1.})
        self.assertEqual(d.recommended_defense,'NORMAL');self.assertIsNone(d.applied_action)
        self.assertEqual(d.current_class_weights,{0:1.,1:1.,2:1.})
        d=c.step({'A':.2,'B':.2})
        self.assertEqual(d.recommended_defense,'TARGETED_DEFENSE')
        self.assertIsNone(d.applied_action)
        self.assertEqual(d.current_class_weights,{0:1.,1:1.,2:1.})
        for f in [{'A':.2},{'A':np.nan,'B':.2},{'A':1.2,'B':.2}]:
            with self.assertRaises(ValueError):c.step(f)
        with self.assertRaises(ValueError):StackelbergIDSController(['A','A'])
        with self.assertRaises(ValueError):StackelbergIDSController(self.names,base_class_weights={0:1})

    def test_replicator_symmetry_and_stability(self):
        p=np.ones(3)/3
        self.assertTrue(np.allclose(StackelbergIDSController._replicator_update(p,np.ones(3)),p))
        u=np.array([-1e6,0.,1e6]);q=StackelbergIDSController._replicator_update(p,u)
        self.assertAlmostEqual(q.sum(),1);self.assertTrue(np.all(q>=.001))
        self.assertGreater(q[2],q[0])
        with self.assertRaises(ValueError):StackelbergIDSController._replicator_update(p,np.ones(2))

    def test_metrics_missing_predictions(self):
        r=metrics(np.array([0,1,2]),np.array([0,0,0]),self.names)
        self.assertEqual(r['per_class']['A']['precision'],0)
        self.assertEqual(r['per_class']['A']['f1'],0)
        self.assertAlmostEqual(r['macro_f1'],1/6)

    def test_train_only_preprocessing_preserves_splits(self):
        train=pd.DataFrame({'x':[1.,2.,np.nan,4.,5.,6.],'z':[1.,2.,3.,4.,5.,6.],'label':['A','B']*3})
        val=pd.DataFrame({'x':[np.nan,100000.],'z':[100.,200.],'label':['A','B']})
        combined,encoder=preprocess_splits(train,val,val)
        self.assertEqual(len(combined),10)
        self.assertEqual(combined[combined.dataset.eq('val')].x.iloc[0],4.)
        processed,features,scaler,_=prepare_features(combined,2)
        self.assertTrue(np.allclose(scaler.mean_,[combined[combined.dataset.eq('train')][f].mean() for f in features]))
        self.assertEqual(processed.dataset.value_counts().to_dict(),{'train':6,'val':2,'test':2})
        bad=val.copy();bad.loc[0,'label']='unknown'
        with self.assertRaises(ValueError):preprocess_splits(train,bad,val)

    def test_missing_data_and_balancing(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(FileNotFoundError,'No dataset substitution'):resolve_data_paths(d)
        X=np.arange(16).reshape(8,2);y=np.array([0]*3+[1]*5)
        a,b=hybrid_balance(X,y,4,4)
        self.assertEqual(np.bincount(b).tolist(),[4,4]);self.assertEqual(a.shape,(8,2))

    def test_baseline_budget_sampling_and_aggregation(self):
        from tools.run_ablation import aggregate
        class Model:
            def __init__(self): self.samples=[]; self.weights=[]
            def fit(self,X,y,**kwargs):
                self.samples.append(X.copy());self.weights.append(kwargs['class_weight'].copy())
                assert kwargs['epochs']==1
            def predict(self,X,**kwargs):return np.eye(3)[np.arange(len(X))%3]
            def save_weights(self,path):open(path,'w').close()
            def load_weights(self,path):pass
        X=np.arange(36).reshape(12,3);y=np.arange(12)%3
        with tempfile.TemporaryDirectory() as out:
            a,b=Model(),Model()
            with patch('game_theory.ids_training.plot_history'):
                ha,_=train_ids(a,X,y,X,y,self.names,epochs=2,batch_fraction=.5,adaptive=False,output_dir=out+'/a',verbose=0)
                hb,_=train_ids(b,X,y,X,y,self.names,epochs=2,batch_fraction=.5,adaptive=True,output_dir=out+'/b',verbose=0)
            for left,right in zip(a.samples,b.samples):self.assertTrue(np.array_equal(left,right))
            self.assertEqual(a.weights[0],a.weights[1])
            self.assertIsNone(ha[0]['decision'])
            runs=[{'method':method,'seed':seed,'test':metrics(y,y,self.names)}
                  for method in ['baseline','proposed'] for seed in [42,43,44]]
            aggregate(runs,__import__('pathlib').Path(out))
            table=pd.read_csv(out+'/individual_metrics.csv')
            self.assertEqual(len(table),6)
            self.assertTrue(np.allclose(table.macro_f1,1.))
            summary=pd.read_csv(out+'/mean_std.csv',header=[0,1],index_col=0)
            self.assertEqual(summary.loc['baseline',('macro_f1','count')],3)
            self.assertEqual(summary.loc['baseline',('macro_f1','std')],0)

    def test_actual_predictions_drive_next_epoch_weights(self):
        # Synthetic spy model: predictions are measured, never injected F1 states.
        class Model:
            def __init__(self):self.weights=[];self.predictions=0
            def fit(self,X,y,**kwargs):
                assert kwargs['epochs']==1
                self.weights.append(kwargs['class_weight'].copy())
            def predict(self,X,**kwargs):
                self.predictions+=1
                # Balanced, imperfect predictions give F1=0.2 for each attack.
                pred=np.array([1]+[2]*4+[2]+[1]*4)
                if self.predictions>1:pred=np.array([1,1]+[2]*3+[2,2]+[1]*3)
                return np.eye(3)[pred]
            def save_weights(self,path):open(path,'w').close()
            def load_weights(self,path):pass
        model=Model();X=np.zeros((10,4));y=np.array([1]*5+[2]*5)
        Xt=np.zeros((12,4));yt=np.array([0,1,2]*4)
        with tempfile.TemporaryDirectory() as out:
            with patch('game_theory.ids_training.plot_history'):
                history,c=train_ids(model,Xt,yt,X,y,self.names,epochs=2,output_dir=out,verbose=0,controller_options={"game_parameters":self.reweight_scenario})
            self.assertEqual(c.game.history[0]['recommended_defense'],'CLASS_REWEIGHTING')
            self.assertEqual(history[0]['decision']['class_f1']['A'],history[0]['validation']['per_class']['A']['f1'])
            self.assertNotEqual(model.weights[0],model.weights[1])
            self.assertEqual(history[0]['decision']['current_class_weights'],history[1]['training_weights'])
            self.assertNotEqual(history[0]['decision']['game_f1'],history[1]['decision']['game_f1'])

if __name__=='__main__':unittest.main()

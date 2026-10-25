"""One fixed, validation-only focal-XSS experiment; reuses existing Q-ARMOR components."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import os
os.environ['MPLBACKEND']='Agg'
import json, time, hashlib, platform, shutil
from dataclasses import asdict
import numpy as np
import pandas as pd
from game_theory.ids_training import metrics,predict_labels,save_json
from game_theory.training_bridge import StackelbergIDSController
from tools.data_pipeline import preprocess_splits,prepare_features,hybrid_balance,validation_partition
from tools.training_runtime import configure_runtime

OUT=Path(__file__).resolve().parent
TARGET='XSS'
SEEDS=[42,43,44]
PROTOCOL=dict(target=TARGET,seeds=SEEDS,warmup_epochs=5,continuation_epochs=1,
    architecture='unchanged original CNN/LSTM/attention; all 34 classes',
    data='exact restored CICIoT2023 bounded archive sample',
    validation='fixed audit half, stratified seed 2026; same rows before and after',
    target_basis='Pre-established weak-class list and historical validation weakness; fixed before these runs.',
    threat_scope='Focused XSS scenario: existing severity API sets XSS=1 and all other attacks=0. This prioritizes the chosen threat; it is not an unrestricted all-threat policy test.',
    game_parameters='Existing original prototype defaults, not empirically calibrated coefficients; no cost/effectiveness tuning in this experiment.',
    defenses=['NORMAL','CLASS_REWEIGHTING'],scale_step=.15,max_scale=2.5,
    training_rows_per_class=1000,batch_size=256,learning_rate=.0005,
    comparison='Paired NORMAL and game-selected continuation; identical starting model/optimizer, training order, branch seed and epoch budget.',
    stopping='Exactly three seeds, five warmup epochs and one continuation epoch per branch; no reruns or post-result tuning to obtain improvement.',
    interpretation='Validation demonstration only. A before/after gain alone is not causal evidence; compare matched NORMAL. Test CSV never read. No artificial performance corruption or fabricated F1.',
    created_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())

def main():
    if (OUT/'protocol.json').exists():
        raise FileExistsError('Experiment already started; refusing to overwrite or retune')
    data=ROOT/'data/CICIoT23_archive_sample_v1'
    PROTOCOL['source_sha256']={n:hashlib.sha256((data/p).read_bytes()).hexdigest() for n,p in [('train','train/train.csv'),('validation','validation/validation.csv')]}
    PROTOCOL['source_hashes']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['tools/ids_model.py','tools/data_pipeline.py','game_theory/training_bridge.py','game_theory/stackelberg.py']}
    save_json(OUT/'protocol.json',PROTOCOL)
    runtime=configure_runtime('cpu',4)
    import tensorflow as tf
    from tools.ids_model import build_enhanced_cnn_lstm_attention_model
    import joblib
    save_json(OUT/'environment.json',dict(runtime=runtime,python=sys.version,platform=platform.platform(),tensorflow=tf.__version__,numpy=np.__version__,pandas=pd.__version__))
    train=pd.read_csv(data/'train/train.csv');val=pd.read_csv(data/'validation/validation.csv')
    _,audit=validation_partition(val.label.to_numpy())
    save_json(OUT/'validation_partition.json',dict(seed=2026,audit_rows=audit.tolist(),test_accessed=False))
    val=val.iloc[audit].reset_index(drop=True)
    rows=[]
    for seed in SEEDS:
        if shutil.disk_usage(OUT).free < 90*1024*1024:
            raise RuntimeError('Need at least 90 MiB free before the next seed; no unrelated files deleted')
        seedout=OUT/str(seed);seedout.mkdir()
        tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(seed)
        combined,encoder=preprocess_splits(train,val,val)
        transformed,features,scaler,selector=prepare_features(combined,seed=seed)
        names=encoder.classes_.tolist();target_id=names.index(TARGET)
        t=transformed[transformed.dataset.eq('train')];v=transformed[transformed.dataset.eq('val')]
        X,y=hybrid_balance(t[features].to_numpy(np.float32),t.label_encoded.to_numpy(int),1000,1000,seed)
        Xv=v[features].to_numpy(np.float32);yv=v.label_encoded.to_numpy(int)
        onehot=np.eye(len(names),dtype=np.float32)[y]
        from sklearn.utils.class_weight import compute_class_weight
        base=compute_class_weight('balanced',classes=np.arange(len(names)),y=y);base/=base.mean()
        joblib.dump(dict(encoder=encoder,features=features,scaler=scaler,selector=selector),seedout/'preprocessing.pkl')
        def build():
            m=build_enhanced_cnn_lstm_attention_model((X.shape[1],),len(names))
            m.compile(optimizer=tf.keras.optimizers.Adam(5e-4,clipnorm=1.),loss='categorical_crossentropy')
            m.optimizer.build(m.trainable_variables)
            return m
        def evaluate(m):return metrics(yv,predict_labels(m,Xv,names),names)
        model=build();rng=np.random.default_rng(seed);warmup=[]
        for epoch in range(5):
            order=rng.permutation(len(X));start=time.perf_counter()
            model.fit(X[order],onehot[order],epochs=1,batch_size=256,shuffle=False,class_weight=dict(enumerate(base)),verbose=0)
            score=evaluate(model)
            warmup.append(dict(epoch=epoch+1,validation=score,seconds=time.perf_counter()-start))
            save_json(seedout/'warmup.json',warmup)
            save_json(OUT/'status.json',dict(status='running',seed=seed,stage='warmup',epoch=epoch+1,completed_seeds=len(rows)))
            print(f'Seed {seed} warmup {epoch+1}/5: XSS F1={score["per_class"][TARGET]["f1"]:.6f}, macro-F1={score["macro_f1"]:.6f}, {time.perf_counter()-start:.1f}s',flush=True)
        before=evaluate(model);snapshot=model.get_weights();optimizer=[v.numpy().copy() for v in model.optimizer.variables]
        model.save_weights(seedout/'before.weights.h5')
        severity={n:float(n==TARGET) for n in names if n!='BenignTraffic'}
        controller=StackelbergIDSController(names,dict(enumerate(base)),severity=severity,scale_step=.15,max_scale=2.5,available_defenses=['NORMAL','CLASS_REWEIGHTING'])
        decision=controller.step({n:d['f1'] for n,d in before['per_class'].items()})
        assert decision.targeted_class in (None,TARGET)
        save_json(seedout/'decision.json',asdict(decision))
        order=rng.permutation(len(X));pair={}
        for method in ['normal_control','qarmor']:
            tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(seed+1000)
            branch=build();branch.set_weights(snapshot)
            if len(branch.optimizer.variables)!=len(optimizer):raise RuntimeError('Optimizer state mismatch')
            for var,value in zip(branch.optimizer.variables,optimizer):var.assign(value)
            assert all(np.array_equal(a,b) for a,b in zip(branch.get_weights(),snapshot))
            assert all(np.array_equal(v.numpy(),a) for v,a in zip(branch.optimizer.variables,optimizer))
            weights=controller.current_class_weights() if method=='qarmor' else dict(enumerate(base))
            start=time.perf_counter()
            branch.fit(X[order],onehot[order],epochs=1,batch_size=256,shuffle=False,class_weight=weights,verbose=0)
            after=evaluate(branch)
            pair[method]=dict(after=after,training_weights=weights,seconds=time.perf_counter()-start)
            if method=='qarmor':branch.save_weights(seedout/'after.weights.h5')
            save_json(seedout/(method+'.json'),pair[method])
            print(f'Seed {seed} {method}: XSS F1={after["per_class"][TARGET]["f1"]:.6f}, macro-F1={after["macro_f1"]:.6f}, {time.perf_counter()-start:.1f}s',flush=True)
        followup=controller.step({n:d['f1'] for n,d in pair['qarmor']['after']['per_class'].items()})
        save_json(seedout/'followup_decision_not_trained.json',asdict(followup))
        result=dict(seed=seed,target=TARGET,before=before,decision=asdict(decision),**pair,
                    matched_start_model_and_optimizer=True,train_order_sha256=hashlib.sha256(order.tobytes()).hexdigest())
        save_json(seedout/'result.json',result)
        q=pair['qarmor']['after'];c=pair['normal_control']['after']
        row=dict(seed=seed,target=TARGET,f1_before=before['per_class'][TARGET]['f1'],decision=decision.recommended_defense,
            selected_class=decision.targeted_class,applied_action=decision.applied_action,scale=decision.class_weight_scale,
            weight_before=float(base[target_id]),weight_applied=float(pair['qarmor']['training_weights'][target_id]),
            f1_after=q['per_class'][TARGET]['f1'],normal_f1_after=c['per_class'][TARGET]['f1'],
            macro_f1_before=before['macro_f1'],macro_f1_after=q['macro_f1'],normal_macro_f1_after=c['macro_f1'])
        row.update(f1_change=row['f1_after']-row['f1_before'],f1_vs_normal=row['f1_after']-row['normal_f1_after'],macro_f1_change=row['macro_f1_after']-row['macro_f1_before'])
        rows.append(row);pd.DataFrame(rows).to_csv(OUT/'results.csv',index=False)
        save_json(OUT/'status.json',dict(status='running',completed_seeds=len(rows),expected_seeds=3))
    table=pd.DataFrame(rows)
    table.select_dtypes(include='number').drop(columns=['seed']).agg(['mean','std']).to_csv(OUT/'mean_std.csv')
    save_json(OUT/'status.json',dict(status='complete',completed_seeds=3,expected_seeds=3,test_accessed=False,implementation_stopped=True))
    print('FINAL EXPERIMENT COMPLETE; no additional implementation or tuning.',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        save_json(OUT/'failure.json',dict(error=repr(exc)))
        raise

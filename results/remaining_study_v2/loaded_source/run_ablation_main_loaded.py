"""Original CICIoT2023 splits only. Fair three-seed baseline vs proposed."""
import argparse
import hashlib
import importlib.metadata
import platform
from pathlib import Path
import json
import numpy as np
from .data_pipeline import resolve_data_paths, load_splits, preprocess_splits, prepare_features, hybrid_balance
from game_theory.ids_training import train_ids, metrics, predict_labels, save_json, WEAK_CLASSES


def aggregate(results, output):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys = ['accuracy','macro_f1','weighted_f1','macro_precision','macro_recall']
    rows = []
    for run in results:
        row = dict(method=run['method'], seed=run['seed'])
        row.update({k:run['test'][k] for k in keys})
        row.update({n:run['test']['per_class'][n]['f1'] for n in WEAK_CLASSES if n in run['test']['per_class']})
        rows.append(row)
    df = pd.DataFrame(rows); df.to_csv(output/'individual_metrics.csv',index=False)
    summary = df.groupby('method').agg({k:['mean','std','count'] for k in df.columns if k not in ['method','seed']})
    summary.to_csv(output/'mean_std.csv')
    fig, ax = plt.subplots(figsize=(10,5))
    for i, method in enumerate(df.method.unique()):
        part=df[df.method.eq(method)]
        ax.bar(np.arange(len(keys)) + i*.35, part[keys].mean(), .35,
               yerr=part[keys].std(ddof=1).fillna(0), label=method)
    ax.set_xticks(np.arange(len(keys))+.175,keys);ax.legend();fig.tight_layout()
    fig.savefig(output/'comparison.png',dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir');p.add_argument('--output',default='results/ablation')
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    p.add_argument('--epochs',type=int,default=10);p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--batch-fraction',type=float,default=.1)
    p.add_argument('--minority-target',type=int,default=300000);p.add_argument('--majority-target',type=int,default=300000)
    p.add_argument('--scale-step',type=float,default=.15);p.add_argument('--max-scale',type=float,default=2.5)
    p.add_argument('--game-config',help='JSON of payoff/cost/effectiveness overrides, calibrated on validation only')
    p.add_argument('--device',choices=['auto','cpu','metal'],default='auto')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--resume',action='store_true',help='Reuse completed runs with identical arguments')
    p.add_argument('--validation-subset',choices=['full','audit'],default='full')
    p.add_argument('--check-data',action='store_true',help='Assess file sizes without loading or training')
    args=p.parse_args(); output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    try: paths=resolve_data_paths(args.data_dir)
    except FileNotFoundError as exc:
        save_json(output/'status.json',{'status':'blocked_missing_original_data','reason':str(exc),'arguments':vars(args)})
        p.exit(2,str(exc)+'\n')
    sizes={k:{'path':str(v.resolve()),'bytes':v.stat().st_size} for k,v in paths.items()}
    if args.data_dir:
        descriptor=Path(args.data_dir).expanduser()/'dataset_manifest.json'
        if descriptor.is_file():
            sizes['dataset_description']=json.loads(descriptor.read_text())
    save_json(output/'data_manifest.json',sizes)
    if args.check_data:
        print(json.dumps(sizes,indent=2)); print('Assess RAM for all splits, MI and 300k/class balancing before a full run.');return
    import tensorflow as tf
    from .training_runtime import configure_runtime
    runtime=configure_runtime(args.device,args.threads)
    from .ids_model import build_enhanced_cnn_lstm_attention_model
    import joblib
    if (output/'environment.json').exists():
        previous=json.loads((output/'environment.json').read_text())['arguments']
        if not args.resume or any(previous.get(k)!=v for k,v in vars(args).items() if k!='resume'):
            p.error('Output exists; use a new directory or --resume with identical arguments')
    save_json(output/'environment.json',{'python':platform.python_version(),'platform':platform.platform(),
       'machine':platform.machine(),'dependencies':{n:importlib.metadata.version(n) for n in
          ['tensorflow','numpy','scikit-learn','imbalanced-learn','cryptography']},
       'devices':[str(d) for d in tf.config.list_physical_devices()], 'runtime':runtime,'arguments':vars(args),
       'determinism':'Seeds and deterministic ops requested; Metal may not be bit-identical across hosts.'})
    for k,v in paths.items():
        h=hashlib.sha256()
        with v.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        sizes[k]['sha256']=h.hexdigest()
    save_json(output/'data_manifest.json',sizes)
    train,val,test=load_splits(args.data_dir)
    if args.validation_subset=='audit':
        from .data_pipeline import validation_partition
        _,audit=validation_partition(val.label.to_numpy())
        save_json(output/'validation_partition.json',{'seed':2026,'rows':audit.tolist(),
            'purpose':'Disjoint from calibration fitting; used for controller feedback and checkpoint selection.'})
        val=val.iloc[audit].reset_index(drop=True)
    game=json.loads(Path(args.game_config).read_text()) if args.game_config else {}
    results=[]; failures=[]
    for seed in args.seeds:
        # One preprocessing/resampling result is reused by both methods for each seed.
        combined,encoder=preprocess_splits(train,val,test)
        transformed,features,scaler,selector=prepare_features(combined,seed=seed)
        arrays=[]
        for split in ['train','val','test']:
            frame=transformed[transformed.dataset.eq(split)]
            arrays.append((frame[features].to_numpy(dtype=np.float32),frame.label_encoded.to_numpy(dtype=int)))
        (Xt,yt),(Xv,yv),(Xs,ys)=arrays
        Xt,yt=hybrid_balance(Xt,yt,args.minority_target,args.majority_target,seed)
        seedout=output/str(seed);seedout.mkdir(exist_ok=True)
        joblib.dump(dict(encoder=encoder,scaler=scaler,selector=selector,features=features),seedout/'preprocessing.pkl')
        save_json(seedout/'weak_classes.json',{'classes':[n for n in WEAK_CLASSES if n in encoder.classes_],
            'basis':'Pre-established project bridge weak-class list; no test-score selection.'})
        for method in ['baseline','proposed']:
            runout=seedout/method
            if args.resume and (runout/'test.json').is_file():
                results.append(json.loads((runout/'test.json').read_text()))
                aggregate(results,output)
                continue
            try:
                tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(seed)
                tf.config.experimental.enable_op_determinism()
                model=build_enhanced_cnn_lstm_attention_model((Xt.shape[1],),len(encoder.classes_))
                model.compile(optimizer=tf.keras.optimizers.Adam(5e-4,clipnorm=1.),loss='categorical_crossentropy',metrics=['accuracy'])
                history,_=train_ids(model,Xt,yt,Xv,yv,encoder.classes_.tolist(),epochs=args.epochs,
                     batch_size=args.batch_size,batch_fraction=args.batch_fraction,seed=seed,adaptive=method=='proposed',
                     controller_options=dict(scale_step=args.scale_step,max_scale=args.max_scale,game_parameters=game,
                                             available_defenses=['NORMAL','CLASS_REWEIGHTING']),output_dir=runout,
                     verbose=0,evidence_label='Archive sample validation' if sizes.get('dataset_description') else 'Provided validation')
                evaluation=metrics(ys,predict_labels(model,Xs,encoder.classes_.tolist()),encoder.classes_.tolist())
                run=dict(method=method,seed=seed,test=evaluation,best_validation_macro_f1=max(r['validation']['macro_f1'] for r in history))
                save_json(runout/'test.json',run);np.savetxt(runout/'confusion_matrix.csv',evaluation['confusion_matrix'],delimiter=',',fmt='%d')
                import matplotlib.pyplot as plt
                fig,ax=plt.subplots(figsize=(12,10))
                image=ax.imshow(evaluation['confusion_matrix'])
                ax.set(xlabel='Predicted',ylabel='Actual',title=f'{method}, seed {seed}: test confusion')
                positions=np.arange(len(encoder.classes_))
                ax.set_xticks(positions,encoder.classes_,rotation=90,fontsize=6)
                ax.set_yticks(positions,encoder.classes_,fontsize=6)
                fig.colorbar(image,ax=ax,label='Test rows')
                fig.tight_layout();fig.savefig(runout/'confusion_matrix.png');plt.close(fig)
                results.append(run);aggregate(results,output)
            except Exception as exc:
                failures.append(dict(seed=seed,method=method,error=repr(exc)))
                save_json(runout/'failure.json',failures[-1])
    save_json(output/'status.json',dict(status='complete' if not failures else 'incomplete',
        completed_runs=len(results),expected_runs=len(args.seeds)*2,failures=failures,
        arguments=vars(args),dataset_description=sizes.get('dataset_description')))
    if failures: raise SystemExit(1)

if __name__=='__main__':main()

"""Clearly labelled synthetic Keras integration smoke test, never CICIoT results."""
import argparse
import numpy as np
from game_theory.ids_training import train_ids,save_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='results/synthetic_smoke');p.add_argument('--original-model',action='store_true');p.add_argument('--reweight-fixture',action='store_true',help='Synthetic unit fixture with costly targeted defence; not calibrated experiment parameters');args=p.parse_args()
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    rng=np.random.default_rng(42)
    # Random labels deliberately keep classes weak, exercising genuine game decisions.
    X=rng.normal(size=(96,25)).astype(np.float32);y=np.tile(np.arange(3),32)
    Xv=rng.normal(size=(30,25)).astype(np.float32);yv=np.tile(np.arange(3),10)
    if args.original_model:
        from .ids_model import build_enhanced_cnn_lstm_attention_model
        model=build_enhanced_cnn_lstm_attention_model((25,),3)
    else:
        model=tf.keras.Sequential([tf.keras.layers.Input((25,)),tf.keras.layers.Dense(12,activation='relu'),tf.keras.layers.Dense(3,activation='softmax')])
    model.compile(optimizer=tf.keras.optimizers.Adam(.001),loss='categorical_crossentropy')
    history,_=train_ids(model,X,y,Xv,yv,['BenignTraffic','A','B'],epochs=3,batch_size=16,output_dir=args.output,verbose=0,evidence_label="SYNTHETIC smoke (not CICIoT2023)",controller_options={'game_parameters':{'resource_cost':{'TARGETED_DEFENSE':1.0}}} if args.reweight_fixture else None)
    assert len(history)==3
    if args.reweight_fixture:
        assert any(row['decision']['applied_action']=='CLASS_REWEIGHTING' for row in history), 'Fixture did not exercise actuator' 
    for previous,current in zip(history,history[1:]):
        assert previous['decision']['current_class_weights']==current['training_weights']
    save_json(args.output+'/status.json',{'status':'passed','dataset':'synthetic unit/smoke only','original_architecture':args.original_model,'epochs':3,'reweight_fixture':args.reweight_fixture,'applied_actions':[r['decision']['applied_action'] for r in history]})
    print('SYNTHETIC smoke passed; not CICIoT2023 evidence.')

if __name__=='__main__':main()

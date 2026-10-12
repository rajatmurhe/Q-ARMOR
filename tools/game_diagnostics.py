"""Synthetic game diagnostics and parameter sensitivity; no IDS research metrics."""
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
from game_theory.training_bridge import StackelbergIDSController
from game_theory.ids_training import save_json,plot_history


def main():
    import pandas as pd
    out=Path('results/component_tests');out.mkdir(parents=True,exist_ok=True)
    names=['BenignTraffic','A','B'];controller=StackelbergIDSController(names)
    states=[{'BenignTraffic':.9,'A':a,'B':b} for a,b in [(.15,.85),(.8,.25),(.35,.7),(.8,.45),(.55,.8)]]
    history=[]
    for i,state in enumerate(states,1):
        d=controller.step(state)
        history.append(dict(epoch=i,decision=asdict(d),validation={'per_class':{n:{'f1':v} for n,v in state.items()}}))
    save_json(out/'synthetic_states.json',{'dataset':'synthetic state transitions only','history':history})
    plot_history(history,names,out,evidence_label="SYNTHETIC state transitions (not CICIoT2023)")
    assert all(np.isclose(row['decision']['p_def'].sum(),1) and np.isclose(row['decision']['p_att'].sum(),1) for row in history)
    assert not np.allclose(history[0]['decision']['p_def'],history[-1]['decision']['p_def'])
    assert not np.allclose(history[0]['decision']['p_att'],history[-1]['decision']['p_att'])
    # Original balanced-class strategy rows collapse to identical uniform rows.
    counts=np.ones(34)*300000;importance=counts.max()/(counts+1e-8);importance/=importance.mean()+1e-12
    legacy_def=np.vstack([np.ones(34),1+1.5*(importance-1),1+.7/(1+importance)])
    legacy_def/=legacy_def.sum(axis=1,keepdims=True)
    assert np.allclose(legacy_def,legacy_def[0])
    rows=[]
    for w in np.linspace(0,1,101):
        for cost in [.15,.3,.6,1.]:
            c=StackelbergIDSController(names,game_parameters={'payoff':{'resource_weight':cost}})
            f={'A':1-float(w),'B':1-float(w)};d=c.step(f)
            rows.append(dict(weakness=float(w),resource_weight=cost,defence=d.recommended_defense,
                             defender_utility=d.defender_utility,attacker_utility=d.attacker_utility))
    pd.DataFrame(rows).to_csv(out/'payoff_sensitivity.csv',index=False)
    default=[r for r in rows if r['resource_weight']==.3]
    save_json(out/'diagnostics.json',{'evidence':'synthetic component tests, no CICIoT2023 training',
        'legacy_balanced_strategy_rows_identical':True,
        'default_uniform_severity_grid_action_counts':{a:sum(r['defence']==a for r in default) for a in controller.DEFENDER_ACTIONS},
        'limitation':'Unimplemented recommendations are logged without actuator. Original default payoff assumptions can prevent reweighting. Calibrate on paired validation probes before research claims.'})
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4))
    for cost in [.15,.3,.6,1.]:
        subset=[r for r in rows if r['resource_weight']==cost]
        ax.plot([r['weakness'] for r in subset],[controller.DEFENDER_ACTIONS.index(r['defence']) for r in subset],label=f'resource weight {cost}')
    ax.set_yticks(range(4),controller.DEFENDER_ACTIONS);ax.set(xlabel='Synthetic class weakness (1 - F1)',title='Game parameter sensitivity; synthetic states')
    ax.legend();fig.tight_layout();fig.savefig(out/'payoff_sensitivity.png');plt.close(fig)
    print('Synthetic game diagnostics passed.')

if __name__=='__main__':main()

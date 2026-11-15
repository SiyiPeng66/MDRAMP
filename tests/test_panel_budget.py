import pandas as pd
from mdramp.panel_budget import choose,SPECIES


def test_budget_protects_resistance_and_greedy_function_coverage():
    rows=[];functions={};protected=set()
    for species in SPECIES:
        for i in range(3):
            pid=f'{species}{i}';rows.append({'record_id':pid,'species':species,'reviewed':False,'annotation_score':1,'length':100})
            functions[pid]={'a'} if i<2 else {'b','c'}
        protected.add(species+'0')
    frame=pd.DataFrame(rows);selected,decisions=choose(frame,functions,protected,{s:2 for s in SPECIES})
    assert selected=={s+str(i) for s in SPECIES for i in (0,2)}
    assert len(decisions)==8
    assert choose(frame.iloc[::-1],functions,protected,{s:2 for s in SPECIES})[0]==selected

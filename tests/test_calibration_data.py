import json
import math
import subprocess
import sys
import pytest
from mdramp.calibration_data import parse_mic,screen_constraint,intersect_constraints,assign_groups,build_tables,peptide_id


def test_screen_intervals_and_strict_boundary():
    left=intersect_constraints([screen_constraint('+',32,'plus'),screen_constraint('+',256,'plus')])
    assert left[:2]==(-math.inf,5.) and left[4]=='usable'
    middle=intersect_constraints([screen_constraint('-',32,'plus'),screen_constraint('+',256,'plus')])
    assert middle[:4]==(5.,8.,False,True)
    right=intersect_constraints([screen_constraint('-',32,'plus'),screen_constraint('-',256,'plus')])
    assert right[:2]==(8.,math.inf)
    contradiction=intersect_constraints([screen_constraint('+',32,'plus'),screen_constraint('-',256,'plus')])
    assert contradiction[4]=='quarantined'
    assert screen_constraint('-',32,'minus')==screen_constraint('+',32,'plus')
    assert intersect_constraints([parse_mic('32'),screen_constraint('-',32,'plus')])[4]=='quarantined'
    assert parse_mic('>256')[:4]==(8.,math.inf,False,False)
    assert intersect_constraints([parse_mic('16'),screen_constraint('+',32,'plus')])[:2]==(4.,4.)


def test_connected_components_transitive_and_fold_isolation():
    sequences=['A'*20,'A'*18+'CC','DD'+'A'*16+'CC']+['C'*20,'G'*20,'K'*20,'L'*20,'V'*20]
    assigned,loads=assign_groups(sequences,11)
    assert len({assigned[seq] for seq in sequences[:3]})==1
    assert set(fold for _,fold in assigned.values())==set(range(5))
    assert sum(loads)==len(sequences)
    assert assign_groups(sequences[::-1],11)==(assigned,loads)
    with pytest.raises(ValueError,match='at least 5'):assign_groups(sequences[:3])


def test_raw_sources_merge_six_strains_and_conflicts():
    headers={'A1':'轮次','B1':'旧原始ID','C1':'采用序列','D1':'A. baumannii 32 μg/mL','E1':'MIC_Ab_WT_μg_mL'}
    seqs=['A'*20,'C'*20,'G'*20,'K'*20,'L'*20,'V'*20]
    rows=[{'row':1,'cells':headers,'formulas':{}}]
    for i,seq in enumerate(seqs,2):rows.append({'row':i,'cells':{f'A{i}':'R0',f'B{i}':f'id{i}',f'C{i}':seq,f'D{i}':'+',f'E{i}':'16'},'formulas':{}})
    other={'sheet':'对比方法','rows':[{'row':1,'cells':{'A1':'模型','B1':'序列','C1':'A. baumannii 32 μg/mL'},'formulas':{}},{'row':2,'cells':{'A2':'Comparator','B2':seqs[0],'C2':'-','R2':'ß'},'formulas':{}}]}
    sequences,observations,sources,excluded,loads=build_tables([{'sheet':'R0-5','rows':rows},other],'plus')
    assert len(sequences)==6 and len(observations)==36 and len(sources)==7
    conflict=observations[(observations.peptide_id==peptide_id(seqs[0]))&(observations.strain_id=='Ab_WT')].iloc[0]
    assert conflict.status=='quarantined' and conflict.source_row_count==2
    assert len(json.loads(conflict.provenance))==2
    assert 'unmapped_nonempty_cell' in set(excluded.reason)
    exact=observations[(observations.peptide_id==peptide_id(seqs[1]))&(observations.strain_id=='Ab_WT')].iloc[0]
    assert exact.status=='usable' and exact.exact_log2==4 and exact.constraint_count==2
    assert (observations[observations.strain_id!='Ab_WT'].status=='missing').all()
    for _,group in observations.groupby('peptide_id'):assert group.fold.nunique()==1 and group.cluster_id.nunique()==1


def test_cli_requires_explicit_symbol():
    result=subprocess.run([sys.executable,'-m','mdramp.calibration_data','--input','absent.xlsx','--output-dir','unused'],capture_output=True,text=True)
    assert result.returncode!=0 and '--positive-symbol' in result.stderr

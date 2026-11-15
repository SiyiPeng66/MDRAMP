from pathlib import Path
import json
import random
from zipfile import ZipFile
from xml.sax.saxutils import escape
import pandas as pd
from src.data.prepare_pairs import NearIdentity,split_peptides,prepare,AA
from src.data.cluster_split import length_aware_identity


def test_compiled_identity_matches_existing_recurrence():
    rng=random.Random(11)
    with NearIdentity() as close:
        for _ in range(150):
            left=''.join(rng.choices(AA,k=rng.randint(10,50)))
            right=list(left)
            for _ in range(rng.randint(0,7)):
                position=rng.randrange(len(right));event=rng.randrange(3)
                if event==0:right[position]=rng.choice(AA)
                elif event==1 and len(right)<50:right.insert(position,rng.choice(AA))
                elif len(right)>10:right.pop(position)
            right=''.join(right)
            assert close(left,right)==(length_aware_identity(left,right)>=.9)


def test_transitive_component_and_public_conflict_quarantine():
    seqs=['ACDEFGHIKL','YCDEFGHIKL','YWDEFGHIKL']
    public=pd.DataFrame({'sequence':[seqs[0],seqs[2]],'cluster_id':['a','b'],'split':['train','test']})
    result,audit=split_peptides(seqs,public)
    assert result.cluster_id.nunique()==1
    assert set(result.split)=={'quarantine'}
    assert audit['pair_graph_edges']>=2


def write_dictionary(path,rows):
    body=''.join(f'<row r="{i}"><c r="A{i}" t="inlineStr"><is><t>{escape(name)}</t></is></c><c r="B{i}" t="inlineStr"><is><t>{seq}</t></is></c></row>' for i,(name,seq) in enumerate(rows,1))
    with ZipFile(path,'w') as archive:
        archive.writestr('xl/worksheets/sheet1.xml','<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'+body+'</sheetData></worksheet>')


def test_preparation_provenance_balance_and_conflicts(tmp_path):
    data=tmp_path/'otherModels/TPepPro/data';data.mkdir(parents=True)
    prep=tmp_path/'outputs/public_prepared';prep.mkdir(parents=True)
    rng=random.Random(1);seqs=[''.join(rng.choices(AA,k=14)) for _ in range(6)]
    pd.DataFrame({'sequence':seqs,'cluster_id':[f'c{i}' for i in range(6)],'split':['train','train','validation','validation','test','test']}).to_csv(prep/'public_sequences.csv',index=False)
    write_dictionary(data/'receptor(14374)-peptide(9594)_dictionary.xlsx',[(f'p{i}',seq) for i,seq in enumerate(seqs)]+[('t1',AA*3),('t2',AA[::-1]*3)])
    lines=[f't1\tp{i}\t1\nt2\tp{i}\t0\n' for i in range(6)]
    (data/'receptor-peptide.actions.tsv').write_text(''.join(lines)+'t1\tp0\t0\nt1\tp2\t1\n')
    output=tmp_path/'new_pairs';manifest=prepare(tmp_path,output)
    assert manifest['counts']['contradictory_sequence_pairs']==1
    assert manifest['source_release']=='local-snapshot' and manifest['official_commit'] is None
    assert len(pd.read_csv(output/'source_actions.csv'))==14
    for filename in ['train_pairs.csv','val_pairs.csv','test_pairs.csv']:
        pairs=pd.read_csv(output/filename)
        assert (pairs.label==1).sum()==(pairs.label==0).sum()>0
        assert pairs.loc[pairs.label==0,'explicit_label'].isna().all()
    assert pd.read_csv(output/'all_pairs.csv').source_record_count.max()==2
    quarantine=pd.read_csv(output/'quarantine.csv')
    assert 'contradictory_sequence_pair_labels' in set(quarantine.quarantine_reason)

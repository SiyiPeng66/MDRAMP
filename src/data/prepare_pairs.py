"""Version a new public TPepPro cold-peptide dataset from the local source snapshot.

This is not recovery of the manuscript's historical 31,248-pair training ledger.
Operational background pairs are not experimentally confirmed non-interactions.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from xml.etree import ElementTree as ET
from zipfile import ZipFile
import numpy as np
import pandas as pd
from src.utils.hash import sha256_file
from src.data.cluster_split import length_aware_identity

AA='ACDEFGHIKLMNPQRSTVWY'
C_SOURCE=r'''
#include <string.h>
int close90(const char *a,const char *b){
 int n=strlen(a),m=strlen(b),pm[51],pl[51],cm[51],cl[51];
 if(n>50||m>50||!n||!m)return 0;
 for(int j=0;j<=m;j++){pm[j]=0;pl[j]=j;}
 for(int i=1;i<=n;i++){
  cm[0]=0;cl[0]=i;
  for(int j=1;j<=m;j++){
   int bm=pm[j-1]+(a[i-1]==b[j-1]),bl=pl[j-1]+1;
   if(pm[j]>bm||(pm[j]==bm&&pl[j]+1<bl)){bm=pm[j];bl=pl[j]+1;}
   if(cm[j-1]>bm||(cm[j-1]==bm&&cl[j-1]+1<bl)){bm=cm[j-1];bl=cl[j-1]+1;}
   cm[j]=bm;cl[j]=bl;
  }
  memcpy(pm,cm,sizeof(pm));memcpy(pl,cl,sizeof(pl));
 }
 return 10*pm[m]>=9*pl[m];
}
'''


def read_dictionary(path):
    """Read the source XLSX ID/sequence columns without an Excel dependency."""
    ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    rows=[]
    with ZipFile(path) as archive:
        strings=[]
        if 'xl/sharedStrings.xml' in archive.namelist():
            strings=[''.join(t.text or '' for t in item.findall('.//m:t',ns)) for item in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('m:si',ns)]
        for name in sorted(n for n in archive.namelist() if n.startswith('xl/worksheets/sheet') and n.endswith('.xml')):
            tree=ET.fromstring(archive.read(name))
            for row in tree.findall('m:sheetData/m:row',ns):
                cells={}
                for cell in row:
                    column=''.join(c for c in cell.get('r','') if c.isalpha())
                    value=cell.findtext('m:v',default='',namespaces=ns)
                    if cell.get('t')=='s':value=strings[int(value)]
                    elif cell.get('t')=='inlineStr':value=''.join(t.text or '' for t in cell.findall('.//m:t',ns))
                    cells[column]=value
                if cells.get('A') or cells.get('B'):
                    rows.append({'source_id':cells.get('A','').strip(),'raw_sequence':cells.get('B',''),
                                 'sequence':''.join(cells.get('B','').split()).upper(),
                                 'dictionary_sheet':name,'dictionary_row':int(row.get('r','0'))})
    return pd.DataFrame(rows)


def stable_id(prefix,sequence):return prefix+hashlib.sha256(sequence.encode()).hexdigest()[:20]


class NearIdentity:
    """Compiled exact existing DP recurrence; safe filters never omit a 90% pair."""
    def __enter__(self):
        compiler=shutil.which('cc') or shutil.which('clang') or shutil.which('gcc')
        if not compiler:raise RuntimeError('A C compiler is required for audited all-pairs cold-peptide clustering')
        self.temp=tempfile.TemporaryDirectory(prefix='mdramp-pair-align-')
        path=Path(self.temp.name);(path/'identity.c').write_text(C_SOURCE)
        result=subprocess.run([compiler,'-O3','-shared','-fPIC',str(path/'identity.c'),'-o',str(path/'identity.so')],capture_output=True,text=True)
        if result.returncode:raise RuntimeError(result.stderr)
        self.library=ctypes.CDLL(str(path/'identity.so'))
        self.function=self.library.close90;self.function.argtypes=[ctypes.c_char_p,ctypes.c_char_p];self.function.restype=ctypes.c_int
        return self
    def __exit__(self,*args):self.temp.cleanup()
    def __call__(self,left,right):return bool(self.function(left.encode(),right.encode()))


def counts(sequences):return np.asarray([[s.count(a) for a in AA] for s in sequences],dtype=np.int16).reshape(-1,20)

def candidates(sequence,reference_counts,lengths):
    length=len(sequence); maximum=np.maximum(lengths,length)
    # Alignment length >= max lengths and matches <= amino-acid multiset intersection.
    mask=10*np.minimum(lengths,length)>=9*maximum
    indices=np.flatnonzero(mask)
    common=np.minimum(reference_counts[indices],counts([sequence])[0]).sum(axis=1)
    return indices[10*common>=9*maximum[indices]]


def split_peptides(sequences,public,seed=11):
    sequences=sorted(set(sequences));n=len(sequences);parent=list(range(n))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):
        i,j=find(i),find(j)
        if i!=j:parent[max(i,j)]=min(i,j)
    vectors=counts(sequences);lengths=np.array(list(map(len,sequences)))
    public=public[['sequence','cluster_id','split']].drop_duplicates()
    if public.sequence.duplicated().any() or not set(public.split).issubset({'train','validation','test'}):
        raise ValueError('Public split mapping must be unique and valid')
    public_seqs=public.sequence.astype(str).tolist();pc=counts(public_seqs);pl=np.array(list(map(len,public_seqs)))
    inherited=[set() for _ in sequences];source_clusters=[set() for _ in sequences]
    comparisons=0;edges=0
    with NearIdentity() as close:
        for i,sequence in enumerate(sequences):
            for j in candidates(sequence,vectors,lengths):
                if j<=i:continue
                comparisons+=1
                if close(sequence,sequences[j]):union(i,int(j));edges+=1
            for j in candidates(sequence,pc,pl):
                if close(sequence,public_seqs[j]):
                    inherited[i].add(str(public.iloc[j]['split']));source_clusters[i].add(str(public.iloc[j]['cluster_id']))
            if (i+1)%250==0:print(f'cold-peptide audit {i+1}/{n}',flush=True)
    groups=defaultdict(list)
    for i in range(n):groups[find(i)].append(i)
    rows=[]
    for members in groups.values():
        values=sorted(sequences[i] for i in members);cluster=stable_id('PC_', '\n'.join(values))
        splits=set().union(*(inherited[i] for i in members))
        if len(splits)>1:split='quarantine';reason='conflicting_public_split_homologues'
        elif splits:split=next(iter(splits));reason='inherited_public_homology'
        else:
            number=int(hashlib.sha256(f'{seed}:{cluster}'.encode()).hexdigest()[:16],16)/2**64
            split='train' if number<.8 else ('validation' if number<.9 else 'test');reason='deterministic_component_hash_80_10_10'
        for i in members:
            rows.append({'sequence':sequences[i],'peptide_id':stable_id('pep_',sequences[i]),'cluster_id':cluster,
                         'split':split,'split_reason':reason,'inherited_public_clusters':json.dumps(sorted(source_clusters[i]))})
    return pd.DataFrame(rows),{'component_count':len(groups),'pair_graph_comparisons_after_safe_prefilter':comparisons,'pair_graph_edges':edges,
                               'identity_definition':'exact existing length_aware_identity recurrence; match-count maximum then shortest alignment; >=0.90',
                               'algorithm_sha256':hashlib.sha256(C_SOURCE.encode()).hexdigest(),'seed':seed,
                               'public_homology_inheritance':'all >=90% public matches; incompatible split components quarantined'}


def prepare(root,output_dir,seed=11):
    root=Path(root).resolve();output=Path(output_dir).absolute()
    if output.exists():raise FileExistsError(output)
    actions=root/'otherModels/TPepPro/data/receptor-peptide.actions.tsv'
    dictionary=root/'otherModels/TPepPro/data/receptor(14374)-peptide(9594)_dictionary.xlsx'
    public_path=root/'outputs/public_prepared/public_sequences.csv'
    if not public_path.exists():public_path=root/'outputs/processed/public_cluster_split.csv'
    inputs={str(p.relative_to(root)):sha256_file(p) for p in (actions,dictionary,public_path)}
    dictionary_rows=read_dictionary(dictionary);lookup={};ambiguous=set()
    for identifier,group in dictionary_rows.groupby('source_id',sort=False):
        seqs=set(group.sequence)
        if len(seqs)!=1:ambiguous.add(identifier)
        else:lookup[identifier]=next(iter(seqs))
    raw=pd.read_csv(actions,sep='\t',header=None,names=['source_target_id','source_peptide_id','source_label'],dtype=str,keep_default_na=False)
    raw.insert(0,'source_action_row',np.arange(1,len(raw)+1))
    raw['target_sequence']=raw.source_target_id.map(lookup);raw['sequence']=raw.source_peptide_id.map(lookup)
    quarantine=[];eligible=[]
    for row in raw.to_dict('records'):
        reason=None
        if row['source_target_id'] in ambiguous or row['source_peptide_id'] in ambiguous:reason='ambiguous_dictionary_id'
        elif not isinstance(row['sequence'],str) or not isinstance(row['target_sequence'],str):reason='missing_dictionary_id'
        elif row['source_label'] not in {'0','1'}:reason='invalid_source_label'
        elif not 10<=len(row['sequence'])<=50 or set(row['sequence'])-set(AA):reason='ineligible_peptide_sequence'
        elif not row['target_sequence'] or set(row['target_sequence'])-set(AA):reason='nonstandard_or_empty_target_sequence'
        if reason:quarantine.append({**row,'quarantine_reason':reason})
        else:eligible.append(row)
    eligible=pd.DataFrame(eligible)
    if eligible.empty:raise ValueError('No valid public pairs')
    pairs=[];conflicts=0
    for (peptide,protein),group in eligible.groupby(['sequence','target_sequence'],sort=True):
        if group.source_label.nunique()>1:
            conflicts+=1
            quarantine.extend({**row,'quarantine_reason':'contradictory_sequence_pair_labels'} for row in group.to_dict('records'))
            continue
        label=int(group.source_label.iloc[0]);pairid=stable_id('pair_',peptide+'|'+protein)
        pairs.append({'pair_id':pairid,'peptide_id':stable_id('pep_',peptide),'target_id':stable_id('protein_',protein),
                      'sequence':peptide,'target_sequence':protein,'label':label,'label_type':'positive' if label else 'background',
                      'explicit_label':1. if label else np.nan,
                      'label_origin':'TPepPro_source_positive' if label else 'TPepPro_operational_background_not_confirmed_noninteraction',
                      'source_action_rows':json.dumps(group.source_action_row.tolist()),
                      'source_peptide_ids':json.dumps(sorted(set(group.source_peptide_id))),
                      'source_target_ids':json.dumps(sorted(set(group.source_target_id))),
                      'source_record_count':len(group)})
    all_pairs=pd.DataFrame(pairs)
    public=pd.read_csv(public_path,dtype={'sequence':str,'cluster_id':str,'split':str})
    peptide_table,audit=split_peptides(all_pairs.sequence,public,seed)
    all_pairs=all_pairs.merge(peptide_table[['peptide_id','cluster_id','split','split_reason']],on='peptide_id',validate='many_to_one')
    all_pairs['selected_for_split']=False;all_pairs['exclusion_reason']=''
    all_pairs.loc[all_pairs.split.eq('quarantine'),'exclusion_reason']='conflicting_public_split_homologues'
    for split in ('train','validation','test'):
        subset=all_pairs[all_pairs.split.eq(split)];budget=min((subset.label==0).sum(),(subset.label==1).sum())
        if budget==0:raise ValueError(f'Cannot form nonempty balanced {split} split')
        for label in (0,1):
            indices=subset[subset.label.eq(label)].sort_values('pair_id').index
            all_pairs.loc[indices[:budget],'selected_for_split']=True
            all_pairs.loc[indices[budget:],'exclusion_reason']='surplus_for_split_1_to_1_balance'
    for row in all_pairs[~all_pairs.selected_for_split].to_dict('records'):
        for index in json.loads(row['source_action_rows']):
            original=raw.iloc[index-1].to_dict()
            quarantine.append({**original,'pair_id':row['pair_id'],'quarantine_reason':row['exclusion_reason']})
    proteins=all_pairs[['target_id','target_sequence']].drop_duplicates().sort_values('target_id')
    proteins['length']=proteins.target_sequence.str.len()
    peptide_table['length']=peptide_table.sequence.str.len()
    output.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix='.'+output.name+'-',dir=output.parent))
    try:
        raw.to_csv(temp/'source_actions.csv',index=False);dictionary_rows.to_csv(temp/'source_dictionary.csv',index=False)
        all_pairs.sort_values('pair_id').to_csv(temp/'all_pairs.csv',index=False)
        peptide_table.sort_values('peptide_id').to_csv(temp/'peptide_sequences.csv',index=False)
        proteins.to_csv(temp/'proteins.csv',index=False)
        pd.DataFrame(quarantine).to_csv(temp/'quarantine.csv',index=False)
        split_counts={}
        for split,filename in [('train','train_pairs.csv'),('validation','val_pairs.csv'),('test','test_pairs.csv')]:
            selected=all_pairs[all_pairs.split.eq(split)&all_pairs.selected_for_split].sort_values('pair_id')
            selected.to_csv(temp/filename,index=False)
            split_counts[split]={'pairs':len(selected),'positive':int(selected.label.sum()),'background':int((selected.label==0).sum()),'peptides':selected.peptide_id.nunique(),'proteins':selected.target_id.nunique()}
        for name,digest in inputs.items():
            if sha256_file(root/name)!=digest:raise RuntimeError('Source changed during preparation')
        manifest={'schema_version':1,'dataset_version':'public_tpeppro_cold_peptide_v1',
                  'dataset_role':'public_pair_candidates_before_teacher_screening','v6_pair_curation_complete':False,
                  'teacher_screening':{'status':'not_performed','high_confidence_threshold':0.8},
                  'created_at':datetime.now(timezone.utc).isoformat(),'source_release':'local-snapshot','official_commit':None,
                  'source_release_date_known':False,'historical_31248_reproduction':False,
                  'input_sha256':inputs,'algorithm':audit,'split_counts':split_counts,
                  'counts':{'source_rows':len(raw),'dictionary_rows':len(dictionary_rows),'eligible_raw_rows':len(eligible),'unique_nonconflicting_pairs':len(all_pairs),'contradictory_sequence_pairs':conflicts,'quarantined_source_rows':len(quarantine),'peptides':len(peptide_table),'proteins':len(proteins)},
                  'quarantine_counts':pd.Series([r['quarantine_reason'] for r in quarantine]).value_counts().to_dict(),
                  'limitations':['New versioned public training dataset, not historical manuscript ledger.','Source label 0 means operational background, not confirmed non-binding.','Cold-peptide only: target proteins may overlap across splits.','Public expert pretraining can expose inherited-public test/validation sequences only in their corresponding held-out roles; maintain upstream split isolation.','Report test performance separately; training-set results are not generalization evidence.'],
                  'output_sha256':{p.name:sha256_file(p) for p in sorted(temp.iterdir())}}
        (temp/'manifest.json').write_text(json.dumps(manifest,indent=2))
        if output.exists():raise FileExistsError(output)
        os.rename(temp,output);temp=None
        return manifest
    finally:
        if temp is not None:shutil.rmtree(temp)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--seed',type=int,default=11)
    a=p.parse_args();print(json.dumps(prepare(a.root,a.output_dir,a.seed),indent=2))
if __name__=='__main__':main()

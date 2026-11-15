"""Freeze label-free panel-stability cohorts and assemble real ESM3 probe vectors."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner
from mdramp.protein_panel_build import sha,dump


def sequence_id(sequence):return 'probe_'+hashlib.sha256(sequence.encode()).hexdigest()[:24]


def make_identity():
    aligner=PairwiseAligner(mode='global',match_score=1000,mismatch_score=0,open_gap_score=-1,extend_gap_score=-1)
    def identity(a,b):
        if not a or not b:return 0.
        if min(len(a),len(b))/max(len(a),len(b))<.9:return 0.
        score=aligner.score(a,b);matches=math.ceil(score/1000)
        gaps=1000*matches-score
        return matches/((len(a)+len(b)+gaps)/2)
    return identity


def prepare(a):
    out=Path(a.output_dir).absolute()
    if out.exists():raise ValueError('Probe cohorts are immutable; use a new output directory')
    protocol=json.loads(Path(a.protocol).read_text())
    sources={str(Path(p).absolute()):sha(p) for p in [a.public,a.generated,a.private_sequences,a.protocol,__file__]}
    public=pd.read_csv(a.public,usecols=['sequence','cluster_id','split'],dtype=str)
    private=pd.read_csv(a.private_sequences,usecols=['sequence'],dtype=str)
    generated=pd.read_csv(a.generated,usecols=['sequence'],dtype=str)
    if public.sequence.duplicated().any() or private.sequence.duplicated().any():raise ValueError('Unique sequence sources required')
    if (public.groupby('cluster_id').split.nunique()>1).any():raise ValueError('Public cluster partition conflict')
    valid=generated.sequence.fillna('').str.fullmatch('[ACDEFGHIKLMNPQRSTVWY]{10,50}')
    generated_audit=generated.copy();generated_audit['valid_sequence']=valid
    gen=sorted(set(generated.loc[valid,'sequence']))
    if len(gen)<30:raise ValueError('Generated stability cohort too small')
    if not private.sequence.str.fullmatch('[ACDEFGHIKLMNPQRSTVWY]{10,50}').all():raise ValueError('Invalid private audit sequence')
    identity=make_identity();n=protocol['public_cohort_size'];selections={};exclusions=[]
    for split,seed,forbidden in [('validation',protocol['seed'],gen),('test',protocol['seed']+1,None)]:
        if split=='test':forbidden=gen+selections['validation']+private.sequence.tolist()
        candidates=public[public.split.eq(split)].copy()
        candidates['sequence_hash']=candidates.sequence.map(sequence_id)
        representatives=candidates.sort_values('sequence_hash').drop_duplicates('cluster_id').sort_values('cluster_id')
        order=np.random.default_rng(seed).permutation(len(representatives));chosen=[]
        for i in order:
            row=representatives.iloc[i];seq=row.sequence
            if any(identity(seq,s)>=.9 for s in forbidden+chosen):
                exclusions.append(dict(split=split,sequence=seq,reason='90pct_sequence_overlap_with_forbidden_or_selected'));continue
            chosen.append(seq)
            if len(chosen)==n:break
        if len(chosen)!=n:raise ValueError('Insufficient isolated public sequence clusters')
        selections[split]=chosen
    cohorts={'generated_development':gen,'public_validation_development':selections['validation'],
             'public_test_confirmation':selections['test'],'private_230_audit':sorted(private.sequence)}
    sequences=sorted(set(s for values in cohorts.values() for s in values),key=sequence_id)
    parent=list(range(len(sequences)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,s in enumerate(sequences):
        for j in range(i):
            if identity(s,sequences[j])>=.9:
                x,y=find(i),find(j)
                if x!=y:parent[max(x,y)]=min(x,y)
    clusters={s:'cluster_'+sequence_id(sequences[find(i)])[6:] for i,s in enumerate(sequences)}
    members=pd.DataFrame([dict(cohort=c,record_id=sequence_id(s),sequence=s,cluster_id=clusters[s]) for c,values in cohorts.items() for s in values])
    test_clusters=set(members.loc[members.cohort.eq('public_test_confirmation'),'cluster_id'])
    if test_clusters&set(members.loc[~members.cohort.eq('public_test_confirmation'),'cluster_id']):raise ValueError('Transitive confirmation-cohort homology overlap')
    probes=pd.DataFrame([dict(record_id=sequence_id(s),sequence=s,cluster_id=clusters[s]) for s in sequences])
    if probes.record_id.duplicated().any():raise ValueError('Probe hash collision')
    # Existing real embeddings cover public and historical sequences; generated-only sequences require new encoding.
    available=set(public.sequence)|set(private.sequence)
    missing=probes[~probes.sequence.isin(available)][['record_id','sequence']]
    out.mkdir(parents=True);probes.to_csv(out/'probes.csv',index=False);members.to_csv(out/'cohorts.csv',index=False)
    missing.to_csv(out/'missing_embeddings.csv',index=False);generated_audit.to_csv(out/'generated_input_audit.csv',index=False)
    pd.DataFrame(exclusions,columns=['split','sequence','reason']).to_csv(out/'sampling_exclusions.csv',index=False)
    if any(sha(p)!=h for p,h in sources.items()):raise ValueError('Probe source changed')
    dump(out/'manifest.json',dict(status='frozen_label_free_cohorts',input_sha256=sources,protocol=protocol,
        counts=members.groupby('cohort').size().to_dict(),unique_probe_count=len(probes),missing_embedding_count=len(missing),
        cluster_counts=members.groupby('cohort').cluster_id.nunique().to_dict(),
        no_mic_values_loaded=True,output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps({'cohorts':{k:len(v) for k,v in cohorts.items()},'missing_embeddings':len(missing)}))


def assemble(a):
    out=Path(a.output_dir).absolute()
    if out.exists():raise ValueError('Never overwrite assembled probe embeddings')
    probes=pd.read_csv(a.probes);cfg=json.loads(Path(a.sources).read_text());weight_hash=sha(cfg['esm_checkpoint'])
    wanted=set(probes.sequence);vectors={};origins={};hashes={str(Path(p).absolute()):sha(p) for p in [a.probes,a.sources,__file__,cfg['esm_checkpoint']]}
    for source in cfg['sources']:
        paths={k:Path(source[k]).absolute() for k in ['npz','metadata','provenance']}
        provenance=json.loads(paths['provenance'].read_text())
        if provenance.get('encoder')!='ESM3' or provenance.get('status')!='complete' or provenance.get('embedding_dimension')!=1536 or provenance.get('binding',{}).get('checkpoint_sha256')!=weight_hash:raise ValueError('Complete matching real ESM3 source required')
        for k in ['npz','metadata']:
            if provenance['output_sha256'].get(paths[k].name)!=sha(paths[k]):raise ValueError('Embedding source hash mismatch')
        hashes.update({str(p):sha(p) for p in paths.values()})
        meta=pd.read_csv(paths['metadata'])
        with np.load(paths['npz'],allow_pickle=False) as z:global_vectors=z['global_embeddings']
        if global_vectors.shape!=(len(meta),1536) or not np.isfinite(global_vectors).all() or meta.record_id.duplicated().any():raise ValueError('Invalid embedding source shape/IDs')
        index=pd.to_numeric(meta.global_index).to_numpy(float)
        if not np.isfinite(index).all() or not np.equal(index,index.astype(int)).all() or len(set(index))!=len(index) or (index<0).any() or (index>=len(meta)).any():raise ValueError('Invalid global index')
        for row in meta[meta.sequence.isin(wanted)].itertuples():
            v=global_vectors[int(row.global_index)]
            if row.sequence in vectors and not np.allclose(vectors[row.sequence],v,atol=2e-5,rtol=2e-5):raise ValueError('Different actual ESM vectors for identical source sequence')
            if row.sequence not in vectors:vectors[row.sequence]=v.copy();origins[row.sequence]=str(paths['provenance'])
    if set(vectors)!=wanted:raise ValueError('Missing real probe embeddings')
    matrix=np.stack([vectors[s] for s in probes.sequence]);out.mkdir(parents=True)
    np.savez_compressed(out/'probes.npz',global_embeddings=matrix)
    meta=probes[['record_id','sequence']].copy();meta['length']=meta.sequence.str.len();meta['global_index']=np.arange(len(meta));meta['source_provenance']=meta.sequence.map(origins)
    meta.to_csv(out/'probes_metadata.csv',index=False)
    if any(sha(p)!=h for p,h in hashes.items()):raise ValueError('Embedding input changed')
    dump(out/'probes_metadata.json',dict(status='complete',encoder='ESM3',embedding_dimension=1536,record_count=len(meta),
        binding={'checkpoint_sha256':weight_hash,'input_sha256':sha(a.probes),'assembly':'actual_source_global_vectors_only_no_surrogate'},
        input_sha256=hashes,output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(f'Assembled {len(meta)} actual ESM3 probe vectors')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    q=sub.add_parser('prepare')
    for n in ['public','generated','private-sequences','protocol','output-dir']:q.add_argument('--'+n,required=True)
    q=sub.add_parser('assemble')
    for n in ['probes','sources','output-dir']:q.add_argument('--'+n,required=True)
    a=p.parse_args();prepare(a) if a.action=='prepare' else assemble(a)


if __name__=='__main__':main()

"""Version an official UniProtKB operational-unlabelled, length-matched background."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import time
import pandas as pd
import requests
from src.data.cluster_split import length_aware_identity
from src.utils.hash import sha256_file

API='https://rest.uniprot.org/uniprotkb/search'
FIELDS='accession,sequence,length,protein_name,keyword,reviewed,cc_function'
EXCLUSION=re.compile(r'antimicrob|anti.?bacter|antifung|defensin|bacteriocin|toxin|venom|\bAMPs?\b',re.I)
AA=set('ACDEFGHIKLMNPQRSTVWY')


def grams(seq):return Counter(seq[i:i+3] for i in range(len(seq)-2))


class HomologyIndex:
    """Lossless q-gram candidate filter followed by manuscript global identity."""
    def __init__(self,sequences):
        self.sequences=list(dict.fromkeys(sequences));self.exact=set(self.sequences);self.index=defaultdict(list)
        self.lengths=[len(s) for s in self.sequences]
        for i,seq in enumerate(self.sequences):
            for gram,count in grams(seq).items():self.index[gram].append((i,count))
    def matches(self,seq):
        if seq in self.exact:return True
        counts=defaultdict(int);n=len(seq)
        for gram,count in grams(seq).items():
            for i,other_count in self.index.get(gram,()):
                m=self.lengths[i]
                if min(n,m)/max(n,m)>=.9:counts[i]+=min(count,other_count)
        for i,shared in counts.items():
            m=self.lengths[i];k=math.floor(max(n,m)/9+1e-9)
            if shared>=max(n,m)-2-3*k and length_aware_identity(seq,self.sequences[i])>=.9:return True
        return False


def eligible(row,index):
    seq=str(row.get('Sequence','')).strip().upper()
    if not 10<=len(seq)<=50 or not set(seq)<=AA:return None,'sequence'
    annotation=' '.join(str(row.get(k,'')) for k in ('Protein names','Keywords','Function [CC]'))
    if EXCLUSION.search(annotation):return None,'annotation'
    if index.matches(seq):return None,'public_homology'
    return {'sequence':seq,'length':len(seq),'accession':row['Entry'],'protein_name':row.get('Protein names',''),'keywords':row.get('Keywords',''),'reviewed':row.get('Reviewed',''),'function':row.get('Function [CC]',''),'label_type':'unlabelled','source':'UniProtKB'},None


def get_page(session,url,directory,number,release):
    raw=directory/f'page_{number:05d}.tsv';meta=raw.with_suffix('.json')
    if raw.exists() or meta.exists():
        if not raw.exists() or not meta.exists():raise ValueError(f'Incomplete page archive: {raw}')
        info=json.loads(meta.read_text())
        if info['url']!=url or sha256_file(raw)!=info['sha256']:raise ValueError('Archived page identity mismatch')
    else:
        for attempt in range(6):
            try:
                response=session.get(url,timeout=(20,120));response.raise_for_status();break
            except requests.RequestException:
                if attempt==5:raise
                time.sleep(min(2**attempt,20))
        release_value=response.headers.get('X-UniProt-Release')
        if not release_value:raise ValueError('Missing UniProt release header')
        info={'url':url,'retrieved_at':datetime.now(timezone.utc).isoformat(),'headers':dict(response.headers),'release':release_value,'next':response.links.get('next',{}).get('url')}
        raw.write_bytes(response.content);info['sha256']=sha256_file(raw);meta.write_text(json.dumps(info,indent=2))
    if release is not None and info['release']!=release:raise ValueError('UniProt release changed across pages')
    return pd.read_csv(raw,sep='\t',keep_default_na=False),info


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--public',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--seed',type=int,default=20261111);p.add_argument('--minimum-candidates',type=int,default=30000)
    p.add_argument('--oversample',type=float,default=2.0);p.add_argument('--resume',action='store_true');p.add_argument('--maximum-pages-per-length',type=int,default=1000)
    a=p.parse_args();directory=Path(a.output_dir)
    if directory.exists() and not a.resume:raise ValueError('Output directory exists; explicit --resume required')
    directory.mkdir(parents=True,exist_ok=True)
    binding={'public_sha256':sha256_file(a.public),'seed':a.seed,'oversample':a.oversample,'minimum_candidates':a.minimum_candidates,'fields':FIELDS,'identity':.9,'code_sha256':sha256_file(__file__)}
    bp=directory/'binding.json'
    if bp.exists() and json.loads(bp.read_text())!=binding:raise ValueError('Background resume input/configuration mismatch')
    if not bp.exists():bp.write_text(json.dumps(binding,indent=2))
    if (directory/'manifest.json').exists():
        manifest=json.loads((directory/'manifest.json').read_text())
        for filename,digest in manifest['outputs'].items():
            if sha256_file(directory/filename)!=digest:raise ValueError('Completed output changed')
        print('Completed background already verified',flush=True);return
    public=pd.read_csv(a.public);required=Counter(public[public.split=='train'].sequence.str.len());index=HomologyIndex(public.sequence.tolist())
    session=requests.Session();session.headers['User-Agent']='MDRAMP-public-background/1.0 (reproducible research)'
    all_rows={};release=None;pages=[];rejections=Counter();shortages={}
    # Length stratification prevents abundant long proteins exhausting short-peptide quotas.
    lengths=list(sorted(required))+[n for n in range(10,51) if n not in required]
    for length in lengths:
        target=max(math.ceil(required.get(length,0)*a.oversample),100)
        if length not in required and len(all_rows)>=a.minimum_candidates:break
        query=f'length:[{length} TO {length}]'
        url=requests.Request('GET',API,params={'query':query,'format':'tsv','fields':FIELDS,'size':500,'sort':'accession asc'}).prepare().url
        folder=directory/'raw'/f'length_{length}';folder.mkdir(parents=True,exist_ok=True)
        count=0
        for number in range(a.maximum_pages_per_length):
            frame,info=get_page(session,url,folder,number,release);release=info['release'];pages.append({'path':str(folder.relative_to(directory)/f'page_{number:05d}.tsv'),**info})
            for row in frame.to_dict('records'):
                item,reason=eligible(row,index)
                if reason:rejections[reason]+=1;continue
                if item['sequence'] in all_rows:rejections['duplicate']+=1;continue
                all_rows[item['sequence']]=item;count+=1
            print(f'length={length} page={number+1} unique_eligible={count} target={target} total={len(all_rows)} release={release}',flush=True)
            if count>=target or not info['next']:break
            url=info['next']
        if count<required.get(length,0):shortages[length]={'required':required[length],'available':count}
    candidates=pd.DataFrame(all_rows.values()).sort_values(['length','accession']).reset_index(drop=True)
    candidates.to_csv(directory/'candidates.csv',index=False)
    report={'schema_version':1,'binding':binding,'release':release,'retrieved_at':datetime.now(timezone.utc).isoformat(),'semantics':'Operational unlabelled; absence of selected known AMP annotations is not experimental proof of inactivity. Reviewed and unreviewed UniProtKB records are allowed.','required_by_length':dict(required),'candidate_counts':candidates.length.value_counts().sort_index().to_dict(),'shortages':shortages,'rejections':dict(rejections),'pages':pages,'official_documentation':'https://www.uniprot.org/help/api_queries'}
    if shortages or len(candidates)<a.minimum_candidates:
        (directory/'incomplete_report.json').write_text(json.dumps(report,indent=2));raise ValueError(f'Insufficient background: shortages={shortages}, candidates={len(candidates)}')
    selected=[]
    for length,count in sorted(required.items()):selected.append(candidates[candidates.length==length].sample(n=count,random_state=a.seed+length))
    matched=pd.concat(selected,ignore_index=True);matched.insert(0,'peptide_id',['uniprot_'+x for x in matched.accession]);matched['split']='train'
    matched.to_csv(directory/'background_matched.csv',index=False)
    if sha256_file(a.public)!=binding['public_sha256']:raise ValueError('Public input changed')
    report['outputs']={name:sha256_file(directory/name) for name in ('candidates.csv','background_matched.csv')}
    (directory/'manifest.json').write_text(json.dumps(report,indent=2));print(f'Background complete: candidates={len(candidates)} matched={len(matched)}',flush=True)


if __name__=='__main__':main()

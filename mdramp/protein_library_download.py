"""Archive full official reference proteomes; no target-panel selection."""
import argparse,json,hashlib,re,time,fcntl
from pathlib import Path
from urllib.request import urlopen,Request
from urllib.parse import urlencode
from datetime import datetime,timezone
SPECIES={'Ab':470,'Sa':1280,'Kp':573,'Pa':287}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rank(row):
 report=row.get('proteomeCompletenessReport',{})
 return (row.get('genomeAssembly',{}).get('level') not in ('full','complete'),report.get('cpdReport',{}).get('status')!='Standard',-report.get('buscoReport',{}).get('score',-1),row['id'])
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',required=True);a=p.parse_args();out=Path(a.output_dir).absolute();raw=out/'raw';raw.mkdir(parents=True,exist_ok=True)
 with (out/'.download.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (out/'download_manifest.json').exists():raise ValueError('Completed library already exists')
  requests=[];releases=set()
  def fetch(url,name):
   path=raw/name;meta=path.with_suffix(path.suffix+'.request.json')
   if path.exists() and meta.exists():
    m=json.loads(meta.read_text())
    if m['url']!=url or m['sha256']!=sha(path):raise ValueError('Cache integrity mismatch')
   else:
    for attempt in range(5):
     try:
      with urlopen(Request(url,headers={'User-Agent':'MDRAMP-reference-library/1'}),timeout=180) as r:body=r.read();headers=dict(r.headers)
      break
     except Exception:
      if attempt==4:raise
      time.sleep(2**attempt)
    path.write_bytes(body);m={'url':url,'headers':headers,'retrieved_at':datetime.now(timezone.utc).isoformat(),'sha256':sha(path),'path':str(path.relative_to(out))};meta.write_text(json.dumps(m,indent=2))
   headers={k.lower():v for k,v in m['headers'].items()};release=headers.get('x-uniprot-release')
   if not release:raise ValueError('Missing official release header')
   releases.add(release)
   if len(releases)!=1:raise ValueError('Release changed between cached/downloaded artifacts')
   requests.append(m);return path.read_bytes(),headers
  def pages(endpoint,query,prefix):
   url='https://rest.uniprot.org/'+endpoint+'/search?'+urlencode({'query':query,'format':'json','size':500});records=[];total=None;n=0
   while url:
    body,h=fetch(url,f'{prefix}_{n:03d}.json');expected=int(h['x-total-results'])
    if total is not None and total!=expected:raise ValueError('Pagination total changed')
    total=expected;records.extend(json.loads(body)['results']);match=re.search(r'<([^>]+)>; rel="next"',h.get('link',''));url=match.group(1) if match else None;n+=1
   if len(records)!=total:raise ValueError('Incomplete pagination')
   return records
  rules={'species_taxonomy':SPECIES,'selection':'Reference proteomes only; complete/full genome assembly first, CPD Standard first, BUSCO score descending, then proteome ID ascending','not_actual_assay_isolates':True,'no_gene_or_reviewed_filter':True,'implementation_sha256':sha(__file__)}
  rulepath=out/'selection_rules.json'
  if rulepath.exists() and json.loads(rulepath.read_text())!=rules:raise ValueError('Download rules changed')
  rulepath.write_text(json.dumps(rules,indent=2));chosen={}
  for species,taxon in SPECIES.items():
   candidates=pages('proteomes',f'taxonomy_id:{taxon} AND proteome_type:REFERENCE',species+'_proteomes')
   if not candidates or any(x.get('proteomeType')!='Reference proteome' for x in candidates):raise ValueError('No valid reference proteomes')
   selected=sorted(candidates,key=rank)[0];upid=selected['id'];chosen[species]={'proteome_id':upid,'taxonomy':selected['taxonomy'],'strain':selected.get('strain'),'assembly':selected.get('genomeAssembly'),'selection_rank':rank(selected),'candidate_count':len(candidates),'declared_protein_count':selected.get('proteinCount')}
   (out/'chosen_proteomes.inprogress.json').write_text(json.dumps(chosen,indent=2))
   print(f'{species}: {upid} ({len(candidates)} reference candidates)',flush=True)
   records=pages('uniprotkb','proteome:'+upid,species+'_'+upid+'_proteins')
   accessions=[r['primaryAccession'] for r in records]
   if len(set(accessions))!=len(accessions):raise ValueError('Duplicate protein accessions')
   body,_=fetch('https://rest.uniprot.org/uniprotkb/stream?'+urlencode({'query':'proteome:'+upid,'format':'fasta'}),species+'_'+upid+'.fasta')
   fasta_ids=re.findall(r'^>[^|]+\|([^|]+)\|',body.decode(),re.M)
   if len(fasta_ids)!=len(records) or set(fasta_ids)!=set(accessions):raise ValueError('FASTA/JSON accession coverage mismatch')
   chosen[species]['downloaded_protein_count']=len(records)
   chosen[species]['proteome_record_count_matches']=len(records)==selected.get('proteinCount')
   if not chosen[species]['proteome_record_count_matches']:raise ValueError('Official proteome proteinCount differs from full library')
  if sha(__file__)!=rules['implementation_sha256']:raise ValueError('Implementation changed during download')
  manifest={**rules,'status':'complete','uniprot_release':list(releases)[0],'chosen_proteomes':chosen,'requests':requests,'output_sha256':{str(p.relative_to(out)):sha(p) for p in raw.iterdir() if p.is_file()}}
  (out/'download_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(chosen,indent=2))
if __name__=='__main__':main()

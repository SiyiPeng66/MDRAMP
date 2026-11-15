"""Freeze a new, explicitly reference-only functional bacterial protein panel."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import urlopen, Request
from urllib.parse import urlencode
import pandas as pd

SPECIES={'Ab':470,'Sa':1280,'Kp':573,'Pa':287}
GENES={'gyrA':'DNA topology','gyrB':'DNA topology','rpoB':'transcription','murA':'peptidoglycan synthesis','murB':'peptidoglycan synthesis','folA':'folate metabolism','ftsZ':'cell division'}
ASSAYS={'Ab_WT':'Ab','CRAB':'Ab','Sa_WT':'Sa','MRSA':'Sa','CRKP':'Kp','CRPA':'Pa'}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select(entries):
    candidates=[]
    for species,records in entries.items():
        for entry in records:
            names={g.get('geneName',{}).get('value','').lower() for g in entry.get('genes',[])}
            seq=entry['sequence']['value']
            if entry.get('entryType')!='UniProtKB reviewed (Swiss-Prot)' or not re.fullmatch('[ACDEFGHIKLMNPQRSTVWY]+',seq):continue
            if entry.get('proteinDescription',{}).get('flag')=='Fragment':continue
            for gene,function in GENES.items():
                if gene.lower() in names:
                    candidates.append(dict(species=species,gene=gene,functional_category=function,record_id=entry['primaryAccession'],sequence=seq,
                      organism=entry['organism']['scientificName'],taxonomy_id=entry['organism']['taxonId'],annotation_score=entry.get('annotationScore',0),
                      protein_description=json.dumps(entry.get('proteinDescription',{}),ensure_ascii=False),
                      source_url='https://www.uniprot.org/uniprotkb/'+entry['primaryAccession'],
                      target_evidence='Reviewed functional annotation; not evidence of peptide binding or strain-specific susceptibility'))
    frame=pd.DataFrame(candidates)
    if frame.empty:raise ValueError('No qualifying reviewed targets')
    ranked=frame.sort_values(['species','gene','annotation_score','record_id'],ascending=[True,True,False,True])
    chosen=ranked.drop_duplicates(['species','gene']).copy()
    return chosen,frame


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',required=True)
    args=parser.parse_args();out=Path(args.output_dir).resolve()
    if out.exists():raise ValueError('Output exists; never overwrite a frozen panel')
    out.mkdir(parents=True);raw=out/'raw';raw.mkdir()
    rules={'schema_version':1,'species_taxonomy':SPECIES,'genes':GENES,'assay_species_mapping':ASSAYS,
      'selection':'reviewed; primary gene exact; complete standard amino acid sequence; one per species/gene ranked annotationScore descending then accession ascending; exact sequence dedup for scoring',
      'purpose':'new_prospective_reference_panel','historical_r0_panel':False,'actual_assay_isolate_match':False,
      'limitations':['Resistance phenotype cannot be inferred from species sequence','Functional relevance does not establish AMP binding or causal killing','Representatives may originate from different reference strains; organism names retained'],
      'implementation_sha256':digest(__file__)}
    (out/'selection_rules.json').write_text(json.dumps(rules,indent=2))
    entries={};requests=[];release=None
    for species,taxon in SPECIES.items():
        query=f'(taxonomy_id:{taxon}) AND (reviewed:true) AND ('+' OR '.join('gene_exact:'+g for g in GENES)+')'
        url='https://rest.uniprot.org/uniprotkb/search?'+urlencode({'query':query,'format':'json','size':500})
        entries[species]=[];page=0
        while url:
            with urlopen(Request(url,headers={'User-Agent':'MDRAMP-prospective-panel/1.0'}),timeout=180) as response:
                body=response.read();headers=dict(response.headers);current=response.headers.get('X-UniProt-Release')
                if not current or (release and current!=release):raise ValueError('Missing or changing UniProt release')
                release=current;link=response.headers.get('Link','')
            path=raw/f'{species}_{page}.json';path.write_bytes(body)
            requests.append({'url':url,'path':str(path.relative_to(out)),'sha256':digest(path),'headers':headers})
            entries[species].extend(json.loads(body)['results']);page+=1
            match=re.search(r'<([^>]+)>; rel="next"',link);url=match.group(1) if match else None
    chosen,candidates=select(entries)
    candidates.to_csv(out/'all_candidates.csv',index=False)
    chosen.to_csv(out/'selected_targets.csv',index=False)
    unique=chosen.drop_duplicates('sequence').copy();unique.to_csv(out/'proteins.csv',index=False)
    (out/'targets.fasta').write_text(''.join(f'>{r.record_id}\n{r.sequence}\n' for r in unique.itertuples()))
    if len(unique)<5 or set(chosen.species)!=set(SPECIES):raise ValueError('Insufficient panel coverage')
    if digest(__file__)!=rules['implementation_sha256']:raise ValueError('Implementation changed')
    gaps=[f'{s}:{g}' for s in SPECIES for g in GENES if not ((chosen.species==s)&(chosen.gene==g)).any()]
    manifest={**rules,'status':'frozen_new_prospective_panel','frozen_at':datetime.now(timezone.utc).isoformat(),'uniprot_release':release,
      'record_ids':unique.record_id.tolist(),'protein_count':len(unique),'coverage_gaps':gaps,'requests':requests,
      'output_sha256':{str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    print(json.dumps({'protein_count':len(unique),'coverage_gaps':gaps,'release':release}))


if __name__=='__main__':main()

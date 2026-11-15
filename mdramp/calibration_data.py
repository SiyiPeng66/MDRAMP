"""Retrospective six-strain calibration ledger from a traceable XLSX workbook.

Symbols have no assumed meaning: --positive-symbol is mandatory. Fixed-dose
screening imposes MIC interval constraints under the explicitly selected active
symbol convention. This does not reconstruct historical label-release times.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path,PurePosixPath
import random
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile
import pandas as pd
from src.data.cluster_split import length_aware_identity
from src.utils.hash import sha256_file

STRAINS=('Ab_WT','Sa_WT','CRAB','MRSA','CRKP','CRPA')
SPECIES=('A. baumannii','S. aureus','Res. A. baumannii','Res. S. aureus','Res. K. pneumoniae','Res. P. aeruginosa')
NS={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
REL='http://schemas.openxmlformats.org/officeDocument/2006/relationships'


def peptide_id(sequence):return 'cal_'+hashlib.sha256(sequence.encode()).hexdigest()[:20]


def provenance_json(value):
    def convert(item):
        if isinstance(item,float) and not math.isfinite(item):return 'inf' if item>0 else '-inf'
        if isinstance(item,dict):return {k:convert(v) for k,v in item.items()}
        if isinstance(item,(list,tuple)):return [convert(v) for v in item]
        return item
    return json.dumps(convert(value),ensure_ascii=False,allow_nan=False)


def read_workbook(path):
    """Read OOXML cell values and coordinates without executing Excel formulas."""
    with zipfile.ZipFile(path) as book:
        strings=[]
        if 'xl/sharedStrings.xml' in book.namelist():
            strings=[''.join(element.itertext()) for element in ET.fromstring(book.read('xl/sharedStrings.xml')).findall('s:si',NS)]
        relationships={element.attrib['Id']:element.attrib['Target'] for element in ET.fromstring(book.read('xl/_rels/workbook.xml.rels'))}
        result=[]
        for sheet in ET.fromstring(book.read('xl/workbook.xml')).findall('s:sheets/s:sheet',NS):
            target=relationships[sheet.attrib['{'+REL+'}id']]
            member=target.lstrip('/') if target.startswith('/') else str(PurePosixPath('xl')/target)
            if '..' in PurePosixPath(member).parts:raise ValueError('Unsupported workbook relationship path')
            rows=[]
            for row in ET.fromstring(book.read(member)).findall('s:sheetData/s:row',NS):
                cells={};formulas={}
                for cell in row:
                    coordinate=cell.attrib['r'];node=cell.find('s:v',NS)
                    if cell.attrib.get('t')=='s':value=strings[int(node.text)] if node is not None else ''
                    elif cell.attrib.get('t')=='inlineStr':value=''.join(cell.find('s:is',NS).itertext())
                    else:value=node.text if node is not None and node.text is not None else ''
                    cells[coordinate]=value
                    formula=cell.find('s:f',NS)
                    if formula is not None:formulas[coordinate]=formula.text or ''
                rows.append({'row':int(row.attrib['r']),'cells':cells,'formulas':formulas})
            result.append({'sheet':sheet.attrib['name'],'rows':rows})
    return result


def normalize_sequence(value):
    sequence=re.sub(r'\s+','',str(value)).upper()
    if not sequence or not set(sequence)<=set('ACDEFGHIKLMNPQRSTVWY'):raise ValueError('invalid_or_missing_standard_amino_acid_sequence')
    return sequence


def parse_mic(value):
    value=str(value).strip().replace('＞','>').replace('＜','<').replace('≥','>=').replace('≤','<=')
    if not value:return None
    matched=re.fullmatch(r'(>=|<=|>|<)?\s*(\d+(?:\.\d+)?)',value)
    if not matched:raise ValueError('unsupported_MIC_value')
    op=matched[1] or '=';number=float(matched[2])
    if number<=0 or not math.isfinite(number):raise ValueError('nonpositive_or_nonfinite_MIC')
    y=math.log2(number)
    if op=='=':return (y,y,True,True,'exact')
    if op in ('>','>='):return (y,math.inf,op=='>=',False,'right_censored')
    return (-math.inf,y,False,op=='<=','left_censored')


def screen_constraint(value,concentration,positive_symbol):
    value=str(value).strip().replace('＋','+').replace('−','-').replace('－','-')
    if not value:return None
    if value not in ('+','-'):raise ValueError('unsupported_screen_symbol')
    positive='+' if positive_symbol=='plus' else '-'
    boundary=math.log2(concentration)
    return (-math.inf,boundary,False,True,'screen') if value==positive else (boundary,math.inf,False,False,'screen')


def intersect_constraints(constraints):
    if not constraints:return (None,None,False,False,'missing',None)
    lower=max(c[0] for c in constraints);upper=min(c[1] for c in constraints)
    lower_inclusive=all(c[2] for c in constraints if c[0]==lower)
    upper_inclusive=all(c[3] for c in constraints if c[1]==upper)
    if lower>upper or (lower==upper and not (lower_inclusive and upper_inclusive)):
        return (lower,upper,lower_inclusive,upper_inclusive,'quarantined','contradictory_MIC_screen_or_repeated_measurements')
    return (lower,upper,lower_inclusive,upper_inclusive,'usable',None)


def assign_groups(sequences,seed=11,fold_count=5):
    """Global 90% identity connected components, then size-balanced whole groups."""
    sequences=sorted(set(sequences));parents=list(range(len(sequences)))
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    for i,left in enumerate(sequences):
        for j in range(i):
            right=sequences[j]
            if min(len(left),len(right))/max(len(left),len(right))>=.9 and length_aware_identity(left,right)>=.9:
                a,b=find(i),find(j)
                if a!=b:parents[b]=a
    components=defaultdict(list)
    for i,sequence in enumerate(sequences):components[find(i)].append(sequence)
    if len(components)<fold_count:raise ValueError(f'Need at least {fold_count} independent sequence clusters; found {len(components)}')
    rng=random.Random(seed);groups=list(components.values());rng.shuffle(groups);groups.sort(key=len,reverse=True)
    loads=[0]*fold_count;assignment={}
    for group in groups:
        available=[i for i,v in enumerate(loads) if v==min(loads)];fold=rng.choice(available)
        cluster='CALC_'+hashlib.sha256('\n'.join(sorted(group)).encode()).hexdigest()[:20]
        for sequence in group:assignment[sequence]=(cluster,fold)
        loads[fold]+=len(group)
    return assignment,loads


def build_tables(workbook,positive_symbol,seed=11):
    if positive_symbol not in ('plus','minus'):raise ValueError('Explicit plus/minus activity convention required')
    evidence=defaultdict(list);sources=[];exclusions=[];valid_sequences=set()
    for sheet in workbook:
        if not sheet['rows']:continue
        header=sheet['rows'][0];headers={re.sub(r'\d','',cell):str(value).strip() for cell,value in header['cells'].items()}
        reverse={value:column for column,value in headers.items() if value}
        sequence_column=reverse.get('采用序列',reverse.get('序列'))
        if not sequence_column:raise ValueError(f"Missing sequence header in sheet {sheet['sheet']}")
        known=set(headers)
        for row in sheet['rows'][1:]:
            values={headers.get(re.sub(r'\d','',cell),cell):value for cell,value in row['cells'].items()}
            raw_sequence=row['cells'].get(sequence_column+str(row['row']),'')
            source_id=f"{sheet['sheet']}!{row['row']}"
            source={'source_row_id':source_id,'sheet':sheet['sheet'],'row':row['row'],'sequence_cell':sequence_column+str(row['row']),'raw_sequence':raw_sequence,'round':values.get('轮次',''),'model':values.get('模型',''),'new_name':values.get('新版名称',''),'original_id':values.get('旧原始ID',values.get('序列ID','')),'old_id':values.get('旧ID',''),'cells_json':json.dumps(row['cells'],ensure_ascii=False),'formulas_json':json.dumps(row['formulas'],ensure_ascii=False)}
            try:sequence=normalize_sequence(raw_sequence)
            except ValueError as error:
                source.update(sequence='',peptide_id='',status='excluded');sources.append(source)
                exclusions.append({'source_row_id':source_id,'reason':str(error),'provenance':source['cells_json']});continue
            pid=peptide_id(sequence);valid_sequences.add(sequence);source.update(sequence=sequence,peptide_id=pid,status='parsed');sources.append(source)
            for cell,value in row['cells'].items():
                if value and re.sub(r'\d','',cell) not in known:exclusions.append({'source_row_id':source_id,'peptide_id':pid,'reason':'unmapped_nonempty_cell','provenance':json.dumps({'cell':cell,'value':value},ensure_ascii=False)})
            for strain,species in zip(STRAINS,SPECIES):
                record={'source_row_id':source_id,'sheet':sheet['sheet'],'row':row['row'],'round':source['round'],'model':source['model'],'original_id':source['original_id'],'new_name':source['new_name'],'old_id':source['old_id'],'cells':[],'constraints':[],'errors':[]}
                for name,kind,concentration in [(f'{species} 256 μg/mL','screen',256),(f'{species} 32 μg/mL','screen',32),(f'MIC_{strain}_μg_mL','MIC',None)]:
                    col=reverse.get(name)
                    if col is None:continue
                    coordinate=col+str(row['row']);value=row['cells'].get(coordinate,'')
                    record['cells'].append({'cell':coordinate,'header':name,'value':value})
                    try:
                        if coordinate in row['formulas']:raise ValueError('formula_cell_requires_independent_resolution')
                        constraint=parse_mic(value) if kind=='MIC' else screen_constraint(value,concentration,positive_symbol)
                        if constraint is not None:record['constraints'].append(constraint)
                    except ValueError as error:record['errors'].append({'cell':coordinate,'value':value,'reason':str(error)})
                evidence[(sequence,strain)].append(record)
    assignment,loads=assign_groups(valid_sequences,seed)
    sequence_rows=[{'peptide_id':peptide_id(seq),'sequence':seq,'length':len(seq),'cluster_id':assignment[seq][0],'fold':assignment[seq][1]} for seq in sorted(valid_sequences)]
    observations=[]
    for sequence in sorted(valid_sequences):
        for strain in STRAINS:
            records=evidence[(sequence,strain)];constraints=[c for record in records for c in record['constraints']];errors=[e for record in records for e in record['errors']]
            lower,upper,li,ui,status,reason=intersect_constraints(constraints)
            if errors:status='quarantined';reason='unsupported_or_unresolved_source_measurement'
            kind='missing' if status=='missing' else ('exact' if lower==upper and li and ui else ('left_censored' if lower==-math.inf else ('right_censored' if upper==math.inf else 'interval')))
            row={'peptide_id':peptide_id(sequence),'strain_id':strain,'lower_log2':lower,'upper_log2':upper,'lower_inclusive':li,'upper_inclusive':ui,'status':status,'observation_type':kind,'exact_log2':lower if kind=='exact' and status=='usable' else None,'cluster_id':assignment[sequence][0],'fold':assignment[sequence][1],'source_row_count':len(records),'constraint_count':len(constraints),'provenance':provenance_json(records),'exclusion_reason':reason or ''}
            observations.append(row)
            if status=='quarantined':exclusions.append({'peptide_id':row['peptide_id'],'strain_id':strain,'reason':reason,'provenance':row['provenance']})
    return pd.DataFrame(sequence_rows),pd.DataFrame(observations),pd.DataFrame(sources),pd.DataFrame(exclusions,columns=['source_row_id','peptide_id','strain_id','reason','provenance']),loads



def apply_corrections(tables, document, source_sha256):
    """Apply explicit, source-bound user adjudications; retain original evidence."""
    if document.get('source_sha256') != source_sha256 or document.get('schema_version') != 1:
        raise ValueError('Correction source hash/schema mismatch')
    seq, obs, sources, exclusions, loads = tables
    obs=obs.copy();exclusions=exclusions.copy();seen=set()
    for correction in document['corrections']:
        pid,strain=correction['peptide_id'],correction['strain_id']
        if (pid,strain) in seen:raise ValueError('Duplicate correction')
        seen.add((pid,strain))
        mask=obs.peptide_id.eq(pid)&obs.strain_id.eq(strain)
        if mask.sum()!=1:raise ValueError('Correction must match exactly one observation')
        index=obs.index[mask][0];old=obs.loc[index]
        evidence=json.loads(old.provenance)
        if not any(r['source_row_id']==correction['source_row_id'] and r['new_name']==correction['new_name'] for r in evidence):
            raise ValueError('Correction source identity mismatch')
        if not correction.get('reason') or not correction.get('authority'):raise ValueError('Correction requires authority and reason')
        interval=parse_mic(str(correction['mic_ug_ml']))
        if interval is None or interval[4]!='exact':raise ValueError('Correction requires exact MIC')
        value=interval[0]
        evidence.append({'user_adjudication':correction,'previous_status':old.status,'previous_exclusion_reason':old.exclusion_reason})
        for key,val in {'lower_log2':value,'upper_log2':value,'lower_inclusive':True,'upper_inclusive':True,'status':'usable','observation_type':'exact','exact_log2':value,'exclusion_reason':'','provenance':provenance_json(evidence)}.items():obs.at[index,key]=val
        removed=exclusions.peptide_id.eq(pid)&exclusions.strain_id.eq(strain)&exclusions.reason.eq('contradictory_MIC_screen_or_repeated_measurements')
        exclusions=exclusions.loc[~removed].copy()
    return seq,obs,sources,exclusions,loads


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',required=True);parser.add_argument('--output-dir',required=True)
    parser.add_argument('--positive-symbol',choices=['plus','minus'],required=True);parser.add_argument('--seed',type=int,default=11)
    parser.add_argument('--corrections',help='Source-hash-bound JSON user adjudications')
    args=parser.parse_args();source=Path(args.input).resolve();output=Path(args.output_dir).resolve();digest=sha256_file(source)
    implementation={str(Path(__file__).resolve()):sha256_file(__file__),str(Path(__file__).parents[1]/'src/data/cluster_split.py'):sha256_file(Path(__file__).parents[1]/'src/data/cluster_split.py')}
    if output.exists():raise ValueError('Output already exists; use a new calibration version')
    corrections_path=Path(args.corrections).resolve() if args.corrections else None
    corrections_hash=sha256_file(corrections_path) if corrections_path else None
    tables=build_tables(read_workbook(source),args.positive_symbol,args.seed)
    if corrections_path:tables=apply_corrections(tables,json.loads(corrections_path.read_text()),digest)
    output.parent.mkdir(parents=True,exist_ok=True)
    with (output.parent/(output.name+'.lock')).open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise ValueError('Calibration output locked') from error
        if output.exists():raise ValueError('Output already exists')
        staging=Path(tempfile.mkdtemp(prefix='.'+output.name+'.',dir=output.parent))
        try:
            shutil.copyfile(source,staging/'source.xlsx')
            if corrections_path:
                shutil.copyfile(corrections_path,staging/'corrections.json')
                if sha256_file(corrections_path)!=corrections_hash or sha256_file(staging/'corrections.json')!=corrections_hash:raise ValueError('Corrections changed during extraction')
            frames=dict(zip(['sequences.csv','observations.csv','source_rows.csv','exclusions.csv'],tables[:4]))
            for name,frame in frames.items():frame.to_csv(staging/name,index=False)
            if sha256_file(source)!=digest or sha256_file(staging/'source.xlsx')!=digest:raise ValueError('Workbook changed during extraction')
            if any(sha256_file(path)!=digest for path,digest in implementation.items()):raise ValueError('Calibration implementation changed during extraction')
            observations=tables[1]
            manifest={'schema_version':1,'purpose':'retrospective_calibration_and_grouped_validation','historical_prospective_reproduction':False,'label_release_dates_available':False,'source':{'path':str(source),'sha256':digest,'archive':'source.xlsx'},'positive_symbol':args.positive_symbol,'mic_units':'ug/mL','screen_interpretation':'active => MIC <= concentration; inactive => MIC > concentration; concentrations 32 and 256 ug/mL; intersect all source constraints','strains':list(STRAINS),'fold_count':5,'seed':args.seed,'grouping':'global >=90% identity connected components; exact duplicate sequences merged across sheets; all six strains share group/fold','fold_balance':'greedy whole-cluster size balance; shuffled deterministic tie order','fold_sequence_counts':tables[4],'cluster_count':int(tables[0].cluster_id.nunique()),'unique_sequences':len(tables[0]),'source_rows':len(tables[2]),'observation_status_counts':observations.status.value_counts().to_dict(),'source_sheet_counts':tables[2].sheet.value_counts().to_dict(),'unsupported_values_and_conflicts':'quarantined; raw values and cell coordinates retained','implementation_sha256':implementation,'output_sha256':{name:sha256_file(staging/name) for name in [*frames,'source.xlsx']}}
            if corrections_path:
                manifest['corrections']={'path':str(corrections_path),'sha256':corrections_hash,'policy':'Explicit user adjudication supersedes conflicting screen constraints for named observation only; original evidence retained.'}
                manifest['output_sha256']['corrections.json']=corrections_hash
            (staging/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
            if output.exists():raise ValueError('Output appeared during extraction')
            os.rename(staging,output)
        finally:
            if staging.exists():shutil.rmtree(staging)
    print(json.dumps({key:manifest[key] for key in ['source_rows','unique_sequences','fold_sequence_counts','observation_status_counts']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()

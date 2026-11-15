"""Join audited baseline and assay-matched sources without selecting proteins."""
import argparse
import json
from pathlib import Path
from mdramp.protein_panel_build import sha,dump


def assemble(root):
    root=Path(root).absolute();destination=root/'experiment_library_manifest.json'
    if destination.exists():raise ValueError('Never overwrite an assembled source manifest')
    base=json.loads((root/'download_manifest.json').read_text())
    assay=json.loads((root/'assay_matched/manifest.json').read_text())
    if base['status']!='complete' or assay['status']!='complete' or base['uniprot_release']!=assay['uniprot_release']:raise ValueError('Complete same-release sources required')
    hashes=dict(base['output_sha256'])
    for field in ['raw_sha256','executed_script_sha256','query_audit_sha256']:
        hashes.update({'assay_matched/'+name:digest for name,digest in assay[field].items()})
    for p in ['download_manifest.json','assay_matched/manifest.json','assay_source_policy.json']:
        hashes[p]=sha(root/p)
    for name,digest in hashes.items():
        if sha(root/name)!=digest:raise ValueError('Source mismatch: '+name)
    chosen={};mapping=assay['assay_mapping']
    expected={'Ab_WT','Sa_WT','CRAB','MRSA','CRPA','CRKP'}
    if len(mapping)!=6 or {r['assay'] for r in mapping}!=expected:raise ValueError('Six unique assay sources required')
    for species,info in base['chosen_proteomes'].items():
        pid=info['proteome_id'];used=[r['assay'] for r in mapping if r['chosen_proxy']==pid]
        if used:
            chosen[pid]={**info,'species':species,'assays':used,
                'protein_json_files':[str(p.relative_to(root)) for p in sorted((root/'raw').glob(f'{species}_{pid}_proteins_*.json')) if not p.name.endswith('.request.json')]}
    for info in assay['chosen_proteomes'].values():
        pid=info['proteome_id'];used=[r['assay'] for r in mapping if r['chosen_proxy']==pid]
        if used:chosen[pid]={**info,'assays':used}
    if {r['chosen_proxy'] for r in mapping}!=set(chosen):raise ValueError('Missing selected proteome source')
    for info in chosen.values():
        for path in info['protein_json_files']:
            if path not in hashes:raise ValueError('Unbound protein source')
    result=dict(status='complete',schema_version=1,uniprot_release=base['uniprot_release'],
        chosen_proteomes=chosen,assay_source_mapping=mapping,auxiliary_uniparc_archives=assay['auxiliary_uniparc_archives'],
        annotated_source_record_count=sum(x['downloaded_protein_count'] for x in chosen.values()),
        actual_assay_isolate_genome_verified=False,purpose='new_prospective_assay_mapped_reference_library',
        implementation_sha256=sha(__file__),output_sha256=hashes)
    dump(destination,result)
    print(json.dumps({'proteomes':len(chosen),'annotated_protein_records':result['annotated_source_record_count']}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--library-dir',required=True);a=p.parse_args();assemble(a.library_dir)


if __name__=='__main__':main()

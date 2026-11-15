"""Independent evidence/identity-contract regressions; no external model runs."""
import json
from pathlib import Path

import pandas as pd
import pytest

from mdramp import protein_panel_build as panel


@pytest.fixture
def rules():
    return json.loads((Path(__file__).parents[1] / 'configs/prospective/protein_panel_v2_rules.json').read_text())


def entry(accession='A', sequence='ACDEFGHIK', **extra):
    return dict(primaryAccession=accession, sequence={'value': sequence},
                organism={'scientificName': 'Acinetobacter baumannii', 'taxonId': 470}, **extra)


@pytest.mark.parametrize('foreign', ['KW-1043', 'KW-1032', 'KW-1053', 'KW-0999'])
def test_foreign_membrane_does_not_imply_bacterial_envelope(rules, foreign):
    row=panel.annotate({'keyword_ids': f'{foreign};KW-0472;KW-0812;KW-1133'}, rules)
    assert not row['envelope_annotation']
    assert row['non_bacterial_location_keywords'] == foreign
    # An independent bacterial localization or wall annotation must survive.
    for specific in ('KW-0134', 'KW-0997', 'KW-0998', 'KW-1003'):
        row=panel.annotate({'keyword_ids': f'{foreign};KW-0472;{specific};KW-0046'},rules)
        assert row['envelope_annotation'] and row['resistance_annotation']
        assert row['envelope_matched_keywords'] == specific


def test_quality_does_not_require_reviewed_and_never_truncates():
    row=panel.normalize(entry(sequence='A'*2050), 'Ab', 'UP1')
    assert row['quality_eligible'] and not row['reviewed']
    assert len(row['sequence'])==2050 and row['long_sequence_review_required']
    row=panel.normalize(entry(proteinDescription={'flag':'Fragment'}), 'Ab', 'UP1')
    assert not row['quality_eligible'] and row['quality_exclusion']=='fragment'
    row=panel.normalize(entry(sequence='ACDX'), 'Ab', 'UP1')
    assert not row['quality_eligible']
    row=panel.normalize(entry(comments=[{'commentType':'SEQUENCE CAUTION','sequenceCautions':[{'type':'Erroneous gene model prediction'}]}]), 'Ab', 'UP1')
    assert row['quality_exclusion']=='erroneous_gene_model'


def test_exact_and_inferred_essentiality_remain_same_species(tmp_path, monkeypatch):
    frame=pd.DataFrame([
        dict(record_id='A',species='Ab',sequence='ACDEFGHIK',quality_eligible=True),
        dict(record_id='B',species='Ab',sequence='ACDEFGHIL',quality_eligible=True),
        dict(record_id='C',species='Sa',sequence='ACDEFGHIK',quality_eligible=True),
        dict(record_id='BAD',species='Ab',sequence='ACDEFGHIK',quality_eligible=False)])
    records=pd.DataFrame([dict(record_id='D',species='Ab',sequence='ACDEFGHIK'),
                          dict(record_id='E',species='Pa',sequence='ACDEFGHIK')])
    def fake(command, log):
        assert '>C\n' not in Path(command[2]).read_text()
        assert '>BAD\n' not in Path(command[2]).read_text()
        assert '>E\n' not in Path(command[3]).read_text()
        Path(command[4]).write_text('A\tD\t1\t1\t1\t1e-20\t50\nB\tD\t0.91\t0.95\t0.95\t1e-10\t40\n')
    monkeypatch.setattr(panel,'execute',fake)
    mapped=panel.essential_matches(frame,records,tmp_path,'mmseqs',1)
    assert mapped.set_index('record_id').mapping.to_dict()=={'A':'exact_sequence','B':'same_species_homology_inference'}
    assert set(mapped.species)=={'Ab'}


def test_nonfinite_alignment_rejected(tmp_path, monkeypatch):
    frame=pd.DataFrame([dict(record_id='A',species='Ab',sequence='ACDEFGHIK',quality_eligible=True)])
    records=pd.DataFrame([dict(record_id='D',species='Ab',sequence='ACDEFGHIL')])
    monkeypatch.setattr(panel,'execute',lambda cmd, log: Path(cmd[4]).write_text('A\tD\tnan\t1\t1\t1e-20\t40\n'))
    with pytest.raises(ValueError):
        panel.essential_matches(frame,records,tmp_path,'mmseqs',1)


@pytest.mark.parametrize('pairs', ['A\tA\n', 'A\tA\nA\tA\n', 'A\tB\nB\tA\n'])
def test_cluster_invalid_memberships_rejected(tmp_path,monkeypatch,pairs):
    frame=pd.DataFrame([dict(record_id=x,species='Ab',evidence_signature='100',sequence='ACDEFGHIK') for x in ('A','B')])
    monkeypatch.setattr(panel,'execute',lambda cmd, log: Path(str(cmd[3])+'_cluster.tsv').write_text(pairs))
    with pytest.raises(ValueError):
        panel.cluster(frame,tmp_path/'clusters','mmseqs',.9,1)


def test_clustering_preserves_species_and_evidence_strata(tmp_path,monkeypatch):
    frame=pd.DataFrame([dict(record_id=rid,species=species,evidence_signature=sig,sequence='ACDEFGHIK')
        for rid,species,sig in [('A','Ab','100'),('B','Ab','100'),('C','Ab','001'),('D','Sa','100')]])
    commands=[]
    def fake(cmd,log):
        commands.append(cmd)
        Path(str(cmd[3])+'_cluster.tsv').write_text('A\tA\nA\tB\n')
    monkeypatch.setattr(panel,'execute',fake)
    m=panel.cluster(frame,tmp_path/'clusters','mmseqs',.9,1)
    assert len(commands)==1 and set(m.member_id)==set(frame.record_id)
    assert m.set_index('member_id').representative_id.to_dict()=={'A':'A','B':'A','C':'C','D':'D'}
    cmd=commands[0]
    assert cmd[cmd.index('--cov-mode')+1]=='0'
    assert cmd[cmd.index('--single-step-clustering')+1]=='1'
    assert cmd[cmd.index('--alignment-mode')+1]=='3'


def test_explicit_pseudogene_name_excluded():
    r=panel.normalize(entry(proteinDescription={'recommendedName':{'fullName':{'value':'ABC transporter pseudogene'}}}), 'Ab','UP1')
    assert not r['quality_eligible'] and 'pseudogene' in r['quality_exclusion']


def test_build_evidence_union_exact_collapse_and_cross_stratum_audit(tmp_path,monkeypatch,rules):
    from types import SimpleNamespace
    root=tmp_path/'source'; (root/'raw').mkdir(parents=True)
    (root/'rules_sources').mkdir()
    ontology=root/'rules_sources/uniprot_keywords.json';ontology.write_text('{}')
    rules['keyword_source_sha256']=panel.sha(ontology)
    rulepath=tmp_path/'rules.json';rulepath.write_text(json.dumps(rules))
    def tagged(rid,seq,ids=(),**kwargs):
        return entry(rid,seq,keywords=[{'id':i,'name':i} for i in ids],**kwargs)
    seq='ACDEFGHIK'
    rows=[tagged('A',seq,['KW-0472']),tagged('B',seq,['KW-0046']),
          tagged('C','ACDEFGHIL'),tagged('NONE','ACDEFGHIM'),
          tagged('FRAGMENT',seq,['KW-0046'],proteinDescription={'flag':'Fragment'})]
    raw=root/'raw/Ab_UP1_proteins_000.json';raw.write_text(json.dumps({'results':rows}))
    manifest=dict(status='complete',output_sha256={str(raw.relative_to(root)):panel.sha(raw)},
                  chosen_proteomes={'Ab':{'proteome_id':'UP1','downloaded_protein_count':5}},uniprot_release='fixture')
    (root/'download_manifest.json').write_text(json.dumps(manifest))
    records=tmp_path/'essential.csv'
    pd.DataFrame([dict(record_id='DEG1',species='Ab',sequence='ACDEFGHIL')]).to_csv(records,index=False)
    essential_manifest=tmp_path/'essential_manifest.json'
    essential_manifest.write_text(json.dumps({'status':'official_snapshot_parsed','outputs_sha256':{records.name:panel.sha(records)},'inputs_sha256':{}}))
    binary=tmp_path/'mmseqs';binary.write_text('fixture binary')
    calls=[]
    def fake(cmd,log):
        calls.append(cmd)
        if cmd[1]=='easy-search':Path(cmd[4]).write_text('')
        else:
            ids=[line[1:] for line in Path(cmd[2]).read_text().splitlines() if line.startswith('>')]
            Path(str(cmd[3])+'_cluster.tsv').write_text(''.join(f'{ids[0]}\t{i}\n' for i in ids))
    monkeypatch.setattr(panel,'execute',fake)
    monkeypatch.setattr(panel.subprocess,'check_output',lambda *args,**kwargs:'fixture-version')
    out=tmp_path/'built'
    panel.build(SimpleNamespace(library_dir=root,output_dir=out,rules=rulepath,essential_records=records,essential_manifest=essential_manifest,mmseqs=str(binary),threads=1))
    evidence=pd.read_csv(out/'library_evidence.csv').set_index('record_id')
    assert set(evidence.index[evidence.selected_by_evidence])=={'A','B','C'}
    assert evidence.loc['C','essential_exact_sequence'] and not evidence.loc['C','essential_homology_only']
    memberships=pd.read_csv(out/'panel_membership.csv')
    assert set(memberships.member_id)=={'A','B','C'}
    assert memberships.set_index('member_id').scoring_record_id.to_dict()=={'A':'A','B':'A','C':'C'}
    report=json.loads((out/'manifest.json').read_text())
    assert (report['raw_count'],report['clean_count'],report['evidence_selected_count'],report['scoring_protein_count'])==(5,4,3,2)
    assert report['residual_global_90pct_redundant_representatives']==1
    assert report['no_mic_or_peptide_score_used'] and not report['historical_r0_panel']
    assert all(panel.sha(out/p)==h for p,h in report['output_sha256'].items())


@pytest.mark.parametrize('second_sequence,second_species,changed_annotation', [('ACDEFGHIK','Ab',False), ('ACDEFGHIL','Ab',False), ('ACDEFGHIK','Sa',False), ('ACDEFGHIK','Ab',True)])
def test_multisource_accession_integrity_and_membership(tmp_path,monkeypatch,rules,second_sequence,second_species,changed_annotation):
    from types import SimpleNamespace
    root=tmp_path/'source';(root/'raw').mkdir(parents=True);(root/'rules_sources').mkdir()
    ontology=root/'rules_sources/uniprot_keywords.json';ontology.write_text('{}')
    rules['keyword_source_sha256']=panel.sha(ontology)
    rulepath=tmp_path/'rules.json';rulepath.write_text(json.dumps(rules))
    chosen={};hashes={}
    for i,(species,sequence,assay) in enumerate([('Ab','ACDEFGHIK','Ab_WT'),(second_species,second_sequence,'CRAB')]):
        record=entry('SHARED',sequence,keywords=[{'id':'KW-0472','name':'Membrane'}])
        if i==1 and changed_annotation:record['proteinDescription']={'flag':'Fragment'}
        path=root/f'raw/source{i}.json';path.write_text(json.dumps({'results':[record]}));hashes[str(path.relative_to(root))]=panel.sha(path)
        chosen[f'source{i}']=dict(species=species,proteome_id=f'UP{i}',assays=[assay],protein_json_files=[str(path.relative_to(root))],downloaded_protein_count=1)
    source_manifest=root/'custom_manifest.json'
    source_manifest.write_text(json.dumps(dict(status='complete',output_sha256=hashes,chosen_proteomes=chosen,uniprot_release='fixture')))
    records=tmp_path/'essential.csv';pd.DataFrame(columns=['record_id','species','sequence']).to_csv(records,index=False)
    em=tmp_path/'essential_manifest.json';em.write_text(json.dumps(dict(status='official_snapshot_parsed',inputs_sha256={},outputs_sha256={records.name:panel.sha(records)})))
    binary=tmp_path/'mmseqs';binary.write_text('fixture')
    monkeypatch.setattr(panel.subprocess,'check_output',lambda *a,**k:'fixture')
    monkeypatch.setattr(panel,'execute',lambda *a,**k:pytest.fail('singleton strata require no external alignment'))
    out=tmp_path/'built'
    args=SimpleNamespace(library_dir=root,library_manifest=source_manifest,output_dir=out,rules=rulepath,essential_records=records,essential_manifest=em,mmseqs=str(binary),threads=1)
    if second_sequence!='ACDEFGHIK' or second_species!='Ab' or changed_annotation:
        with pytest.raises(ValueError,match='Conflicting repeated library accession|annotation'):panel.build(args)
        return
    panel.build(args)
    sources=pd.read_csv(out/'library_source_membership.csv')
    assert len(sources)==2 and set(sources.proteome_id)=={'UP0','UP1'}
    proteins=pd.read_csv(out/'proteins.csv')
    assert len(proteins)==1 and proteins.iloc[0].proteome_id=='UP0;UP1'
    assert proteins.iloc[0].represented_assays=='Ab_WT;CRAB'
    assert pd.read_csv(out/'panel_membership.csv').iloc[0].source_assays=='Ab_WT;CRAB'


def test_near_variants_from_distinct_strain_sources_are_not_clustered(tmp_path,monkeypatch):
    frame=pd.DataFrame([dict(record_id='A',species='Ab',proteome_id='UP1',evidence_signature='100',sequence='A'*99+'C'),
                        dict(record_id='B',species='Ab',proteome_id='UP2',evidence_signature='100',sequence='A'*99+'D')])
    monkeypatch.setattr(panel,'execute',lambda *a,**k:pytest.fail('cross-source approximate clustering is forbidden'))
    m=panel.cluster(frame,tmp_path/'cluster','mmseqs',.9,1)
    assert set(m.representative_id)=={'A','B'}
    assert set(m.source_stratum)=={'UP1','UP2'}

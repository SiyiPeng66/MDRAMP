"""Explicit adjudication must change only its source-bound peptide/strain row."""
import copy
import json
import pandas as pd
import pytest
from mdramp.calibration_data import apply_corrections


def fixture():
    evidence=[{'source_row_id':'R0-5!97','new_name':'AMP-DN05','cells':[
        {'cell':'M97','value':'-'},{'cell':'U97','value':'32'}]}]
    obs=pd.DataFrame([{'peptide_id':'p1','strain_id':strain,'provenance':json.dumps(evidence),
        'status':'quarantined' if strain=='MRSA' else 'usable',
        'exclusion_reason':'contradictory_MIC_screen_or_repeated_measurements' if strain=='MRSA' else '',
        'lower_log2':5.,'upper_log2':5.,'lower_inclusive':False if strain=='MRSA' else True,
        'upper_inclusive':True,'observation_type':'interval' if strain=='MRSA' else 'exact',
        'exact_log2':None if strain=='MRSA' else 5.,'cluster_id':'c1','fold':2}
        for strain in ['MRSA','CRAB']])
    sequences=pd.DataFrame([{'peptide_id':'p1','sequence':'ACDEFGHIKL','cluster_id':'c1','fold':2}])
    sources=pd.DataFrame([{'source_row_id':'R0-5!97','raw_sequence':'ACDEFGHIKL'}])
    exclusions=pd.DataFrame([{'peptide_id':'p1','strain_id':'MRSA','reason':'contradictory_MIC_screen_or_repeated_measurements','provenance':json.dumps(evidence)}])
    document={'schema_version':1,'source_sha256':'sourcehash','corrections':[
        {'peptide_id':'p1','strain_id':'MRSA','source_row_id':'R0-5!97','new_name':'AMP-DN05',
         'mic_ug_ml':32,'authority':'explicit user correction','reason':'MIC confirmed 32'}]}
    return (sequences,obs,sources,exclusions,[0,0,1,0,0]),document


def test_exact_adjudication_retains_evidence_and_only_target_changes():
    tables,document=fixture();before=[x.copy(deep=True) if isinstance(x,pd.DataFrame) else x[:] for x in tables]
    result=apply_corrections(tables,document,'sourcehash')
    target=result[1].iloc[0]
    assert target.status=='usable' and target.observation_type=='exact'
    assert target.lower_log2==target.upper_log2==target.exact_log2==5
    assert target.lower_inclusive and target.upper_inclusive
    assert not target.exclusion_reason and result[3].empty
    provenance=json.loads(target.provenance)
    assert provenance[:-1]==json.loads(before[1].iloc[0].provenance)
    assert provenance[-1]['user_adjudication']==document['corrections'][0]
    assert provenance[-1]['previous_status']=='quarantined'
    pd.testing.assert_series_equal(result[1].iloc[1],before[1].iloc[1])
    for i in [0,2]:pd.testing.assert_frame_equal(result[i],before[i])
    for i in range(4):pd.testing.assert_frame_equal(tables[i],before[i])
    assert result[4]==before[4]


@pytest.mark.parametrize('field,value',[
    ('peptide_id','other'),('strain_id','CRPA'),('source_row_id','R0-5!98'),
    ('new_name','AMP-DN01'),('mic_ug_ml','>32'),('mic_ug_ml',0),('authority',''),('reason','')])
def test_identity_and_exact_value_rejections(field,value):
    tables,document=fixture();document['corrections'][0][field]=value
    with pytest.raises(ValueError):apply_corrections(tables,document,'sourcehash')


def test_source_binding_and_duplicate_adjudications_rejected():
    tables,document=fixture()
    with pytest.raises(ValueError,match='hash/schema'):apply_corrections(tables,document,'otherhash')
    document['corrections'].append(copy.deepcopy(document['corrections'][0]))
    with pytest.raises(ValueError,match='Duplicate'):apply_corrections(tables,document,'sourcehash')

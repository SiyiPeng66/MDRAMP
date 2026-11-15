import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from mdramp.input_formats import read_sequences, read_experiments, aggregate_best

@pytest.mark.parametrize('suffix,separator',[('.csv',','),('.tsv','\t')])
def test_sequence_tables(tmp_path,suffix,separator):
    p=tmp_path/('input'+suffix);pd.DataFrame({'sequence':[' flr rlvkwlkkm ','MLKALKNLIRKLL']}).to_csv(p,sep=separator,index=False)
    f=read_sequences(p)
    assert f.sequence.tolist()==['FLRRLVKWLKKM','MLKALKNLIRKLL']
    assert f.id.is_unique

def test_fasta_and_conflicting_ids(tmp_path):
    p=tmp_path/'a.fa';p.write_text('>one\nflrrlv\nkwlkkm\n>two\nMLKALKNLIRKLL\n')
    assert read_sequences(p).sequence.iloc[0]=='FLRRLVKWLKKM'
    p.write_text('>same\nAAA\n>same\nCCC\n')
    with pytest.raises(ValueError,match='Row 3.*conflicting'):read_sequences(p)

def test_excel_first_sheet(tmp_path):
    # Read-only OOXML fixture assembled with stdlib; no spreadsheet authoring dependency.
    from zipfile import ZipFile
    p=tmp_path/'a.xlsx'
    with ZipFile(p,'w') as z:
        z.writestr('[Content_Types].xml','<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>')
        z.writestr('_rels/.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        z.writestr('xl/workbook.xml','<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Inputs" sheetId="1" r:id="rId1"/></sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>')
        z.writestr('xl/worksheets/sheet1.xml','<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>sequence</t></is></c></row><row r="2"><c r="A2" t="inlineStr"><is><t>AAA</t></is></c></row></sheetData></worksheet>')
    assert read_sequences(p).sequence.tolist()==['AAA']

@pytest.mark.parametrize('mic',['0','-1','NaN','inf','<8','abc',''])
def test_invalid_mic(tmp_path,mic):
    p=tmp_path/'e.csv';pd.DataFrame({'sequence':['AAA'],'mic':[mic]}).to_csv(p,index=False)
    with pytest.raises(ValueError,match='Row 2'):read_experiments(p)

def test_best_censoring_and_conflict(tmp_path):
    p=tmp_path/'e.csv';pd.DataFrame({'sequence':['AAA','AAA','CCC'],'mic':['>32','128','>64'],'strain':['a','b','a']}).to_csv(p,index=False)
    g=aggregate_best(read_experiments(p),10)
    assert g.observation_type.tolist()==['interval','right_censored']
    assert g.original_lower_log2.tolist()==[5,6]
    assert g.original_upper_log2.iloc[0]==7 and math.isinf(g.upper_log2.iloc[1])
    assert not g.lower_inclusive.any()
    assert np.isclose(g.lower_log2.iloc[0],5-math.log2(10))
    pd.DataFrame({'sequence':['AAA','AAA'],'mic':[8,16],'strain':['a','a']}).to_csv(p,index=False)
    with pytest.raises(ValueError,match='conflicting MIC'):read_experiments(p)

def test_demo_matches_contract():
    root=Path(__file__).resolve().parents[1]
    seq=read_sequences(root/'demo/predict.fasta')
    assert len(seq)==3
    pd.testing.assert_frame_equal(seq,read_sequences(root/'demo/predict.csv'))
    obs=read_experiments(root/'demo/experiments.csv')
    assert len(obs)==18
    best=aggregate_best(obs)
    assert np.exp2(best.original_lower_log2).tolist()==[4.,2.,2.]
    assert np.allclose(np.exp2(best.lower_log2),[.4,.2,.2])

def test_invalid_sequence_and_empty(tmp_path):
    p=tmp_path/'e.csv';p.write_text('sequence\nABX\n')
    with pytest.raises(ValueError,match='Row 2'):read_sequences(p)
    p.write_text('sequence\n')
    with pytest.raises(ValueError,match='no data'):read_sequences(p)

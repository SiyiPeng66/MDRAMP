import gzip
import io
from pathlib import Path
import numpy as np
import pytest
from src.data.prepare_tpeppro_structures import contact_from_chain,StructureCache

class Atom:
    def __init__(self,coord):self.coord=np.asarray(coord,dtype=float)
    def get_altloc(self):return ' '
    def get_occupancy(self):return 1.
class Residue(dict):
    def __init__(self,name,index,coord):
        super().__init__({} if coord is None else {'CA':Atom(coord)})
        self.resname=name;self.id=(' ',index,' ')
class Chain(list):id='A'


def test_exact_observed_sequence_contacts_and_no_self_loops():
    chain=Chain([Residue('ALA',1,[0,0,0]),Residue('CYS',2,[7.9,0,0]),Residue('ASP',3,[8,0,0])])
    contact,evidence=contact_from_chain(chain,'ACD')
    assert contact.shape==(3,3) and np.array_equal(contact,contact.T)
    assert not contact.diagonal().any()
    assert contact[0,1]==1 and contact[0,2]==0
    assert evidence['observed_sequence']=='ACD'
    with pytest.raises(ValueError,match='mismatch'):contact_from_chain(chain,'ACDE')
    with pytest.raises(ValueError,match='missing_CA'):contact_from_chain(Chain([Residue('ALA',1,None)]),'A')


def test_official_download_cache_reuse_and_tamper_rejection(tmp_path,monkeypatch):
    import src.data.prepare_tpeppro_structures as module
    calls=[]
    class Response(io.BytesIO):headers={}
    def fetch(request,timeout):
        calls.append(request.full_url)
        return Response(gzip.compress(b'data_fixture\n#\n'))
    monkeypatch.setattr(module,'urlopen',fetch)
    cache=StructureCache(tmp_path)
    path,receipt=cache.fetch('1abc')
    assert calls==['https://files.rcsb.org/download/1abc.cif.gz']
    assert cache.fetch('1abc')[1]==receipt
    assert len(calls)==1
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash_mismatch'):cache.fetch('1abc')


def test_failed_download_is_explicit_and_cached(tmp_path,monkeypatch):
    import src.data.prepare_tpeppro_structures as module
    from urllib.error import HTTPError
    calls=[]
    def fetch(request,timeout):
        calls.append(request.full_url);raise HTTPError(request.full_url,404,'missing',{},None)
    monkeypatch.setattr(module,'urlopen',fetch)
    cache=StructureCache(tmp_path)
    with pytest.raises(ValueError,match='download_failed'):cache.fetch('1abc')
    with pytest.raises(ValueError,match='cached_download_failure'):cache.fetch('1abc')
    assert len(calls)==1


def test_preparation_quarantines_unverifiable_pair_and_resumes(tmp_path, monkeypatch):
    import json
    import pandas as pd
    import src.data.prepare_tpeppro_structures as module
    source = tmp_path / 'outputs/public_pairs/all_pairs.csv'
    source.parent.mkdir(parents=True)
    dictionary = tmp_path / 'otherModels/TPepPro/data/receptor(14374)-peptide(9594)_dictionary.xlsx'
    dictionary.parent.mkdir(parents=True)
    dictionary.write_bytes(b'fixture dictionary')
    pd.DataFrame([
        dict(pair_id='pair1', peptide_id='pep_1', target_id='protein_1', sequence='AC', target_sequence='ACD', label_type='background', split='train'),
        dict(pair_id='pair2', peptide_id='pep_2', target_id='protein_1', sequence='DE', target_sequence='ACD', label_type='background', split='test'),
        dict(pair_id='positive', peptide_id='pep_3', target_id='protein_1', sequence='EF', target_sequence='ACD', label_type='positive', split='train'),
    ]).to_csv(source, index=False)
    monkeypatch.setattr(module, 'read_dictionary', lambda path: pd.DataFrame([
        {'source_id':'1abc_A', 'sequence':'AC'},
        {'source_id':'2abc_A', 'sequence':'AC'},
        {'source_id':'3abc_A', 'sequence':'ACD'},
        {'source_id':'4abc_A', 'sequence':'DE'},
    ]))
    calls = []
    monkeypatch.setattr(module.StructureCache, 'fetch', lambda self, pdb: (Path(pdb), {'url': f'https://files.rcsb.org/download/{pdb}.cif.gz', 'sha256':'fixture'}))
    def parse(path, pdb, chain, sequence):
        calls.append(pdb)
        if pdb in {'1abc','4abc'}:
            raise ValueError('observed_sequence_mismatch')
        return np.ones((len(sequence), len(sequence)), dtype=np.uint8) - np.eye(len(sequence), dtype=np.uint8), {'observed_sequence':sequence}
    monkeypatch.setattr(module, 'parse_chain', parse)
    output = tmp_path/'result'
    manifest = module.prepare(tmp_path, output, tmp_path/'cache', workers=1)
    assert manifest['verified_pairs'] == 1 and manifest['rejected_pairs'] == 1
    assert manifest['teacher_screening'] == 'not_performed'
    assert not manifest['v6_curation_complete']
    assert pd.read_csv(output/'teacher_pairs.csv').to_dict('records') == [{'receptor':'protein_1','peptide':'pep_1','pair_id':'pair1'}]
    record = json.loads((output/'records/pep_1.json').read_text())
    assert record['selected_source_id'] == '2abc_A'
    assert record['rejected_alternatives'][0]['source_id'] == '1abc_A'
    assert not (output/'contacts/pep_2.npz').exists()
    before = list(calls)
    module.prepare(tmp_path, output, tmp_path/'cache', workers=1, resume=True)
    assert calls == before
    source.write_text(source.read_text()+'\n')
    with pytest.raises(ValueError, match='changed'):
        module.prepare(tmp_path, output, tmp_path/'cache', workers=1, resume=True)

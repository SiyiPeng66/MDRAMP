from mdramp.protein_library_download import rank


def test_reference_selection_quality_precedes_identifier():
    def candidate(identifier,level='full',status='Standard',score=99):
        return {'id':identifier,'genomeAssembly':{'level':level},'proteomeCompletenessReport':{'cpdReport':{'status':status},'buscoReport':{'score':score}}}
    values=[candidate('UP9'),candidate('UP0','contig'),candidate('UP1',score=90),candidate('UP2',status='Outlier'),candidate('UP8')]
    assert sorted(values,key=rank)[0]['id']=='UP8'

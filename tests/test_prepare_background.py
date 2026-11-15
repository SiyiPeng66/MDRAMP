from src.data.prepare_background import HomologyIndex,eligible


def test_homology_filter_handles_substitution_and_indel():
    index=HomologyIndex(['ACDEFGHIKLMNPQRSTVWY'])
    assert index.matches('ACDEFGHIKLMNPQRSTVWY')
    assert index.matches('ACDEFGHIKLMNPQRSTVWA')
    assert index.matches('ACDEFGHIKLMNPQRSTVW')
    assert not index.matches('AAAAAAAAAAAAAAAAAAA')


def test_annotation_and_semantics():
    index=HomologyIndex([])
    row={'Entry':'TEST','Sequence':'ACDEFGHIKL','Protein names':'bacteriocin'}
    assert eligible(row,index)[1]=='annotation'
    row['Protein names']='Uncharacterized protein'
    result,_=eligible(row,index)
    assert result['label_type']=='unlabelled'


def test_qgram_shortlist_never_misses_global_identity_neighbors():
    import random
    from src.data.cluster_split import length_aware_identity
    rng=random.Random(23)
    alphabet='ACDEFGHIKLMNPQRSTVWY'
    references=[''.join(rng.choices(alphabet,k=n)) for n in range(10,51)]
    index=HomologyIndex(references)
    queries=[]
    for reference in references:
        for _ in range(3):
            chars=list(reference)
            for _ in range(rng.randrange(1,5)):
                event=rng.randrange(3);pos=rng.randrange(len(chars))
                if event==0:chars[pos]=rng.choice(alphabet)
                elif event==1:chars.insert(pos,rng.choice(alphabet))
                else:chars.pop(pos)
            queries.append(''.join(chars))
    for query in queries:
        expected=any(length_aware_identity(query,ref)>=.9 for ref in references)
        assert index.matches(query)==expected

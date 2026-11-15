from mdramp.functional_panel import classify,family


def test_scope_exclusions_and_positive_function_evidence():
    assert classify('Uncharacterized membrane protein',True,False,False)[0]==[]
    assert classify('Hypothetical DUF123 protein',False,True,False)[0]==[]
    assert 'functionally_named_essential' in classify('DNA gyrase subunit A',False,True,False)[0]
    assert 'cell_wall_envelope' in classify('Penicillin-binding protein 1A',True,False,False)[0]
    assert 'functional_transport' not in classify('Glucose transporter',False,False,False)[0]
    assert 'functional_transport' in classify('Glucose transporter',True,False,False)[0]
    assert classify('Unknown protein',False,False,True)[0]==['resistance_annotation']


def test_pfam_composition_not_invented_domain_order():
    x={'uniProtKBCrossReferences':[{'database':'Pfam','id':'PF02'},{'database':'Pfam','id':'PF01'}]}
    f=family(x);assert f[0]==('Pfam:PF01','Pfam:PF02');assert f[2]==()


def test_generic_essential_names_are_not_specific_function_evidence():
    for name in ['Membrane protein','putative protein','UPF0123 protein','DUF 123 protein']:
        assert 'functionally_named_essential' not in classify(name,True,True,False)[0]
    assert 'cell_wall_envelope' in classify('Putative peptidoglycan synthase',True,True,False)[0]
    assert 'membrane_lipid_homeostasis' in classify('Phosphatidylglycerol synthase',True,False,False)[0]

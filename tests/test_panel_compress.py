from mdramp.panel_compress import domains,true


def test_function_signature_includes_domains_and_nested_ec():
    entry={'uniProtKBCrossReferences':[{'database':'Pfam','id':'PF001'},{'database':'InterPro','id':'IPR001'},{'database':'PDB','id':'1ABC'}],
           'proteinDescription':{'recommendedName':{'ecNumbers':[{'value':'1.2.3.4'}]}}}
    assert domains(entry)==('EC:1.2.3.4','InterPro:IPR001','Pfam:PF001')
    assert domains({})==()
    assert true(True) and true('True') and not true(False)

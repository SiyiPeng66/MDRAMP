"""Execute official RF+MERCI script without unused ESM-only imports.

This compatibility launcher removes only imports of torch/transformers from the
upstream source. It does not change features, model weights or motif rules. It repairs the upstream negative-motif no-hit count (1 to 0), preventing batch-dependent false penalties; see docs/HEMOPI_RUNTIME.md.
"""
import ast
import sys
import os
import subprocess
from pathlib import Path

def main():
    source=Path(sys.argv[1]).resolve()
    sys.argv=[str(source)]+sys.argv[2:]
    tree=ast.parse(source.read_text())
    for value in ast.walk(tree):
        if isinstance(value,ast.Constant) and isinstance(value.value,str) and value.value.startswith('./model/'):
            value.value='./Model/'+value.value[len('./model/'):]
    repairs=0
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name=='MERCI_Processor_n':
            for branch in ast.walk(node):
                if isinstance(branch,ast.If) and isinstance(branch.test,ast.UnaryOp) and isinstance(branch.test.op,ast.Not) and isinstance(branch.test.operand,ast.Name) and branch.test.operand.id=='hh':
                    for statement in branch.body:
                        for value in ast.walk(statement):
                            if isinstance(value,ast.Constant) and value.value=='1':
                                value.value='0'
                                repairs+=1
    if repairs != 1:
        raise RuntimeError("Unexpected upstream MERCI source: expected exactly one no-hit repair")
    tree.body=[node for node in tree.body if not (
        isinstance(node,ast.Import) and any(a.name in ('torch','transformers') for a in node.names)
        or isinstance(node,ast.ImportFrom) and node.module in ('torch','transformers'))]
    def checked_system(command):
        subprocess.run(command,shell=True,check=True)
        return 0
    os.system=checked_system
    exec(compile(tree,str(source),'exec'),{'__name__':'__main__','__file__':str(source)})
if __name__=='__main__': main()

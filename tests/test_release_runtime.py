import numpy as np
import pandas as pd
import torch
from mdramp.release_runtime import Runtime

def test_duplicate_sequence_ids_preserve_order_without_duplicate_inference():
 r=Runtime.__new__(Runtime);r.model=torch.nn.Linear(1571,1)
 def extract(seq):
  assert seq==['AAA','CCC']
  return np.zeros((2,1571),dtype=np.float32),pd.DataFrame({'sequence':seq,'p_cons':[.5,.8]})
 r.extract=extract
 f=pd.DataFrame({'id':['a','b','c'],'sequence':['AAA','CCC','AAA']})
 out=r.predict(f)
 assert out.id.tolist()==['a','b','c']
 assert out.predicted_mic.iloc[0]==out.predicted_mic.iloc[2]

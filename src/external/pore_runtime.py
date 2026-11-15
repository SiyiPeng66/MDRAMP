"""Device-aware original PF-Teacher inference (ProtBert-BFD + supplied RF)."""
from pathlib import Path
import numpy as np
import torch
import joblib

class PoreTeacherRuntime:
    def __init__(self, model_root='otherModels/Pore-Forming', device='cpu'):
        from transformers import AutoTokenizer, AutoModel
        root=Path(model_root)
        self.tokenizer=AutoTokenizer.from_pretrained(root/'prot_bert_bfd',do_lower_case=False,local_files_only=True)
        self.model=AutoModel.from_pretrained(root/'prot_bert_bfd',local_files_only=True).to(device).eval()
        self.device=device
        self.classifier=joblib.load(root/'model.pkl')
        if not hasattr(self.classifier,'predict_proba'):
            raise ValueError('PF-Teacher must expose continuous probabilities')
    @torch.inference_mode()
    def score(self, sequence):
        sequence = str(sequence).strip().upper()
        if not sequence or set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
            raise ValueError("PF input must use standard amino acids")
        inputs=self.tokenizer(' '.join(sequence.upper()),return_tensors='pt')
        embedding=self.model(**{k:v.to(self.device) for k,v in inputs.items()}).last_hidden_state.cpu().numpy().reshape(-1)
        # Identical to upstream predict.py: retain special tokens, flatten, pad/truncate.
        embedding=np.pad(embedding,(0,max(0,43008-len(embedding))))[:43008]
        classes=list(self.classifier.classes_)
        value=float(self.classifier.predict_proba(embedding.reshape(1,-1))[0,classes.index(1)])
        if not np.isfinite(value) or not 0<=value<=1: raise ValueError('Invalid PF output')
        return value

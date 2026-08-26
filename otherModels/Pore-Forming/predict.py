import os
import warnings
warnings.filterwarnings("ignore")
import re
import numpy as np
import joblib
import torch
from transformers import AutoTokenizer, AutoModel, pipeline


# ==============================
# Global load (只加载一次)
# ==============================

print("Loading BERT model...")
tokenizer = AutoTokenizer.from_pretrained('./prot_bert_bfd', do_lower_case=False)
model = AutoModel.from_pretrained("./prot_bert_bfd")
fe = pipeline('feature-extraction', model=model, tokenizer=tokenizer, device=0)

print("Loading Random Forest model...")
clf_choice = joblib.load('model.pkl')

print("Model loaded successfully.\n")


# ==============================
# Feature extraction
# ==============================

def BERT_Extracting_seq_features(seq):
    sequences_Example = [" ".join(seq)]
    sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]

    embedding = fe(sequences_Example)
    embedding = np.array(embedding)
    embedding = embedding.flatten()

    if len(embedding) < 43008:
        pad_0_number = 43008 - len(embedding)
        embedding = np.pad(embedding, (0, pad_0_number))
    else:
        embedding = embedding[:43008]

    return [embedding]


# ==============================
# Prediction function
# ==============================

def predict_amp(sequence):
    """
    输入:
        sequence (str): 蛋白序列，例如 "GLFDIVKKVVGALGSL"
    
    输出:
        int: 0 或 1
            1 = AMP
            0 = non-AMP
    """

    sequence = sequence.strip().upper()

    embedding = BERT_Extracting_seq_features(sequence)
    pred = clf_choice.predict(embedding)

    return int(pred[0])


# ==============================
# Example usage
# ==============================

if __name__ == "__main__":
    test_seq = "VWVNGLPLRYL"
    result = predict_amp(test_seq)

    print("Input sequence:", test_seq)
    print("Prediction result:", result)
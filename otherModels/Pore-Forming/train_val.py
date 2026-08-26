# BERT-RF classifier for AMP prediction
# independent training code modified from Ruihan Dong version
# 24-03-07

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import warnings
warnings.filterwarnings("ignore")

import re
import numpy as np
import pandas as pd
from time import time
from tqdm.auto import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, matthews_corrcoef, roc_auc_score, f1_score

import torch
from transformers import AutoTokenizer, AutoModel, pipeline


def show_metrics(y_true, y_pred):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
        if y_true[i] == 1 and y_pred[i] == 0:
            fn += 1
        if y_true[i] == 0 and y_pred[i] == 1:
            fp += 1
        if y_true[i] == 0 and y_pred[i] == 0:
            tn += 1
    
    sensitivity = float(tp) / (float(tp) + float(fn))
    specificity = float(tn) / (float(tn) + float(fp))
    auroc = roc_auc_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    print('sensitivity {:.4f}, specificity {:.4f}, accuracy {:.4f}, precision {:.4f}, mcc {:.4f}, F1 {:.4f}, auroc {:.4f}'.format(
        sensitivity, specificity, accuracy, precision, mcc, f1, auroc))


# ==============================
# BERT feature extraction
# ==============================

tokenizer = AutoTokenizer.from_pretrained("./prot_bert_bfd", do_lower_case=False)
model = AutoModel.from_pretrained("./prot_bert_bfd")
fe = pipeline('feature-extraction', model=model, tokenizer=tokenizer, device=0)


def BERT_Extracting_seq_features(seq):
    sequences_Example = [" ".join(seq)]
    sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]

    embedding = fe(sequences_Example)
    embedding = np.array(embedding)
    embedding = embedding.flatten()

    # 固定长度 43008
    if len(embedding) < 43008:
        pad_0_number = 43008 - len(embedding)
        embedding = np.pad(embedding, (0, pad_0_number))
    else:
        embedding = embedding[:43008]
    return embedding


def read_dataset(filename):
    df = pd.read_csv(filename)
    encoding_seqs = []
    seqs = df['Seq'].values.tolist()

    for seq in tqdm(seqs):
        emb = BERT_Extracting_seq_features(seq)
        encoding_seqs.append(emb)

    labels = df['Label'].values.tolist()
    return encoding_seqs, labels


# ==============================
# Load dataset
# ==============================

datapath = './dataset/benchmark/'  # or cls_benchmark_imbalanced

train_reps, train_labels = read_dataset(datapath + 'training_data.csv')
val_reps, val_labels = read_dataset(datapath + 'val_data.csv')


# ==============================
# Random Forest Grid Search
# ==============================

n_estimators_list = [100, 200, 300]
max_depth_list = [None, 10, 20, 30]

print('Total trials: ', len(n_estimators_list) * len(max_depth_list))

best_mcc = 0
best_params = {
    'n_estimators': n_estimators_list[0],
    'max_depth': max_depth_list[0],
    'random_state': 42
}

for n_estimators in n_estimators_list:
    for max_depth in max_depth_list:
        grid_rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )

        grid_rf.fit(train_reps, train_labels)
        val_mcc = matthews_corrcoef(val_labels, grid_rf.predict(val_reps))

        print('Grid search running: n_estimators {}, max_depth {}, val_mcc {:.4f}'.format(
            n_estimators, max_depth, val_mcc))

        if val_mcc > best_mcc:
            best_mcc = val_mcc
            best_params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'random_state': 42
            }

print('best params: ', best_params)


# ==============================
# Train final model
# ==============================

t0 = time()
model_rf = RandomForestClassifier(**best_params, n_jobs=-1)
model_rf.fit(train_reps, train_labels)
t1 = time()

print('Training time: {:.3f} min'.format((t1 - t0) / 60))


# ==============================
# Evaluation
# ==============================

val_pre_label = model_rf.predict(val_reps)
show_metrics(val_labels, val_pre_label)

import joblib
joblib.dump(model_rf, 'model.pkl')
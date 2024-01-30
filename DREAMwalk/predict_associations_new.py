import argparse
import pickle
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

from utils import set_seed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_file', type=str, required=True)
    parser.add_argument('--pair_file', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--model_checkpoint', type=str, default='clf.pkl')
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    
    args = parser.parse_args()
    args = {'embeddingf':args.embedding_file,
     'pairf':args.pair_file,
     'seed':args.seed,
     'patience':args.patience,
     'modelf':args.model_checkpoint,
     'testr':args.test_ratio,
     'validr':args.validation_ratio
     }
    return args

def split_dataset(pairf, embeddingf, validr, testr, seed):

  with open(embeddingf, 'rb') as f:
    embeddings = pickle.load(f)

  nodes = [...] # liste des nodes

  xs, ys = [], []

  with open(pairf, 'r') as fin:
    for line in lines[1:]:
      drug = line[0]
      dis = line[1] 
     
      drug_idx = nodes.index(drug)  
      dis_idx = nodes.index(dis)

      drug_emb = embeddings[drug_idx]
      dis_emb = embeddings[dis_idx]

      xs.append(drug_emb - dis_emb)  
      ys.append(int(line[2]))
      
  # Split dataset

  return x, y

def return_scores(target_list, pred_list):
    metric_list = [
        accuracy_score, 
        roc_auc_score, 
        average_precision_score, 
        f1_score
    ] 
    
    scores = []
    for metric in metric_list:
        if metric in [roc_auc_score, average_precision_score]:
            scores.append(metric(target_list,pred_list))
        else: # accuracy_score, f1_score
            scores.append(metric(target_list, pred_list.round())) 
    return scores

def predict_associations():

  set_seed()

  x,y = split_dataset()

  clf = XGBClassifier() 
  clf.fit(x['train'], y['train'])

  # prediction and evaluation

  if __name__ == '__main__':
     predict_associations()
import joblib
import json
import csv
import pickle
import argparse
import numpy as np
import pandas as pd
import os

from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,TensorDataset
import random
from AAmodel_class import *
from AAmetric import *

def get_train_data(fea1,fea2,k,path):
    select_file = path + f"{k}.txt"
    with open(select_file, 'r') as f:
        indices_str = f.read().strip().split(',')
        selected_indices = [int(idx) for idx in indices_str]  

    fea1 = pd.read_csv(fea1, header=0, index_col=None)
    fea2 = pd.read_csv(fea2, header=None, index_col=None)
    fea = pd.concat([fea2, fea1], axis=1)

    features = fea.iloc[:, :2889]
    features = features.iloc[:, selected_indices]
    sequence=fea1.iloc[:,-1]
    return features,sequence

def test(Model,test_data,sequence,path):
    Model.load_state_dict(torch.load(path + 'model.pkl'))
    Model.eval()
    pp = []
    ss=None
    with torch.no_grad():
        test_dataa = torch.tensor(test_data.to_numpy(), dtype=torch.float32)
        predict = Model(test_dataa)
        pred = predict.detach().cpu().numpy().ravel().tolist()
        ss = sequence.tolist()
        if isinstance(pred, list):
            pp.extend(pred)
        else:
            pp.extend([pred])
    result = {'seq': ss, 'pred': pp}
    resultt = pd.DataFrame(result)
    resultt.to_csv(path + 'predict.csv', index_label=False, index=False)

def analyse(path):
    combined_results = []
    model_predictions = pd.read_csv(path + 'predict.csv')
    sequences = model_predictions.iloc[:, 0].values
    predictions = model_predictions.iloc[:, 1].values
    binary_predictions = np.where(predictions > 0.5, 1, 0)
    combined_results.append(binary_predictions)
    combined_df = pd.DataFrame({
        'sequence': sequences,  # 添加第一列（sequence）
        'model_pred': np.column_stack(combined_results).flatten()  # 二值化后的预测值
    })
    

    combined_df.to_csv(path + 'prediction.csv', index=False)
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('fea1', type=str,help='fea1:56 dim')
    parser.add_argument('fea2', type=str, help='fea2:2560 dim')
    parser.add_argument('path', type=str,help='save path')
    parser.add_argument('k', type=int, default=2400,help='feature selection index')

    args = parser.parse_args()

    test_data ,sequence= get_train_data(args.fea1,args.fea2,args.k,args.path)
    model_full = FullModel(in_dim=args.k)
    # device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model_full.to(device)
    test(model_full,test_data,sequence,args.path)
    analyse(args.path)


if __name__ == "__main__":
    main()



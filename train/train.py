import joblib
import json
import csv
import pickle
import argparse
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer,r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,TensorDataset
import random
import time
from Model import *
from Metric import *


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
seed_value = 42
set_seed(seed_value)

k_list=[1012]
def get_finetune_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_data', type=str,
                        default="",
                        help='train-features')
    parser.add_argument('--save_path', type=str,
                        default="./",
                        help='save file')

    parser.add_argument('--batchsize', type=int, default=64, help='batch_size 5 30')
    parser.add_argument('--epoch', type=int, default=15, help='epoch_numbers')
    parser.add_argument('--learning_rate',type=float, default=1e-04,help='training-learning_rate')
    parser.add_argument('--kfold', type=str,default=5)
    args = parser.parse_args()
    return args
def get_train_data(path,k):
    select_file = args.save_path + f"{k}.txt"
    with open(select_file, 'r') as f:
        indices_str = f.read().strip().split(',')
        selected_indices = [int(idx) for idx in indices_str]
    data = pd.read_csv(path, header=None, index_col=None)
 
    features = features.iloc[:, selected_indices]
    label = data.iloc[:, -3]
    sequence = data.iloc[:, -2]
    return features,label,sequence

def cal_average(cv_res,k):
    ave_se = [d['se'] for d in cv_res.values()]
    ave_see = sum(ave_se) / len(ave_se)
    ave_sp = [d['sp'] for d in cv_res.values()]
    ave_spp = sum(ave_sp) / len(ave_sp)
    ave_pre = [d['pre'] for d in cv_res.values()]
    ave_pree = sum(ave_pre) / len(ave_pre)
    ave_mcc = [d['mcc'] for d in cv_res.values()]
    ave_mccc = sum(ave_mcc) / len(ave_mcc)
    ave_acc = [d['acc'] for d in cv_res.values()]
    ave_accc = sum(ave_acc) / len(ave_acc)
    ave_auc = [d['auc_prc'] for d in cv_res.values()]
    ave_auccc = sum(ave_auc) / len(ave_auc)
    ave_auc_roc = [d['auc_roc'] for d in cv_res.values()]
    ave_aucccc = sum(ave_auc_roc) / len(ave_auc_roc)

    logger.info('CV{}-average metrics result:  se:{}, sp:{}, pre:{},mcc:{},acc:{},auc:{},auc_roc:{}'.
                format(args.kfold, ave_see, ave_spp, ave_pree, ave_mccc, ave_accc, ave_auccc, ave_aucccc))
    result_data = {
        'index': k,
        'acc': ave_accc,
        'auc_roc': ave_aucccc,
        'mcc': ave_mccc,
        'se': ave_see,
        'sp': ave_spp,
        'pre': ave_pree,
        'auc': ave_auccc,

    }
    # Convert the result data to a DataFrame
    df = pd.DataFrame(result_data, index=[0])
    # Save the result to CSV
    df.to_csv(args.save_path+'train_result.csv', mode='a', header=False, index=False)  # Append to the file if it exists
def train_model(model_dim,train_data,train_labelS,save_path,k):
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model_full = FullModel(in_dim=model_dim)
    model_full.to(device)
    best_res = {'acc': 0}
    cv_res = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for cv, (train_index, test_index) in enumerate(kf.split(train_data)):
        X_train, X_test = train_data.iloc[train_index, :], train_data.iloc[test_index, :]
        y_train, y_test = train_labelS.iloc[train_index], train_labelS.iloc[test_index]
        train_input = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
        val_input = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
        train_label = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
        val_label = torch.tensor(y_test.to_numpy(), dtype=torch.float32)

        datasets_train = TensorDataset(train_input, train_label)
        datasets_val = TensorDataset(val_input, val_label)
        trainloader = DataLoader(datasets_train, batch_size=args.batchsize, shuffle=True)
        valloader = DataLoader(datasets_val, batch_size=args.batchsize, shuffle=True)

        for epoch in range(args.epoch):
            train_loss = 0.0
            trainnum=len(train_input)

            for inputs, labels in trainloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                criterion =  nn.BCELoss(reduction='none')
                optimizer = optim.Adam(model_full.parameters(), lr=args.learning_rate)
                output=model_full(inputs)
                labels = labels.unsqueeze(1)
                loss = criterion(output, labels)
                loss = loss.sum()
                loss.backward()
                optimizer.step()
                train_loss = train_loss+loss.item()
                print('batch over')
            train_loss = train_loss / trainnum
            model_full.eval()
            eval_metric = Meter()
            with torch.no_grad():
                for tinputs, tlabels in valloader:
                    tinputs = tinputs.to(device)
                    tlabels = tlabels.to(device)

                    predict = model_full(tinputs)
                    true = tlabels.detach().cpu().numpy().squeeze().tolist()
                    pred = predict.detach().cpu().numpy().ravel().tolist()

                    if isinstance(true, list) and isinstance(pred, list):
                        eval_metric.update(pred, true)
                    else:
                        eval_metric.update([pred], [true])
                eval_res = eval_metric.compute_metric('classification', ['se', 'sp', 'pre', 'mcc', 'acc', 'auc_prc','auc_roc'])

                logger.info(f'fold {cv +1} epoch {epoch + 1} train-loss {train_loss} eval-loss {eval_res}')

            print(epoch, 'test over!')
            PATH_save_bert = save_path + 'model_{}'.format(str(cv+1)) + '.pkl'
            if eval_res['acc'] > best_res['acc']:
                best_res = eval_res
                torch.save(model_full.state_dict(), PATH_save_bert)
                logger.info(f'fold {cv + 1} epoch {epoch + 1} saved the model')

        logger.info(f'fold {cv + 1} best result is {best_res}')
        cv_res[cv] = best_res
        cv += 1
        best_res = {'acc': 0}
    best_fold = max(cv_res, key=lambda x: cv_res[x]['acc'])
    logger.info(f'CV finished, best fold is {best_fold}, and its result is {cv_res[best_fold]}')
    cal_average(cv_res,k)
    logger.info('next is outside dataset test')


if __name__ == "__main__":
    import os
    args = get_finetune_config()
    for k in k_list:
        save_path=args.save_path+f"{k}/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        train_data, train_label,seqq = get_train_data(args.train_data,k)
        model_dim=k
        train_model(model_dim,train_data, train_label,save_path,k)
        print( "over!")


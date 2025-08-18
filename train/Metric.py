import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score, confusion_matrix, precision_recall_curve, auc, r2_score
import logging
from scipy.stats import pearsonr
import argparse
def get_param():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str,
                        default="HLF_human_n",
                        help='sequence-file:HLF_intestine HLF_human_n HLF_rat_n')
    parser.add_argument('--fea_path', type=str,
                        default="/home/ubuntu/Documents/work-zll/half-life/data/half-life_data_split/2/",
                        help='sequence-file')
    parser.add_argument('--batch_size', type=int, default=30, help='split data')
    parser.add_argument('--model_dim', type=int,default=2560,help='esm_model dim')
    parser.add_argument('--rep_layer', type=int, default=33, help='esm_model layers')
    parser.add_argument('--gpu', type=int, default=0, help='cuda index')
    parser.add_argument('--use_cuda', default=False, help='cuda use')
    parser.add_argument('--learning_rate',type=float, default=1e-05, help='Train-learning_rate')
    parser.add_argument('--epoch', type=int, default=15, help='Train-Epoch')
    parser.add_argument('--save_path',type=str,default='/home/ubuntu/Documents/work-zll/half-life/1result/intesine/no6_ESM_ANN/lr_1e_04/',help='Train-model_save_path')
    parser.add_argument('--model_path', type=str,
                        default='/home/ubuntu/Documents/work-zll/half-life/1result/intesine/no6_ESM_ANN/lr_1e_04/',
                        help='Train-model_save_path')
    # parser.add_argument('--test_file', type=str,
    #                     default='/home/ubuntu/Documents/work-zll/half-life/data/half-life_data_split/2/HLF_intestine_test.csv',
    #                     help='test file ')


    parser.add_argument('--kfold',type=int,default=5)
    args = parser.parse_args()
    return args
class Meter(object):
    '''
    steps:
    meter.update()
    meter.gather()
    '''
    def __init__(self):
        self.y_score = []
        self.y_true = []
    def update(self, y_score, y_true):
        self.y_score.extend(y_score)
        self.y_true.extend(y_true)
    def calculate_cmat(self):
        # self.y_score = [item if isinstance(item, (int, float)) else item[0] for item in self.y_score]
        self.y_score = [item if isinstance(item, (float)) else item[0] for item in self.y_score]
        self.y_score=np.array(self.y_score)
        y_pred = np.where(self.y_score > 0.5, 1, 0)
        tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred, labels=[0, 1]).ravel()
        return tn, fp, fn, tp
    @staticmethod
    def calculate_se(tn, fp, fn, tp):
        se = tp / (tp + fn)
        return se

    @staticmethod
    def calculate_sp(tn, fp, fn, tp):
        sp = tn / (tn + fp)
        return sp

    @staticmethod
    def calculate_pre(tn, fp, fn, tp):
        pre = tp / (tp + fp)
        return pre

    @staticmethod
    def calculate_acc(tn, fp, fn, tp):
        acc = (tp + tn) / (tn + fp + fn + tp)
        return acc

    @staticmethod
    def calculate_mcc(tn, fp, fn, tp):
        mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return mcc

    @staticmethod
    def calculate_f1(tn, fp, fn, tp):
        f1 = (2 * tp) / (2 * tp + fp + fn)
        return f1

    def calculate_auc_roc(self):
        auc_roc = roc_auc_score(self.y_true, self.y_score)
        return auc_roc

    def calculate_auc_prc(self):
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_score, pos_label=1)
        auc_prc = auc(recall, precision)
        return auc_prc

    def calculate_rmse(self):
        self.y_score = [item if isinstance(item, (float)) else item[0] for item in self.y_score]
        self.y_true = [item if isinstance(item, (float)) else item[0] for item in self.y_true]

        rmse = np.sqrt(mean_squared_error(y_true=self.y_true,
                                          y_pred=self.y_score))

        return rmse

    def calculate_pearson(self):
        y_true = self.y_true
        y_score = self.y_score
        pcc = pearsonr(y_true, y_score)[0]
        return pcc

    def calculate_r2(self):

        r2 = r2_score(self.y_true, self.y_score)
        return r2
    def calculate_mae(self):
        mae = mean_squared_error(self.y_true, self.y_score)
        return mae
    def compute_metric(self, task_type, metric_name):
        '''
        task_type should be 'classification' or 'regression'
        metric_name should be a list,
        ['se', 'sp', 'pre', 'acc', 'mcc', 'f1', 'auc_roc', 'auc_prc', 'rmse']
        '''
        res = {}
        if task_type == 'regression':
            for task in metric_name:
                if task == 'rmse':
                    res['rmse'] = self.calculate_rmse()
                if task == 'pcc':
                    res['pcc'] = self.calculate_pearson()
                if task == 'r2':
                    res['r2'] = self.calculate_r2()
                if task == 'mae':
                    res['mae'] = self.calculate_mae()
        elif task_type == 'classification':
            tn, fp, fn, tp = self.calculate_cmat()
            for task in metric_name:
                if task == 'se':
                    res['se'] = self.calculate_se(tn, fp, fn, tp)
                if task == 'sp':
                    res['sp'] = self.calculate_sp(tn, fp, fn, tp)
                if task == 'pre':
                    res['pre'] = self.calculate_pre(tn, fp, fn, tp)
                if task == 'acc':
                    res['acc'] = self.calculate_acc(tn, fp, fn, tp)
                if task == 'mcc':
                    res['mcc'] = self.calculate_mcc(tn, fp, fn, tp)
                if task == 'f1':
                    res['f1'] = self.calculate_f1(tn, fp, fn, tp)
                if task == 'auc_roc':
                    res['auc_roc'] = self.calculate_auc_roc()
                if task == 'auc_prc':
                    res['auc_prc'] = self.calculate_auc_prc()
        else:
            raise ValueError('wrong task type')

        return res
# class Meter(object):
#     '''
#     steps:
#     meter.update()
#     meter.gather()
#     '''
#     def __init__(self):
#         self.y_score = []
#         self.y_true = []
#     def update(self, y_score, y_true):
#         self.y_score.extend(y_score)
#         self.y_true.extend(y_true)
#     def calculate_cmat(self):
#
#         y_pred = np.where(self.y_score > 0.5, 1, 0)
#         tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred, labels=[0, 1]).ravel()
#         return tn, fp, fn, tp
#     @staticmethod
#     def calculate_se(tn, fp, fn, tp):
#         se = tp / (tp + fn)
#         return se
#
#     @staticmethod
#     def calculate_sp(tn, fp, fn, tp):
#         sp = tn / (tn + fp)
#         return sp
#
#     @staticmethod
#     def calculate_pre(tn, fp, fn, tp):
#         pre = tp / (tp + fp)
#         return pre
#
#     @staticmethod
#     def calculate_acc(tn, fp, fn, tp):
#         acc = (tp + tn) / (tn + fp + fn + tp)
#         return acc
#
#     @staticmethod
#     def calculate_mcc(tn, fp, fn, tp):
#         mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
#         return mcc
#
#     @staticmethod
#     def calculate_f1(tn, fp, fn, tp):
#         f1 = (2 * tp) / (2 * tp + fp + fn)
#         return f1
#
#     def calculate_auc_roc(self):
#         auc_roc = roc_auc_score(self.y_true, self.y_score)
#         return auc_roc
#
#     def calculate_auc_prc(self):
#         precision, recall, _ = precision_recall_curve(self.y_true, self.y_score, pos_label=1)
#         auc_prc = auc(recall, precision)
#         return auc_prc
#
#     def calculate_rmse(self):
#         rmse = np.sqrt(mean_squared_error(self.y_true,self.y_score))
#
#         return rmse
#
#     def calculate_pearson(self):
#         y_true = self.y_true
#         y_score = self.y_score
#         pcc = pearsonr(y_true, y_score)[0]
#         return pcc
#
#     def calculate_r2(self):
#
#         r2 = r2_score(self.y_true, self.y_score)
#         return r2
#     def calculate_mae(self):
#         mae = mean_squared_error(self.y_true, self.y_score)
#         return mae
#     def compute_metric(self, task_type, metric_name):
#         '''
#         task_type should be 'classification' or 'regression'
#         metric_name should be a list,
#         ['se', 'sp', 'pre', 'acc', 'mcc', 'f1', 'auc_roc', 'auc_prc', 'rmse']
#         '''
#         res = {}
#         if task_type == 'regression':
#             for task in metric_name:
#                 if task == 'rmse':
#                     res['rmse'] = self.calculate_rmse()
#                 if task == 'pcc':
#                     res['pcc'] = self.calculate_pearson()
#                 if task == 'r2':
#                     res['r2'] = self.calculate_r2()
#                 if task == 'mae':
#                     res['mae'] = self.calculate_mae()
#         elif task_type == 'classification':
#             tn, fp, fn, tp = self.calculate_cmat()
#             for task in metric_name:
#                 if task == 'se':
#                     res['se'] = self.calculate_se(tn, fp, fn, tp)
#                 if task == 'sp':
#                     res['sp'] = self.calculate_sp(tn, fp, fn, tp)
#                 if task == 'pre':
#                     res['pre'] = self.calculate_pre(tn, fp, fn, tp)
#                 if task == 'acc':
#                     res['acc'] = self.calculate_acc(tn, fp, fn, tp)
#                 if task == 'mcc':
#                     res['mcc'] = self.calculate_mcc(tn, fp, fn, tp)
#                 if task == 'f1':
#                     res['f1'] = self.calculate_f1(tn, fp, fn, tp)
#                 if task == 'auc_roc':
#                     res['auc_roc'] = self.calculate_auc_roc()
#                 if task == 'auc_prc':
#                     res['auc_prc'] = self.calculate_auc_prc()
#         else:
#             raise ValueError('wrong task type')
#
#         return res
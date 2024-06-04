import numpy as np
import pandas as pd
import skrf as rf
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, random_split
import torch.nn.functional as F
import torch
from torch import nn
import csv


loss_fn = nn.BCELoss()
loss_fn_2 = nn.BCELoss(reduction='none')
loss_fn_3 = nn.CrossEntropyLoss()


def obtain_posterior_from_net_out(D, cost_function_v):
    if cost_function_v == 5:
        R = (1-D)/D
    elif cost_function_v == 3:
        R = torch.exp(D)  # because linear layer is used
    return R


def save_dict_lists_csv(path, dictionary):
    with open(path, "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(dictionary.keys())
        writer.writerows(zip(*dictionary.values()))


def to_categorical(y, num_classes, t_tensor=False, dtype="uint8"):
    if t_tensor:
        return F.one_hot(y, num_classes=num_classes)
    else:
        return np.eye(num_classes, dtype=dtype)[y.astype(int).squeeze()]


def sl_cost_fcn(out_1, out_2, data_tx, num_classes, alpha):
    loss_1 = sl_first(out_1.squeeze(), data_tx, num_classes)
    loss_2 = sl_sec(out_2.squeeze())
    loss = loss_1 + alpha * loss_2
    return loss


def sl_first(y_pred, data_tx, num_classes, t_tensor=True):
    loss_1 = torch.matmul(y_pred, torch.transpose(data_tx.float(), 0, 1))
    loss_1 = torch.diagonal(loss_1, 0)
    loss_1 = torch.mean(loss_1)
    return loss_1


def sl_sec(y_pred):
    eps = 1e-6
    log_pred = torch.log(y_pred + eps) - y_pred
    sum_log_pred = torch.mean(log_pred, dim=1)
    loss = torch.mean(sum_log_pred)
    return -loss


def from_numpy_to_dataframe(numpy_dataset):
    df = pd.DataFrame(numpy_dataset).reset_index(drop=True)
    return df


def from_s4p_to_df(path, filter_frequencies=[]):
    ntwk = rf.Network(path)
    df = ntwk.to_dataframe('s')
    df = df.reset_index()
    df = df.rename({'index': 'Frequency'}, axis=1)
    transmission_coefficients = True
    if filter_frequencies:
        df = df[df["Frequency"] > filter_frequencies[0]]
        df = df[df["Frequency"] < filter_frequencies[1]]
    if transmission_coefficients:
        df = df.drop("s 11", axis='columns')
        df = df.drop("s 22", axis='columns')
        df = df.drop("s 33", axis='columns')
        df = df.drop("s 44", axis='columns')
    return df

def compute_module(a):
    return np.sqrt((a.real)**2 + (a.imag)**2)
from __future__ import print_function, division
import numpy as np
import scipy.io as sio
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
import torch
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, random_split
from torch import nn
import torch.nn.functional as F
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils import *
from classes import *
from datetime import datetime
import mat4py


def load_dataset_v0(list_paths_datasets):
    num_features = 320
    composed_transform = transforms.Compose([ToTensor()])
    dataframe = pd.DataFrame(columns=range(num_features))
    for idx, path in enumerate(list_paths_datasets):
        data = mat4py.loadmat(path)
        new_data = list(map(list, zip(*data['H{}'.format(idx)])))
        tmp_dataframe = pd.DataFrame(new_data, columns=range(num_features))
        for i in range(tmp_dataframe.shape[0]):
            for j in range(tmp_dataframe.shape[1]):
                tmp_dataframe.iloc[i][j] = tmp_dataframe.iloc[i][j][0]
        scaler = StandardScaler()
        scaler = scaler.fit(tmp_dataframe)
        std_tmp_dataframe = from_numpy_to_dataframe(scaler.transform(tmp_dataframe))
        std_tmp_dataframe['label'] = idx
        dataframe = pd.concat([dataframe, tmp_dataframe], ignore_index=True)
    dataset = MyDataset(dataframe, composed_transform)
    return dataset

def load_dataset_raw(type):
    list_paths_datasets = ["Datasets/Raw/Empty", "Datasets/Raw/Person",
                           "Datasets/Raw/Two_People"
                           ]
    num_classes = len(list_paths_datasets)
    if type == "raw_reduced":
        frequencies_extremes = [4e9, 4.5e9]
    else:
        frequencies_extremes = [0, np.inf]

    columns_list_tmp = ["Frequency", "s 11", "s 21", "s 31", "s 41", "s 12", "s 22", "s 32", "s 42", "s 13", "s 23",
                    "s 33", "s 43", "s 14", "s 24", "s 34", "s 44"]
    columns_list = ["Frequency", "s 11", "s 21", "s 31", "s 41", "s 12", "s 22", "s 32", "s 42", "s 13", "s 23",
                    "s 33", "s 43", "s 14", "s 24", "s 34", "s 44", "Number_of_people"]
    df = pd.DataFrame()
    for idx, dir in enumerate(list_paths_datasets):
        for filename in os.listdir(dir):
            f = os.path.join(dir, filename)
            # Obtain dataframe starting from s4p file
            df_s_params_tmp = from_s4p_to_df(f, filter_frequencies=frequencies_extremes)
            for column in df_s_params_tmp.columns:
                df_s_params_tmp[column] = df_s_params_tmp[column].apply(compute_module)
            new_df_s_params_tmp = pd.DataFrame(np.reshape(df_s_params_tmp.values, (1, -1)))
            new_df_s_params_tmp['Number_of_people'] = idx
            df = pd.concat([df, new_df_s_params_tmp])

    df = df.reset_index(drop=True)
    # Split features from the label
    y = df["Number_of_people"]
    x = df.drop(columns=["Number_of_people"], axis=1)
    scaler = StandardScaler()
    scaler = scaler.fit(x)
    x = scaler.transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)
    composed_transform = transforms.Compose([ToTensor()])
    train_dataset = MyDataset(from_numpy_to_dataframe(x_train), from_numpy_to_dataframe(y_train), composed_transform)
    test_dataset = MyDataset(from_numpy_to_dataframe(x_test), from_numpy_to_dataframe(y_test), composed_transform)
    return train_dataset, test_dataset, x_train.shape[1], num_classes


def print_dataset_raw(type):

    list_paths_datasets = ["Datasets/Raw/Empty", "Datasets/Raw/Person",
                           "Datasets/Raw/Two_People"
                           ]
    num_classes = len(list_paths_datasets)
    if type == "raw_reduced":
        frequencies_extremes = [4e9, 4.5e9]
    else:
        frequencies_extremes = [0, np.inf]

    columns_list_tmp = ["Frequency", "s 11", "s 21", "s 31", "s 41", "s 12", "s 22", "s 32", "s 42", "s 13", "s 23",
                        "s 33", "s 43", "s 14", "s 24", "s 34", "s 44"]
    columns_list = ["Frequency", "s 11", "s 21", "s 31", "s 41", "s 12", "s 22", "s 32", "s 42", "s 13", "s 23",
                    "s 33", "s 43", "s 14", "s 24", "s 34", "s 44", "Number_of_people"]

    df = pd.DataFrame()
    for idx, dir in enumerate(list_paths_datasets):
        for filename in os.listdir(dir):
            f = os.path.join(dir, filename)
            # Obtain dataframe starting from s4p file
            df_s_params_tmp = from_s4p_to_df(f, filter_frequencies=frequencies_extremes)
            for column in df_s_params_tmp.columns:
                df_s_params_tmp[column] = df_s_params_tmp[column].apply(compute_module)
            df_s_params_tmp['Number_of_people'] = idx
            df = pd.concat([df, df_s_params_tmp])
    s_21_0 = df.query("Number_of_people = '0'")["s 21"]
    s_21_1 = df.query("Number_of_people = '1'")["s 21"]
    s_21_2 = df.query("Number_of_people = '2'")["s 21"]


def load_dataset_lambdas(list_paths_datasets):
    keys_list = ['Lambda01', 'Lambda02', 'Lambda03', 'Lambda04', 'Lambda1', 'Lambda2', 'Lambda3', 'Lambda4',
                 'Lambda1', 'Lambda2', 'Lambda3', 'Lambda4']
    num_features_part = 4096
    num_features = num_features_part*4
    num_classes = 3
    composed_transform = transforms.Compose([ToTensor()])
    x_dataframe_0 = pd.DataFrame()
    x_dataframe_1 = pd.DataFrame()
    x_dataframe_2 = pd.DataFrame()
    y_dataframe = pd.DataFrame(columns=['label'])
    for idx, path in enumerate(list_paths_datasets):
        data = mat4py.loadmat(path)
        new_data = list(map(list, zip(*data[keys_list[idx]])))
        tmp_x_dataframe = pd.DataFrame(new_data, columns=range(num_features_part))
        if idx<4:
            x_dataframe_0 = pd.concat([x_dataframe_0, tmp_x_dataframe], axis=1)
        elif idx>=4 and idx<8:
            x_dataframe_1 = pd.concat([x_dataframe_1, tmp_x_dataframe], axis=1)
        else:
            x_dataframe_2 = pd.concat([x_dataframe_2, tmp_x_dataframe], axis=1)

    x_dataframe = pd.concat([x_dataframe_0, x_dataframe_1, x_dataframe_2], ignore_index=True)
    y_dataframe['label'] = [0] * 320 + [1] * 320 + [2] * 320
    scaler = StandardScaler()
    scaler = scaler.fit(x_dataframe)
    x_dataframe = scaler.transform(x_dataframe)
    x_train, x_test, y_train, y_test = train_test_split(x_dataframe, y_dataframe, test_size=0.2, random_state=0,
                                                        stratify=y_dataframe)
    train_dataset = MyDataset(from_numpy_to_dataframe(x_train), from_numpy_to_dataframe(y_train), composed_transform)
    test_dataset = MyDataset(from_numpy_to_dataframe(x_test), from_numpy_to_dataframe(y_test), composed_transform)
    return train_dataset, test_dataset, num_features, num_classes


def load_dataset_cauchy():
    list_paths_datasets = ["Datasets/Cauchy/H0_21_Cauchy_poly.mat", "Datasets/Cauchy/H0_31_Cauchy_poly.mat",
                           "Datasets/Cauchy/H0_32_Cauchy_poly.mat", "Datasets/Cauchy/H0_41_Cauchy_poly.mat",
                           "Datasets/Cauchy/H0_42_Cauchy_poly.mat", "Datasets/Cauchy/H0_43_Cauchy_poly.mat",
                           "Datasets/Cauchy/H1_21_Cauchy_poly.mat", "Datasets/Cauchy/H1_31_Cauchy_poly.mat",
                           "Datasets/Cauchy/H1_32_Cauchy_poly.mat", "Datasets/Cauchy/H1_41_Cauchy_poly.mat",
                           "Datasets/Cauchy/H1_42_Cauchy_poly.mat", "Datasets/Cauchy/H1_43_Cauchy_poly.mat",
                           "Datasets/Cauchy/H2_21_Cauchy_poly.mat", "Datasets/Cauchy/H2_31_Cauchy_poly.mat",
                           "Datasets/Cauchy/H2_32_Cauchy_poly.mat", "Datasets/Cauchy/H2_41_Cauchy_poly.mat",
                           "Datasets/Cauchy/H2_42_Cauchy_poly.mat", "Datasets/Cauchy/H2_43_Cauchy_poly.mat"
                           ]
    keys_list = ['H0_21', 'H0_31', 'H0_32', 'H0_41', 'H0_42', 'H0_43',
                 'H1_21', 'H1_31', 'H1_32', 'H1_41', 'H1_42', 'H1_43',
                 'H2_21', 'H2_31', 'H2_32', 'H2_41', 'H2_42', 'H2_43']

    num_features_part = 67
    num_features = num_features_part * 6
    num_classes = 3
    x_dataframe_0 = pd.DataFrame()
    x_dataframe_1 = pd.DataFrame()
    x_dataframe_2 = pd.DataFrame()
    y_dataframe = pd.DataFrame(columns=['label'])
    composed_transform = transforms.Compose([ToTensor()])
    for idx, path in enumerate(list_paths_datasets):
        data = mat4py.loadmat(path)
        new_data = list(map(list, zip(*data[keys_list[idx]])))
        tmp_x_dataframe = pd.DataFrame(new_data, columns=range(num_features_part))
        if keys_list[idx][1]=="0":
            x_dataframe_0 = pd.concat([x_dataframe_0, tmp_x_dataframe], axis=1)
        elif keys_list[idx][1]=="1":
            x_dataframe_1 = pd.concat([x_dataframe_1, tmp_x_dataframe], axis=1)
        else:
            x_dataframe_2 = pd.concat([x_dataframe_2, tmp_x_dataframe], axis=1)

    x_dataframe = pd.concat([x_dataframe_0, x_dataframe_1, x_dataframe_2], ignore_index=True)
    x_dataframe.columns = range(x_dataframe.shape[1])
    y_dataframe['label'] = [0] * x_dataframe_0.shape[0] + [1] * x_dataframe_1.shape[0] + [2] * x_dataframe_2.shape[0]
    x_dataframe.replace(np.inf, 1e6, inplace=True)
    x_dataframe.replace(-np.inf, -1e6, inplace=True)
    scaler = StandardScaler()
    scaler = scaler.fit(x_dataframe)
    x_dataframe = scaler.transform(x_dataframe)
    x_train, x_test, y_train, y_test = train_test_split(x_dataframe, y_dataframe, test_size=0.1, random_state=0,
                                                        stratify=y_dataframe)
    train_dataset = MyDataset(from_numpy_to_dataframe(x_train), from_numpy_to_dataframe(y_train), composed_transform)
    test_dataset = MyDataset(from_numpy_to_dataframe(x_test), from_numpy_to_dataframe(y_test), composed_transform)

    return train_dataset, test_dataset, num_features, num_classes


def train_model(model, train_dataset, test_dataset, cost_function_v=1, num_classes=10, batch_size=100, epochs=10, device="cpu",
                verbose=True, save_epochs=[], save_training_loss=False, lr=0.001, alpha=1, random_seed=0):

    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.00001)
    model.train()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    losses = []
    for epoch in range(epochs):
        if verbose:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Starting epoch training at time =", current_time)
            print("EPOCH: ", epoch+1)
        loss_batch = []
        total = 0
        correct = 0
        for sample_batched in train_dataloader:
            data_rx = sample_batched[0].squeeze().to(device)
            data_tx = sample_batched[1].squeeze().long().to(device)
            current_batch_size = len(sample_batched[0])
            optimizer.zero_grad()
            data_y = get_random_batch(train_dataset, batch_size=current_batch_size).to(device)
            out_1, out_2 = model(data_rx, data_y)
            R_all = obtain_posterior_from_net_out(out_1, cost_function_v)
            _, predicted = R_all.max(1)
            total += data_tx.size(0)
            correct += predicted.eq(data_tx).sum().item()
            accuracy_1 = 100. * correct / total
            loss = compute_loss_divergence(cost_function_v, out_1, out_2, data_tx, num_classes, current_batch_size, alpha, device)
            loss.backward()
            optimizer.step()
            loss_batch.append(loss.item())
        print("Epoch loss: ", np.mean(loss_batch))
        losses.append(np.mean(loss_batch))
    if save_training_loss:
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss cost function v{}".format(cost_function_v))
        plt.savefig("LossPlots/Loss cost function v{}_epochs{}.png".format(cost_function_v, epochs))
    return model


def test_model(model, test_dataset, cost_function_v=1, device="cpu"):

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    model.eval()
    test_size = len(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=0)

    with torch.no_grad():
        total = 0
        correct = 0
        for sample_batched in test_dataloader:
            data_rx = sample_batched[0].squeeze().to(device)
            data_tx = sample_batched[1].squeeze().to(device)
            D_all, _ = model(data_rx, data_rx)  # get all density-ratios
            R_all = obtain_posterior_from_net_out(D_all, cost_function_v)
            # Compute the accuracy
            _, predicted = R_all.max(1)
            total += data_tx.size(0)
            correct += predicted.eq(data_tx).sum().item()
        accuracy = 100. * correct / total
        print("Test accuracy: ", accuracy)

        return accuracy


def convergence_study(train_dataset, test_dataset, input_dim, num_classes, main_opt_params,
                      main_proc_params, epochs_list, type="raw"):

    max_num_train_epochs = int(np.max(epochs_list) + 1)
    list_cf_v = [3,5]
    for random_seed in main_proc_params['random_seed']:
        for cost_func_v in list_cf_v:
            torch.manual_seed(random_seed)
            random.seed(random_seed)
            np.random.seed(random_seed)

            model = choose_nn_model(input_dim, num_classes, cost_func_v, main_proc_params['device'])
            # Train
            trained_net = train_model(model, train_dataset, test_dataset, cost_function_v=cost_func_v, num_classes=num_classes, batch_size=main_proc_params['batch_size'], epochs=max_num_train_epochs, device=main_proc_params['device'],
                    verbose=True, save_epochs=epochs_list, save_training_loss=False, lr=main_opt_params['learning_rate'], alpha=main_proc_params['alpha'], random_seed=random_seed)
            # Test
            accuracy = test_model(trained_net, test_dataset, cost_function_v=cost_func_v, device=main_proc_params['device'])
            save_dict_lists_csv("ClassificationResults/divergence_{}_seed_{}_type_{}_epochs_{}.csv".format(cost_func_v, random_seed, type, max_num_train_epochs),
                                {'Accuracy': [accuracy]})


def main_convergence_study_lambdas(main_opt_params, main_proc_params):
    print("Convergence study lambdas...")
    list_paths_lambdas = ["Datasets/Lambdas/Lambda1_Empty.mat", "Datasets/Lambdas/Lambda2_Empty.mat",
                          "Datasets/Lambdas/Lambda3_Empty.mat", "Datasets/Lambdas/Lambda4_Empty.mat",
                          "Datasets/Lambdas/Lambda1_Person.mat", "Datasets/Lambdas/Lambda2_Person.mat",
                          "Datasets/Lambdas/Lambda3_Person.mat", "Datasets/Lambdas/Lambda4_Person.mat",
                          "Datasets/Lambdas/Two_Lambda1.mat", "Datasets/Lambdas/Two_Lambda2.mat",
                          "Datasets/Lambdas/Two_Lambda3.mat", "Datasets/Lambdas/Two_Lambda4.mat"]
    train_dataset, test_dataset, input_dim, num_classes = load_dataset_lambdas(list_paths_lambdas)
    latent_dim = 1000
    print("train_dataset.len: ", len(train_dataset))
    print("test_dataset.len: ", len(test_dataset))
    print("input_dim: ", input_dim)
    print("num_classes: ", num_classes)
    epochs_list = range(10)
    type = "lambdas"
    convergence_study(train_dataset, test_dataset, input_dim, num_classes, main_opt_params,
                              main_proc_params, epochs_list, type=type)


def main_convergence_study_cauchy(main_opt_params, main_proc_params):
    print("Convergence study cauchy...")
    train_dataset, test_dataset, input_dim, num_classes = load_dataset_cauchy()
    print("train_dataset.len: ", len(train_dataset))
    print("test_dataset.len: ", len(test_dataset))
    print("input_dim: ", input_dim)
    print("num_classes: ", num_classes)
    epochs_list = range(10)
    type = "cauchy"
    convergence_study(train_dataset, test_dataset, input_dim, num_classes, main_opt_params,
                      main_proc_params, epochs_list, type=type)


def main_convergence_study_raw_data(main_opt_params, main_proc_params):
    print("Convergence study raw data...")
    type = "raw"  #  "raw_reduced"
    train_dataset, test_dataset, input_dim, num_classes = load_dataset_raw(type)
    epochs_list = range(10)
    convergence_study(train_dataset, test_dataset, input_dim, num_classes, main_opt_params,
                      main_proc_params, epochs_list, type=type)

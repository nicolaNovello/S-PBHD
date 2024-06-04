from torch import nn
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
import torch
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import sklearn
from utils import *
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, random_split


class ToTensor(object):
    def __call__(self, sample):
        x, y = sample
        return (torch.tensor([x]).float(),
                torch.tensor([y]).float())

class MyDataset(Dataset):
    def __init__(self, x_dataframe, y_dataframe, transform=None):
        self.transform = transform
        self.x_dataframe = x_dataframe
        self.y_dataframe = y_dataframe
        self.data = []
        for i in range(len(x_dataframe)):
            self.data.append((self.x_dataframe.iloc[i], self.y_dataframe.iloc[i]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


class Discriminator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(100, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(100, output_dim),
        )

    def forward(self, input_tensor):
        output_tensor = self.main(input_tensor)
        return output_tensor

class CombinedArchitecture(nn.Module):
    def __init__(self, single_architecture, cost_function_v=1):
        super(CombinedArchitecture, self).__init__()
        self.div_to_act_func = {
            3: nn.Identity(),
            5: nn.Softmax()
        }
        self.cost_function_version = cost_function_v
        self.single_architecture = single_architecture
        self.final_activation = self.div_to_act_func[cost_function_v]

    def forward(self, input_tensor_1, input_tensor_2):
        intermediate_1 = self.single_architecture(input_tensor_1)
        output_tensor_1 = self.final_activation(intermediate_1)
        intermediate_2 = self.single_architecture(input_tensor_2)
        output_tensor_2 = self.final_activation(intermediate_2)

        return output_tensor_1, output_tensor_2


def get_random_batch(dataset, batch_size=32, random_seed=0):

    train_dataloader_random = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                         worker_init_fn=lambda id: np.random.seed(random_seed))
    my_testiter = iter(train_dataloader_random)
    random_batch, target = next(my_testiter)
    return random_batch


def compute_loss_divergence(cost_function_v, out_1, out_2, data_tx, num_classes, current_batch_size, alpha, device):
    loss_fn = nn.BCELoss()
    loss_fn_2 = nn.BCELoss(reduction='none')
    loss_fn_3 = nn.CrossEntropyLoss()

    data_tx_categorical = torch.Tensor(to_categorical(data_tx, t_tensor=True, num_classes=num_classes))

    if cost_function_v == 3:  # cross-entropy / KL
        loss = loss_fn_3(out_1.squeeze(), data_tx.squeeze().long())
    elif cost_function_v == 5:  # SL
        loss = sl_cost_fcn(out_1, out_2, data_tx_categorical, num_classes, alpha)

    return loss


def compute_loss_divergence_old(cost_function_v, out_1, out_2, data_tx, num_classes, current_batch_size, alpha, device):
    loss_fn = nn.BCELoss()
    loss_fn_2 = nn.BCELoss(reduction='none')
    loss_fn_3 = nn.CrossEntropyLoss()

    data_tx_categorical = torch.Tensor(to_categorical(data_tx, t_tensor=True, num_classes=num_classes))

    if cost_function_v == 3:  # cross-entropy
        loss = loss_fn_3(out_1.squeeze(), data_tx.squeeze().long())
    elif cost_function_v == 5:  # SL
        loss = sl_cost_fcn(out_1, out_2, data_tx_categorical, num_classes, alpha)
    return loss


class MyException(Exception):
    pass


def choose_nn_model(input_dim, num_classes, cost_func_v, device):

    model = load_simple_net(input_dim, num_classes, convolutional_discr=True,
                            cost_function_v=cost_func_v).to(device)
    return model

def load_simple_net(input_dim, num_classes, convolutional_discr=False, cost_function_v=4):

    partial_net = Discriminator(input_dim, num_classes)
    combined = CombinedArchitecture(partial_net, cost_function_v=cost_function_v)
    return combined


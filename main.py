from __future__ import print_function, division

import numpy as np
import scipy.io as sio
import argparse
import matplotlib.pyplot as plt

import pandas as pd
import random
import os
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn import metrics, manifold
import torch
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, random_split
from torch import nn
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from PIL import Image
import torch.nn.functional as F
import time
from torchvision.utils import make_grid, save_image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils import *
from classes import *
from main_functions import *
from datetime import datetime
import mat4py


if __name__ == '__main__':
    # Define input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='Pre-processing mode: Lambdas, Cauchy, No', default="Lambdas")
    parser.add_argument('--batch_size', help='Number of data samples to train on at once', default=128)
    parser.add_argument('--learning_rate', help='Learning rate of the optimizer for the neural network',
                        default=0.0001)
    parser.add_argument('--alpha', default=1)
    args = parser.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)


    main_opt_params = {
        'learning_rate': float(args.learning_rate)
    }

    main_proc_params = {
        'batch_size': int(args.batch_size),
        'alpha': float(args.alpha),
        'device': "cpu",
        'random_seed': [0]
    }

    if args.mode == "Lambdas":
        main_convergence_study_lambdas(main_opt_params, main_proc_params)
    elif args.mode == "Cauchy":
        main_convergence_study_cauchy(main_opt_params, main_proc_params)
    else:
        main_convergence_study_raw_data(main_opt_params, main_proc_params)


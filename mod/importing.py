#https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
#https://kaggle2.blob.core.windows.net/forum-message-attachments/250927/8325/nn.cfg.logx

import logging
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from torch import nn, optim
from torch.autograd import Variable
import pandas as pd
import os
import numpy as np
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import train_test_split
from scipy.special import erfinv
import matplotlib.pyplot as plt
import dill as pickle


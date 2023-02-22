import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_data():
    df = pd.read_csv('csv/BasicModelSmall_regpca0_dimn.csv')

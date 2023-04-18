from BasicModel import BasicModel
import torch
from torch import nn
import matplotlib.pyplot as plt
from config import Config
from train_uncertainty_loss import train
from utils import functions_to_names, run_models
import numpy as np
import pandas as pd

config = Config()
LIMIT = config.LIMIT

class UncertaintyLossModel(BasicModel):
    pass

def run_uncertaintylossmodel(functions):
    seeds = [1, 2, 3]
    test_sizes = np.linspace(0.05, 0.95, 19)
    modelname = 'UncertaintyLossModel'
    df = run_models(UncertaintyLossModel, modelname, train, test_sizes, seeds, functions)
    df.to_csv(f'full_results/{modelname}/{modelname}.csv')

if __name__ == '__main__':
    functions = ['a+b', 'a-b', 'a*b']
    run_uncertaintylossmodel(functions)
    
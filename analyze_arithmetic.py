import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_test_split():
    df = pd.read_csv('mod arithmetic data.csv')
    lw = 0.4
    plt.scatter(df['test_prop'], df['only'], label = 'a+b data only', color = 'b')
    plt.plot(df['test_prop'], df['only'], color = 'b', lw = lw)
    plt.scatter(df['test_prop'], df['both'], label = 'a+b and a-b data', color = 'r')
    plt.plot(df['test_prop'], df['both'], color = 'r', lw = lw)
    plt.title('Modular Arithmetic Model Performance')
    plt.xlabel('Test Set Proportion')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def plot_splits():
    df_only_reg = pd.read_csv('mod_arith_only.csv')
    df_only_fine = pd.read_csv('mod_arith_only_fine.csv')
    df_only = pd.concat((df_only_reg, df_only_fine))
    df_only = df_only.sort_values(by=['test_size'])
    
    df_both_reg = pd.read_csv('mod_arith_both.csv')
    df_both_fine = pd.read_csv('mod_arith_both_fine.csv')
    df_both = pd.concat((df_both_reg, df_both_fine))
    df_both= df_both.sort_values(by=['test_size'])

    only_mean, both_mean, only_std, both_std = [], [], [], []
    both_mean_minus, both_std_minus = [],[]
    test_sizes = df_only['test_size'].unique()
    for ts in test_sizes:
        df_ts_only = df_only[df_only['test_size'] ==  ts]
        df_ts_both = df_both[df_both['test_size'] == ts]

        only_mean.append(df_ts_only['test_acc'].mean())
        only_std.append(df_ts_only['test_acc'].std()/df_ts_only.shape[0])
        both_mean.append(df_ts_both['test_apb_acc'].mean())
        both_std.append(df_ts_both['test_apb_acc'].std()/df_ts_both.shape[0])

        both_mean_minus.append(df_ts_both['test_amb_acc'].mean())
        both_std_minus.append(df_ts_both['test_amb_acc'].std()/df_ts_both.shape[0])

    print(len(both_mean))
    plt.errorbar(test_sizes, both_mean, yerr = both_std, label = 'a + b acc (both model)', color = 'r', fmt = 'o')
    plt.plot(test_sizes, both_mean, lw = 0.2, color = 'r')

    plt.errorbar(test_sizes, only_mean, yerr = only_std, label = 'a + b acc (only model)', color = 'b', fmt = 'o')
    plt.plot(test_sizes, only_mean, lw = 0.2, color = 'b')

    #plt.errorbar(test_sizes, both_mean_minus, yerr = both_std_minus, label = 'a - b acc (both model)', color = 'orange')
    #plt.plot(test_sizes, both_mean_minus, lw = 0.2, color = 'orange')
    

    plt.title('Modular Arithmetic Performance')
    plt.xlabel('Test Set Proportion')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_splits()
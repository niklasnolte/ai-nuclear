import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from BasicModelSmallBoth import BasicModelSmallBoth
from train_arith_combined import get_data, onehot_stacky, get_y_pred, calc_acc

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
    dir = 'good_mod_arith_data'
    #df_only_reg = pd.read_csv(f'{dir}/mod_arith_only.csv')
    #df_only_fine = pd.read_csv(f'{dir}/mod_arith_only_fine.csv')
    #df_only = pd.concat((df_only_reg, df_only_fine))
    df_only = pd.read_csv(f'{dir}/mod_arith_only_all.csv')
    df_only['test_apb_acc'] = df_only['test_acc']
    
    #df_both_reg = pd.read_csv(f'{dir}/mod_arith_both.csv')
    #df_both_fine = pd.read_csv(f'{dir}/mod_arith_both_fine.csv')
    #df_both_before = pd.concat((df_both_reg, df_both_fine))
    
    #df_both_before['test_acc'] = (df_both_before['test_apb_acc']+df_both_before['test_amb_acc'])/2
    #df_both_before['train_acc'] = (df_both_before['train_apb_acc']+df_both_before['train_amb_acc'])/2

    #df_both_after = pd.read_csv(f'{dir}/mod_arith_both_fine_all.csv')
    df_comb = pd.read_csv(f'{dir}/mod_arith_combined_all.csv')

    #df_comb_reg = pd.read_csv(f'{dir}/mod_arith_combined.csv')
    #df_comb_fine = pd.read_csv(f'{dir}/mod_arith_combined_fine.csv')
    #df_comb = pd.concat((df_comb_reg, df_comb_fine))
    #df_comb = pd.read_csv(f'{dir}/mod_arith_both_all.csv')

    df_apbambatb = pd.read_csv('mod_arith_mult_apb_amb_atb.csv')
    
    df_apbatb_first = pd.read_csv('mod_arith_mult_apb_atb.csv')
    df_apbatb_second = pd.read_csv('mod_arith_mult_apb_atb_other.csv')
    df_apbatb = pd.concat((df_apbatb_first, df_apbatb_second))

    colors = ['r', 'g', 'b', 'purple']
    labels = ['a+b, a-b, a*b model', 'a+b and a*b model', 'a+b and a-b model', 'a+b only model']
    dfs = [df_apbambatb,df_apbatb, df_comb, df_only]

    for i,df in enumerate(dfs):
    
        df = df.sort_values(by = 'test_size')
        mean = []
        std = []
        for ts in df['test_size']:
            df_ts = df[df['test_size'] == ts]
            if len(df_ts)>0:
                mean.append(df_ts['test_apb_acc'].mean())
                std.append(df_ts['test_apb_acc'].std()/df_ts.shape[0])

        plt.errorbar(df['test_size'], mean, yerr = std, label = labels[i], color = colors[i], fmt = 'o', alpha  = 1)
        plt.plot(df['test_size'], mean, lw = 0.2, color = colors[i])
    plt.xlim((0,1))
    plt.title('Modular Arithmetic Performance')
    plt.xlabel('Test Set Proportion')
    plt.ylabel('A+B Test Accuracy')
    plt.legend()
    plt.show()

def get_model(model_title, dir):
    path = f"models/{dir}/{model_title}"
    sd = torch.load(f'{path}/best.pt')
    model = torch.load(f"{path}/model.pt")
    model.load_state_dict(sd)
    return model

def get_model_testapb_acc(model, test_size, seed, run = 'both'):
    X, y, _, test_mask, _ = get_data(test_size = test_size, seed = seed, run = run)
    y_pred = model(X)
    shape = y_pred.shape
    apb_mask = torch.hstack((torch.ones(shape[0], 1), torch.zeros(shape[0], 1))).bool()

    y_pred_apb = get_y_pred(y_pred, apb_mask)
    y_act_apb = get_y_pred(y, apb_mask)
    y_pred_apb_test = y_pred_apb[test_mask[:,0]] #gets only test items where we see a+b target
    y_act_apb_test = y_act_apb[test_mask[:,0]].flatten()

    print(y_pred_apb_test.argmax(dim=1))
    print(y_act_apb_test)
    print(X[apb_mask])

    apb_acc = (y_act_apb_test == y_pred_apb_test.argmax(dim=1)).float().mean()
    return apb_acc.item()


def get_df_apb_acc(df, dir, run = 'combined'):
    testapb_acc = []
    for index, row in df.iterrows():
        model_title, seed, test_size = row['title'], row['seed'], row['test_size']
        model = get_model(model_title, dir)
        testapb_acc.append(get_model_testapb_acc(model, test_size, seed, run = run))
        print(testapb_acc[-1])
        break
    #df['test_apb_acc'] = testapb_acc



if __name__ == '__main__':
    #dir = 'good_mod_arith_data'
    #csv_file = 'mod_arith_both_fine_all'
    #df_comb_fine = pd.read_csv(f'{dir}/{csv_file}.csv')
    #get_df_apb_acc(df_comb_fine,csv_file, run = 'both')
    #print(df_comb_fine)
    plot_splits()

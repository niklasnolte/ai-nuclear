import pandas as pd
import matplotlib.pyplot as plt


#plt.ylabel(metric)

plt.show()
if __name__ == '__main__':
    #df = df_reg[:list(df_reg['Iteration']).index(100000)]
    val = 'reg1'
    df_reg1 = pd.read_csv(f'csv/BasicLinear_{val}_heavy15.csv')
    val = 'reg1e_1'
    df_reg10 = pd.read_csv(f'csv/BasicLinear_{val}_heavy15.csv')
    val = 'reg2e_2'
    df_reg50 = pd.read_csv(f'csv/BasicLinear_{val}_heavy15.csv')
    val = 'reg2e_3'
    df_reg500 = pd.read_csv(f'csv/BasicLinear_{val}_heavy15.csv')

    metrics = ['Loss', 'Proton_Entropy', 'Neutron_Entropy']
    n=20
    n2 = 200
    figsize = (3,1)
    fig, axs = plt.subplots(nrows=3, figsize = tuple(5*i for i in figsize), sharex=False)
    for i in range(len(metrics)):
        ax, metric = axs[i], metrics[i]
        if i==0:
            ax.set_yscale('log')
        ax.title.set_text(metric)
        #ax.plot(df['Iteration'][n:], df[metric][n:], label = 'Basic Model no regularization')
        ax.plot(df_reg1['Iteration'][n:n2], df_reg1[metric][n:n2], label = 'reg = proton_ent + neutron_ent')
        ax.plot(df_reg10['Iteration'][n:n2], df_reg10[metric][n:n2], label = 'reg = 1/10 * (proton_ent + neutron_ent)')
        ax.plot(df_reg50['Iteration'][n:n2], df_reg50[metric][n:n2], label = 'reg = 1/50 * (proton_ent + neutron_ent)')
        ax.plot(df_reg500['Iteration'][n:n2], df_reg500[metric][n:n2], label = 'reg = 1/500 * (proton_ent + neutron_ent)')
        ax.legend()
        ax.set_xlabel('Epoch')

    plt.show()

    '''
    df_trans = pd.read_csv('TransformerModel.csv')
    plt.plot(df_trans['Iteration'], df_trans['Loss'], label = 'Transformer')
    df_basic = pd.read_csv('BasicModel.csv')
    plt.plot(df_basic['Iteration'], df_basic['Loss'], label = 'Basic')
    plt.legend()
    plt.show()
    '''
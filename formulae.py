import os
import numpy as np
import pandas as pd
import warnings

def radius_formula(data):
    N = data[["n"]].values
    Z = data[["z"]].values
    A = Z+N
    r0 = 1.2
    fake_rad = r0*A**(1/3)  # fm
    return fake_rad*(Z < 110)


def delta(Z, N):
    A = Z+N
    aP = 9.87
    delta0 = aP/A**(1/2)
    for i in range(len(A)):
        if ((N % 2 == 0) & (Z % 2 == 0))[i]:
            pass
        elif ((N % 2 == 1) & (Z % 2 == 1))[i]:
            delta0[i] = -delta0[i]
        else:
            delta0[i] = 0
    return delta0

def shell(Z, N):
    #calculates the shell effects according to "Mutual influence of terms in a semi-empirical" Kirson
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    alpham = -1.9
    betam = 0.14
    magic = [2, 8, 20, 28, 50, 82, 126, 184]

    def find_nearest(lst, target):
        return min(lst, key=lambda x: abs(x - target))
    nup = np.array([abs(x - find_nearest(magic, x)) for x in Z])
    nun = np.array([abs(x - find_nearest(magic, x)) for x in N])
    P = nup*nun/(nup+nun)
    P[np.isnan(P)] = 0
    return alpham*P + betam*P**2

def E_BW(df):
    N = df["n"].values
    Z = df["z"].values
    A = N+Z    
    
    aV = 15.36
    aS = -16.42
    aC = -0.691
    aA = -22.53
    
    Eb = aV*A + aS*A**(2/3) + aC*Z**2/(A**(1/3)) + \
        aA*(N-Z)**2/A + delta(Z, N)
    
    return Eb
        
def E_BW2(df):
    N = df["n"].values
    Z = df["z"].values
    A = N+Z    
    
    aV = 16.58
    aS = -26.95
    aC = -0.774
    aA = -31.51
    axC = 2.22
    aW = -43.4
    ast = 55.62
    aR = 14.77
    
    Eb = aV*A + aS*A**(2/3) + aC*Z**2/(A**(1/3)) + \
        aA*(N-Z)**2/A + delta(Z, N) + shell(Z, N) + aR*A**(1/3) + axC*Z**(4/3)/A**(1/3) + \
        aW*abs(N-Z)/A + ast*(N-Z)**2/A**(4/3)
    return Eb


    
def E_WS4(df):  

    N = df["n"].values
    Z = df["z"].values
    A = N+Z 
    Da = 931.494102    
    mp = 938.78307
    mn = 939.56542


    file_path = os.path.join(os.path.dirname(__file__), 'tables', 'WS4.txt')

    df_WS4 = pd.read_fwf(file_path, widths=[9, 9, 15, 15])
    
    df_WS4['Z'] = df_WS4['Z'].astype(float)
    df_WS4['N'] = df_WS4['A'].astype(float) - df_WS4['Z']
    
    # Merge the two dataframes based on 'Z' and 'N'
    merged_df = pd.merge(df, df_WS4, how='left', left_on=['z', 'n'], right_on=['Z', 'N'])
    
    merged_df['WS4'] = Z*mp + N*mn - merged_df['WS4'].astype(float) - A*Da
    
    # Create a new column 'WS4' in the merged dataframe and fill it with values from 'WS4' column in df_WS4
    merged_df['WS4'] = merged_df['WS4'].fillna(0)
    
    # Drop unnecessary columns from the merged dataframe
    merged_df = merged_df.drop(['A', 'Z', 'N', 'WS4+RBF'], axis=1)
    
    Eb = merged_df['WS4'].values.astype(float)
        
    return Eb   
    

def binding_formula(df, model, TMS):
    
    binding_formulae = {'BW' : E_BW, 'BW2' : E_BW2, 'WS4' : E_WS4}

    def E_binding(x, model):
        return binding_formulae[model](x)

    N = df["n"].values
    Z = df["z"].values
    A = N+Z 
    
    #calculates E_b/nucleon for the Bethe–von Weizsäcker semi-empirical mass formula (formula = 'BW'), 
    #its extension (formula = 'BW2') and the micro-macro Weizsäcker-Skyrme (WS) model  (formula = 'WS4')
    Eb = E_binding(df, model)
        
    Eb[Eb < 0] = 0
    
    if TMS != 'TMS':
        for i in range(len(A)):
            if (df['binding_sys'].values[i] == 'Y') or (df['binding_unc'].values[i]*A[i] > 100):
                Eb[i] = 0
    return Eb/A #MeV

def PySR_formula(df, TMS):
    N = df["n"].values
    Z = df["z"].values
    A = N+Z
    x0 = Z
    x1 = N
    Eb = (((-10.931704 / ((0.742612 / x0) + x1)) +
          7.764321) + np.sin(x0 * 0.03747635))
    if TMS != 'TMS':
        for i in range(len(A)):
            if (df['binding_sys'].values[i] == 'Y') or (df['binding_unc'].values[i]*A[i] > 100):
                Eb[i] = 0
    return Eb
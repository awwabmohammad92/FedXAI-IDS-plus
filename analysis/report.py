import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_rows',500); pd.set_option('display.max_columns',30)
s=pd.read_csv('tables/summary.csv'); t=pd.read_csv('tables/tests.csv')
def fam(df, **kw):
    base=dict(SPLIT='default',PARTITION='natural',ATTACK='none',EPS_S=8.0,AGG='cclip',N_TREES=50,DEPTH=5)
    base.update(kw); m=np.ones(len(df),bool)
    for k,v in base.items(): m&=(df[k]==v).values
    return df[m]
def piv(df, val='f1_mean'):
    return df.pivot_table(index='method', columns='dataset', values=val).round(3)

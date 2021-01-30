import os

import pandas as pd
import numpy as np

all_data = ['banknote','breastcancer','cifar10_resnet','cifar10_vgg','htru2','mnist']

for data in all_data:
    path_csv = os.path.join('csv', data+'.csv')
    df = pd.read_csv(path_csv, sep=',')

    df = df.drop(columns=['Unnamed: 0'])

    # table 1 (same as example)
    df1 = df.pivot_table(index=['Attack', 'Epsilon'],columns=['Defence'],values=['Score','False Positive Rate'])\
        .rename(columns={"baard_2stage":"BAARD2","baard_3stage":"BAARD3","fs":"FS","lid":"LID","rc":"RC","Block Rate":"Blocking Rate","False Positive Rate":"FPR"})
    df1.columns = df1.columns.swaplevel(0,1).rename(["",""])
    df1.sort_index(axis=1,level=0,inplace=True)
    df1.drop('FPR',inplace=True,axis=1)
    save_path = os.path.join('tables',data+'_1.xlsx')
    print(df1)
    df1.to_excel(save_path)

    # table 2 (without FPR)
    df1 = df.pivot_table(index=['Attack', 'Epsilon'],columns=['Defence'],values=['Block Rate','False Positive Rate'])\
        .rename(columns={"baard_2stage":"BAARD2","baard_3stage":"BAARD3","fs":"FS","lid":"LID","rc":"RC","Block Rate":"Blocking Rate","False Positive Rate":"FPR"})
    df1.columns = df1.columns.swaplevel(0,1).rename(["",""])
    df1.sort_index(axis=1,level=0,inplace=True)

    save_path = os.path.join('tables',data+'_1.xlsx')
    print(df1.columns)
    df1.to_excel(save_path)

    # table 3 (swap with Blocking Rate)
    df1 = df.pivot_table(index=['Attack', 'Epsilon'],columns=['Defence'],values=['Block Rate','False Positive Rate'])\
        .rename(columns={"baard_2stage":"BAARD2","baard_3stage":"BAARD3","fs":"FS","lid":"LID","rc":"RC","Block Rate":"Blocking Rate","False Positive Rate":"FPR"})
    df1.columns = df1.columns.swaplevel(0,1).rename(["",""])
    df1.sort_index(axis=1,level=0,inplace=True)

    save_path = os.path.join('tables',data+'_1.xlsx')
    print(df1.columns)
    df1.to_excel(save_path)
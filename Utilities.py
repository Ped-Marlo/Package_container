#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 19:37:06 2020

@author: Pedro_Martínez
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 19:46:52 2020

@author: Pedro_Martínez

Instance methods need a class instance and can access the instance through self.
Class methods don’t need a class instance. They can’t access the instance (self) but they have access to the class itself via cls.
Static methods don’t have access to cls or self. They work like regular functions but belong to the class’s namespace.


"""
import pandas as pd
import numpy as np
import os 
import statistics

class Data:
    
    def __init__(self, filename, sep=None):
        self.filename = filename
        self.filepath = os.getcwd()
        if filename.endswith('.csv'):
            self.df_raw_data = pd.read_csv(os.path.join(self.filepath, filename),sep)
        self.nan_cols = self.find_nan(self.df_raw_data)
#        self.df_filled_na = self.replace_nan(self.df_raw_data, self.nan_cols)
        
    
    def find_nan(self,df):
        nan_values = df.isna()
        nan_columns = nan_values.any()
        cols = df.columns[nan_columns].tolist()
        return cols
    

    def replace_nan(self, df, cols):
        df_na=df.copy()
        for col in cols:
            try:
                mean_col = df[col].mean()
                df_na[col]=df[col].fillna(mean_col)
                
            except:#mean impossible to perform--> categorical data
                mode_col = statistics.mode(df[col])
                       
                if pd.isnull(mode_col):
                    df_na[col]= pd.get_dummies(df[col],dummy_na=True)[np.nan]
                else:
                    df_na[col]=df[col].fillna(mode_col)   
        return df_na


class CSV(Data):

    def _read_csv(self, index='PassengerId'):
        df_raw = pd.read_csv(os.path.join(self.filepath, self.filename))
#        df_raw = df_raw.set_index(index, drop=True)
        return df_raw


class Txt(Data):

    def __init__(self, fname):
        f = open(fname, 'r')
        Lines = f.readlines()
        f.close()
        Lines.append('\n')
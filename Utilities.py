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
    __slots__ = ['filename', 'data']

    def __init__(self, filename='DataFrame', data=pd.DataFrame()):
        self.df_raw_data = data
        self.filename = filename
        self.filepath = os.getcwd()

    @classmethod
    def find_nan(cls, df):
        nan_values = df.isna()
        nan_columns = nan_values.any()
        cols = df.columns[nan_columns].tolist()
        return cols

    @classmethod
    def replace_nan(cls, df, cols):
        df_na = df.copy()
        for col in cols:
            try:
                mean_col = df[col].mean()
                df_na[col] = df[col].fillna(mean_col)

            except TypeError:  # mean impossible to perferm--> categorical data
                mode_col = statistics.mode(df[col])

                if pd.isnull(mode_col):
                    df_na[col] = pd.get_dummies(df[col], dummy_na=True)[np.nan]
                    df_na[col] = ~df_na[col].astype(bool)
                else:
                    df_na[col] = df[col].fillna(mode_col)
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

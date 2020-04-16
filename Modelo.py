#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:30:52 2020

@author: Pedro_Martínez
"""
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error, roc_auc_score as ROC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


class Model:

    def train_test(self, Data, target):

        X_train, X_test, y_train, y_test = train_test_split(
                                                    Data.drop(target, axis=1),
                                                    Data[target],
                                                    train_size=0.8,
                                                    test_size=0.2,
                                                    random_state=55
                                                    )
        return X_train, X_test, y_train, y_test

    def standarize(self, X):
        scaler = StandardScaler()

        for col in X.select_dtypes(include=[float]).columns:
            X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1))
#        self.Mean = scaler.mean_
#        self.Std_dev = scaler.scale_
        return X

    def select_inputs(self, X_train, X_test, y_train, k):

        fs = SelectKBest(f_classif, k=k)
        fs.fit(X_train, y_train)
        cols = fs.get_support(indices=True)
        X_train_fs = X_train.iloc[:, cols]
        X_test_fs = X_test.iloc[:, cols]

        for pos, val in enumerate(fs.scores_):
            print(X_train.columns[pos], ': %f' % (fs.scores_[pos]))

        return X_train_fs, X_test_fs,

    def categorize(self, X):
        X1 = X.copy()
        for col in X.select_dtypes(include=[object]):
            vals = X[col].drop_duplicates()
            X1[col] = X[col].map(dict(zip(vals, range(len(vals)))))
        return X1

    def intervals(self, X, n):
        X1 = X.copy()
        for col in X.select_dtypes(include=[float]):
            bins = np.linspace(min(X1[col]), max(X1[col]), n)
            X1[col] = pd.cut(x=X[col], bins=bins, labels=list(range(1,n)))
        return X1

    def fit(self, X_train, y_train, metric, test_set):
        self.my_model.fit(
                          X_train,
                          y_train,
                          eval_metric=metric,
                          eval_set=test_set,
                          verbose=False
                          )

    def predict(self, X_test):
        return self.my_model.predict(X_test)

    def fit_grid(self, X_train, y_train, metric, test_set):
        self.grid_mse = GridSearchCV(
                                    estimator=self.my_model,
                                    param_grid=self.gbm_param_grid,
                                    scoring='neg_mean_squared_error',
                                    cv=5,
                                    verbose=False,
                                    n_jobs=-1
                                    )
        self.grid_mse.fit(
                            X_train,
                            y_train,
                            eval_metric=metric,
                            verbose=False,
                            eval_set=test_set,
                            early_stopping_rounds=5
                            )

    def predict_grid(self, X_test):
        return self.grid_mse.predict(X_test)


class Classifier(Model):

    def __init__(self):

        self.my_model = XGBClassifier(objective='reg:logistic')
        self.gbm_param_grid = {
                                'learning_rate': [0.1, 0.3, 0.5],
                                'colsample_bytree': np.linspace(0.1, 0.9, 8),
                                'n_estimators': [50, 100, 200, 300],
                                'max_depth': [10, 15, 20, 25]
                                }

    def _error(self, text, y_test, predictions):

        print(text+' roc_auc_score: ', ROC(y_test, predictions))
        return ROC(y_test, predictions)


class Regressor(Model):

    def __init__(self):

        self.my_model = XGBRegressor(objective='reg:squarederror')
        self.gbm_param_grid = {
                    'colsample_bytree': np.linspace(0.1, 0.9, 8),
                    'n_estimators': [100, 200],
                    'max_depth': [10, 15, 20, 25]
                    }


#        if  tipo == 'Classifier':
#            print('roac_auc_score: ',roc_auc_score(self.y_test,self.predictions))
##            print('roac_auc_score_cv: ',roc_auc_score(self.y_test,self.predictions_cv))
#
#
#            
#        elif tipo == 'Regressor':
#            r2 = self.my_model.score(self.y_test,self.predictions)
#            n = self.X.shape[0]
#            p = self.y.shape[1]
#            adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
#            
#            print("Mean Absolute Error: " + str(mean_absolute_error(self.predictions, self.y_test)))
#            print("Mean Absolute Error CV: " + str(mean_absolute_error(self.predictions_cv, self.y_test)))

    
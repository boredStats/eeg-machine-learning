# -*- coding: UTF-8 -*-

"""Sandbox for new functions."""

import os
import math
import utils
import numpy as np
import pandas as pd
import pickle as pkl

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import VotingRegressor


def try_svr():
    x_train, x_holdout, y_train, y_holdout = create_holdout_data(
        ratio=.10,
        seed=13,
        targets='distress_TQ',
        )

    scaler = RobustScaler()
    est = SVR(kernel='rbf')

    param_grid = {
        'svr__C': (1, 10, 100, 1000),
        'svr__gamma': (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
        }

    pipe = Pipeline([('scaler', scaler), ('svr', est)])

    grid = GridSearchCV(
        pipe,
        cv=3,
        scoring='neg_mean_absolute_error',
        param_grid=param_grid,
        refit=True)
    grid.fit(x_train, y_train)
    score = grid.score(x_holdout, y_holdout)
    print(score)
    with open('./data/SVR_distress_TQ.pkl', 'wb') as file:
        pkl.dump(grid, file)


def check_svr():
    x_train, x_holdout, y_train, y_holdout = create_holdout_data(
        ratio=.10,
        seed=13,
        targets='distress_TQ',
        )

    with open('./data/SVR_distress_TQ.pkl', 'rb') as file:
        grid = pkl.load(file)

    print(grid.cv_results_)
    # grid.set_params({'scoring': 'neg_mean_absolute_error'})
    # mae = grid.score(x_holdout, y_holdout)
    # print(mae)


def try_gbr():
    x_train, x_holdout, y_train, y_holdout = create_holdout_data(
        ratio=.10,
        seed=13,
        targets='distress_TQ',
        )

    scaler = RobustScaler()
    est = GradientBoostingRegressor(n_estimators=1000, random_state=13)

    param_grid = {
        'gbr__loss': ('ls', 'lad', 'huber'),
        'gbr__max_features': ('auto', 'sqrt', 'log2')
        }

    pipe = Pipeline([('scaler', scaler), ('gbr', est)])

    grid = GridSearchCV(
        pipe,
        cv=3,
        scoring='neg_mean_absolute_error',
        param_grid=param_grid,
        refit=True)
    grid.fit(x_train, y_train)
    score = grid.score(x_holdout, y_holdout)
    print(score)
    with open('./data/GBR_distress_TQ.pkl', 'wb') as file:
        pkl.dump(grid, file)


def create_holdout_data(targets='both', ratio=.20, seed=None, outfile=None):
    # Recommend setting seed for consistency
    if seed is None:
        seed = 13  # hard-coding a seed
    all_data = utils.load_data()
    if targets == 'both':
        targets = ['distress_TQ', 'loudness_VAS10']
    features = [f for f in list(all_data) if f not in targets]
    connectivity_data, target_data = all_data[features], all_data[targets]

    x_train, x_holdout, y_train, y_holdout = train_test_split(
        connectivity_data, target_data,
        test_size=ratio,
        random_state=seed,
        shuffle=False,
    )
    if outfile is not None:
        holdout_split = {
            'connectivity training set': x_train,
            'connectivity holdout set': x_holdout,
            'target training set': y_train,
            'target holdout set': y_holdout
        }
        if '.xls' in outfile:
            utils.save_xls(holdout_split, outfile)
        elif '.pkl' in outfile:
            with open(outfile, 'wb') as f:
                pkl.dump(holdout_split, f)
    return x_train, x_holdout, y_train, y_holdout


def voting_test():
    # create_holdout_data(outfile='./data/holdout_split.pkl')

    x_train, x_holdout, y_train, y_holdout = create_holdout_data(
        ratio=.10,
        seed=13,
        targets='distress_TQ',
        )

    estimators = [
        ('svm', SVR(kernel='rbf')),
        ('etree', ExtraTreesRegressor(1000, 'mae', random_state=13)),
        # ('gb', GradientBoostingRegressor())
        ]
    params = {
        # 'svm__kernel': ('linear', 'rbf'),
        'svm__C': (1, 10, 100, 100),
        'svm__gamma': (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
        # 'svm__degree': (2, 3, 4, 5),
        # 'etree__n_estimators': (100, 500, 1000),
        # 'etree__criterion': ('mse', 'mae'),
    }

    reg = VotingRegressor(estimators=estimators)
    grid = GridSearchCV(estimator=reg, param_grid=params, cv=3, verbose=2)
    grid.fit(x_train, y_train)
    print(grid.best_params_)
    gridfile = './data/distress_TQ_VotingRegressor_GridSearchCV.pkl'
    with open(gridfile, 'wb') as file:
        pkl.dump(grid, file)


def main():
    # try_svr()
    # check_svr()
    try_gbr()


main()

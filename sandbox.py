# -*- coding: UTF-8 -*-

"""Sandbox for new functions."""

import os
import math
import utils
import numpy as np
import pandas as pd
import pickle as pkl

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.SVM import SVR
from sklearn.ensemble import VotingRegressor


def create_holdout_data(holdout_ratio=.20, seed=None, outfile=None):
    # Recommend setting seed for consistency
    if seed is None:
        seed = 13  # hard-coding a seed
    all_data = utils.load_data()
    targets = ['distress_TQ', 'loudness_VAS10']
    features = [f for f in list(all_data) if f not in targets]
    connectivity_data, target_data = all_data[features], all_data[targets]

    x_train, x_holdout, y_train, y_holdout = train_test_split(
        connectivity_data, target_data,
        test_size=holdout_ratio,
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


def main():
    pass


if __name__ == "__main__":
    create_holdout_data(outfile='./data/holdout_split.pkl')

# Common functions for this project

import os, time, datetime
import numpy as np

def ctime():
    t = time.time()
    f = '%Y-%m-%d %H:%M:%S '
    return datetime.datetime.fromtimestamp(t).strftime(f)

def load_connectivity_data(currrent_data_path=None, drop_behavior=True):
    currrent_data_path = './../../data_raw_labeled.pkl'
    data_path = os.path.abspath(currrent_data_path)
    raw_data = np.load(data_path)

    if drop_behavior:
        behavior_variables = ['distress_TQ', 'loudness_VAS10']
        raw_data.drop(columns=behavior_variables, inplace=True)

    return raw_data


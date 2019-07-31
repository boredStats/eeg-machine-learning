# Common functions for this project

import os, time, datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from scipy.stats import zscore
from copy import deepcopy


def ctime():
    t = time.time()
    f = '%Y-%m-%d %H:%M:%S '
    return datetime.datetime.fromtimestamp(t).strftime(f)


def load_connectivity_data(currrent_data_path=None, drop_behavior=True):
    if currrent_data_path is None:
        currrent_data_path = './../data/data_raw_labeled.pkl'
    # data_path = os.path.abspath(currrent_data_path)
    raw_data = np.load(currrent_data_path, allow_pickle=True)

    if drop_behavior:
        behavior_variables = ['distress_TQ', 'loudness_VAS10']
        raw_data.drop(columns=behavior_variables, inplace=True)

    return raw_data


def load_behavior_data(current_behavior_path=None):
    if current_behavior_path is None:
        current_behavior_path = './../data/features_nonEEG.xlsx'

    sheets = ['vars_continuous', 'vars_categorical']
    dfs = []
    for sheet in sheets:
        if 'categorical' in sheet:
            dtype = 'category'
        else:
            dtype = 'float'
        behavior_df = pd.read_excel(current_behavior_path, sheet_name=sheet, dtype=dtype)
        dfs.append(behavior_df)

    final = pd.concat(dfs, sort=False, axis=1)
    final.dropna(inplace=True)
    return final


def get_group_indices(full_sides=True):

    def _preprocess_side_data(side_series):
        # Convert asymmetrical side category to LR category
        cleaned_side_data = deepcopy(side_series)
        for s, subj_data in enumerate(side_data):
            if subj_data < 0:
                cleaned_side_data.iloc[s] = -1
            elif subj_data == 0:
                cleaned_side_data.iloc[s] = 0
            else:
                cleaned_side_data.iloc[s] = 1

        return cleaned_side_data

    behavior_df = load_behavior_data()

    type_data = behavior_df['tinnitus_type']
    tin_types = pd.unique(type_data)

    side_data = behavior_df['tinnitus_side']
    if full_sides:
        tin_sides = pd.unique(side_data)
    else:
        new_side_data = _preprocess_side_data(side_data)
        tin_sides = pd.unique(new_side_data)
        side_data = new_side_data

    type_1, type_2, type_3 = [], [], []
    side_1, side_2, side_3, side_4, side_5 = [], [], [], [], []
    for subj in range(len(behavior_df.index)):
        if type_data.iloc[subj] == tin_types[0]:
            type_1.append(subj)
        elif type_data.iloc[subj] == tin_types[1]:
            type_2.append(subj)
        elif type_data.iloc[subj] == tin_types[2]:
            type_3.append(subj)
        else:
            print('Subject %d did not have type data' % subj)

        if side_data.iloc[subj] == tin_sides[0]:
            side_1.append(subj)
        elif side_data.iloc[subj] == tin_sides[1]:
            side_2.append(subj)
        elif side_data.iloc[subj] == tin_sides[2]:
            side_3.append(subj)
        else:
            print('Subject %d did not have side data' % subj)
        if full_sides:
            if side_data.iloc[subj] == tin_sides[3]:
                side_4.append(subj)
            elif side_data.iloc[subj] == tin_sides[4]:
                side_5.append(subj)
            else:
                print('Subject %d did not have side data' % subj)

    res = {'type_%d_subj_indices' % tin_types[0]: type_1,
           'type_%d_subj_indices' % tin_types[1]: type_2,
           'type_%d_subj_indices' % tin_types[2]: type_3,
           'side_%d_subj_indices' % tin_sides[0]: side_1,
           'side_%d_subj_indices' % tin_sides[1]: side_2,
           'side_%d_subj_indices' % tin_sides[2]: side_3}
    if full_sides:
        res['side_%d_subj_indices' % tin_sides[3]] = side_4
        res['side_%d_subj_indices' % tin_sides[4]] = side_5

    return res


def generate_test_df(n=100, c=10, normalize=True):
    test_data = np.random.rand(n, c)
    if normalize:
        test_data = zscore(test_data, ddof=1)
    column_names = ['Column_%d' % x for x in range(c)]
    test_df = pd.DataFrame(test_data, columns=column_names)

    return test_df


def clean_df_to_numpy(df):
    # Dumb function to give networkx a numpy array
    n_rows = len(df.index)
    n_cols = len(list(df))
    new_array = np.ndarray(shape=(n_rows, n_cols))

    for x in range(n_rows):
        for y in range(n_cols):
            new_array[x, y] = df.iloc[x, y]

    return new_array


def load_data_full_subjects():
    # Remove EEG subjects that don't have behavior data
    behavior_df = load_behavior_data()
    conn_df = load_connectivity_data()
    filt_df = conn_df.filter(items=behavior_df.index, axis=0)  # Remove EEG subjects with missing rowvals in behavior_df
    return behavior_df, filt_df


def dummy_code_binary(categorical_series):
    # Sex: 1M, -1F
    string_categorical_series = pd.DataFrame(index=categorical_series.index, columns=list(categorical_series))

    for colname in list(categorical_series):
        string_series = []
        for value in categorical_series[colname].values:
            if value == 1:
                if 'sex' in colname:
                    string_series.append('male')
                else:
                    string_series.append('yes')
            elif value == -1:
                if 'sex' in colname:
                    string_series.append('female')
                else:
                    string_series.append('no')
        string_categorical_series[colname] = string_series

    dummy_series = pd.get_dummies(string_categorical_series)
    old_names = list(dummy_series)
    return dummy_series.rename(columns=dict(zip(old_names, ['categorical_%s' % d for d in old_names])))


def main():
    # Sandbox stuff
    print(ctime())

    conn_df = load_connectivity_data()
    print(conn_df.head())

    behavior = load_behavior_data()
    print(behavior)
    print(behavior.select_dtypes(include='category'))
    get_group_indices(full_sides=False)


if __name__ == "__main__":
    main()

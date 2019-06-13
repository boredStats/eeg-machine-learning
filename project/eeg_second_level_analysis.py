import os
import logging
import numpy as np
import pandas as pd
import pickle as pkl
import proj_utils
from scipy.stats import ttest_1samp as ttest


logging.basicConfig(level=logging.INFO, filename='./logs/eeg_second_level.log', filemode='w')


def create_new_df_from_indices(index_list, data_df):
    new_df = pd.DataFrame(index=range(len(index_list)), columns=list(data_df))
    for i, index in enumerate(index_list):
        new_df.iloc[i] = data_df.iloc[index]
    return new_df


def run_second_level(group_df):
    mus = np.zeros(len(list(group_df)))
    tvals, pvals = ttest(proj_utils.clean_df_to_numpy(group_df), popmean=mus, axis=0)

    res_df = pd.DataFrame(index=['t_values', 'p_values'], columns=list(group_df))
    res_df.iloc[0] = tvals
    del tvals
    res_df.iloc[1] = pvals
    del pvals

    return res_df


def fdr_thresh(output_path, alpha=.00001):
    from statsmodels.stats.multitest import fdrcorrection

    keywords = ['side', 'type']
    for keyword in keywords:
        res_list = []
        for file in os.listdir(output_path):
            if 'thresh' in file:
                continue
            if keyword in file:
                with open(os.path.join(output_path, file), 'rb') as f:
                    res_df = pkl.load(f)
                p_values = np.ndarray.flatten(proj_utils.clean_df_to_numpy(res_df.loc[['p_values']]))
                res_list.append(p_values)
        res_stack = np.hstack(res_list)
        truth, fdr_p = fdrcorrection(res_stack, alpha=alpha, is_sorted=False)
        falses = truth[truth==False]
        print(keyword, len(falses))

    for file in sorted(os.listdir(output_path)):
        if 'thresh' in file:
            continue
        key = file.replace('.pkl', '')
        with open (os.path.join(output_path, file), 'rb') as f:
            res_df = pkl.load(f)
        p_values = np.ndarray.flatten(proj_utils.clean_df_to_numpy(res_df.loc[['p_values']]))
        truth, fdr_p = fdrcorrection(p_values, alpha=alpha, is_sorted=False)

        thresh_df = pd.DataFrame(index=['t_values', 'p_thresh'], columns=list(res_df))
        thresh_df.iloc[0] = res_df.iloc[0]
        thresh_df.iloc[1] = fdr_p

        # print(thresh_df.head())
        falses = truth[truth==False]
        print(key, len(falses))

        with open(os.path.join(output_path, '%s_thresh.pkl' % key), 'wb') as f:
            pkl.dump(thresh_df, f)


def main():
    output_dir = './../data/eeg_second_level/'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    data_df = proj_utils.load_connectivity_data()
    print(data_df.head())
    index_dict = proj_utils.get_group_indices(full_sides=False)
    for key in index_dict:
        indices = index_dict[key]
        print(key, len(indices))
        logging.info('%s: Running second level for %s' % (proj_utils.ctime(), key))
        index_list = index_dict[key]
        res_df = run_second_level(create_new_df_from_indices(index_list, data_df))

        with open(os.path.join(output_dir, '%s.pkl' % key), 'wb') as file:
            pkl.dump(res_df, file)
    fdr_thresh(output_dir)


main()

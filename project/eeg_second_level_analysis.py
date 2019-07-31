import os
import logging
import statsmodels.api as sm
import numpy as np
import pandas as pd
import pickle as pkl
import proj_utils
from copy import deepcopy
from scipy.stats import ttest_1samp as ttest
from statsmodels.stats import multitest


logging.basicConfig(level=logging.INFO, filename='./logs/eeg_second_level.log', filemode='w')


def dummy_code_categorical(data):
    def _create_dummy_code(series):
        levels = pd.unique(series)
        variable_name = series.name
        column_names = ['%s_%d' % (variable_name, i) for i in range(len(levels))]
        new_dummy_df = pd.DataFrame(columns=column_names, index=series.index, dtype='float')
        for c, col in enumerate(column_names):
            dummy = np.zeros(len(series.index))
            level = levels[c]

            for subj in range(len(series.index)):
                value = series.iloc[subj]
                if value == level:
                    dummy[subj] = 1

            new_dummy_df[col] = dummy

        return new_dummy_df

    continuous_data = data.select_dtypes(include='float')
    categorical_data = data.select_dtypes(include='category')
    cat_list = [continuous_data]

    for cat_var in list(categorical_data):
        cat_data = categorical_data[cat_var]
        dummy_df = _create_dummy_code(cat_data)
        cat_list.append(dummy_df)

    return pd.concat(cat_list, axis=1)


def run_ancova(connectivity_data, covariates, where_zero='ind', output_path=None):
    logging.info('%s: Running second level analysis' % proj_utils.ctime())
    intercept = np.zeros(len(covariates.index))

    # if where_zero is 'ind':
    #     covariates['intercept'] = intercept
    # elif where_zero is 'dep':
    #     pass

    res_df = pd.DataFrame(index=['F', 'P'], columns=list(connectivity_data))
    for c, conn_var in enumerate(list(connectivity_data)):
        if where_zero is 'ind':
            dep_ = connectivity_data[conn_var].values
            covariates['intercept'] = intercept
            ind_ = covariates.values
        elif where_zero is 'dep':
            dep_ = intercept
            covariates['predictor'] = connectivity_data[conn_var].values
            ind_ = covariates.values

        model = sm.OLS(dep_, ind_, hasconst=False)
        results = model.fit()

        res_df.loc['F'].iloc[c] = results.fvalue
        res_df.loc['P'].iloc[c] = results.f_pvalue

    if output_path is not None:
        with open(output_path, 'wb') as file:
            pkl.dump(res_df, file)

    logging.info('%s: Finished second level analysis' % proj_utils.ctime())

    return res_df


def multiple_correction_ancova_res(pickle_file, method='fdr', alpha=.05):
    # method params {'fdr_bh', 'bonferroni', 'sidak'}
    with open(pickle_file, 'rb') as f:
        ancova_df = pkl.load(f)

    p_vals = ancova_df.loc['P'].values
    truth, corrected_p, _, _ = multitest.multipletests(pvals=p_vals, alpha=alpha, method=method)
    # truth, fdr_p = fdrcorrection(p_vals, alpha=alpha, method='indep', is_sorted=False)

    corrected_df = pd.DataFrame(index=['F', 'corrected_p'], columns=list(ancova_df))
    corrected_df.loc['F'] = ancova_df.loc['F'].values
    corrected_df.loc['corrected_p'] = corrected_p

    return truth, corrected_df


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


def full_matrix_second_level(data_df, output_path=None):
    logging.info('%s: Running second level for all subjects' % proj_utils.ctime())
    full_matrix_res = run_second_level(data_df)
    if output_path is not None:
        with open(output_path, 'wb') as f:
            pkl.dump(full_matrix_res, f)


def group_matrices_second_level(data_df, index_dict, output_dir=None):
    for key in index_dict:
        logging.info('%s: Running second level for %s' % (proj_utils.ctime(), key))
        index_list = index_dict[key]
        res_df = run_second_level(create_new_df_from_indices(index_list, data_df))

        with open(os.path.join(output_dir, '%s.pkl' % key), 'wb') as file:
            pkl.dump(res_df, file)


def fdr_thresh(output_path, alpha=.00001):
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
        truth, fdr_p = multitest.fdrcorrection(p_values, alpha=alpha, is_sorted=False)

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

    behavior = proj_utils.load_behavior_data()
    ml_targets = ['loudness_VAS', 'distress_TQ', 'distress_VAS']

    covariates_df = dummy_code_categorical(behavior)
    covariates_df.drop(labels=ml_targets, axis=1, inplace=True)

    data_df = proj_utils.load_connectivity_data()
    data_df_filt = data_df.filter(items=covariates_df.index, axis=0)

    wz = 'dep'
    if wz is 'ind':
        output_path = os.path.join(output_dir, 'second_level_f_tests_zero_as_independent_var.pkl')
    else:
        output_path = os.path.join(output_dir, 'second_level_f_tests_zero_as_dependent_var.pkl')
    res = run_ancova(connectivity_data=data_df_filt, covariates=covariates_df, where_zero=wz, output_path=output_path)
    print(res)

    truth, corrected_p_df = multiple_correction_ancova_res(output_path, method='bonferroni', alpha=1e-20)
    print(corrected_p_df)
    print(np.count_nonzero(truth))

    # full_matrix_second_level(data_df, output_path=os.path.join(output_dir, 'all_subjects_second_level.pkl'))

    # index_dict = proj_utils.get_group_indices(full_sides=False)
    # group_matrices_second_level(data_df, index_dict, output_dir)

    # for key in index_dict:
    #     indices = index_dict[key]
    #     print(key, len(indices))
    #     logging.info('%s: Running second level for %s' % (proj_utils.ctime(), key))
    #     index_list = index_dict[key]
    #     res_df = run_second_level(create_new_df_from_indices(index_list, data_df))
    #
    #     with open(os.path.join(output_dir, '%s.pkl' % key), 'wb') as file:
    #         pkl.dump(res_df, file)
    # fdr_thresh(output_dir)


main()

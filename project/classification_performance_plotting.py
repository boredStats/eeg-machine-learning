import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

sns.set(style='darkgrid')


def load_performance(tin_variable, acc_measure='Balanced accuracy', filterby=None):
    """
    Folder naming convention: tin_variable-classifier-covariate_check-resampling_method

    tin_variable options: tin_side, tin_type, TQ_grade, TQ_high_low
    acc_measure options: Balanced accuracy, Chance accuracy, f1 scores
    filterby options:
        - If extra_trees, svm, or knn is used, performance will be returned for that specific classifier
        - If with_covariates or without_covariates is used, performance will be returned for that covariate option
        - If no_resample, ROS, RUS, or SMOTE is used, performance will be returned for that resampler
        - If None is used, all performance for the tinnitus variable will be returned
        Note: a tuple of options can be used
    """
    data_dir = './../data/eeg_classification'
    tin_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    var_dirs = [os.path.join(data_dir, d) for d in tin_dirs if tin_variable in d]
    if filterby is not None:
        if isinstance(filterby, tuple):
            perf_dirs = var_dirs
            for f in filterby:
                str_filt = [d for d in var_dirs if f in d]
                perf_dirs = list(set(perf_dirs).intersection(set(str_filt)))
        else:
            perf_dirs = [d for d in var_dirs if filterby in d]
    else:
        perf_dirs = var_dirs

    output = {}
    for d in perf_dirs:
        dname = d.split(" ")
        clf = dname[1]
        covariate_check = dname[2]
        resampler = dname[3]

        if acc_measure is 'Balanced accuracy' or 'Chance accuracy':
            sheet_name = 'accuracy scores'
        elif acc_measure is 'f1 scores':
            sheet_name = 'f1 scores'
        else:
            raise ValueError('Invalid accuracy measure')

        acc_df = pd.read_excel(os.path.join(d, 'performance.xlsx'), sheet_name=sheet_name, index_col=0)
        key = '%s %s %s %s' % (tin_variable, clf, covariate_check, resampler)
        output[key] = acc_df

    return output


def test_load_performance():
    tin_variable = 'tin_side'
    filterby = ('svm', 'with_covariates')
    res = load_performance(tin_variable, filterby=filterby, acc_measure='f1 scores')
    print(list(res))


def acc_line_plots(plot_data, x_group, y, hue_group=None, order=None, ax=None, linestyles='-', color='b', markers='.'):
    sns.set(style='darkgrid')
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    g = sns.pointplot(x=x_group, y=y, hue=hue_group, ax=ax, data=plot_data, order=order,
                      linestyles=linestyles, color=color, markers=markers, dodge=False)
    return g


def pretty_keys(keysplit):
    tvar, clf, cv, rs = keysplit[0], keysplit[1], keysplit[2], keysplit[3]
    if tvar == 'tin_side':
        pretty_tvar = 'Percept Side'
    elif tvar == 'tin_type':
        pretty_tvar = 'Percept Type'
    elif tvar == 'TQ_grade':
        pretty_tvar = 'TQ (Grade)'
    elif tvar == 'TQ_high_low':
        pretty_tvar = 'TQ (High/Low)'
    else:
        raise ValueError('tvar not read in correctly')

    if clf == 'extra_trees':
        pretty_clf = 'ExtraTrees'
    elif clf == 'svm':
        pretty_clf = 'SVM'
    elif clf == 'knn':
        pretty_clf = 'k-Nearest Neighbors'
    else:
        raise ValueError('clf not read in correctly')

    if cv == 'with_covariates':
        pretty_cv = 'With covariates'
    elif cv == 'without_covariates':
        pretty_cv = 'Without covariates'
    else:
        raise ValueError('cv not read in correctly')

    if rs == 'no_resample':
        pretty_rs = 'None'
    elif rs == 'RUS':
        pretty_rs = 'Undersampling'
    elif rs == 'ROS':
        pretty_rs = 'Oversampling'
    elif rs == 'SMOTE':
        pretty_rs = 'SMOTE'
    else:
        raise ValueError('rs not read in correctly')

    return pretty_tvar, pretty_clf, pretty_cv, pretty_rs


def accuracy_lineplot_resampling_comparison():
    mpl.rcParams['axes.labelsize'] = 18
    perf = load_performance(tin_variable='tin_side')

    def _create_plot_df(accuracy_data, acc='Balanced accuracy'):
        plot_df = pd.DataFrame(
            columns=['Accuracy', 'Tinnitus variable', 'Classifier', 'Covariates', 'Resampling method'])
        for key in list(accuracy_data):
            keysplit = key.split(" ")
            tvar, clf, cv, rs = pretty_keys(keysplit)

            scores = accuracy_data[key][acc]
            val = scores.loc['Average']
            new_row = {'Accuracy': val,
                       'Tinnitus variable': tvar,
                       'Classifier': clf,
                       'Covariates': cv,
                       'Resampling method': rs}
            plot_df = plot_df.append(new_row, ignore_index=True)

        return plot_df
    df = _create_plot_df(perf)

    resampling_order = ['None', 'Undersampling', 'SMOTE', 'Oversampling']

    classifiers = ['ExtraTrees', 'SVM', 'k-Nearest Neighbors']
    color_options = ['tab:green', 'tab:blue', 'tab:red']

    covariate_check = ['With covariates', 'Without covariates']
    line_options = ['-', '--']

    fig, ax = plt.subplots(figsize=(12, 9))
    plot_data = {}
    x = np.arange(1, 5)
    for clf_index, clf in enumerate(classifiers):
        clf_slice = df[df['Classifier'] == clf]

        for cov_index, cov in enumerate(covariate_check):
            cov_slice = clf_slice[clf_slice['Covariates'] == cov]
            cov_slice.set_index('Resampling method', inplace=True)

            temp = cov_slice.loc[resampling_order]
            y = temp['Accuracy'].values
            line_data = (x, y, color_options[clf_index], line_options[cov_index])
            plot_data['%s %s' % (clf, cov)] = line_data
            color = color_options[clf_index]
            line = line_options[cov_index]
            plt.plot(x, y, c=color, linestyle=line)

    ax.set_ylim([0, 1])
    ax.set_xlim([0.8, 4.2])
    ax.set_xticks(x)
    ax.set_xticklabels(resampling_order)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Resampling method')
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=13)

    fig.savefig('test.png', bbox_inches='tight')
    # plt.show()


accuracy_lineplot_resampling_comparison()

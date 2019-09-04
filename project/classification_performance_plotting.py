import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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


def acc_line_plots(plot_data, x_group, y, hue_group=None, order=None, outdir=None):
    sns.set(style='darkgrid')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.pointplot(x=x_group, y=y, hue=hue_group, data=plot_data, ax=ax, order=order)
    ax.set_ylim([0, 1])
    if outdir is not None:
        fig.savefig(outdir)
    else:
        return ax


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


def plot_avg_accuracy(tin_variable, acc='Balanced accuracy', filters=None, outdir=None):
    # acc options: 'Balanced accuracy', 'Chance accuracy'
    accuracy_data = load_performance(tin_variable=tin_variable, filterby=filters)
    plot_data = pd.DataFrame(columns=['Accuracy', 'Tinnitus variable', 'Classifier', 'Covariates', 'Resampling method'])
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
        plot_data = plot_data.append(new_row, ignore_index=True)
    print(plot_data)

    order = ['None', 'Undersampling', 'SMOTE', 'Oversampling']
    cov_ver = acc_line_plots(plot_data, x_group='Resampling method', y='Accuracy', hue_group='Classifier', order=order)



plot_avg_accuracy(tin_variable='tin_side', filters=('with_covariates'))

from os.path import join, isdir
from os import mkdir, listdir
from itertools import repeat
from copy import deepcopy
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import proj_utils as pu


def plot_confusion_matrix(confusion_matrix, ordered_strings=None, new_strings=None, norm=True, fname=None):
    if ordered_strings is not None:
        temp = confusion_matrix[ordered_strings]   # Create temp df with reordered columns
        plot_matrix = pd.DataFrame(columns=ordered_strings)  # Final df with reordered rows
        for ord_str in ordered_strings:
            plot_matrix.loc[ord_str] = temp.loc[ord_str].values
    else:
        plot_matrix = confusion_matrix

    if new_strings is not None:
        old_strings = list(plot_matrix)
        map = dict(zip(old_strings, new_strings))
        plot_matrix.rename(map, axis=0, inplace=True)
        plot_matrix.rename(map, axis=1, inplace=True)
    plot_matrix.fillna(value=np.nan, inplace=True)

    if norm:
        vmin, vmax = 0, 1
        fmt = ".2g"
    else:
        vmin, vmax = None, None
        fmt = "g"
    cmap = plt.cm.Blues
    sns.set(context='notebook', font_scale=1.3)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data=plot_matrix, vmin=vmin, vmax=vmax, fmt=fmt, annot=True, ax=ax, cmap=cmap)
    ax.set_xlabel('Predicted label', fontsize='large')
    ax.set_ylabel('True label', fontsize='large')
    # ax.set_xticklabels(plt.xticks()[1], rotation=45)  # apply xtick rotation
    ax.set_yticklabels(plt.yticks()[1], va='center')  # manual vertical alignment of yticks

    if fname is not None:
        fig.savefig(fname, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


def format_accuracy_df_for_box_plot(accuracy_df):
    columns = list(accuracy_df)
    n_rows = len(accuracy_df.index)

    rows_stacked, columns_stacked, data_stacked = [], [], []
    for col in columns:
        data_stacked.extend(accuracy_df[col].values)
        rows_stacked.extend(repeat('Fold', n_rows))
        columns_stacked.extend(repeat(col, n_rows))

    formatted_df = pd.DataFrame(data=data_stacked, columns=['Accuracy'])
    # formatted_df['Fold_info'] = rows_stacked
    formatted_df['Accuracy Type'] = columns_stacked
    return formatted_df


def plot_single_accuracy(accuracy_series, fname=None):
    plt.clf()

    sns.set(style='darkgrid')
    ax = sns.boxplot(y=accuracy_series)

    if fname is not None:
        plt.savefig(fname)
    else:
        plt.show()


def plot_both_accuracy(accuracy_data, fname=None):
    plt.clf()

    sns.set(style='darkgrid')
    ax = sns.boxplot(x='Accuracy Type', y='Accuracy', data=accuracy_data)

    if fname is not None:
        plt.savefig(fname)
    else:
        plt.show()


def boxplot_testing():
    model = 'extra_trees'
    tin_variable = 'tinnitus_side'
    output_dir = './../data/%s/' % model
    tin_dir = join(output_dir, tin_variable)
    acc_sheet = pd.read_excel(join(tin_dir, 'performance.xlsx'), sheet_name='accuracy scores', index_col=0)
    fold_rows = [r for r in acc_sheet.index if 'Fold' in r]
    acc_sheet = acc_sheet.filter(fold_rows, axis='index')

    plot_single_accuracy(acc_sheet['Balanced accuracy'])
    f = format_accuracy_df_for_box_plot(acc_sheet)
    plot_both_accuracy(f)


def conf_mat_testing(fname=None):
    model = 'extra_trees'
    tin_variable = 'tinnitus_side'
    output_dir = './../data/%s/' % model
    tin_dir = join(output_dir, tin_variable)

    or_str = ['Left', 'Left>Right', 'Bilateral', 'Right>Left', 'Right']
    new_str = ['L', 'L>R', 'L=R', 'R>L', 'R']

    xls = pd.ExcelFile(join(tin_dir, 'confusion_matrices.xlsx'))
    cm_arrays = []
    for sheet in xls.sheet_names:
        cm_sheet = pd.read_excel(xls, sheet_name=sheet, index_col=0)
        cm_arrays.append(cm_sheet.values)
    avg_cm_array = np.sum(np.asarray(cm_arrays), axis=0)
    cm_df = pd.DataFrame(avg_cm_array, index=list(cm_sheet), columns=list(cm_sheet))
    plot_confusion_matrix(cm_df, ordered_strings=or_str, new_strings=new_str, norm=False, fname=fname)

    cm_sheet = pd.read_excel(join(tin_dir, 'confusion_matrices.xlsx'), sheet_name='Fold 01', index_col=0)
    plot_confusion_matrix(cm_sheet, ordered_strings=or_str, new_strings=new_str, norm=True, fname=fname)


def replot_confusion_matrices():
    models = ['svm', 'extra_trees', 'sgd', 'knn']
    variables = ['tinnitus_side', 'tinnitus_type', 'TQ_grade', 'TQ_high_low']

    for tin_variable in variables:
        if 'side' in tin_variable:
            or_str = ['Left', 'Left>Right', 'Bilateral', 'Right>Left', 'Right']
            new_str = ['L', 'L>R', 'L=R', 'R>L', 'R']
        elif 'type' in tin_variable:
            or_str = ['NBN', 'PT_and_NBN', 'PT']
            new_str = ['NBN', 'PT+NBN', 'PT']
        else:
            or_str, new_str = None, None

        output_dir = './../data/%s/' % tin_variable
        for model in listdir(output_dir):
            tin_dir = join(output_dir, model)
            xls = pd.ExcelFile(join(tin_dir, 'confusion_matrices_normalized.xlsx'))
            cm_arrays = []
            for sheet in xls.sheet_names:
                cm_sheet = pd.read_excel(xls, sheet_name=sheet, index_col=0)
                cm_arrays.append(cm_sheet.values)
            avg_cm_array = np.mean(np.asarray(cm_arrays), axis=0)
            avg_cm = pd.DataFrame(avg_cm_array, index=list(cm_sheet), columns=list(cm_sheet))
            fname = join(tin_dir, 'average confusion matrix normalized.png')
            plot_confusion_matrix(avg_cm, ordered_strings=or_str, new_strings=new_str, norm=True, fname=fname)


def plot_extra_trees_features():
    variables = ['tinnitus_side', 'tinnitus_type', 'TQ_grade', 'TQ_high_low']

    conn_data = pu.load_connectivity_data()
    conn_variables = list(conn_data)
    band_list, roi_list = [], []
    for c in conn_variables:
        band = c.split('_')[0]
        roi_1 = c.split('_')[1]
        roi_2 = c.split('_')[2]
        band_list.append(band)
        roi_list.append(roi_1)
        roi_list.append(roi_2)

    bands = pd.unique(band_list)
    rois = pd.unique(roi_list)

    conn_matrix = pd.DataFrame(index=rois, columns=rois)
    band_matrices_master = {}
    for band in bands:
        band_matrices_master[band] = conn_matrix

    for tin_variable in variables:
        output_dir = './../data/%s/' % tin_variable
        for model in listdir(output_dir):
            if 'extra_trees' in model:
                tin_dir = join(output_dir, model)
                xls = pd.ExcelFile(join(tin_dir, 'coefficients.xlsx'))
                fold_data = {}
                for sheet in xls.sheet_names:
                    band_matrices = deepcopy(band_matrices_master)
                    feature_df = pd.read_excel(xls, sheet_name=sheet, index_col=0)
                    for feat in list(feature_df):
                        feat_str = feat.split('_')
                        if any([True for b in bands if b in feat_str]):  # check if feature is connectivity data
                            feat_band, r1, r2 = feat_str[0], feat_str[1], feat_str[2]
                            input_matrix = band_matrices[feat_band]
                            input_matrix.loc[r1][r2] = feature_df[feat].values[0]
                    fold_data[sheet] = band_matrices

                with open(join(tin_dir, 'connectivity_matrices.pkl'), 'wb') as file:
                    pkl.dump(obj=fold_data, file=file)


def get_variable_data():
    def _count_data(data_to_count, vartype):
        data_df = pd.DataFrame(data_to_count, columns=[vartype])
        count_df = data_df[vartype].value_counts()
        return count_df

    output_dir = './../data/eeg_classification'
    if not isdir(output_dir):
        mkdir(output_dir)
    behavior_data, conn_data = pu.load_data_full_subjects()

    side_data = pu.convert_tin_to_str(behavior_data['tinnitus_side'].values.astype(float), 'tinnitus_side')
    side_count = _count_data(side_data, 'Side')

    type_data = pu.convert_tin_to_str(behavior_data['tinnitus_type'].values.astype(float), 'tinnitus_type')
    type_count = _count_data(type_data, 'Type')

    tq_data = behavior_data['distress_TQ'].values
    high_low_thresholds = [0, 46, 84]
    binned_high_low = np.digitize(tq_data, bins=high_low_thresholds, right=True)
    tq_high_low = ['Low' if t < 2 else 'High' for t in binned_high_low]
    hl_count = _count_data(tq_high_low, 'TQ (High/Low)')

    grade_thresholds = [0, 30, 46, 59, 84]
    binned_grade = np.digitize(tq_data, bins=grade_thresholds, right=True)
    tq_grade = ['Grade_%d' % t for t in binned_grade]
    grade_count = _count_data(tq_grade, 'TQ (Grade)')

    gender = behavior_data['sex']
    gender_str = ['Male' if g > 0 else 'Female' for g in gender.values]
    gender_count = _count_data(gender_str, 'Gender')

    # categorical_variables = ['smoking', 'deanxit_antidepressants', 'rivotril_antianxiety', 'sex']
    # categorical_data = behavior_data[categorical_variables]

    output = {'side': side_count,
              'type': type_count,
              'tq_high_low': hl_count,
              'tq_grade': grade_count,
              'gender': gender_count}
    pu.save_xls(output, join(output_dir, 'tin_variables_classcount.xlsx'))


def plot_age_historgram(output_dir=None):
    behavior_data, conn_data = pu.load_data_full_subjects()
    age = behavior_data['age']
    sns.set_style('darkgrid')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.distplot(age.values, kde=False, ax=ax, hist_kws={"alpha": .75, "color": 'b'})
    ax.set_xlabel('Age')
    ax.set_ylabel('Frequency')
    if output_dir is None:
        plt.show()
    else:
        fig.savefig(join(output_dir, 'age_hist.png'))


if __name__ == "__main__":
    models = ['extra_trees']  # ['svm', 'extra_trees', 'sgd', 'knn']
    variables = ['tinnitus_side', 'tinnitus_type', 'TQ_grade', 'TQ_high_low']
    # boxplot_testing()
    # conf_mat_testing(fname='test_confusion_matrix.png')
    # replot_confusion_matrices()
    # plot_extra_trees_features()

    # get_variable_data()

    plot_age_historgram(output_dir='./../data/eeg_classification')

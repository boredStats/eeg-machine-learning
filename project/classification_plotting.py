from os.path import join, isdir
from os import mkdir
from itertools import repeat
import numpy as np
import pandas as pd
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

    for model in models:
        output_dir = './../data/%s/' % model
        for tin_variable in variables:
            if 'side' in tin_variable:
                or_str = ['Left', 'Left>Right', 'Bilateral', 'Right>Left', 'Right']
                new_str = ['L', 'L>R', 'L=R', 'R>L', 'R']
            elif 'type' in tin_variable:
                or_str = ['NBN', 'PT_and_NBN', 'PT']
                new_str = ['NBN', 'PT+NBN', 'PT']
            else:
                or_str, new_str = None, None

            tin_dir = join(output_dir, tin_variable)

            xls = pd.ExcelFile(join(tin_dir, 'confusion_matrices.xlsx'))
            cm_arrays = []
            for sheet in xls.sheet_names:
                cm_sheet = pd.read_excel(xls, sheet_name=sheet, index_col=0)
                cm_arrays.append(cm_sheet.values)

            avg_cm_array = np.sum(np.asarray(cm_arrays), axis=0)
            avg_cm = pd.DataFrame(avg_cm_array, index=list(cm_sheet), columns=list(cm_sheet))
            fname = join(tin_dir, 'average confusion matrix.png')
            plot_confusion_matrix(avg_cm, ordered_strings=or_str, new_strings=new_str, norm=False, fname=fname)

            xls = pd.ExcelFile(join(tin_dir, 'confusion_matrices_normalized.xlsx'))
            cm_arrays = []
            for sheet in xls.sheet_names:
                cm_sheet = pd.read_excel(xls, sheet_name=sheet, index_col=0)
                cm_arrays.append(cm_sheet.values)

            avg_cm_array = np.mean(np.asarray(cm_arrays), axis=0)
            avg_cm = pd.DataFrame(avg_cm_array, index=list(cm_sheet), columns=list(cm_sheet))
            fname = join(tin_dir, 'average confusion matrix normalized.png')
            plot_confusion_matrix(avg_cm, ordered_strings=or_str, new_strings=new_str, norm=True, fname=fname)


if __name__ == "__main__":
    models = ['svm', 'extra_trees', 'sgd', 'knn']
    variables = ['tinnitus_side', 'tinnitus_type', 'TQ_grade', 'TQ_high_low']
    # boxplot_testing()
    # conf_mat_testing(fname='test_confusion_matrix.png')
    replot_confusion_matrices()

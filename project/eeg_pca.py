# -*- coding: utf-8 -*-
"""Running PCAs on connectivity data."""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import proj_utils as pu

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, RobustScaler

pd.options.display.float_format = '{:.3f}'.format


def pretty_pca_res(p):
    # Return pca results in a nicer way
    cols = [
        'Singular values',
        'Explained variance',
        'Explained variance ratio',
        ]
    df = pd.DataFrame(columns=cols)
    df['Singular values'] = p.singular_values_
    df['Explained variance'] = p.explained_variance_
    df['Explained variance ratio'] = p.explained_variance_ratio_ * 100

    cumulative_variance_ratio = []
    i = 0
    for comp in df['Explained variance ratio']:
        i += comp
        cumulative_variance_ratio.append(i)
    df['Cumulative variance ratio'] = cumulative_variance_ratio
    return df


def find_n_components(data, max_n_components=None, step=1):
    # Use cross-validation to find best number of components to use

    if max_n_components is None:
        if pu.df_or_np(data):
            max_n_components = data.values.shape[1]
        else:
            max_n_components = data.shape[1]

    scores = []
    pca = PCA()
    for n in np.arange(1, max_n_components, step):
        pca.n_components = n
        scores.append(np.mean(cross_val_score(pca, data, cv=3)))

    df = pd.DataFrame(columns=['Cross validation scores'])
    df['Cross validation scores'] = scores
    print(df.head())
    return df


def reconstruct_pca(data):
    # Reconstruct data after removing the first component
    # https://bit.ly/2rGNlXn
    X = data.values
    mu = np.mean(X, axis=0)

    pca = PCA()
    pca.fit(X)

    raw_num_comp = pca.n_components_
    Xhat = np.dot(
        pca.transform(X)[:, 1:raw_num_comp],
        pca.components_[1:raw_num_comp, :]
        )
    Xhat += mu

    return Xhat


def plot_scree(pretty_res, percent=True, pvals=None, kaiser=False, fname=None):
    # Create a scree plot using pretty_pca_res output
    mpl.rcParams.update(mpl.rcParamsDefault)

    eigs = pretty_res['Singular values'].values
    percent_var = pretty_res['Explained variance ratio'].values
    if len(eigs) > 30:
        n_comp_to_plot = 30
        eigs = eigs[:n_comp_to_plot]
        percent_var = percent_var[:n_comp_to_plot]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Scree plot", fontsize='xx-large')
    ax.plot(np.arange(1, len(eigs) + 1), eigs, 'ok')
    ax.set_ylim([0, (max(eigs) * 1.2)])
    ax.set_ylabel('Eigenvalues', fontsize='xx-large')
    ax.set_xlabel('Principal Components', fontsize='xx-large')

    if percent:
        ax2 = ax.twinx()
        ax2.plot(np.arange(1, len(percent_var) + 1), percent_var, '-k')
        ax2.set_ylim(0, max(percent_var) * 1.2)
        ax2.set_ylabel('Percentage of variance explained', fontsize='xx-large')

    if pvals is not None and len(pvals) == len(eigs):
        # TO-DO: add p<.05 legend?
        p_check = [i for i, t in enumerate(pvals) if t < .05]
        eigen_check = [e for i, e in enumerate(eigs) for j in p_check if i == j]
        ax.plot(np.add(p_check, 1), eigen_check, 'ob', markersize=12)

    if kaiser:
        ax.axhline(1, color='k', linestyle=':', linewidth=2)

    if fname:
        fig.savefig(fname, bbox_inches='tight')
    else:
        plt.show()
    # return fig, ax, ax2


def perm_pca(data, n_iters=1000, n_components=None):
    if n_iters == 0:
        pass

    if n_components is None:
        n_components = np.min(data.shape)
    n = 0
    permutation_dataframes = {}
    while n != n_iters:
        n += 1
        print('Running permutation PCA - Iteration %04d' % n)
        permuted_dataset = resample(data, replace=False)
        pca = IncrementalPCA(n_components=n_components, whiten=True)
        pca.fit(permuted_dataset)
        perm_df = pretty_pca_res(pca)
        permutation_dataframes['Permutation %04d' % n] = perm_df
        del pca

    return permutation_dataframes


def p_from_perm_data(observed_df, perm_data, variable='Singular values'):
    perm_results = {}
    for perm in perm_data:
        p_df = perm_data[perm]

        perm_results[perm] = p_df[variable].values

    p_values = []
    for i, perm in enumerate(perm_results):
        perm_array = perm_results[perm]
        observed = observed_df.iloc[i][variable]
        p = permutation_p(observed, perm_array)
        p_values.append(p)

    return p_values


def permutation_p(observed, perm_array):
    """Non-parametric null hypothesis testing
    see Phipson & Smyth 2010 for more information
    """
    n_iters = len(perm_array)
    n_hits = np.where(np.abs(perm_array) >= np.abs(observed))
    return (len(n_hits[0]) + 1) / (n_iters + 1)


def split_connectivity_by_band(connectivity_df):
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    colnames = list(connectivity_df)
    connectivity_by_band = {}
    for band in bands:
        band_columns = [c for c in colnames if band in c]
        band_df = connectivity_df[band_columns]
        connectivity_by_band[band] = band_df

    return connectivity_by_band


def pca_by_band(data, n_iters=1000, res_dir=None):
    if res_dir is None:
        res_dir = os.path.dirname(__file__)
    conn_by_band = split_connectivity_by_band(data)
    band_results = {}

    for b in conn_by_band:
        band_df = conn_by_band[b]
        print(pu.ctime() + 'Running PCA on %s' % b)

        # scaled_data = norm_to_ss1(band_df.values)
        # scaled_data = RobustScaler().fit_transform(band_df.values)
        scaled_data = StandardScaler().fit_transform(band_df)
        pca = PCA(.97)
        pca.fit(scaled_data)

        band_res = pretty_pca_res(pca)
        band_results[b] = band_res
        print(pca.n_components_)
        del pca

        perm_res = perm_pca(data=band_df, n_iters=n_iters)
        p_values = p_from_perm_data(observed_df=band_res, perm_data=perm_res)

        plot_scree(
            band_res,
            pvals=p_values,
            percent=False,
            fname=os.path.join(res_dir, '%s_pca_scree.png' % b)
        )
        band_res.to_excel(os.path.join(res_dir, '%s_pca_res.xlsx' % b))


def norm_to_ss1(matrix):
    # Alternate method for scaling, see Abdi & Williams, 2010 (PLS methods)
    centered = matrix - np.mean(matrix, axis=0)
    sum_of_squares = np.sum(centered ** 2, axis=0)

    rescaled_matrix = np.ndarray(shape=matrix.shape)
    for i, ss in enumerate(sum_of_squares):
        rescaled_matrix[:, i] = centered[:, i] / np.sqrt(ss)

    return rescaled_matrix


def _test_cross_val_score_method(data):
    pca = PCA(n_components=5, whiten=True)
    pca.fit(data)
    scores = pca.score_samples(data.values)
    print(scores.shape)
    score_df = find_n_components(data, step=5)
    score_df.to_excel('./pca_cross_val_scores_test.xlsx')


def grand_pca(data, res_dir=None):
    if res_dir is None:
        res_dir = os.path.dirname(__file__)
    print(pu.ctime() + 'Running grand PCA')
    pca = PCA(n_components=.99, whiten=True)
    zdata = StandardScaler().fit_transform(data)
    pca.fit(zdata)
    print(pca.n_components_)
    true_df = pretty_pca_res(pca)

    plot_scree(
        true_df,
        percent=False,
        fname=os.path.join(res_dir, 'grand_pca_scree.png'))
    true_df.to_excel(os.path.join(res_dir, 'grand_pca_res.xlsx'))


if __name__ == "__main__":
    print(pu.ctime() + 'Loading data')
    data = pu.load_connectivity_data()
    res_dir = os.path.abspath('./../results/pca')
    if not os.path.isdir(res_dir):
        os.mkdir(res_dir)

    grand_pca(data)

    pca_by_band(data, n_iters=0, res_dir=res_dir)

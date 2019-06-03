import utils
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

pd.options.display.float_format = '{:.3f}'.format


def pretty_pca_res(p):
    # Return pca results in a nicer way
    cols = ['Singular values', 'Explained variance', 'Explained variance ratio']
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


def find_n_components(data, max_n_components, step=1):
    # Use cross-validation to find best number of components to use
    scores = []
    pca = PCA()
    for n in np.arange(1, max_n_components, step):
        pca.n_components = n
        scores.append(np.mean(cross_val_score(p, data, cv=3, verbose=1)))

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
    Xhat = np.dot(pca.transform(X)[:, 1:raw_num_comp], pca.components_[1:raw_num_comp, :])
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
    ax.plot(np.arange(1, len(eigs) + 1), eigs, 'or')
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
        ax.plot(np.add(p_check, 1), eigen_check, 'ob', markersize=10)

    if kaiser:
        ax.axhline(1, color='k', linestyle=':', linewidth=2)

    if fname:
        fig.savefig(fname, bbox_inches='tight')
    else:
        plt.show()
    return fig, ax, ax2


if __name__ == "__main__":
    print(utils.ctime() + 'Loading data')
    data = utils.load_connectivity_data()
    n_components = np.min(data.shape)

    print(utils.ctime() + 'Running PCA with all components')
    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(data)
    df_raw = pretty_pca_res(pca)
    print(df_raw.head())
    plot_scree(df_raw, percent=True, fname='./scree_raw.png')
    df_raw.to_excel('./pca_raw.xlsx')

    print(utils.ctime() + 'Removing first component')
    data_cleaned = reconstruct_pca(data)

    print(utils.ctime() + 'Re-running pca on cleaned data')
    pca.fit(data_cleaned)
    df_cleaned = pretty_pca_res(pca)
    print(df_cleaned.head())
    sum_variances = np.ceil(np.sum(df_cleaned['Explained variance ratio'].values))
    print('Checking work - sum of variance ratios is: ' + str(sum_variances))
    plot_scree(df_cleaned, percent=True, fname='./scree_cleaned.png')
    df_cleaned.to_excel('./pca_first_component_removed.xlsx')

    print(utils.ctime() + 'Finding best n_components using k-fold cross-validation')
    crossval_df = find_n_components(data, n_components=18)
    crossval_df.to_excel('./crossval_res.xlsx')
    print(utils.ctime() + 'Finished')
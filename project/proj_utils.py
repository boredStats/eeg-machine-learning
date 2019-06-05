# Common functions for this project

import os, time, datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from scipy.stats import zscore


def ctime():
    t = time.time()
    f = '%Y-%m-%d %H:%M:%S '
    return datetime.datetime.fromtimestamp(t).strftime(f)


def load_connectivity_data(currrent_data_path=None, drop_behavior=True):
    if currrent_data_path is None:
        currrent_data_path = './../../data_raw_labeled.pkl'
    # data_path = os.path.abspath(currrent_data_path)
    raw_data = np.load(currrent_data_path)

    if drop_behavior:
        behavior_variables = ['distress_TQ', 'loudness_VAS10']
        raw_data.drop(columns=behavior_variables, inplace=True)

    return raw_data


class VisualizeData:
    # Visualize data before running analyses
    def __init__(self, seaborn_format=None, colors=None):
        if seaborn_format:
            sns.set(seaborn_format)  # seaborn_format must be a valid dict
        else:
            mpl.rcParams.update(mpl.rcParamsDefault)

        if not colors:
            colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:yellow']
        self.colors = colors

    @staticmethod
    def _check_data(data):
        print(data)

    def plot_single_scatter(self, vector, **kwargs):
        """Plot a single variable as a scatter

        Parameters:
        -----------
        vector : 1-D numpy array or pandas Series
            if numpy array, will be converted to a pd.Series to generate axes labels

        Returns:
        --------
        fig : matplotlib figure handle
        """


def generate_test_df(n=100, c=10, normalize=True):
    test_data = np.random.rand(n, c)
    if normalize:
        test_data = zscore(test_data, ddof=1)
    column_names = ['Column_%d' % x for x in range(c)]
    test_df = pd.DataFrame(test_data, columns=column_names)

    return test_df


def main():
    # Sandbox stuff
    print(ctime())
    test_df = generate_test_df()


if __name__ == "__main__":
    main()

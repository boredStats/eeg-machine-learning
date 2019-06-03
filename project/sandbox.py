
import numpy as np
import pandas as pd
import networkx

#
# rand_matrix = np.random.randint(low=0, high=2, size=(84, 84))
# G = networkx.from_numpy_matrix(rand_matrix)
# graph_degree = G.degree
# graph_global_eff = networkx.algorithms.efficiency.global_efficiency(G)
#
# print(graph_degree)
# print(graph_global_eff)
#

import os
import logging
import networkx
import numpy as np
import pandas as pd
import pickle as pkl
import proj_utils


logging.basicConfig(level=logging.INFO)


def create_adjacency_dict(data_df, bands):
    colnames = list(data_df)

    def _extract_band_colnames(band, colnames):
        names_extracted = []
        for name in colnames:
            if band in name:
                names_extracted.append(name)

        return names_extracted

    mat_dict = {}
    for band in bands:
        band_variables = _extract_band_colnames(band, colnames)
        adjacency_df = pd.DataFrame(columns=band_variables)
        for bv in band_variables:
            adjacency_df[bv] = data_df[bv]
        mat_dict[band] = adjacency_df
    return mat_dict


def parse_roi_names(colnames):
    roi_list = []
    for name in colnames:
        if "delta" in name:
            parsed = name.split("_")
            roi = parsed[2]
            if roi not in roi_list:
                roi_list.append(roi)
    return roi_list


def create_subj_adjacency_mats(adj_dict, bands, rois, dpath=None):
    if not dpath:
        subj_adj_dict = {}

    for band in bands:
        band_df = adj_dict[band]
        subjects = band_df.index
        for s, subj in enumerate(subjects):
            logging.info('Creating matrix for %s_%s' % (band, str(subj)))
            subject_matrix = pd.DataFrame(index=rois, columns=rois)
            key = '%s_%s' % (band, str(subj))
            for r1, roi_1 in enumerate(rois):
                for r2, roi_2 in enumerate(rois):
                    strcmp_1 = "%s_%s_%s" % (band, roi_1, roi_2)
                    strcmp_2 = "%s_%s_%s" % (band, roi_2, roi_1)
                    for varname in list(band_df):
                        if strcmp_1==str(varname) or strcmp_2==str(varname):
                            col = band_df[varname].values
                            val = col[s]
                            subject_matrix.iloc[r1, r2] = val

            for r1 in range(len(rois)):
                for r2 in range(len(rois)):
                    if r1 == r2:
                        subject_matrix.iloc[r1, r2] = 1

            if dpath is not None:
                fpath = os.path.join(dpath, 'adjacency_%s.pkl' % key)
                with open(fpath, 'wb') as file:
                    pkl.dump(subject_matrix, file)
            else:
                subj_adj_dict['%s_%s' % (band, str(subj))] = subject_matrix

    return subj_adj_dict


def calc_full_zscores(adjacency_df):
    from scipy.stats import zscore
    output_dict = {}
    for band in list(adjacency_df):
        band_df = adjacency_df[band]
        band_data = band_df.values
        z_data = zscore(band_data, axis=None, ddof=1)
        z_df = pd.DataFrame(z_data, index=band_df.index, columns=list(band_df))
        output_dict[band] = z_df
    return output_dict


if __name__ == "__main__":
    data_df = proj_utils.load_connectivity_data(drop_behavior=True)
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']

    rois = parse_roi_names(list(data_df))
    print(rois)

    print('%s: Creating adjacency dicts' % proj_utils.ctime())
    adj_dict = create_adjacency_dict(data_df, bands)
    adj_z_dict = calc_full_zscores(adj_dict)

    print('%s: Creating subject adjacency matrices' % proj_utils.ctime())
    dpath = os.path.abspath('./../../subject_z_adjacency_matrices/')
    if not os.path.isdir(dpath):
        os.mkdir(dpath)
    create_subj_adjacency_mats(adj_z_dict, bands, rois, dpath)

    x = np.random.randint(low=0, high=2, size=(84, 84))
    print(x)
    graph = networkx.from_numpy_matrix(x)
    # ecc = networkx.eccentricity(graph)
    print(graph.degree)
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
    def _union(list_1, list_2):
        return list(set(list_1) | set(list_2))

    roi_list_1, roi_list_2 = [], []
    for name in colnames:
        if "delta" in name:
            parsed = name.split("_")
            roi_1 = parsed[1]
            roi_2 = parsed[2]
            if roi_1 not in roi_list_1:
                roi_list_1.append(roi_1)
            if roi_2 not in roi_list_2:
                roi_list_2.append(roi_2)

    return _union(roi_list_1, roi_list_2)


def create_subj_adjacency_mats(adj_dict, bands, rois, dpath=None):
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


def calc_graph_measures(data_matrix, thresh=0):
    from networkx import eccentricity
    from networkx.algorithms.efficiency import global_efficiency
    from networkx.algorithms.shortest_paths.generic import average_shortest_path_length
    from networkx.algorithms.centrality import betweenness_centrality
    from networkx.algorithms.cluster import average_clustering
    from networkx.algorithms.community.modularity_max import greedy_modularity_communities
    from networkx.algorithms.community.quality import performance

    def _avg_values(results):
        values = []
        if isinstance(results, dict):
            for k in results:
                values.append(results[k])
        elif isinstance(results, list):
            for tup in results:
                values.append(tup[1])

        return np.mean(values)

    below_thresh_indices = data_matrix < thresh
    data_matrix[below_thresh_indices] = 0
    if isinstance(data_matrix, np.ndarray):
        graph = networkx.convert_matrix.from_numpy_matrix(np.real(data_matrix))
    if isinstance(data_matrix, pd.DataFrame):
        graph = networkx.convert_matrix.from_pandas_adjacency(data_matrix)

    graph_dict = {'degree': _avg_values(list(graph.degree)),
                  'eccentricity': _avg_values(eccentricity(graph)),
                  'global_efficiency': global_efficiency(graph),
                  'characteristic_path_length': average_shortest_path_length(graph),
                  'betweenness_centrality': _avg_values(betweenness_centrality(graph)),
                  'clustering_coefficient': average_clustering(graph),
                  'modularity': performance(graph, greedy_modularity_communities(graph))}

    return graph_dict


def test_graph_functions():
    x = np.random.rand(84, 84)
    test_res = calc_graph_measures(x, thresh=.9)
    for test_key in test_res:
        print(test_key)
        print(test_res[test_key])

    return test_res


def clean_df_to_numpy(df):
    # Dumb function to give networkx a numpy array
    n_rows = len(df.index)
    n_cols = len(list(df))
    new_array = np.ndarray(shape=(n_rows, n_cols))

    for x in range(n_rows):
        for y in range(n_cols):
            new_array[x, y] = df.iloc[x, y]

    return new_array


if __name__ == "__main__":
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    data_df = proj_utils.load_connectivity_data(drop_behavior=True)

    rois = parse_roi_names(list(data_df))
    print('%s: Creating adjacency dicts' % proj_utils.ctime())
    adj_dict = create_adjacency_dict(data_df, bands)

    print('%s: Creating subject adjacency matrices' % proj_utils.ctime())
    dpath = os.path.abspath('./../../subject_adjacency_matrices/')
    if not os.path.isdir(dpath):
        os.mkdir(dpath)
    create_subj_adjacency_mats(adj_dict, bands, rois, dpath)

    test_res = test_graph_functions()

    final_dict = {}
    columns = list(test_res)
    subjects = np.arange(0, len(os.listdir(dpath)))
    for band in bands:
        graph_res_df = pd.DataFrame(index=subjects, columns=columns)
        s = 0
        for matrix_file in sorted(os.listdir(dpath)):
            if band in matrix_file:
                with open(os.path.join(dpath, matrix_file), 'rb') as f:
                    df = pkl.load(f)

                conn_res = calc_graph_measures(clean_df_to_numpy(df))
                for r, res_key in enumerate(conn_res):
                    graph_res_df.iloc[s, r] = conn_res[res_key]
                s += 1

        final_dict[band] = graph_res_df

    with open(os.path.abspath('./../../graph_theory_res.pkl'), 'wb') as f:
        pkl.dump(final_dict, f)

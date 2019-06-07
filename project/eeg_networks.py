import os
import logging
import networkx
import numpy as np
import pandas as pd
import pickle as pkl
import proj_utils


logging.basicConfig(filename='./logs/eeg_networks.log', filemode='w', level=logging.INFO)


def create_adjacency_dict(data_df, bands):
    column_names = list(data_df)

    def _extract_band_colnames(band_name, colnames):
        names_extracted = []
        for name in colnames:
            if band_name in name:
                names_extracted.append(name)

        return names_extracted

    adjacency_dict = {}
    for band in bands:
        band_variables = _extract_band_colnames(band, column_names)
        adjacency_df = pd.DataFrame(columns=band_variables)
        for bv in band_variables:
            adjacency_df[bv] = data_df[bv]
        adjacency_dict[band] = adjacency_df

    return adjacency_dict


def parse_roi_names(colnames):
    def _union(list_1, list_2):
        return list(set(list_1) | set(list_2))

    roi_list_1, roi_list_2 = [], []
    for name in colnames:
        if "delta" in name:
            parsed = name.split("_")
            roi_1, roi_2 = parsed[1], parsed[2]
            if roi_1 not in roi_list_1:
                roi_list_1.append(roi_1)
            if roi_2 not in roi_list_2:
                roi_list_2.append(roi_2)

    return _union(roi_list_1, roi_list_2)


def create_subj_adjacency_mats(adj_dict, bands, rois, dpath=None):
    def _create_subject_matrix(roi_list, band_data, diag=1):
        subj_df = pd.DataFrame(index=roi_list, columns=roi_list)
        for r1, roi_1 in enumerate(roi_list):
            for r2, roi_2 in enumerate(roi_list):
                strcmp_1 = "%s_%s_%s" % (band, roi_1, roi_2)
                strcmp_2 = "%s_%s_%s" % (band, roi_2, roi_1)
                for varname in list(band_data):
                    if strcmp_1 == str(varname) or strcmp_2 == str(varname):
                        col = band_data[varname].values
                        val = col[s]
                        subj_df.iloc[r1, r2] = val

        # Converting diagonals to specified value
        for r1 in range(len(roi_list)):
            for r2 in range(len(roi_list)):
                if r1 == r2:
                    subj_df.iloc[r1, r2] = diag

        return subj_df

    for band in bands:
        band_df = adj_dict[band]
        subjects = band_df.index
        for s, subj in enumerate(subjects):
            key = '%s_%s' % (band, str(subj))
            logging.info('Creating matrix for %s' % key)

            subject_df = _create_subject_matrix(rois, band_df)
            if dpath is not None:
                fpath = os.path.join(dpath, 'adjacency_%s.pkl' % key)
                with open(fpath, 'wb') as file:
                    pkl.dump(subject_df, file)


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

    below_thresh_indices = np.abs(data_matrix) < thresh
    data_matrix[below_thresh_indices] = 0
    if isinstance(data_matrix, np.ndarray):
        graph = networkx.convert_matrix.from_numpy_matrix(np.real(data_matrix))
    if isinstance(data_matrix, pd.DataFrame):
        graph = networkx.convert_matrix.from_pandas_adjacency(data_matrix)

    degree = list(graph.degree)
    global_eff = global_efficiency(graph)
    b_central = betweenness_centrality(graph)
    modularity = performance(graph, greedy_modularity_communities(graph))
    try:
        ecc = eccentricity(graph)
    except networkx.exception.NetworkXError:
        ecc = [0]

    try:
        clust = average_clustering(graph)
    except networkx.exception.NetworkXError:
        clust = 0

    try:
        char_path = average_shortest_path_length(graph)
    except networkx.exception.NetworkXError:
        char_path = 0
    
    graph_dict = {'degree': _avg_values(degree),
                  'eccentricity': _avg_values(ecc),
                  'global_efficiency': global_eff,
                  'characteristic_path_length': char_path,
                  'betweenness_centrality': _avg_values(b_central),
                  'clustering_coefficient': clust,
                  'modularity': modularity}

    return graph_dict


def test_graph_functions():
    x = np.random.rand(84, 84)
    test_res = calc_graph_measures(x, thresh=.9)
    print('Test graph functions...')
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


def run_graph_theory(band, filelist, subjects, columns, outpath):
    thresholds = [0, .2, .4, .6, .8, .9]
    for thresh in thresholds:
        logging.info('%s: Running %s at %.2f' % (proj_utils.ctime(), band, thresh))
        s = 0
        graph_df = pd.DataFrame(index=subjects, columns=columns)
        for adj_file in filelist:
            if band in adj_file:
                with open(adj_file, 'rb') as f:
                    data_df = pkl.load(f)

                conn_res = calc_graph_measures(clean_df_to_numpy(data_df), thresh)
                for r, res_key in enumerate(conn_res):
                    graph_df[s, r] = conn_res[res_key]
                s += 1

        outfile = os.path.join(outpath, 'graph_results_%s_%.2f_thresh' % (band, thresh))
        with open(outfile, 'wb') as f:
            pkl.dump(graph_df, f)


def main():
    logging.info('%s: Starting script' % proj_utils.ctime())

    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    data_df = proj_utils.load_connectivity_data(drop_behavior=True)

    dpath = os.path.abspath('./../data/subject_adjacency_matrices/')
    if not os.path.isdir(dpath):
        os.mkdir(dpath)

    # print('%s: Creating adjacency dicts' % proj_utils.ctime())
    # adj_dict = create_adjacency_dict(data_df, bands)
    #
    # print('%s: Creating subject adjacency matrices' % proj_utils.ctime())
    # rois = parse_roi_names(list(data_df))
    # create_subj_adjacency_mats(adj_dict, bands, rois, dpath)

    test_res = test_graph_functions()

    logging.info('%s: Running graph theory analyses' % proj_utils.ctime())
    columns = list(test_res)
    subjects = np.arange(0, len(data_df.index))
    outpath = './../data/graph_theory_res/'
    if not os.path.isdir(outpath):
        os.mkdir(outpath)

    for band in bands:
        filelist = [os.path.join(dpath, f) for f in os.listdir(dpath) if band in f]
        run_graph_theory(band, filelist, subjects, columns, outpath)

    logging.info('%s: Finished' % proj_utils.ctime())


if __name__ == "__main__":
    logging.info(networkx.__version__)  # checking venv version
    main()

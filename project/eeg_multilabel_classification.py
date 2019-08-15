from os.path import isdir, join
from os import mkdir
import numpy as np
import pandas as pd
import pickle as pkl
import proj_utils as pu
from imblearn.over_sampling import RandomOverSampler
from sklearn import ensemble, feature_selection, model_selection, preprocessing, metrics
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from eeg_classification import extra_trees, knn


seed = 13


def resample_multilabel(data, target):
    """
    Apply LP-transformation to create balanced classes, then convert back to multilabel targets
    """
    target = target.astype(int)

    def convert_hads_to_str(hads_data, hads_type):
        hads_strs = []
        for val in hads_data:
            if val == 0:
                str_convert = '%s_normal' % hads_type
            elif val == 1:
                str_convert = '%s_borderline' % hads_type
            elif val == 2:
                str_convert = '%s_abnormal' % hads_type
            hads_strs.append(str_convert)
        return hads_strs

    def convert_str_to_hads(hads_tuples):
        hads_array = np.ndarray(shape=(len(hads_tuples), 2))
        for t, tup in enumerate(hads_tuples):
            for s, str in enumerate(tup):
                if '_normal' in str:
                    hads_array[t, s] = 0
                elif '_borderline' in str:
                    hads_array[t, s] = 1
                elif '_abnormal' in str:
                    hads_array[t, s] = 2
        return hads_array

    anx_strings = convert_hads_to_str(target[:, 0], 'anxiety')
    dep_strings = convert_hads_to_str(target[:, 1], 'depression')
    multilabel_hads = [(anx_strings[n], dep_strings[n]) for n in range(len(anx_strings))]
    mlb = preprocessing.MultiLabelBinarizer()
    binary_matrix = mlb.fit_transform(multilabel_hads)

    from skmultilearn.problem_transform import LabelPowerset
    lp = LabelPowerset()
    resampler = RandomOverSampler(sampling_strategy='not majority', random_state=seed)
    target_lp_transformed = lp.transform(binary_matrix)

    data_resampled, target_lp_transformed_resampled = resampler.fit_sample(data, target_lp_transformed)
    binary_matrix_resampled = lp.inverse_transform(target_lp_transformed_resampled)
    target_resampled_multilabel = mlb.inverse_transform(binary_matrix_resampled)
    target_resampled_multilabel_array = convert_str_to_hads(target_resampled_multilabel)
    return data_resampled, target_resampled_multilabel_array


def feature_selection_with_covariates(x_train, x_test, y_train, continuous_indices, categorical_indices, feature_names):
    # Split data for continuous, categorical preprocessing
    x_train_cont, x_test_cont = x_train[:, continuous_indices], x_test[:, continuous_indices]
    x_train_cat, x_test_cat = x_train[:, categorical_indices], x_test[:, categorical_indices]

    # Standardization for continuous data
    preproc = preprocessing.StandardScaler().fit(x_train_cont)
    x_train_z = preproc.transform(x_train_cont)
    x_test_z = preproc.transform(x_test_cont)

    # Variance threshold for categorical data
    varthresh = feature_selection.VarianceThreshold(threshold=0).fit(x_train_cat)
    x_train_v = varthresh.transform(x_train_cat)
    x_test_v = varthresh.transform(x_test_cat)

    x_train_data = np.hstack((x_train_z, x_train_v))
    x_test_data = np.hstack((x_test_z, x_test_v))

    # Feature selection with extra trees
    extra_tree_fs = ensemble.ExtraTreesClassifier(random_state=seed)
    feature_model = feature_selection.SelectFromModel(extra_tree_fs, threshold="2*mean")

    # Transform train and test data with feature selection model
    x_train_feature_selected = feature_model.fit_transform(x_train_data, y_train)
    x_test_feature_selected = feature_model.transform(x_test_data)
    feature_indices = feature_model.get_support(indices=True)
    cleaned_features = [feature_names[i] for i in feature_indices]

    return x_train_feature_selected, x_test_feature_selected, cleaned_features


def feature_selection_without_covariates(x_train, x_test, y_train, feature_names):
    # Standardization for continuous data
    preproc = preprocessing.StandardScaler().fit(x_train)
    x_train_z = preproc.transform(x_train)
    x_test_z = preproc.transform(x_test)

    # Feature selection with extra trees
    extra_tree_fs = ensemble.ExtraTreesClassifier(random_state=seed)
    feature_model = feature_selection.SelectFromModel(extra_tree_fs, threshold="2*mean")

    # Transform train and test data with feature selection model
    x_train_feature_selected = feature_model.fit_transform(x_train_z, y_train)
    x_test_feature_selected = feature_model.transform(x_test_z)
    feature_indices = feature_model.get_support(indices=True)
    cleaned_features = [feature_names[i] for i in feature_indices]

    return x_train_feature_selected, x_test_feature_selected, cleaned_features


@ignore_warnings(category=ConvergenceWarning)
def eeg_multilabel_classify(ml_data, target_data, target_type, model, outdir):
    target_outdir = join(outdir, target_type)
    if not isdir(target_outdir):
        mkdir(target_outdir)

    feature_names = list(ml_data)

    # Create score dataframes, k-fold splitter
    n_splits = 10
    skf = model_selection.StratifiedKFold(n_splits=n_splits, random_state=seed)

    # Oversample connectivity data, apply k-fold splitter
    """Note: LP-transformation has to be applied for resampling, even though we're not treating it as a OVR problem"""
    x_res, y_res = resample_multilabel(ml_data, target_data)
    print(x_res.shape)
    print(y_res.shape)
    skf.get_n_splits(x_res, y_res)

    fold_count = 0
    classifier_objects, classifier_coefficients, cm_dict, norm_cm_dict = {}, {}, {}, {}
    balanced_acc, chance_acc, f1_scores = [], [], []

    for train_idx, test_idx in skf.split(x_res, y_res):
        fold_count += 1
        print('%s: Running FOLD %d for %s' % (pu.ctime(), fold_count, target_type))
        foldname = 'Fold %02d' % fold_count

        # Stratified k-fold splitting
        x_train, x_test = x_res[train_idx], x_res[test_idx]
        y_train, y_test = y_res[train_idx], y_res[test_idx]

        if "categorical_sex_male" in feature_names:
            continuous_features = [f for f in feature_names if 'categorical' not in f]
            continuous_indices = [ml_data.columns.get_loc(cont) for cont in continuous_features]

            categorical_features = [f for f in feature_names if 'categorical' in f]
            categorical_indices = [ml_data.columns.get_loc(cat) for cat in categorical_features]

            x_train_feature_selected, x_test_feature_selected, cleaned_features = feature_selection_with_covariates(
                x_train, x_test, y_train, continuous_indices, categorical_indices, feature_names)
        else:
            x_train_feature_selected, x_test_feature_selected, cleaned_features = feature_selection_without_covariates(
                x_train, x_test, y_train, feature_names)


print('%s: Loading data' % pu.ctime())
behavior_data, conn_data = pu.load_data_full_subjects()
conn_data.astype(float)

categorical_variables = ['smoking', 'deanxit_antidepressants', 'rivotril_antianxiety', 'sex']
categorical_data = behavior_data[categorical_variables]
dummy_coded_categorical = pu.dummy_code_binary(categorical_data)
covariate_data = pd.concat([behavior_data['age'], dummy_coded_categorical], axis=1)

ml_data = pd.concat([conn_data, covariate_data], axis=1)
multilabel_models = ['extra_trees', 'knn']
for model in multilabel_models:
    output_dir = './../data/%s_multilabel/' % model
    if not isdir(output_dir):
        mkdir(output_dir)
    # 0-7 (normal); 8-10 (borderline); 11-21 (abnormal)
    hads_thresholds = [8, 11, 21]
    print('%s: Running OVR classification on HADS' % pu.ctime())
    anx = behavior_data['anxiety_score'].values.astype(float)
    anx_binned = np.digitize(anx, bins=hads_thresholds, right=True)  # right=True: bin < x <= bin if ascending

    dep = behavior_data['depression_score'].values.astype(float)
    dep_binned = np.digitize(dep, bins=hads_thresholds, right=True)

    multi_target = np.vstack((anx_binned, dep_binned)).T
    eeg_multilabel_classify(ml_data, multi_target, target_type='hads', model=model, outdir=output_dir)
    break

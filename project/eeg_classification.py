import numpy as np
import pandas as pd
import pickle as pkl
import proj_utils as pu
from os.path import isdir, join
from os import mkdir
from copy import deepcopy
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn import ensemble, feature_selection, model_selection, preprocessing, svm, metrics, neighbors
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


seed = 13


def calc_scores(y_test, predicted):
    balanced = metrics.balanced_accuracy_score(y_test, predicted)
    chance = metrics.balanced_accuracy_score(y_test, predicted, adjusted=True)
    f1 = metrics.f1_score(y_test, predicted, average=None)
    return balanced, chance, f1


def save_scores(f1_scores, balanced_scores, chance_scores, class_labels):
    # Calculate average performance and tack it onto the end of the score list, save to nice df
    n_folds = len(balanced_scores)
    f1_array = np.asarray(f1_scores)
    if n_folds != f1_array.shape[0]:
        raise ValueError("Number of folds does not match")

    rownames = ['Fold %02d' % (n+1) for n in range(n_folds)]
    rownames.append('Average')

    f1_class_averages = np.mean(f1_array, axis=0)
    f1_data = np.vstack((f1_array, f1_class_averages))
    f1_df = pd.DataFrame(f1_data, index=rownames, columns=class_labels)

    balanced_scores.append(np.mean(balanced_scores))
    chance_scores.append(np.mean(chance_scores))
    accuracy_data = np.asarray([balanced_scores, chance_scores]).T
    score_df = pd.DataFrame(data=accuracy_data, index=rownames, columns=['Balanced accuracy', 'Chance accuracy'])
    return f1_df, score_df


def svmc(x_train, y_train, x_test, cleaned_features):
    clf = svm.LinearSVC(fit_intercept=False, random_state=seed)
    clf.fit(x_train, y_train)
    target_classes = clf.classes_
    target_classes = [str(c) for c in target_classes]

    predicted = clf.predict(x_test)

    if len(target_classes) == 2:
        idx_label = ['coefficients']
    else:
        idx_label = target_classes
    coef_df = pd.DataFrame(clf.coef_, index=idx_label, columns=cleaned_features)
    return predicted, coef_df, clf


def extra_trees(x_train, y_train, x_test, cleaned_features):
    clf = ensemble.ExtraTreesClassifier(random_state=seed)
    clf.fit(x_train, y_train)
    predicted = clf.predict(x_test)
    feature_df = pd.DataFrame(columns=cleaned_features)
    feature_df.loc['feature_importances'] = clf.feature_importances_
    return predicted, feature_df, clf


def knn(x_train, y_train, x_test):
    clf = neighbors.KNeighborsClassifier()
    clf.fit(x_train, y_train)
    predicted = clf.predict(x_test)
    return predicted, clf


def convert_hads_to_single_label(hads_array):
    hads_array = hads_array.astype(int)

    vartypes = ['anxiety', 'depression']
    hads_single_label = []
    for row in range(hads_array.shape[0]):
        str_combos = []
        for col in range(hads_array.shape[1]):
            val = hads_array[row, col]
            if val == 0:
                str_convert = '%s_normal' % vartypes[col]
            elif val == 1:
                str_convert = '%s_borderline' % vartypes[col]
            elif val == 2:
                str_convert = '%s_abnormal' % vartypes[col]
            str_combos.append(str_convert)
        hads_combined = '%s-%s' % (str_combos[0], str_combos[1])
        hads_single_label.append(hads_combined)
    return hads_single_label


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
def eeg_classify(eeg_data, target_data, target_type, model, outdir, resample=None):
    target_outdir = join(outdir, target_type)
    if not isdir(target_outdir):
        mkdir(target_outdir)

    feature_names = list(eeg_data)
    if "categorical_sex_male" in feature_names:
        cv_check = 'with_covariates'
    else:
        cv_check = 'without_covariates'

    if resample is None:
        x_res, y_res = eeg_data.values, np.asarray(target_data)
        rs_check = 'no_resample'
    elif resample is 'over':
        # resampler = RandomOverSampler(sampling_strategy='not majority', random_state=seed)
        resampler = SMOTE(sampling_strategy='not majority', random_state=seed)
        x_res, y_res = resampler.fit_resample(eeg_data, target_data)
        rs_check = 'oversampled'
    elif resample is 'under':
        resampler = RandomUnderSampler(sampling_strategy='not minority', random_state=seed)
        x_res, y_res = resampler.fit_resample(eeg_data, target_data)
        rs_check = 'undersampled'

    model_outdir = join(target_outdir, '%s_%s_%s' % (model, cv_check, rs_check))
    if not isdir(model_outdir):
        mkdir(model_outdir)
    print('%s: Running classification - %s %s %s %s' % (pu.ctime(), target_type, model, cv_check, rs_check))

    # Apply k-fold splitter
    n_splits = 10
    skf = model_selection.StratifiedKFold(n_splits=n_splits, random_state=seed)
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
            continuous_indices = [eeg_data.columns.get_loc(cont) for cont in continuous_features]

            categorical_features = [f for f in feature_names if 'categorical' in f]
            categorical_indices = [eeg_data.columns.get_loc(cat) for cat in categorical_features]

            x_train_fs, x_test_fs, cleaned_features = feature_selection_with_covariates(
                x_train, x_test, y_train, continuous_indices, categorical_indices, feature_names)
        else:
            x_train_fs, x_test_fs, cleaned_features = feature_selection_without_covariates(
                x_train, x_test, y_train, feature_names)

        if model is 'svm':
            predicted, coef_df, clf = svmc(x_train_fs, y_train, x_test_fs, cleaned_features)
            classifier_coefficients[foldname] = coef_df

        elif model is 'extra_trees':
            predicted, feature_importances, clf = extra_trees(x_train_fs, y_train, x_test_fs, cleaned_features)
            classifier_coefficients[foldname] = feature_importances

        elif model is 'knn':
            predicted, clf = knn(x_train_fs, y_train, x_test_fs)

        classifier_objects[foldname] = clf

        # Calculating fold performance scores
        balanced, chance, f1 = calc_scores(y_test, predicted)
        balanced_acc.append(balanced)
        chance_acc.append(chance)
        f1_scores.append(f1)

        # Calculating fold confusion matrix
        cm = metrics.confusion_matrix(y_test, predicted)
        normalized_cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]

        cm_dict[foldname] = pd.DataFrame(cm, index=clf.classes_, columns=clf.classes_)
        norm_cm_dict[foldname] = pd.DataFrame(normalized_cm, index=clf.classes_, columns=clf.classes_)

    # Saving performance scores
    f1_df, score_df = save_scores(f1_scores, balanced_acc, chance_acc, class_labels=clf.classes_)
    scores_dict = {'accuracy scores': score_df,
                   'f1 scores': f1_df}

    pu.save_xls(scores_dict, join(model_outdir, 'performance.xlsx'))

    # Saving coefficients
    if bool(classifier_coefficients):
        pu.save_xls(classifier_coefficients, join(model_outdir, 'coefficients.xlsx'))
    pu.save_xls(cm_dict, join(model_outdir, 'confusion_matrices.xlsx'))
    pu.save_xls(norm_cm_dict, join(model_outdir, 'confusion_matrices_normalized.xlsx'))

    # Saving classifier object
    with open(join(model_outdir, 'classifier_object.pkl'), 'wb') as file:
        pkl.dump(classifier_objects, file)


def side_classification_drop_asym(ml_data, behavior_data, output_dir, models=None):
    print('%s: Running classification on tinnitus side, dropping asymmetrical subjects' % pu.ctime())
    ml_copy = deepcopy(ml_data)
    if models is None:
        models = ['extra_trees']
    resample_methods = [None, 'over', 'under']
    t = pu.convert_tin_to_str(behavior_data['tinnitus_side'].values.astype(float), 'tinnitus_side')
    t_df = pd.DataFrame(t, index=ml_copy.index)
    asym_indices = []
    for asym in ['Right>Left', 'Left>Right']:
        asym_indices.extend([i for i, s in enumerate(t) if asym == s])

    asym_data = ml_copy.iloc[asym_indices]
    ml_copy.drop(index=asym_data.index, inplace=True)
    t_df.drop(index=asym_data.index, inplace=True)
    target_cleaned = np.ravel(t_df.values)

    for model in models:
        for res in resample_methods:
            eeg_classify(ml_copy, target_cleaned, 'tinnitus_side_no_asym', model, output_dir, resample=res)


def type_classification_drop_mixed(ml_data, behavior_data, output_dir, models=None):
    print('%s: Running classification on tinnitus type, dropping mixed type subjects' % pu.ctime())
    ml_copy = deepcopy(ml_data)
    if models is None:
        models = ['extra_trees']
    resample_methods = [None, 'over', 'under']
    t = pu.convert_tin_to_str(behavior_data['tinnitus_type'].values.astype(float), 'tinnitus_type')
    t_df = pd.DataFrame(t, index=ml_copy.index)
    mixed_indices = [i for i, s in enumerate(t) if s == 'PT_and_NBN']

    type_data = ml_copy.iloc[mixed_indices]
    ml_copy.drop(index=type_data.index, inplace=True)
    t_df.drop(index=type_data.index, inplace=True)
    target_cleaned = np.ravel(t_df.values)
    for model in models:
        for res in resample_methods:
            eeg_classify(ml_copy, target_cleaned, 'tinnitus_type_no_mixed', model, output_dir, resample=res)


def classification_main(ml_data, behavior_data, output_dir, models=None):
    if models is None:
        models = ['extra_trees']
    resample_methods = ['over']  # [None, 'over', 'under']

    targets = {}
    side_data = pu.convert_tin_to_str(behavior_data['tinnitus_side'].values.astype(float), 'tinnitus_side')
    targets['tinnitus_side'] = side_data

    type_data = pu.convert_tin_to_str(behavior_data['tinnitus_type'].values.astype(float), 'tinnitus_type')
    targets['tinnitus_type'] = type_data

    tq_data = behavior_data['distress_TQ'].values
    high_low_thresholds = [0, 46, 84]
    tq_high_low = np.digitize(tq_data, bins=high_low_thresholds, right=True)
    targets['TQ_high_low'] = tq_high_low

    grade_thresholds = [0, 30, 46, 59, 84]
    binned_target = np.digitize(tq_data, bins=grade_thresholds, right=True)
    tq_grade = ['Grade_%d' % t for t in binned_target]
    targets['TQ_grade'] = tq_grade

    hads_thresholds = [8, 11, 21]  # 0-7 (normal); 8-10 (borderline); 11-21 (abnormal)
    anx_binned = np.digitize(behavior_data['anxiety_score'].values.astype(float), bins=hads_thresholds, right=True)
    dep_binned = np.digitize(behavior_data['depression_score'].values.astype(float), bins=hads_thresholds, right=True)
    targets['hads_OVR'] = convert_hads_to_single_label(np.vstack((anx_binned, dep_binned)).T)

    for target in targets:
        target_data = targets[target]
        for model in models:
            for res in resample_methods:
                eeg_classify(ml_data, target_data, target_type=target, model=model, outdir=output_dir, resample=res)


if __name__ == "__main__":
    output_dir = './../data/'
    print('%s: Loading data' % pu.ctime())
    behavior_data, conn_data = pu.load_data_full_subjects()
    conn_data.astype(float)

    categorical_variables = ['smoking', 'deanxit_antidepressants', 'rivotril_antianxiety', 'sex']
    categorical_data = behavior_data[categorical_variables]
    dummy_coded_categorical = pu.dummy_code_binary(categorical_data)
    covariate_data = pd.concat([behavior_data['age'], dummy_coded_categorical], axis=1)

    ml_data = pd.concat([conn_data, covariate_data], axis=1)

    models = ['svm', 'extra_trees', 'knn']
    # classification_main(ml_data, behavior_data, output_dir, models=models)
    # side_classification_drop_asym(ml_data, behavior_data, output_dir, models=models)
    # type_classification_drop_mixed(ml_data, behavior_data, output_dir, models=models)

    print('%s: Finished' % pu.ctime())


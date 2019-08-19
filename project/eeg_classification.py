from os.path import isdir, join
from os import mkdir
import numpy as np
import pandas as pd
import pickle as pkl
import proj_utils as pu
from imblearn.over_sampling import RandomOverSampler
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


@ignore_warnings(category=ConvergenceWarning)
def eeg_classify(eeg_data, target_data, target_type, model, outdir):
    target_outdir = join(outdir, target_type)
    if not isdir(target_outdir):
        mkdir(target_outdir)

    feature_names = list(eeg_data)
    
    continuous_features = [f for f in feature_names if 'categorical' not in f]
    continuous_indices = [eeg_data.columns.get_loc(cont) for cont in continuous_features]

    categorical_features = [f for f in feature_names if 'categorical' in f]
    categorical_indices = [eeg_data.columns.get_loc(cat) for cat in categorical_features]

    # Create score dataframes, k-fold splitter
    n_splits = 10
    skf = model_selection.StratifiedKFold(n_splits=n_splits, random_state=seed)

    # Oversample connectivity data, apply k-fold splitter
    resampler = RandomOverSampler(sampling_strategy='not majority', random_state=seed)
    x_res, y_res = resampler.fit_resample(eeg_data, target_data)
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
        x_train_fs = feature_model.fit_transform(x_train_data, y_train)
        x_test_fs = feature_model.transform(x_test_data)
        feature_indices = feature_model.get_support(indices=True)
        cleaned_features = [feature_names[i] for i in feature_indices]

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
    f1_array = np.asarray(f1_scores)
    f1_class_averages = np.mean(f1_array, axis=0)
    f1_data = np.vstack((f1_array, f1_class_averages))

    balanced_acc_avg = np.mean(balanced_acc)
    chance_acc_avg = np.mean(chance_acc)

    balanced_acc.append(balanced_acc_avg)
    chance_acc.append(chance_acc_avg)

    accuracy_data = np.asarray([balanced_acc, chance_acc]).T

    rownames = ['Fold %02d' % (n+1) for n in range(n_splits)]
    rownames.append('Average')
    score_df = pd.DataFrame(data=accuracy_data, index=rownames, columns=['Balanced accuracy', 'Chance accuracy'])

    f1_df = pd.DataFrame(data=np.asarray(f1_data), index=rownames, columns=clf.classes_)
    scores_dict = {'accuracy scores': score_df,
                   'f1 scores': f1_df}

    pu.save_xls(scores_dict, join(target_outdir, 'performance.xlsx'))

    # Saving coefficients
    if bool(classifier_coefficients):
        pu.save_xls(classifier_coefficients, join(target_outdir, 'coefficients.xlsx'))
    pu.save_xls(cm_dict, join(target_outdir, 'confusion_matrices.xlsx'))
    pu.save_xls(norm_cm_dict, join(target_outdir, 'confusion_matrices_normalized.xlsx'))

    # Saving classifier object
    with open(join(target_outdir, 'classifier_object.pkl'), 'wb') as file:
        pkl.dump(classifier_objects, file)


def convert_tinnitus_data_to_str(tinnitus_data, data_type):
    str_data = []
    if data_type is 'tinnitus_side':
        for t in tinnitus_data:
            if t == -2:
                str_data.append('Left')
            elif t == -1:
                str_data.append('Left>Right')
            elif t == 0:
                str_data.append('Bilateral')
            elif t == 1:
                str_data.append('Right>Left')
            elif t == 2:
                str_data.append('Right')
        if len(str_data) != len(tinnitus_data):
            raise ValueError('Side data not parsed correctly')
    elif data_type is 'tinnitus_type':
        for t in tinnitus_data:
            if t == 0:
                str_data.append('PT')
            elif t == 1:
                str_data.append('PT_and_NBN')
            elif t == 2:
                str_data.append('NBN')
        if len(str_data) != len(tinnitus_data):
            raise ValueError('Type data not parsed correctly')
    return str_data


def side_classification_drop_asym(ml_data, behavior_data):
    print('%s: Running classification on tinnitus side, dropping asymmetrical subjects' % pu.ctime())
    t = convert_tinnitus_data_to_str(behavior_data['tinnitus_side'].values.astype(float) * 2, 'tinnitus_side')
    t_df = pd.DataFrame(t, index=ml_data.index)
    asym_indices = []
    for asym in ['Right>Left', 'Left>Right']:
        asym_indices.extend([i for i, s in enumerate(t) if asym == s])

    asym_data = ml_data.iloc[asym_indices]
    ml_data.drop(index=asym_data.index, inplace=True)
    t_df.drop(index=asym_data.index, inplace=True)

    models = ['svm', 'extra_trees', 'knn']
    for model in models:
        output_dir = './../data/%s/' % model
        if not isdir(output_dir):
            mkdir(output_dir)
        eeg_classify(ml_data, np.ravel(t_df.values), 'tinnitus_side_no_asym', model, output_dir)


def type_classification_drop_mixed(ml_data, behavior_data):
    print('%s: Running classification on tinnitus type, dropping mixed type subjects' % pu.ctime())
    t = convert_tinnitus_data_to_str(np.add(behavior_data['tinnitus_type'].values.astype(int), 1), 'tinnitus_type')
    t_df = pd.DataFrame(t, index=ml_data.index)
    mixed_indices = [i for i, s in enumerate(t) if s == 'PT_and_NBN']

    type_data = ml_data.iloc[mixed_indices]
    ml_data.drop(index=type_data.index, inplace=True)
    t_df.drop(index=type_data.index, inplace=True)

    models = ['svm', 'extra_trees', 'knn']
    for model in models:
        output_dir = './../data/%s/' % model
        if not isdir(output_dir):
            mkdir(output_dir)
        eeg_classify(ml_data, np.ravel(t_df.values), 'tinnitus_type_no_mixed', model, output_dir)


def classification_main(ml_data, behavior_data):
    models = ['svm', 'extra_trees', 'knn']
    for model in models:
        output_dir = './../data/%s/' % model
        if not isdir(output_dir):
            mkdir(output_dir)

        # print('%s: Running classification on tinnitus side' % pu.ctime())
        # # Left, Left>Right, Bil/Holo, Right>Left, Right
        # t = convert_tinnitus_data_to_str(behavior_data['tinnitus_side'].values.astype(float) * 2, 'tinnitus_side')
        # eeg_classify(eeg_data=ml_data, target_data=t, target_type='tinnitus_side', model=model, outdir=output_dir)
        #
        # print('%s: Running classification on tinnitus type' % pu.ctime())
        # # PureTone, PureTone+NBN, NBN
        # t = convert_tinnitus_data_to_str(np.add(behavior_data['tinnitus_type'].values.astype(int), 1), 'tinnitus_type')
        # eeg_classify(eeg_data=ml_data, target_data=t, target_type='tinnitus_type', model=model, outdir=output_dir)
        #
        # print('%s: Running classification on TQ - high/low' % pu.ctime())
        # target = behavior_data['distress_TQ'].values
        # high_low_thresholds = [0, 46, 84]
        # binned_target = np.digitize(target, bins=high_low_thresholds, right=True)
        # target = ['TQ_high' if t > 1 else 'TQ_low' for t in binned_target]
        # eeg_classify(eeg_data=ml_data, target_data=target, target_type='TQ_high_low', model=model, outdir=output_dir)
        #
        # print('%s: Running classification on TQ - grade' % pu.ctime())
        # target = behavior_data['distress_TQ'].values
        # grade_thresholds = [0, 30, 46, 59, 84]
        # binned_target = np.digitize(target, bins=grade_thresholds, right=True)
        # target = ['Grade_%d' % t for t in binned_target]
        # eeg_classify(eeg_data=ml_data, target_data=target, target_type='TQ_grade', model=model, outdir=output_dir)

        # 0-7 (normal); 8-10 (borderline); 11-21 (abnormal)
        hads_thresholds = [8, 11, 21]
        print('%s: Running OVR classification on HADS' % pu.ctime())
        anx = behavior_data['anxiety_score'].values.astype(float)
        anx_binned = np.digitize(anx, bins=hads_thresholds, right=True)  # right=True: bin < x <= bin if ascending
        dep = behavior_data['depression_score'].values.astype(float)
        dep_binned = np.digitize(dep, bins=hads_thresholds, right=True)
        multi_hads = np.vstack((anx_binned, dep_binned)).T
        single_hads = convert_hads_to_single_label(multi_hads)
        eeg_classify(eeg_data=ml_data, target_data=single_hads, target_type='hads_OVR', model=model, outdir=output_dir)

    print('%s: Finished' % pu.ctime())


if __name__ == "__main__":
    print('%s: Loading data' % pu.ctime())
    behavior_data, conn_data = pu.load_data_full_subjects()
    conn_data.astype(float)

    categorical_variables = ['smoking', 'deanxit_antidepressants', 'rivotril_antianxiety', 'sex']
    categorical_data = behavior_data[categorical_variables]
    dummy_coded_categorical = pu.dummy_code_binary(categorical_data)
    covariate_data = pd.concat([behavior_data['age'], dummy_coded_categorical], axis=1)

    ml_data = pd.concat([conn_data, covariate_data], axis=1)

    classification_main(ml_data, behavior_data)

    # type_classification_drop_mixed(ml_data, behavior_data)
    # side_classification_drop_asym(ml_data, behavior_data)

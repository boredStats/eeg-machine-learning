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
import eeg_classification as eeg_class


def resample_multilabel(data, target):
    target = target.astype(int)

    def combine_anx_dep_to_str(hads_data):
        vartypes = ['anxiety', 'depression']
        combined_classes = []
        for row in range(hads_data.shape[0]):
            str_combos = []
            for col in range(hads_data.shape[1]):
                val = hads_data[row, col]
                if val == 0:
                    str_convert = '%s_normal' % vartypes[col]
                elif val == 1:
                    str_convert = '%s_borderline' % vartypes[col]
                elif val == 2:
                    str_convert = '%s_abnormal' % vartypes[col]
                str_combos.append(str_convert)
            hads_combined = '%s-%s' % (str_combos[0], str_combos[1])
            combined_classes.append(hads_combined)
        return combined_classes

    hads_to_string = combine_anx_dep_to_str(target)
    lb = preprocessing.LabelBinarizer()
    indicator_matrix = lb.fit_transform(hads_to_string)
    class_labels = list(lb.classes_)

    from skmultilearn.problem_transform import LabelPowerset
    lp = LabelPowerset()
    resampler = RandomOverSampler(sampling_strategy='not majority', random_state=13)
    target_transformed = lp.transform(indicator_matrix)

    data_resampled, target_resampled = resampler.fit_sample(data, target_transformed)
    target_resampled_binary = lp.inverse_transform(target_resampled)
    return data_resampled, target_resampled, class_labels, target_resampled_binary


@ignore_warnings(category=ConvergenceWarning)
def eeg__multilabel_classify(eeg_data, target_data, target_type, model, outdir):
    target_outdir = join(outdir, target_type)
    if not isdir(target_outdir):
        mkdir(target_outdir)

    feature_names = list(eeg_data)

    continuous_features = [f for f in feature_names if 'categorical' not in f]
    continuous_indices = [eeg_data.columns.get_loc(cont) for cont in continuous_features]

    categorical_features = [f for f in feature_names if 'categorical' in f]
    categorical_indices = [eeg_data.columns.get_loc(cat) for cat in categorical_features]

    n_splits = 10
    data_resampled, target_resampled, target_classes, _ = resample_multilabel(ml_data, target_data)

    skf = model_selection.StratifiedKFold(n_splits=n_splits, random_state=13)
    x_res = data_resampled
    y_res = target_resampled
    skf.get_n_splits(x_res, y_res)

    fold_count = 0
    classifier_objects, classifier_coefficients, cm_dict, norm_cm_dict = {}, {}, {}, {}
    balanced_acc, chance_acc, f1_scores, cm_list, cm_norm_list = [], [], [], [], []

    for train_idx, test_idx in skf.split(x_res, y_res):
        fold_count += 1
        foldname = 'Fold %02d' % fold_count
        print('%s: Running FOLD %d for %s' % (pu.ctime(), fold_count, target_type))

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
        clf = ensemble.ExtraTreesClassifier()
        feature_model = feature_selection.SelectFromModel(clf, threshold="2*mean")

        # Transform train and test data with feature selection model
        x_train_fs = feature_model.fit_transform(x_train_data, y_train)
        x_test_fs = feature_model.transform(x_test_data)
        feature_indices = feature_model.get_support(indices=True)
        cleaned_features = [feature_names[i] for i in feature_indices]

        if model is 'svm':
            predicted, coef_df, clf = eeg_class.svmc(x_train_fs, y_train, x_test_fs, cleaned_features)
            classifier_coefficients[foldname] = coef_df

        elif model is 'extra_trees':
            predicted, feature_importances, clf = eeg_class.extra_trees(x_train_fs, y_train, x_test_fs, cleaned_features)
            classifier_coefficients[foldname] = feature_importances

        elif model is 'knn':
            predicted, clf = eeg_class.knn(x_train_fs, y_train, x_test_fs)

        # Calculating fold performance scores
        balanced, chance, f1 = eeg_class.calc_scores(y_test, predicted)
        balanced_acc.append(balanced)
        chance_acc.append(chance)
        f1_scores.append(f1)

        # Calculating fold confusion matrix
        cm = metrics.confusion_matrix(y_test, predicted)
        normalized_cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]

        cm_dict[foldname] = pd.DataFrame(cm, index=clf.classes_, columns=clf.classes_)
        norm_cm_dict[foldname] = pd.DataFrame(normalized_cm, index=clf.classes_, columns=clf.classes_)

        cm_list.append(cm)
        cm_norm_list.append(normalized_cm)

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

    eeg_class.save_xls(scores_dict, join(target_outdir, 'performance.xlsx'))
    if bool(classifier_coefficients):
        eeg_class.save_xls(classifier_coefficients, join(target_outdir, 'coefficients.xlsx'))

    pu.save_xls(cm_dict, join(target_outdir, 'confusion_matrices.xlsx'))
    pu.save_xls(norm_cm_dict, join(target_outdir, 'confusion_matrices_normalized.xlsx'))

    # Saving classifier object
    with open(join(target_outdir, 'classifier_object.pkl'), 'wb') as file:
        pkl.dump(clf, file)


if __name__ == "__main__":
    models = ['svm', 'extra_trees', 'knn']

    print('%s: Loading data' % pu.ctime())
    behavior_data, conn_data = pu.load_data_full_subjects()
    conn_data.astype(float)

    categorical_variables = ['smoking', 'deanxit_antidepressants', 'rivotril_antianxiety', 'sex']
    categorical_data = behavior_data[categorical_variables]
    dummy_coded_categorical = pu.dummy_code_binary(categorical_data)
    covariate_data = pd.concat([behavior_data['age'], dummy_coded_categorical], axis=1)

    ml_data = pd.concat([conn_data, covariate_data], axis=1)
    for model in models:
        output_dir = './../data/%s/' % model
        if not isdir(output_dir):
            mkdir(output_dir)
        # 0-7 (normal); 8-10 (borderline); 11-21 (abnormal)
        hads_thresholds = [8, 11, 21]
        print('%s: Running classification on HADS' % pu.ctime())
        anx = behavior_data['anxiety_score'].values.astype(float)
        anx_binned = np.digitize(anx, bins=hads_thresholds, right=True)  # right=True: bin < x <= bin if ascending

        dep = behavior_data['depression_score'].values.astype(float)
        dep_binned = np.digitize(dep, bins=hads_thresholds, right=True)

        multi_target = np.vstack((anx_binned, dep_binned)).T
        eeg__multilabel_classify(ml_data, target_data=multi_target, target_type='hads', model=model, outdir=output_dir)

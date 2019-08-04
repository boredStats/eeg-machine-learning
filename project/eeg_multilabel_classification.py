from os.path import isdir, join
from os import mkdir
import numpy as np
import pandas as pd
import pickle as pkl
import proj_utils as pu
from imblearn.over_sampling import RandomOverSampler
from sklearn import ensemble, feature_selection, model_selection, preprocessing, svm, metrics, linear_model, neighbors
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


def save_xls(dict_df, path):
    # Save a dictionary of dataframes to an excel file, with each dataframe as a seperate page
    writer = pd.ExcelWriter(path)
    for key in list(dict_df):
        dict_df[key].to_excel(writer, '%s' % key)
    writer.save()


def calc_scores(y_test, predicted):
    balanced = metrics.balanced_accuracy_score(y_test, predicted)
    chance = metrics.balanced_accuracy_score(y_test, predicted, adjusted=True)
    f1 = metrics.f1_score(y_test, predicted, average=None)
    return balanced, chance, f1


def svmc(x_train, y_train, x_test, target_classes, cleaned_features):
    clf = svm.LinearSVC(fit_intercept=False)
    clf.fit(x_train, y_train)

    predicted = clf.predict(x_test)

    if len(target_classes) == 2:
        idx_label = ['support_vector_coefficients']
    else:
        idx_label = target_classes
    # coef_df = pd.DataFrame(clf.coef_, index=idx_label, columns=cleaned_features)
    coef_df = pd.DataFrame(clf.coef_, columns=idx_label, index=cleaned_features)
    return predicted, coef_df


def extra_trees(x_train, y_train, x_test, cleaned_features):
    clf = ensemble.ExtraTreesClassifier()
    clf.fit(x_train, y_train)
    predicted = clf.predict(x_test)
    feature_df = pd.DataFrame(clf.feature_importances_, index=cleaned_features)
    return predicted, feature_df


def knn(x_train, y_train, x_test):
    clf = neighbors.KNeighborsClassifier()
    clf.fit(x_train, y_train)
    predicted = clf.predict(x_test)
    return predicted


def test_labelpowerset(data, target):
    def _label_to_str(hads_data, hads_type):
        out = []
        for val in hads_data:
            if val == 0:
                out.append('%s_normal' % hads_type)
            elif val == 1:
                out.append('%s_borderline' % hads_type)
            elif val == 2:
                out.append('%s_abnormal' % hads_type)
        return out
    from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
    target = target.astype(int)
    anx_data, dep_data = list(target[:, 0]), list(target[:, 1])
    anx_str = _label_to_str(anx_data, 'anxiety')
    dep_str = _label_to_str(dep_data, 'depression')

    # ## Binarizer (stacking)
    # lb = LabelBinarizer()
    # anx_matrix = lb.fit_transform(anx_data)
    #
    # dep_matrix = lb.fit_transform(dep_data)
    #
    # indicator_matrix = np.hstack((anx_matrix, dep_matrix))
    # print(indicator_matrix.shape)

    # ## Binarizer (multilabel)
    mlb = MultiLabelBinarizer()


    from skmultilearn.problem_transform import LabelPowerset
    lp = LabelPowerset()
    ros = RandomOverSampler()
    target_transformed = lp.transform(indicator_matrix)
    print(np.unique(target_transformed))

    data_, target_ = ros.fit_sample(data, target_transformed)

    target_resampled = lp.inverse_transform(target_).toarray()
    print(target_resampled.shape)


def resample_multilabel(data, target):
    def _label_to_str(hads_data, hads_type):
        out = []
        for val in hads_data:
            if val == 0:
                out.append('%s_normal' % hads_type)
            elif val == 1:
                out.append('%s_borderline' % hads_type)
            elif val == 2:
                out.append('%s_abnormal' % hads_type)
        return out
    from sklearn.preprocessing import LabelBinarizer
    target = target.astype(int)
    anx_data, dep_data = list(target[:, 0]), list(target[:, 1])
    anx_str = _label_to_str(anx_data, 'anxiety')
    dep_str = _label_to_str(dep_data, 'depression')

    class_labels = []
    lb = LabelBinarizer()
    anx_matrix = lb.fit_transform(anx_str)
    for c in list(lb.classes_):
        class_labels.append(c)

    dep_matrix = lb.fit_transform(dep_str)
    for c in list(lb.classes_):
        class_labels.append(c)
    indicator_matrix = np.hstack((anx_matrix, dep_matrix))

    from skmultilearn.problem_transform import LabelPowerset
    lp = LabelPowerset()
    resampler = RandomOverSampler(sampling_strategy='not majority')
    target_transformed = lp.transform(indicator_matrix)

    data_resampled, target_resampled = resampler.fit_sample(data, target_transformed)
    target_resampled_binary = lp.inverse_transform(target_resampled)
    return data_resampled, target_resampled, class_labels, target_resampled_binary


@ignore_warnings(category=ConvergenceWarning)
def eeg_classify(eeg_data, target_data, target_type, model, outdir):
    feature_names = list(eeg_data)

    continuous_features = [f for f in feature_names if 'categorical' not in f]
    continuous_indices = [eeg_data.columns.get_loc(cont) for cont in continuous_features]

    categorical_features = [f for f in feature_names if 'categorical' in f]
    categorical_indices = [eeg_data.columns.get_loc(cat) for cat in categorical_features]

    n_splits = 10
    data_resampled, target_resampled, target_classes, _ = resample_multilabel(ml_data, target_data)
    print(target_classes)
    # # Oversample connectivity data, apply k-fold splitter
    # from sklearn.preprocessing import LabelBinarizer
    # target = target_data.astype(int)
    # anx_data, dep_data = list(target[:, 0]), list(target[:, 1])
    #
    # lb = LabelBinarizer()
    # anx_matrix = lb.fit_transform(anx_data)
    # dep_matrix = lb.fit_transform(dep_data)
    # indicator_matrix = np.hstack((anx_matrix, dep_matrix))
    #
    # from skmultilearn.problem_transform import LabelPowerset
    # lp = LabelPowerset()
    # resampler = RandomOverSampler(sampling_strategy='not majority')
    # target_transformed = lp.transform(indicator_matrix)
    #
    # ml_data_resampled, target_resampled = resampler.fit_sample(ml_data, target_transformed)
    # target_resampled_binary = lp.inverse_transform(target_resampled)

    skf = model_selection.StratifiedKFold(n_splits=n_splits)
    x_res = data_resampled
    y_res = target_resampled
    skf.get_n_splits(x_res, y_res)

    fold_count = 0
    classifier_objects, classifier_coefficients = {}, {}

    # Create score dataframes
    rownames = ['Fold %02d' % (n + 1) for n in range(n_splits)]
    score_df = pd.DataFrame(index=rownames, columns=['Balanced accuracy', 'Chance accuracy'])
    f1_df = pd.DataFrame(index=rownames, columns=target_classes)

    for train_idx, test_idx in skf.split(x_res, y_res):
        foldname = rownames[fold_count]
        fold_count += 1
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
            # predicted = svmc(x_train_fs, y_train, x_test_fs, target_classes, cleaned_features)
            predicted, coef_df = svmc(x_train_fs, y_train, x_test_fs, target_classes, cleaned_features)
            classifier_coefficients[foldname] = coef_df
        elif model is 'extra_trees':
            predicted, feature_importances = extra_trees(x_train_fs, y_train, x_test_fs, cleaned_features)
            classifier_coefficients[foldname] = feature_importances
        elif model is 'knn':
            predicted = knn(x_train_fs, y_train, x_test_fs)

        balanced, chance, f1 = calc_scores(y_test, predicted)
        print(balanced)
        print(chance)
        print(f1)

        # Saving results
        score_df.loc[foldname]['Balanced accuracy'] = balanced
        score_df.loc[foldname]['Chance accuracy'] = chance
        f1_df.loc[foldname][:] = f1

    scores_dict = {'accuracy scores': score_df,
                   'f1 scores': f1_df
                   }

    target_outdir = join(outdir, target_type)
    if not isdir(target_outdir):
        mkdir(target_outdir)

    save_xls(scores_dict, join(target_outdir, 'performance.xlsx'))
    if bool(classifier_coefficients):
        save_xls(classifier_coefficients, join(target_outdir, 'coefficients.xlsx'))


if __name__ == "__main__":
    models = ['svm', 'extra_trees', 'knn']
    model = 'svm'
    output_dir = './../data/eeg_classification/%s/' % model
    if not isdir(output_dir):
        mkdir(output_dir)

    print('%s: Loading data' % pu.ctime())
    behavior_data, conn_data = pu.load_data_full_subjects()
    conn_data.astype(float)

    categorical_variables = ['smoking', 'deanxit_antidepressants', 'rivotril_antianxiety', 'sex']
    categorical_data = behavior_data[categorical_variables]
    dummy_coded_categorical = pu.dummy_code_binary(categorical_data)
    covariate_data = pd.concat([behavior_data['age'], dummy_coded_categorical], axis=1)

    ml_data = pd.concat([conn_data, covariate_data], axis=1)

    # 0-7 (normal); 8-10 (borderline); 11-21 (abnormal)
    hads_thresholds = [8, 11, 21]
    print('%s: Running classification on HADS' % pu.ctime())
    anx = behavior_data['anxiety_score'].values.astype(float)
    anx_binned = np.digitize(anx, bins=hads_thresholds, right=True)  # right=True: bin < x <= bin if ascending

    dep = behavior_data['depression_score'].values.astype(float)
    dep_binned = np.digitize(dep, bins=hads_thresholds, right=True)

    multilabel_target = np.vstack((anx_binned, dep_binned)).T

    # test_labelpowerset(ml_data, multilabel_target)
    eeg_classify(ml_data, target_data=multilabel_target, target_type='multilabel_hads', model=model, outdir=output_dir)

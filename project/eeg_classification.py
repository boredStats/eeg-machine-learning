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
    coef_df = pd.DataFrame(clf.coef_, index=idx_label, columns=cleaned_features)
    return predicted, coef_df


def extra_trees(x_train, y_train, x_test, cleaned_features):
    clf = ensemble.ExtraTreesClassifier()
    clf.fit(x_train, y_train)
    predicted = clf.predict(x_test)
    feature_df = pd.DataFrame(clf.feature_importances_, index=['feature_importances'], columns=cleaned_features)
    return predicted, feature_df


def sgd(x_train, y_train, x_test, target_classes, cleaned_features):
    clf = linear_model.SGDClassifier(loss="hinge", penalty="l2", max_iter=1000)
    clf.fit(x_train, y_train)
    predicted = clf.predict(x_test)
    if len(target_classes) == 2:
        idx_label = ['support_vector_coefficients']
    else:
        idx_label = target_classes
    coef_df = pd.DataFrame(clf.coef_, index=idx_label, columns=cleaned_features)
    return predicted, coef_df


def knn(x_train, y_train, x_test):
    clf = neighbors.KNeighborsClassifier()
    clf.fit(x_train, y_train)
    predicted = clf.predict(x_test)
    return predicted


@ignore_warnings(category=ConvergenceWarning)
def eeg_classify(eeg_data, target_data, target_type, model, outdir):
    feature_names = list(eeg_data)
    
    continuous_features = [f for f in feature_names if 'categorical' not in f]
    continuous_indices = [eeg_data.columns.get_loc(cont) for cont in continuous_features]

    categorical_features = [f for f in feature_names if 'categorical' in f]
    categorical_indices = [eeg_data.columns.get_loc(cat) for cat in categorical_features]

    target_classes = ['%s %d' % (target_type, t) for t in np.unique(target_data)]

    # Create score dataframes, k-fold splitter
    n_splits = 10
    skf = model_selection.StratifiedKFold(n_splits=n_splits)

    rownames = ['Fold %02d' % (n+1) for n in range(n_splits)]
    score_df = pd.DataFrame(index=rownames, columns=['Balanced accuracy', 'Chance accuracy'])
    f1_df = pd.DataFrame(index=rownames, columns=target_classes)

    # Oversample connectivity data, apply k-fold splitter
    resampler = RandomOverSampler(sampling_strategy='not majority')
    x_res, y_res = resampler.fit_resample(eeg_data, target_data)
    skf.get_n_splits(x_res, y_res)

    fold_count = 0
    classifier_objects, classifier_coefficients = {}, {}
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
            predicted, coef_df = svmc(x_train_fs, y_train, x_test_fs, target_classes, cleaned_features)
            classifier_coefficients[foldname] = coef_df
        elif model is 'extra_trees':
            predicted, feature_importances = extra_trees(x_train_fs, y_train, x_test_fs, cleaned_features)
            classifier_coefficients[foldname] = feature_importances
        elif model is 'sgd':
            predicted, coef_df = sgd(x_train_fs, y_train, x_test_fs, target_classes, cleaned_features)
            classifier_coefficients[foldname] = coef_df
        elif model is 'knn':
            predicted = knn(x_train_fs, y_train, x_test_fs)

        balanced, chance, f1 = calc_scores(y_test, predicted)

        # Saving results
        score_df.loc[foldname]['Balanced accuracy'] = balanced
        score_df.loc[foldname]['Chance accuracy'] = chance
        f1_df.loc[foldname][:] = f1

    scores_dict = {'accuracy scores': score_df,
                   'f1 scores': f1_df}

    target_outdir = join(outdir, target_type)
    if not isdir(target_outdir):
        mkdir(target_outdir)

    save_xls(scores_dict, join(target_outdir, 'performance.xlsx'))
    if bool(classifier_coefficients):
        save_xls(classifier_coefficients, join(target_outdir, 'coefficients.xlsx'))


if __name__ == "__main__":
    models = ['svm', 'extra_trees', 'sgd', 'knn']
    model = 'extra_trees'
    output_dir = './../data/%s/' % model
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

    print('%s: Running classification on tinnitus side' % pu.ctime())
    target = behavior_data['tinnitus_side'].values.astype(float) * 2
    eeg_classify(eeg_data=ml_data, target_data=target, target_type='tinnitus_side', model=model, outdir=output_dir)

    print('%s: Running classification on tinnitus type' % pu.ctime())
    target = np.add(behavior_data['tinnitus_type'].values.astype(int), 1)
    eeg_classify(eeg_data=ml_data, target_data=target, target_type='tinnitus_type', model=model, outdir=output_dir)

    target = behavior_data['distress_TQ'].values
    print('%s: Running classification on TQ - high/low' % pu.ctime())
    high_low_thresholds = [0, 46, 84]
    binned_target = np.digitize(target, bins=high_low_thresholds, right=True)
    eeg_classify(eeg_data=ml_data, target_data=binned_target, target_type='TQ_high_low', model=model, outdir=output_dir)

    print('%s: Running classification on TQ - grade' % pu.ctime())
    grade_thresholds = [0, 30, 46, 59, 84]
    binned_target = np.digitize(target, bins=grade_thresholds, right=True)
    eeg_classify(eeg_data=ml_data, target_data=binned_target, target_type='TQ_grade', model=model, outdir=output_dir)

    print('%s: Finished' % pu.ctime())

from os.path import isdir, join
from os import mkdir
import numpy as np
import pandas as pd
import pickle as pkl
import proj_utils as pu
from imblearn.over_sampling import RandomOverSampler
from sklearn import ensemble, feature_selection, model_selection, preprocessing, svm, metrics


def save_xls(dict_df, path):
    # Save a dictionary of dataframes to an excel file, with each dataframe as a seperate page
    writer = pd.ExcelWriter(path)
    for key in list(dict_df):
        dict_df[key].to_excel(writer, '%s' % key)
    writer.save()


def load_data():
    # Remove EEG subjects that don't have behavior data
    behavior_df = pu.load_behavior_data()
    conn_df = pu.load_connectivity_data()
    filt_df = conn_df.filter(items=behavior_df.index, axis=0)  # Remove EEG subjects with missing rowvals in behavior_df
    return behavior_df, filt_df


def eeg_classify(eeg_data, target_data, target_type, outdir=None):
    feature_names = list(eeg_data)
    target_classes = ['%s %d' % (target_type, t) for t in np.unique(target_data)]
    # Create score dataframes, k-fold splitter
    n_splits = 2
    skf = model_selection.StratifiedKFold(n_splits=n_splits)

    rownames = ['Fold %02d' % (n+1) for n in range(n_splits)]
    score_df = pd.DataFrame(index=rownames, columns=['Balanced accuracy', 'Chance accuracy'])
    # f1_colnames = ['%s %d' % (target_type, label) for label in np.unique(target_data)]  # names of the target classes
    f1_df = pd.DataFrame(index=rownames, columns=target_classes)

    # Oversample connectivity data, apply k-fold splitter
    resampler = RandomOverSampler(sampling_strategy='not majority')
    x_res, y_res = resampler.fit_resample(eeg_data, target_data)
    skf.get_n_splits(x_res, y_res)

    fold_count = 0
    svm_classifiers, svm_coefficents = {}, {}
    for train_idx, test_idx in skf.split(x_res, y_res):
        foldname = rownames[fold_count]
        fold_count += 1

        # Stratified k-fold splitting
        x_train, x_test = x_res[train_idx], x_res[test_idx]
        y_train, y_test = y_res[train_idx], y_res[test_idx]

        # Standardization
        preproc = preprocessing.StandardScaler().fit(x_train)
        x_train_z = preproc.transform(x_train)
        x_test_z = preproc.transform(x_test)

        # Feature selection with extra trees
        clf = ensemble.ExtraTreesClassifier()
        model = feature_selection.SelectFromModel(clf, threshold="2*mean")

        # Transform train and test data with feature selection model
        x_train_fs = model.fit_transform(x_train_z, y_train)
        x_test_fs = model.transform(x_test_z)
        feature_indices = model.get_support(indices=True)
        cleaned_features = [feature_names[i] for i in feature_indices]

        # SVM training
        svm_classifier = svm.LinearSVC(fit_intercept=False)
        svm_classifier.fit(x_train_fs, y_train)

        # Scoring
        predicted = svm_classifier.predict(x_test_fs)
        balanced = metrics.balanced_accuracy_score(y_test, predicted)
        chance = metrics.balanced_accuracy_score(y_test, predicted, adjusted=True)
        f1 = metrics.f1_score(y_test, predicted, average=None)

        # Saving results
        score_df.loc[foldname]['Balanced accuracy'] = balanced
        score_df.loc[foldname]['Chance accuracy'] = chance
        f1_df.loc[foldname][:] = f1
        coef_df = pd.DataFrame(svm_classifier.coef_, index=target_classes, columns=cleaned_features)

        svm_classifiers[foldname] = svm_classifier
        svm_coefficents[foldname] = coef_df

    scores_dict = {'accuracy scores': score_df,
                   'f1 scores': f1_df}

    if outdir is not None:
        target_outdir = join(outdir, target_type)
        if not isdir(target_outdir):
            mkdir(target_outdir)

        save_xls(scores_dict, join(target_outdir, 'svm_performance.xlsx'))
        save_xls(svm_coefficents, join(target_outdir, 'svm_coefficients.xlsx'))

        with open(join(target_outdir, 'svm_classifiers.pkl'), 'wb') as file:
            pkl.dump(svm_classifiers, file)


def bin_continuous_targets(target_vector, thresholds):
    if isinstance(target_vector, pd.Series):
        target_vector = target_vector.values

    binned_vector = []
    for value in target_vector:
        for bin_class, thresh in enumerate(thresholds):
            if value > thresh:
                pass
            elif value <= thresh:
                binned_vector.append(bin_class)
                break

    return np.ndarray.flatten(np.asarray(binned_vector))


def dummy_code_binary(categorical_series):
    # Sex: 1M, -1F

    string_categorical_series = pd.DataFrame(index=categorical_series.index, columns=list(categorical_series))

    for colname in list(categorical_series):
        string_series = []
        for value in categorical_series[colname].values:
            if value == 1:
                if 'sex' in colname:
                    string_series.append('male')
                else:
                    string_series.append('yes')
            elif value == -1:
                if 'sex' in colname:
                    string_series.append('female')
                else:
                    string_series.append('no')
        string_categorical_series[colname] = string_series

    dummy_series = pd.get_dummies(string_categorical_series)
    old_names = list(dummy_series)
    return dummy_series.rename(columns=dict(zip(old_names, ['categorical_%s' % d for d in old_names])))


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    categorical_variables = ['smoking', 'deanxit_antidepressants', 'rivotril_antianxiety', 'sex']

    output_dir = './../data/eeg_classification/'
    if not isdir(output_dir):
        mkdir(output_dir)

    behavior_data, conn_data = load_data()
    conn_data.astype(float)

    categorical_data = behavior_data[categorical_variables]
    dummy_coded_categorical = dummy_code_binary(categorical_data)

    target = behavior_data['tinnitus_side'].values.astype(float) * 2
    eeg_classify(eeg_data=conn_data, target_data=target, target_type='tinnitus_side', outdir=output_dir)

    # target = np.add(behavior_data['tinnitus_type'].values.astype(int), 1)
    # eeg_classify(eeg_data=conn_data, target_data=target, target_type='tinnitus_type', outdir=output_dir)

    # logging.info('%s: Running classification on TQ' % pu.ctime())
    # target = behavior_data['distress_TQ'].values
    # binned_target = bin_continuous_targets(target, thresholds=[47, np.max(target)])
    # print(np.unique(binned_target))
    # eeg_classify(eeg_data=conn_data, target_data=binned_target, target_type='distress_TQ', outdir=output_dir)

    logging.info('%s: Finished' % pu.ctime())

import os
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


def eeg_regression(eeg_data, target_data, target_type, outdir=None):
    feature_names = list(eeg_data)

    # Create output objects, k-fold splitter
    n_splits = 10
    skf = model_selection.KFold(n_splits=n_splits)

    rownames = ['Fold %02d' % (n+1) for n in range(n_splits)]
    score_df = pd.DataFrame(index=rownames, columns=['Explained variance score', 'Mean squared error'])

    x_res, y_res = eeg_data.values, target_data
    skf.get_n_splits(x_res, y_res)

    fold_count = 0
    classifier_dict, coef_dict = {}, {}
    for train_idx, test_idx in skf.split(x_res):
        foldname = rownames[fold_count]
        fold_count += 1

        # K-fold splitting
        x_train, x_test = x_res[train_idx], x_res[test_idx]
        y_train, y_test = y_res[train_idx], y_res[test_idx]

        # Standardization
        preproc = preprocessing.StandardScaler().fit(x_train)
        x_train_z = preproc.transform(x_train)
        x_test_z = preproc.transform(x_test)

        # Feature selection with extra trees
        clf = ensemble.ExtraTreesRegressor()
        model = feature_selection.SelectFromModel(clf, threshold="2*mean")

        # Transform train and test data with feature selection model
        x_train_fs = model.fit_transform(x_train_z, y_train)
        x_test_fs = model.transform(x_test_z)
        feature_indices = model.get_support(indices=True)
        cleaned_features = [feature_names[i] for i in feature_indices]

        # SVM training
        svm_classifier = svm.SVR(kernel='linear')
        svm_classifier.fit(x_train_fs, y_train)

        # Scoring
        predicted = svm_classifier.predict(x_test_fs)
        score_df.loc[foldname]['Explained variance score'] = metrics.explained_variance_score(y_test, predicted)
        score_df.loc[foldname]['Mean squared error'] = metrics.mean_squared_error(y_test, predicted)

        # Saving pipeline results
        classifier_dict['svm_%s' % foldname] = svm_classifier
        coef_dict['svm_%s' % foldname] =  pd.DataFrame(svm_classifier.coef_, index=['coef'], columns=cleaned_features)

    if outdir is not None:
        save_xls(coef_dict, outdir+'%s_svm_performance.xlsx' % target_type)
        with open(outdir+'%s_svm_classifiers.pkl' % target_type, 'wb') as file:
            pkl.dump(classifier_dict, file)


if __name__ == "__main__":
    output_dir = './../data/eeg_regression/'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    behavior_data, conn_data = load_data()
    conn_data.astype(float)

    target = behavior_data['loudness_VAS'].values.astype(float)
    eeg_regression(eeg_data=conn_data, target_data=target, target_type='loudness_VAS', outdir=output_dir)

    target = behavior_data['distress_TQ']
    eeg_regression(eeg_data=conn_data, target_data=target, target_type='distress_TQ', outdir=output_dir)

    target = behavior_data['distress_VAS']
    eeg_regression(eeg_data=conn_data, target_data=target, target_type='distress_VAS', outdir=output_dir)

    target = behavior_data['anxiety_score']
    eeg_regression(eeg_data=conn_data, target_data=target, target_type='anxiety_score', outdir=output_dir)

    target = behavior_data['depression_score']
    eeg_regression(eeg_data=conn_data, target_data=target, target_type='depression_score', outdir=output_dir)

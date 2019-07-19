import os
import numpy as np
import pandas as pd
import pickle as pkl
import proj_utils as pu
from sklearn import ensemble, feature_selection, model_selection, preprocessing, svm, metrics


def load_data():
    # Remove EEG subjects that don't have behavior data
    behavior_df = pu.load_behavior_data()
    conn_df = pu.load_connectivity_data()
    filt_df = conn_df.filter(items=behavior_df.index, axis=0)  # Remove EEG subjects with missing rowvals in behavior_df
    return behavior_df, filt_df


def eeg_regression(eeg_data, target_data, target_type, outdir=None):
    feature_names = list(eeg_data)

    # Create output objects
    n_splits = 10
    foldnames = ['Fold %02d' % (n+1) for n in range(n_splits)]
    score_df = pd.DataFrame(index=foldnames, columns=['ExplainedVar', 'MaxErr', 'MAE', 'MSE', 'r2'])
    coef_df = pd.DataFrame()
    classifier_dict = {}

    # Split data
    random_state = 13  # for reproducibility
    skf = model_selection.KFold(n_splits=n_splits, random_state=random_state)
    skf.get_n_splits(eeg_data.values, target_data)

    fold_count = 0
    for train_idx, test_idx in skf.split(eeg_data.values):
        foldname = foldnames[fold_count]
        fold_count += 1

        # K-fold splitting
        x_train, x_test = eeg_data.values[train_idx], eeg_data.values[test_idx]
        y_train, y_test = target_data[train_idx], target_data[test_idx]

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
        score_df.loc[foldname]['ExplainedVar'] = metrics.explained_variance_score(y_test, predicted)
        score_df.loc[foldname]['MaxErr'] = metrics.max_error(y_test, predicted)
        score_df.loc[foldname]['MAE'] = metrics.mean_absolute_error(y_test, predicted)
        score_df.loc[foldname]['MSE'] = metrics.mean_squared_error(y_test, predicted)
        score_df.loc[foldname]['r2'] = metrics.r2_score(y_test, predicted)

        # Saving pipeline results
        classifier_dict['svm_%s' % foldname] = svm_classifier
        fold_df = pd.DataFrame()
        fold_df['%s features' % foldname] = cleaned_features
        fold_df['%s coef' % foldname] = np.ndarray.flatten(svm_classifier.coef_)
        fold_df['_'] = ['' for cf in cleaned_features]
        pd.concat([coef_df, fold_df], ignore_index=True, axis=1)

    if outdir is not None:
        score_df.to_excel('%s_performance_measures.xlsx' % target_type)
        coef_df.to_excel('%s_feature_coefficients.xlsx' % target_type)

        with open(outdir+'%s_linear_classifiers.pkl' % target_type, 'wb') as file:
            pkl.dump(classifier_dict, file)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    output_dir = './../data/eeg_regression/'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    behavior_data, conn_data = load_data()
    conn_data.astype(float)

    targets = ['loudness_VAS', 'distress_TQ', 'distress_VAS', 'anxiety_score', 'depression_score']
    for target in targets:
        print(target)
        target_vect = behavior_data[target].values.astype(float)
        logging.info('%s Running regression on %s' % (pu.ctime(), target))
        eeg_regression(eeg_data=conn_data, target_data=target_vect, target_type=target, outdir=output_dir)
import os
import numpy as np
import pandas as pd
import pickle as pkl
import proj_utils as pu
import tensorflow as tf
from tensorflow import keras
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


def feature_selection_pipeline(conn_data, target):
    # --- sklearn pipeline --- #
    x_train, x_test, y_train, y_test = model_selection.train_test_split(conn_data, target, test_size=0.2)

    # Whiten data
    preproc = preprocessing.StandardScaler().fit(x_train)
    x_train_z = preproc.fit_transform(x_train)
    x_test_z = preproc.transform(x_test)

    # Feature selection with extra trees
    clf = ensemble.ExtraTreesClassifier(n_estimators=100)
    model = feature_selection.SelectFromModel(clf, threshold="2*mean")

    # Transform train and test data with feature selection model
    x_train_fs = model.fit_transform(x_train_z, y_train)
    x_test_fs = model.transform(x_test_z)

    return x_train_fs, x_test_fs, y_train, y_test


def eeg_classify(eeg_data, target_data, target_type, outdir=None):
    feature_names = list(eeg_data)
    target_classes = ['%s %s' % (target_type, t) for t in np.unique(target_data)]
    # Create score dataframes, k-fold splitter
    n_splits = 10
    skf = model_selection.StratifiedKFold(n_splits=n_splits)

    rownames = ['Fold %02d' % (n+1) for n in range(n_splits)]
    score_df = pd.DataFrame(index=rownames, columns=['Balanced accuracy', 'Chance accuracy'])
    f1_colnames = ['%s %d' % (target_type, label) for label in np.unique(target_data)]  # names of the target classes
    f1_df = pd.DataFrame(index=rownames, columns=f1_colnames)

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
        save_xls(scores_dict, outdir+'%s_svm_performance.xlsx' % target_type)
        save_xls(svm_coefficents, outdir+'%s_svm_coefficients.xlsx' % target_type)

        with open(outdir+'%s_svm_classifiers.pkl' % target_type, 'wb') as file:
            pkl.dump(svm_classifiers, file)


def deep_learning():
    behavior_data, conn_data = load_data()
    conn_data = conn_data.values.astype(float)
    target = np.add(behavior_data['tinnitus_type'].values.astype(int), 1)
    # target = np.add((behavior_data['tinnitus_side'].values.astype(float) * 2), 2)

    resampler = RandomOverSampler(sampling_strategy='not majority')
    x_res, y_res = resampler.fit_resample(conn_data, target)
    x_train_fs, x_test_fs, y_train, y_test = feature_selection_pipeline(x_res, y_res)

    # --- Deep learning --- #
    n_labels = len(pd.unique(target))
    print('N classes:', n_labels)
    model = keras.Sequential([
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(n_labels, activation=tf.nn.softmax)
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(x_train_fs, y_train, epochs=5)

    test_loss, test_acc = model.evaluate(x_test_fs, y_test)
    print('Test accuracy:', test_acc)


if __name__ == "__main__":
    output_dir = './../data/eeg_classification/'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    behavior_data, conn_data = load_data()
    conn_data.astype(float)

    target = behavior_data['tinnitus_side'].values.astype(float) * 2
    eeg_classify(eeg_data=conn_data, target_data=target, target_type='tinnitus_side', outdir=output_dir)

    target = np.add(behavior_data['tinnitus_type'].values.astype(int), 1)
    eeg_classify(eeg_data=conn_data, target_data=target, target_type='tinnitus_type', outdir=output_dir)

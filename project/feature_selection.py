import os
import numpy as np
import pandas as pd
import pickle as pkl
import proj_utils as pu
import tensorflow as tf
from tensorflow import keras
from imblearn.over_sampling import RandomOverSampler
from sklearn import ensemble, feature_selection, model_selection, preprocessing, svm, metrics


def dummy_code_categorical(data):
    def _create_dummy_code(series):
        levels = pd.unique(series)
        variable_name = series.name
        column_names = ['%s_%d' % (variable_name, i) for i in range(len(levels))]
        new_dummy_df = pd.DataFrame(columns=column_names, index=series.index, dtype='float')
        for c, col in enumerate(column_names):
            dummy = np.zeros(len(series.index))
            level = levels[c]

            for subj in range(len(series.index)):
                value = series.iloc[subj]
                if value == level:
                    dummy[subj] = 1

            new_dummy_df[col] = dummy

        return new_dummy_df

    continuous_data = data.select_dtypes(include='float')
    categorical_data = data.select_dtypes(include='category')
    cat_list = [continuous_data]

    for cat_var in list(categorical_data):
        cat_data = categorical_data[cat_var]
        dummy_df = _create_dummy_code(cat_data)
        cat_list.append(dummy_df)

    return pd.concat(cat_list, axis=1)


def load_data():
    ml_targets = ['loudness_VAS', 'distress_TQ', 'distress_VAS']

    behavior_data = pu.load_behavior_data()
    covariates_df = dummy_code_categorical(behavior_data)
    covariates_df.drop(labels=ml_targets, axis=1, inplace=True)

    conn_data = pu.load_connectivity_data()
    conn_data_filt = conn_data.filter(items=covariates_df.index, axis=0)

    return behavior_data, conn_data_filt


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


def classify_with_svm_resampling_and_kfolds(target_type='side', outdir=None):
    behavior_data, conn_data = load_data()
    conn_data = conn_data.values.astype(float)

    if target_type is 'side':
        target = behavior_data['tinnitus_side'].values.astype(float) * 2
    elif target_type is 'type':
        target = np.add(behavior_data['tinnitus_type'].values.astype(int), 1)

    # Create score dataframes, k-fold splitter
    n_splits = 10
    skf = model_selection.StratifiedKFold(n_splits=n_splits)

    df_idx_names = ['Fold %02d' % (n+1) for n in range(n_splits)]
    score_df = pd.DataFrame(index=df_idx_names, columns=['Balanced accuracy', 'Chance accuracy'])
    f1_colnames = ['%s %d' % (target_type, label) for label in np.unique(target)]
    f1_df = pd.DataFrame(index=df_idx_names, columns=f1_colnames)

    # Oversample connectivity data, apply k-fold splitter
    resampler = RandomOverSampler(sampling_strategy='not majority')
    x_res, y_res = resampler.fit_resample(conn_data, target)
    skf.get_n_splits(x_res, y_res)

    fold_count = 0
    for train_idx, test_idx in skf.split(x_res, y_res):
        foldname = df_idx_names[fold_count]
        fold_count += 1

        if outdir is not None:
            fname = outdir + '%s_svm_fold%02d_classifier.pkl' % (target_type, fold_count)
            if os.path.exists(fname):
                continue

        # Stratified k-fold splitting
        x_train, x_test = x_res[train_idx], x_res[test_idx]
        y_train, y_test = y_res[train_idx], y_res[test_idx]

        # Standardization
        preproc = preprocessing.StandardScaler().fit(x_train)
        x_train_z = preproc.transform(x_train)
        x_test_z = preproc.transform(x_test)

        # Feature selection with extra trees
        clf = ensemble.ExtraTreesClassifier(n_estimators=100)
        model = feature_selection.SelectFromModel(clf, threshold="2*mean")

        # Transform train and test data with feature selection model
        x_train_fs = model.fit_transform(x_train_z, y_train)
        x_test_fs = model.transform(x_test_z)

        # SVM classification
        svm_classifier = svm.LinearSVC(fit_intercept=False)
        svm_classifier.fit(x_train_fs, y_train)

        # Scoring
        predicted = svm_classifier.predict(x_test_fs)
        balanced = metrics.balanced_accuracy_score(y_test, predicted)
        chance = metrics.balanced_accuracy_score(y_test, predicted, adjusted=True)
        f1 = metrics.f1_score(y_test, predicted, average=None)

        score_df.loc[foldname]['Balanced accuracy'] = balanced
        score_df.loc[foldname]['Chance accuracy'] = chance
        f1_df.loc[foldname][:] = f1

        if outdir is not None:
            with open(fname, 'wb') as file:
                pkl.dump(svm_classifier, file)

    res_dict = {'accuracy scores': score_df,
                'f1 scores': f1_df}

    if outdir is not None:
        with open(outdir+'%s_svm_performance.pkl' % target_type, 'wb') as file:
            pkl.dump(res_dict, file)


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
    output_dir = './../data/feature_selection/'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    classify_with_svm_resampling_and_kfolds(outdir=output_dir)
    # deep_learning()
import os
import math
import numpy as np
import pandas as pd
import proj_utils as pu
from sklearn import ensemble, feature_selection, model_selection, preprocessing, pipeline, svm


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


def pipeline_test():
    behavior_data, conn_data = load_data()

    target = behavior_data['tinnitus_type']
    # target = behavior_data['tinnitus_side'].values.astype(float) * 2

    x_train, x_test, y_train, y_test = model_selection.train_test_split(conn_data, target, test_size=0.2)

    preproc = preprocessing.StandardScaler().fit(x_train)
    x_trainz = preproc.transform(x_train)
    x_testz = preproc.transform(x_test)

    m = math.floor(conn_data.shape[1] * 0.1)  # max_number of features to extract

    # ---Pipeline objects--- #
    variance_filter = feature_selection.VarianceThreshold(threshold=(.8 * (1 - .8)))
    tree_classifier = ensemble.ExtraTreesClassifier(n_estimators=100)
    feature_select = feature_selection.SelectFromModel(tree_classifier, prefit=False)  # , max_features=m)
    svm_classifier = svm.LinearSVC(multi_class='crammer_singer')

    steps = [
        ('whiten data', preprocessing.StandardScaler()),
        ('variance cleaning', variance_filter),
        ('extra trees cleaning', feature_select),
        ('classify step', svm_classifier),
    ]

    pipe = pipeline.Pipeline(steps=steps)
    scores = model_selection.cross_val_score(pipe, conn_data, target, cv=3)
    print(scores.mean())
    # pipe.fit(x_trainz, y_train)
    #
    # prediction = pipe.predict(x_testz)
    # print(prediction)
    # score = pipe.score(x_testz, y_test)
    # print(score)


def feature_selection_test():
    behavior_data, conn_data = load_data()
    # target = behavior_data['tinnitus_type']
    target = behavior_data['tinnitus_side'].values.astype(float) * 2

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        conn_data, target, test_size=0.2)

    preproc = preprocessing.StandardScaler().fit(X_train)
    X_train_z = preproc.transform(X_train)
    X_test_z = preproc.transform(X_test)

    sel = feature_selection.VarianceThreshold(threshold=(.8 * (1 - .8)))
    var_cleaned = sel.fit_transform(X_train_z)
    print(var_cleaned.shape)

    clf = ensemble.ExtraTreesClassifier(n_estimators=100, criterion='entropy')
    clf = clf.fit(var_cleaned, y_train)
    model = feature_selection.SelectFromModel(clf, prefit=True, threshold="2*mean")
    clf_cleaned = model.transform(var_cleaned)
    print(clf_cleaned.shape)


if __name__ == "__main__":
    output_dir = './../data/feature_selection/'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # feature_selection_test()
    pipeline_test()

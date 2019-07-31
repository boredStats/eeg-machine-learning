import numpy as np
import pandas as pd
import proj_utils as pu
from sklearn import metrics, linear_model, preprocessing, feature_selection, ensemble


def lars():
    behavior_data, conn_data = pu.load_data_full_subjects()
    conn_data.astype(float)

    categorical_variables = ['smoking', 'deanxit_antidepressants', 'rivotril_antianxiety', 'sex']
    categorical_data = behavior_data[categorical_variables]
    dummy_coded_categorical = pu.dummy_code_binary(categorical_data)
    covariate_data = pd.concat([behavior_data['age'], dummy_coded_categorical], axis=1)

    ml_data = pd.concat([conn_data, covariate_data], axis=1)
    target = behavior_data['distress_TQ'].values.astype(float)

    feature_names = list(ml_data)
    continuous_features = [f for f in feature_names if 'categorical' not in f]
    continuous_indices = [ml_data.columns.get_loc(cont) for cont in continuous_features]

    categorical_features = [f for f in feature_names if 'categorical' in f]
    categorical_indices = [ml_data.columns.get_loc(cat) for cat in categorical_features]

    ml_continuous = ml_data.values[:, continuous_indices]
    ml_categorical = ml_data.values[:, categorical_indices]

    # Standardization for continuous data
    preproc = preprocessing.StandardScaler().fit(ml_continuous)
    ml_z = preproc.transform(ml_continuous)

    # Variance threshold for categorical data
    varthresh = feature_selection.VarianceThreshold(threshold=0).fit(ml_categorical)
    ml_v = varthresh.transform(ml_categorical)

    ml_preprocessed = np.hstack((ml_z, ml_v))

    # Feature selection with extra trees
    clf = ensemble.ExtraTreesRegressor()
    model = feature_selection.SelectFromModel(clf, threshold="2*mean")
    # Transform train and test data with feature selection model
    ml_cleaned = model.fit_transform(ml_preprocessed, target)
    feature_indices = model.get_support(indices=True)
    cleaned_features = [feature_names[i] for i in feature_indices]

    lars_classifier = linear_model.LarsCV(cv=3, normalize=False, fit_intercept=False)

    lars_classifier.fit(ml_cleaned, target)
    predicted = lars_classifier.predict(ml_cleaned)

    r2 = lars_classifier.score(ml_cleaned, target)

    exp_var = metrics.explained_variance_score(target, predicted)
    max_err = metrics.max_error(target, predicted)
    mae = metrics.mean_absolute_error(target, predicted)
    mse = metrics.mean_squared_error(target, predicted)
    print(r2)


lars()

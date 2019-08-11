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


def svmc(x_train, y_train, x_test, cleaned_features):
    clf = svm.LinearSVC(fit_intercept=False)
    clf.fit(x_train, y_train)
    target_classes = clf.classes_
    target_classes = [str(c) for c in target_classes]

    predicted = clf.predict(x_test)

    if len(target_classes) == 2:
        idx_label = ['support_vector_coefficients']
    else:
        idx_label = target_classes
    coef_df = pd.DataFrame(clf.coef_, index=idx_label, columns=cleaned_features)
    return predicted, coef_df, clf


def extra_trees(x_train, y_train, x_test, cleaned_features):
    clf = ensemble.ExtraTreesClassifier(random_state=13)
    clf.fit(x_train, y_train)
    predicted = clf.predict(x_test)
    feature_df = pd.DataFrame(clf.feature_importances_, columns=['feature_importances'], index=cleaned_features)
    return predicted, feature_df, clf


def sgd(x_train, y_train, x_test, cleaned_features):
    clf = linear_model.SGDClassifier(loss="hinge", penalty="l2", max_iter=1000)
    clf.fit(x_train, y_train)
    target_classes = clf.classes_
    target_classes = [str(c) for c in target_classes]

    predicted = clf.predict(x_test)
    if len(target_classes) == 2:
        idx_label = ['support_vector_coefficients']
    else:
        idx_label = target_classes
    coef_df = pd.DataFrame(clf.coef_, index=idx_label, columns=cleaned_features)
    return predicted, coef_df, clf


def knn(x_train, y_train, x_test):
    clf = neighbors.KNeighborsClassifier()
    clf.fit(x_train, y_train)
    predicted = clf.predict(x_test)
    return predicted, clf


@ignore_warnings(category=ConvergenceWarning)
def eeg_classify(eeg_data, target_data, target_type, model, outdir):
    feature_names = list(eeg_data)
    
    continuous_features = [f for f in feature_names if 'categorical' not in f]
    continuous_indices = [eeg_data.columns.get_loc(cont) for cont in continuous_features]

    categorical_features = [f for f in feature_names if 'categorical' in f]
    categorical_indices = [eeg_data.columns.get_loc(cat) for cat in categorical_features]

    # Create score dataframes, k-fold splitter
    n_splits = 10
    skf = model_selection.StratifiedKFold(n_splits=n_splits, random_state=13)

    # rownames = ['Fold %02d' % (n+1) for n in range(n_splits)]
    # score_df = pd.DataFrame(index=rownames, columns=['Balanced accuracy', 'Chance accuracy'])
    balanced_acc, chance_acc, f1_scores = [], [], []

    # Oversample connectivity data, apply k-fold splitter
    resampler = RandomOverSampler(sampling_strategy='not majority', random_state=13)
    x_res, y_res = resampler.fit_resample(eeg_data, target_data)
    skf.get_n_splits(x_res, y_res)

    fold_count = 0
    classifier_objects, classifier_coefficients = {}, {}
    balanced_acc, chance_acc, f1_scores = [], [], []
    for train_idx, test_idx in skf.split(x_res, y_res):
        foldname = 'Fold %02d' % fold_count + 1
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
        extra_tree_fs = ensemble.ExtraTreesClassifier()
        feature_model = feature_selection.SelectFromModel(extra_tree_fs, threshold="2*mean")

        # Transform train and test data with feature selection model
        x_train_fs = feature_model.fit_transform(x_train_data, y_train)
        x_test_fs = feature_model.transform(x_test_data)
        feature_indices = feature_model.get_support(indices=True)
        cleaned_features = [feature_names[i] for i in feature_indices]

        if model is 'svm':
            predicted, coef_df, clf = svmc(x_train_fs, y_train, x_test_fs, cleaned_features)
            classifier_coefficients[foldname] = coef_df
            target_classes = clf.classes_
        elif model is 'extra_trees':
            predicted, feature_importances, clf = extra_trees(x_train_fs, y_train, x_test_fs, cleaned_features)
            classifier_coefficients[foldname] = feature_importances
            target_classes = clf.classes_
        elif model is 'sgd':
            predicted, coef_df, clf = sgd(x_train_fs, y_train, x_test_fs, cleaned_features)
            classifier_coefficients[foldname] = coef_df
            target_classes = clf.classes_
        elif model is 'knn':
            predicted, clf = knn(x_train_fs, y_train, x_test_fs)
            target_classes = clf.classes_

        balanced, chance, f1 = calc_scores(y_test, predicted)

        # Saving results
        # score_df.loc[foldname]['Balanced accuracy'] = balanced
        # score_df.loc[foldname]['Chance accuracy'] = chance
        balanced_acc.append(balanced)
        chance_acc.append(chance)
        f1_scores.append(f1)  # f1_df.loc[foldname][:] = f1

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

    f1_df = pd.DataFrame(data=np.asarray(f1_data), index=rownames, columns=target_classes)
    scores_dict = {'accuracy scores': score_df,
                   'f1 scores': f1_df}

    target_outdir = join(outdir, target_type)
    if not isdir(target_outdir):
        mkdir(target_outdir)

    save_xls(scores_dict, join(target_outdir, 'performance.xlsx'))
    if bool(classifier_coefficients):
        save_xls(classifier_coefficients, join(target_outdir, 'coefficients.xlsx'))

    with open(join(target_outdir, 'classifier_object.pkl'), 'wb') as file:
        pkl.dump(clf, file)


def convert_tinnitus_data_to_str(tinnitus_data, data_type):
    str_data = []
    if data_type is 'tinnitus_side':
        for t in tinnitus_data:
            if t == -2:
                str_data.append('Left')
            elif t == -1:
                str_data.append('Left_over_Right')
            elif t == 0:
                str_data.append('Bilateral')
            elif t == 1:
                str_data.append('R_over_Left')
            elif t == 2:
                str_data.append('Right')
        if len(str_data) != len(tinnitus_data):
            raise ValueError('Side data not parsed correctly')
    elif data_type is 'tinnitus_type':
        for t in tinnitus_data:
            if t == 0:
                str_data.append('PT')
            elif t == 1:
                str_data.append('PT_and_NBN')
            elif t == 2:
                str_data.append('NBN')
        if len(str_data) != len(tinnitus_data): raise ValueError('Type data not parsed correctly')
    return str_data


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

    print('%s: Running classification on tinnitus side' % pu.ctime())  # Left, Left>Right, Bil/Holo, Right>Left, Right
    target = convert_tinnitus_data_to_str(behavior_data['tinnitus_side'].values.astype(float) * 2, 'tinnitus_side')
    eeg_classify(eeg_data=ml_data, target_data=target, target_type='tinnitus_side', model=model, outdir=output_dir)

    # print('%s: Running classification on tinnitus type' % pu.ctime())  # PureTone, PureTone+NBN, NBN
    # target = convert_tinnitus_data_to_str(np.add(behavior_data['tinnitus_type'].values.astype(int), 1), 'tinnitus_type')
    # eeg_classify(eeg_data=ml_data, target_data=target, target_type='tinnitus_type', model=model, outdir=output_dir)
    #
    # print('%s: Running classification on TQ - high/low' % pu.ctime())
    # target = behavior_data['distress_TQ'].values
    # high_low_thresholds = [0, 46, 84]
    # binned_target = np.digitize(target, bins=high_low_thresholds, right=True)
    # target = ['TQ_high' if t > 1 else 'TQ_low' for t in binned_target]
    # eeg_classify(eeg_data=ml_data, target_data=target, target_type='TQ_high_low', model=model, outdir=output_dir)
    #
    # print('%s: Running classification on TQ - grade' % pu.ctime())
    # target = behavior_data['distress_TQ'].values
    # grade_thresholds = [0, 30, 46, 59, 84]
    # binned_target = np.digitize(target, bins=grade_thresholds, right=True)
    # target = ['Grade_%d' % t for t in binned_target]
    # eeg_classify(eeg_data=ml_data, target_data=target, target_type='TQ_grade', model=model, outdir=output_dir)

    print('%s: Finished' % pu.ctime())

# -*- coding: utf-8 -*-
"""Classifier tools for this project."""

import os
import pandas as pd
import numpy as np

import imblearn
from sklearn import model_selection, ensemble, svm, neighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix

class EEG_Classifier:

    def __init__(
            self,
            resample_type=None,
            classifier_type='ExtraTrees',
            kfold_type='stratified',
            n_splits=10, seed=None):
        self.resample_type = resample_type
        self.classifier_type = classifier_type
        self.kfold_type = kfold_type
        self.n_splits = n_splits
        self.seed = seed
        return self

    @staticmethod
    def _splitter(type='stratified', n_splits=10, random_state=None):
        if type == 'stratified':
            splitter = model_selection.StratifiedKFold(
                n_splits=n_splits, random_state=random_state)
        elif type == 'random':
            splitter = model_selection.KFold(
                n_splits=n_splits, random_state=random_state)
        else:
            raise ValueError(
                "type={type} not recognized. Options are 'stratified'"
                " or 'random'".format(type=repr(type)))
        return splitter

    @staticmethod
    def _calc_scores(y_test, predicted):
        balanced = balanced_accuracy_score(y_test, predicted)
        chance = balanced_accuracy_score(y_test, predicted, adjusted=True)
        f1 = f1_score(y_test, predicted, average=None)
        return balanced, chance, f1

    def feature_selector(
            self, x_train, x_test, y_train,
            continuous_indices=None,
            categorical_indices=None,
            thresh="2*mean"):
        if continuous_indices is None:
            preproc = StandardScaler().fit(x_train)
            x_train_data = preproc.transform(x_train)
            x_test_data = preproc.transform(x_test)

        else:
            x_train_cont = x_train[:, continuous_indices]
            x_test_cont = x_test[:, continuous_indices]
            x_train_cat = x_train[:, categorical_indices]
            x_test_cat = x_test[:, categorical_indices]

            # Standardization for continuous data
            preproc = preprocessing.StandardScaler().fit(x_train_cont)
            x_train_z = preproc.transform(x_train_cont)
            x_test_z = preproc.transform(x_test_cont)

            # Variance threshold for categorical data
            varthresh = VarianceThreshold(threshold=0).fit(x_train_cat)
            x_train_v = varthresh.transform(x_train_cat)
            x_test_v = varthresh.transform(x_test_cat)

            x_train_data = np.hstack((x_train_z, x_train_v))
            x_test_data = np.hstack((x_test_z, x_test_v))

        clf = ensemble.ExtraTreesClassifier(random_state=self.seed)
        fs_model = SelectFromModel(clf, threshold=thresh)

        self.x_train_fs = fs_model.fit_transform(x_train_data, y_train)
        self.x_test_fs = fs_model.transform(x_test_data)
        self.feature_indices = fs_model.get_support(indices=True)

        return self

    def classify(self, eeg_data, target_data, gridsearch=None):
        feature_names, cont_indices, cat_indices = check_eeg_data(eeg_data)

        resampler = _create_resampler(
            type=self.resample_type,
            random_state=self.seed)
        x_res, y_res = resample.fit_resample(eeg_data, target_data)

        kfolder = _splitter(
            type=self.kfold_type,
            n_splits=self.n_splits
            random_state=self.seed)

        clf = _create_classifier(
            type=self.classifier_type,
            random_state=self.seed)

        if gridsearch is not None:
            searcher = model_selection.GridSearchCV(
                clf, gridsearch, scoring='r2', cv=self.n_splits)
            kwargs = searcher.best_params_
            clf.set_params(**kwargs)
            grid_df = searcher.cv_results_
        else:
            grid_df = None

        features_by_fold, confusion_matrices,  = {}
        balanced_acc, chance_acc, f1_scores = [], [], []
        for t, train_idx, test_idx in enumerate(kfolder.split(x_res, y_res)):
            x_train, x_test = x_res[train_idx], x_res[test_idx]
            y_train, y_test = y_res[train_idx], y_res[test_idx]

            self.feature_selector(
                x_train, x_test, y_train,
                continuous_indices=cont_indices,
                categorical_indices=cat_indices)

            cleaned_features = [feature_names[i] for i in self.feature_indices]
            clf.fit(self.x_train_fs, y_train)
            predicted = clf.predict(self.x_test_fs)

            try:
                importances = np.ndarray.flatten(clf.feature_importances_)
                feature_df = pd.DataFrame(columns=cleaned_features)
                feature_df.loc['Feature Importances'] = importances
            except AttributeError:
                pass

            try:
                classes = [str(c) for c in clf.classes_]
                if len(classes) == 2:
                    idx_label = ['Coefficients']
                else:
                    idx_label = ['%s coefficients' % c for c in classes]
                coef = np.ndarray.flatten(clf.coef_)

                feature_df = pd.DataFrame(
                    coef, index=idx_label, columns=cleaned_features)
            except AttributeError:
                pass

            if feature_df is not None:
                features_by_fold['Fold %03d' % (t+1)] = feature_df

            balanced, chance, f1 = self._calc_scores(y_test, predicted)
            balanced_acc.append(balanced)
            chance_acc.append(chance)
            f1_scores.append(f1)

            # Calculating fold confusion matrix
            cm = confusion_matrix(y_test, predicted)
            confusion_matrices['Fold %03d' % (t+1)] = pd.DataFrame(
                cm, index=clf.classes_, columns=clf.classes_)

        f1_df, score_df = _save_scores(
            f1_scores=f1_scores,
            balanced_scores=balanced_acc,
            chance_scores=chance_acc,
            class_labels=clf.classes_)
        scores_dict = {
            'accuracy scores': score_df,
            'f1 scores': f1_df}

        return scores_dict, confusion_matrices, feature_df, grid_df


def check_eeg_data(eeg_df):
    if type(eeg_df) != pd.DataFrame:
        pass

    feature_names = list(eeg_df)
    if "categorical_sex_male" in feature_names:
        cont_feats = [f for f in feature_names if 'categorical' not in f]
        cont_indices = [eeg_df.columns.get_loc(f) for f in cont_feats]

        cat_feats = [f for f in feature_names if 'categorical' in f]
        cat_indices = [eeg_df.columns.get_loc(f) for f in cat_feats]
    else:
        cont_indices, cat_indices = None, None
    return feature_names, cont_indices, cat_indices


def _create_resampler(type=None, random_state=None):
    if type is None:
        class NoResample:
            def fit_resample(a, b):
                return a.values, np.asarray(b)
        resampler = NoResample()
    elif type == 'under':
        resampler = imblearn.under_sampling.RandomUnderSampler(
            sampling_strategy='not minority',
            random_state=random_state)
    elif type == 'over':
        resampler = imblearn.over_sampling.RandomOverSampler(
            sampling_strategy='not_majority',
            random_state=random_state)
    elif type == 'smote':
        resampler = imblearn.over_sampling.SMOTE(
            sampling_strategy='not_majority',
            random_state=random_state)

    return resampler


def _create_classifier(type='ExtraTrees', kwargs=None, random_state=None):
    if type == 'ExtraTrees':
        clf = ensemble.ExtraTreesClassifier(random_state=random_state)
    elif type == 'SVM':
        clf = svm.SVC()
    elif type == 'KNN':
        clf = neighbors.KNeighborsClassifier()

    if kwargs is not None:
        clf.set_params(**kwargs)
    return clf


def _save_scores(f1_scores, balanced_scores, chance_scores, class_labels):
    # Calculate average performance, save to nice dataframes
    n_folds = len(balanced_scores)
    f1_array = np.asarray(f1_scores)
    if n_folds != f1_array.shape[0]:
        raise ValueError("Number of folds does not match")

    rownames = ['Fold %02d' % (n+1) for n in range(n_folds)]
    rownames.append('Average')

    f1_class_averages = np.mean(f1_array, axis=0)
    f1_data = np.vstack((f1_array, f1_class_averages))
    f1_df = pd.DataFrame(f1_data, index=rownames, columns=class_labels)

    balanced_scores.append(np.mean(balanced_scores))
    chance_scores.append(np.mean(chance_scores))
    accuracy_data = np.asarray([balanced_scores, chance_scores]).T

    score_df = pd.DataFrame(
        data=accuracy_data,
        index=rownames,
        columns=['Balanced accuracy', 'Chance accuracy'])
    return f1_df, score_df


def _performance_testing():
    import proj_utils as pu
    print('%s: Loading data' % pu.ctime())
    behavior_data, conn_data = pu.load_data_full_subjects()
    ml_data_without_covariates = conn_data.astype(float)

    side_data = behavior_data['tinnitus_side'].values.astype(float)
    side_target = pu.convert_tin_to_str(side_data, 'tinnitus_side')

    print('%s: Testing performance' % pu.ctime())
    C = EEG_Classifier(n_splits=2, seed=13)
    scores_dict, confusion_matrices, feature_df, grid_df = C.classify(
        eeg_data=ml_data_without_covariates,
        target_data=side_target)
    print('%s: Finished performance testing' % pu.ctime())


if __name__ == "__main__":
    _performance_testing()

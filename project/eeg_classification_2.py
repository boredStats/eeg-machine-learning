from os.path import join, isdir
from os import mkdir
import proj_utils as pu
import pandas as pd
import numpy as np
from classification_tools import EEG_Classifier


seed = 13

output_dir = './../results/'
if not isdir(output_dir):
    mkdir(output_dir)


def load_data(behavior, covariates=True):
    behavior_data, conn_data = pu.load_data_full_subjects()

    if behavior == 'TQ_high_low':
        tq_data = behavior_data['distress_TQ'].values
        high_low_thresholds = [0, 46, 84]
        tq_hl = np.digitize(
            tq_data, bins=high_low_thresholds, right=True)
        target_as_str = ['TQ_High' if t > 1 else 'TQ_low' for t in tq_hl]
    elif behavior == 'TQ_Grade':
        tq_data = behavior_data['distress_TQ'].values
        grade_thresholds = [0, 30, 46, 59, 84]
        tq_grade = np.digitize(tq_data, bins=grade_thresholds, right=True)
        target_as_str = ['Grade %d' % t for t in tq_grade]
    else:
        target_as_float = behavior_data[behavior].values.astype(float)
        target_as_str = pu.convert_tin_to_str(target_as_float, behavior)
    target_data = pd.DataFrame(target_as_str, index=conn_data.index)

    if not covariates:
        ml_data = conn_data.astype(float)
    else:
        categorical_variables = ['smoking', 'deanxit_antidepressants', 'rivotril_antianxiety', 'sex']
        categorical_data = behavior_data[categorical_variables]
        dummy_coded_categorical = pu.dummy_code_binary(categorical_data)
        covariate_data = pd.concat([behavior_data['age'], dummy_coded_categorical], axis=1)

        ml_data = pd.concat([conn_data, covariate_data], axis=1)
    return ml_data, target_data


def save_output(
        output_dir,
        behavior,
        scores,
        confusion_matrices,
        features,
        grid_df=None,
        model=None,
        resamp_method=None,
        covariates=True):
    if covariates:
        cov_check = 'with_covariates'
    else:
        cov_check = 'without_covariates'

    folder_name = '%s %s %s %s' % (behavior, model, resamp_method, cov_check)
    res_dir = join(output_dir, folder_name)
    if not isdir(res_dir):
        mkdir(res_dir)

    pu.save_xls(scores, join(res_dir, 'performance.xlsx'))
    if features is not None:
        pu.save_xls(features, join(res_dir, 'coefficients.xlsx'))
    pu.save_xls(confusion_matrices, join(res_dir, 'confusion_matrices.xlsx'))

    normalized_cms = {}
    for fold in confusion_matrices:
        cm = confusion_matrices[fold]
        norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        normalized_cms[fold] = norm
    pu.save_xls(normalized_cms, join(res_dir, 'confusion_matrices_normalized.xlsx'))


def behavior_classification(behavior='tinnitus_side', covariates=True):
    ml_data, side_data = load_data(behavior, covariates=covariates)
    models = ['SVM', 'ExtraTrees', 'KNN']
    resample_methods = [None, 'under', 'over', 'smote']

    for model in models:
        for resamp in resample_methods:
            prog = '%s %s' % (model, resamp)
            print('%s: Running %s classification with %s' % (pu.ctime(), behavior, prog))
            EC = EEG_Classifier(
                    n_splits=10,
                    seed=seed,
                    classifier_type=model,
                    resample_type=resamp)
            scores, confusion_matrices, features, grid_df = EC.classify(
                eeg_data=ml_data,
                target_data=side_data)
            print('%s: Saving output for %s' % (pu.ctime(), prog))
            save_output(
                output_dir=output_dir,
                behavior=behavior,
                scores=scores,
                confusion_matrices=confusion_matrices,
                features=features,
                grid_df=grid_df,
                model=model,
                resamp_method=resamp,
                covariates=covariates)


if __name__ == "__main__":
    behaviors = [
        'tinnitus_side',
        'tinnitus_type',
        'TQ_high_low',
        'TQ_Grade'
    ]
    for b in behaviors:
        behavior_classification(behavior=b, covariates=True)
        behavior_classification(behavior=b, covariates=False)


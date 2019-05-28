
import numpy as np
import pandas as pd


# Load raw data
data_raw = np.load('data_raw_labeled.pkl')
print(data_raw.head())

print('Removing behavior variables from dataframe...')

def remove_behavior_variables(data_raw):
    #Removes TQ, VAS variables from dataframe in-place
    behavior_variables = ['distress_TQ', 'loudness_VAS10']

    data_raw.drop(columns=behavior_variables, inplace=True)
    return data_raw

remove_behavior_variables(data_raw)
print(data_raw.head())


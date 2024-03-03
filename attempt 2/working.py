import numpy as np
import pandas as pd
import os
import json
import matplotlib
import seaborn as sns
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB

print(f'Read data')
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
train_labels_df = pd.read_csv('train_labels.csv')
specs_df = pd.read_csv('specs.csv')
sample_submission_df = pd.read_csv('sample_submission.csv')
print(f"train shape: {train_df.shape}")
print(f"test shape: {test_df.shape}")
print(f"train labels shape: {train_labels_df.shape}")
print(f"specs shape: {specs_df.shape}")
print(f"sample submission shape: {sample_submission_df.shape}")

train_df = train_df.loc[(train_df.event_code == 4100) |(train_df.event_code == 4110)]

extracted_event_data = pd.io.json.json_normalize(train_df.event_data.apply(json.loads))
print(1)

specs_args_extracted = pd.DataFrame()
for i in range(0, specs_df.shape[0]):
    for arg_item in json.loads(specs_df.args[i]) :
        new_df = pd.DataFrame({'event_id': specs_df['event_id'][i],\
                               'info':specs_df['info'][i],\
                               'args_name': arg_item['name'],\
                               'args_type': arg_item['type'],\
                               'args_info': arg_item['info']}, index=[i])
        specs_args_extracted = specs_args_extracted.append(new_df)
        
print(2)


def extract_time_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    df['year'] = df['timestamp'].dt.year
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['weekofyear'] = df['timestamp'].dt.weekofyear
    df['dayofyear'] = df['timestamp'].dt.dayofyear
    df['quarter'] = df['timestamp'].dt.quarter
    df['is_month_start'] = df['timestamp'].dt.is_month_start
    return df

train_df = extract_time_features(train_df)
test_df = extract_time_features(test_df)

print(3)

numerical_columns = ['game_time', 'month', 'dayofweek', 'hour']
categorical_columns = ['type', 'world']

comp_train_df = pd.DataFrame({'installation_id': train_df['installation_id'].unique()})
comp_train_df.set_index('installation_id', inplace = True)
"""
def get_numeric_columns(df, column):
    df = df.groupby('installation_id').agg({f'{column}': ['mean', 'sum', 'min', 'max', 'std', 'skew']})
    df[column].fillna(df[column].mean(), inplace = True)
    df.columns = [f'{column}_mean', f'{column}_sum', f'{column}_min', f'{column}_max', f'{column}_std', f'{column}_skew']
    return df

print(4)
for i in numerical_columns:
    comp_train_df = comp_train_df.merge(get_numeric_columns(train_df, i), left_index = True, right_index = True)
print(5
      )
# get the mode of the title"""
labels_map = dict(train_labels_df.groupby('title')['accuracy_group'].agg(lambda x:x.value_counts().index[0]))
# merge target
labels = train_labels_df[['installation_id', 'title', 'accuracy_group']]
# replace title with the mode
labels['title'] = labels['title'].map(labels_map)
# join train with labels
comp_train_df = labels.merge(comp_train_df, on = 'installation_id', how = 'left')
print('We have {} training rows'.format(comp_train_df.shape[0]))

print(comp_train_df.head())
#print(labels.shape)
#print(test_df.shape)


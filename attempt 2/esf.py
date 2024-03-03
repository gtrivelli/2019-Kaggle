import numpy as np
import pandas as pd
import os
import json
import matplotlib
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics

print(f'Read data')
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
train_labels_df = pd.read_csv('train_labels.csv')
#specs_df = pd.read_csv('specs.csv')
#sample_submission_df = pd.read_csv('sample_submission.csv')

train_df = train_df.loc[(train_df.event_code == 4100) |(train_df.event_code == 4110)]
test_df = test_df.loc[(test_df.event_code == 4100) |(test_df.event_code == 4110)]

print(f"train shape: {train_df.shape}")
print(f"test shape: {test_df.shape}")
print(f"train labels shape: {train_labels_df.shape}")

labels = train_labels_df[['installation_id', 'title', 'accuracy_group']]

#print(f"specs shape: {specs_df.shape}")
#print(f"sample submission shape: {sample_submission_df.shape}")
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
#print(train_df.dtypes)
#print(test_df.dtypes)

print(train_df.head())
print(test_df.head())
le = LabelEncoder()
train_df['event_id'] = le.fit_transform(train_df['event_id'])
train_df['game_session'] = le.fit_transform(train_df['game_session'])
train_df['event_data'] = le.fit_transform(train_df['event_data'])
train_df['installation_id'] = le.fit_transform(train_df['installation_id'])
train_df['title'] = le.fit_transform(train_df['title'])
train_df['type'] = le.fit_transform(train_df['type'])
train_df['world'] = le.fit_transform(train_df['world'])
train_df['is_month_start'] = le.fit_transform(train_df['is_month_start'])
train_df['date'] = le.fit_transform(train_df['date'])

labels['installation_id'] = le.fit_transform(labels['installation_id'])
labels['title'] = le.fit_transform(labels['title'])
labels['accuracy_group'] = le.fit_transform(labels['accuracy_group'])

test_df['event_id'] = le.fit_transform(test_df['event_id'])
test_df['game_session'] = le.fit_transform(test_df['game_session'])
test_df['event_data'] = le.fit_transform(test_df['event_data'])
test_df['installation_id'] = le.fit_transform(test_df['installation_id'])
test_df['title'] = le.fit_transform(test_df['title'])
test_df['type'] = le.fit_transform(test_df['type'])
test_df['world'] = le.fit_transform(test_df['world'])
test_df['is_month_start'] = le.fit_transform(test_df['is_month_start'])
test_df['date'] = le.fit_transform(test_df['date'])

train_df = train_df.drop(columns=['timestamp'])
test_df = test_df.drop(columns=['timestamp'])

print(train_df.head())
print(test_df.head())
print(labels.head())

#pca = PCA(n_components=.99)
#pca_df = pca.fit_transform(train_df)
#print(pca_df.shape)

#feature_train, feature_test, label_train, label_test = train_test_split(train_df, labels, test_size=0.3)
gnb = GaussianNB()
gnb.fit(train_df, labels)
label_pred_gnb = gnb.predict(test_df)
#print("Accuracy, Naive Bayes:",metrics.accuracy_score(label_test, label_pred_gnb))
print(label_pred_gnb)
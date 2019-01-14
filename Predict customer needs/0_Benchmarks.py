import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import math
script_start_time = time.time()

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
%matplotlib inline
plt.rcParams["figure.figsize"] = 20,15
sns.set(rc={'figure.figsize':(20,15)})
plt.rcParams["figure.figsize"] = 15,10
sns.set(rc={'figure.figsize':(15,10)})
plt.rcParams["figure.figsize"] = 12,8
sns.set(rc={'figure.figsize':(12,8)})
plt.style.use('fivethirtyeight')

# Settings
project_path = os.getcwd()
data_path = os.path.join(project_path, 'DataFile')
raw_data_path = os.path.join(data_path, 'RawData')
processed_data_path = os.path.join(data_path, 'ProcessedData')
meta_feature_dir = os.path.join(data_path, 'MetaData')
submission_path = os.path.join(project_path, 'Submission')

scoringMethod = 'neg_mean_squared_error'
SEED = 2018

# Useful functions
def check_data(data):
    print('Shape: %s'%(str(data.shape)))
    print('NA: ')
    print(data.isnull().sum())
    print()
    print('Unique: ')
    print(data.nunique())
    print('-'*50)

def evaluate_result(y_actual, y_predicted):
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    rms = sqrt(mean_squared_error(y_actual, y_predicted))
    print(rms)
    return rms

def get_rmse(neg_mse):
    from math import sqrt
    return math.sqrt(abs(neg_mse))

if 'model_evaluations' not in globals(): model_evaluations = {}
def evaluate_model(model, model_name):
    from sklearn import model_selection
    cv_split = model_selection.KFold(n_splits = 5, random_state = SEED)
    # cv_split = model_selection.ShuffleSplit(n_splits = 5, test_size = .3, train_size = .6, random_state = SEED)
    results = model_selection.cross_val_score(model, X_train, y_train, cv=cv_split, scoring=scoringMethod)
    print("RMSE: %f (%f)" % (get_rmse(results.mean()), get_rmse(results.std())))
    model_evaluations[model_name] = get_rmse(results.mean())

def check_grid_result(grid_result):
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (get_rmse(mean), get_rmse(stdev), param))

def submit_prediction_with_model(model, submission_name):
    y_test_pred = model.predict(test[feature_columns])
    test['predictions'] = y_test_pred
    test[['predictions']].to_csv(os.path.join(submission_path, submission_name), index = False)
    test.drop(columns = ['predictions'])
    print("Prediction mean: %d"%(y_test_pred.mean()))
    return y_test_pred

def submit_prediction(y_test_pred, submission_name):
    test['predictions'] = y_test_pred
    test[['predictions']].to_csv(os.path.join(submission_path, submission_name), index = False)
    test.drop(columns = ['predictions'])
    print("Prediction mean: %d"%(y_test_pred.mean()))

# 1. Prepare data =============================================
train = pd.read_csv(os.path.join(raw_data_path, 'train.csv'), sep = ';', decimal=',')
test = pd.read_csv(os.path.join(raw_data_path, 'test.csv'))
target = 'MOD_VOLUME_CONSUMPTION'
global_mean = train[target].mean()

train_columns = train.columns.tolist()
test_columns = test.columns.tolist()
train_columns_feature = train.columns.tolist()
train_columns_feature.remove(target)
train_columns
test_columns
for i in range(len(test_columns)):
    train.rename(columns={train_columns_feature[i]: test_columns[i]}, inplace = True)
train.rename(columns={'TIMESTAMP - Year': 'Year', 'TIMESTAMP - Month': 'Month', 'Sum of Sales_CR': 'Sum_of_Sales_CR'}, inplace = True)
test.rename(columns={'TIMESTAMP - Year': 'Year', 'TIMESTAMP - Month': 'Month', 'Sum of Sales_CR': 'Sum_of_Sales_CR'}, inplace = True)

month_map = dict(zip(['December', 'November', 'October', 'September', 'August', 'July', 'June', 'May', 'April', 'March', 'February', 'January'], \
                    [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]))
train['Month'] = train['Month'].map(month_map)
test['Month'] = test['Month'].map(month_map)

test['has_NA'] = test.isnull().any(axis=1)
test.fillna(test['ZIPcode'].value_counts().reset_index().iloc[0]['index'], inplace = True)
test.drop(columns = ['has_NA'], inplace = True)

train['has_NA'] = train.isnull().any(axis=1)
train = train[pd.notna(train['ID'])]
train.fillna(test['ZIPcode'].value_counts().reset_index().iloc[0]['index'], inplace = True)
train.drop(columns = ['has_NA'], inplace = True)

for column in ['ID', 'Year', 'Month', 'ZIPcode']:
    train[column] = train[column].astype(int)
    test[column] = test[column].astype(int)

train['Month_cnt'] = (train['Year'] - 2015) * 12 + train['Month'] - 11
test['Month_cnt'] = (test['Year'] - 2015) * 12 + test['Month'] - 11

train = train.drop(columns = ['Year','Month'])
test = test.drop(columns = ['Year','Month'])

test.head(10)
test[['ID', 'Month_cnt']].drop_duplicates()

test_index = test[['ID', 'Month_cnt']] # @NOTE for submission merge
train_index = train[['ID', 'Month_cnt']] # @NOTE for submission merge
test_ID = test['ID'].unique().tolist()
train_ID = train['ID'].unique().tolist()
only_in_train_ID = list(set(train_ID) - set(test_ID))


check_data(train)
# check_data(test)
# train.head()
# test.head()

# train = train[train['Month_cnt'] < 19]
# test = train[train['Month_cnt'] > 13]
# train = train[train['Month_cnt'] <=13]

# # A. Benchmark with all average ------------------------------------------------
# # 654425.7665358948
# test['prediction'] = train['MOD_VOLUME_CONSUMPTION'].mean()
# evaluate_result(test['prediction'], test['MOD_VOLUME_CONSUMPTION'])
#
#
# # B. Benchmark with ID average ------------------------------------------------
# # 427306.68501270027
# ID_sales_mean = train.groupby(['ID'])['MOD_VOLUME_CONSUMPTION'].mean().reset_index().rename(columns = {'MOD_VOLUME_CONSUMPTION': 'prediction'})
# test = test.merge(ID_sales_mean, how = 'left', on = 'ID')
# test_ = test[-pd.isna(test['prediction'])]
# evaluate_result(test_['prediction'], test_['MOD_VOLUME_CONSUMPTION'])
#
#
# # C. Benchmark with ID median ------------------------------------------------
# # 512244.84437995753
# ID_sales_mean = train.groupby(['ID'])['MOD_VOLUME_CONSUMPTION'].median().reset_index().rename(columns = {'MOD_VOLUME_CONSUMPTION': 'prediction'})
# test = test.merge(ID_sales_mean, how = 'left', on = 'ID').fillna(train['MOD_VOLUME_CONSUMPTION'].mean())
# test_ = test[-pd.isna(test['prediction'])]
# evaluate_result(test_['prediction'], test_['MOD_VOLUME_CONSUMPTION'])
#
#


# C. walk-forward validation
index_cols = ['GAS', 'ID', 'MARKET_DOMAIN_DESCR', 'Month_cnt', 'ZIPcode', 'Sum_of_Sales_CR']
cols_to_shift = list(train.columns.difference(index_cols))
cols_to_shift

ID_mean_map = train.groupby(['ID'])['MOD_VOLUME_CONSUMPTION'].mean()

id_mean_alpha = 1
global_mean_alpha = 0

shift_range_max = 7
shift_range = range(1, shift_range_max)
for month_shift in tqdm(shift_range):
    train_shift = train[index_cols + cols_to_shift].copy()

    train_shift['Month_cnt'] = train_shift['Month_cnt'] + month_shift
    train_shift.head()

    foo = lambda x: '{}_lag_{}'.format(x, month_shift) if x in cols_to_shift else x
    train_shift = train_shift.rename(columns=foo)
    train_shift

    # all_data = pd.merge(all_data, train_shift, on=index_cols, how='left').fillna(0)
    # train = pd.merge(train, train_shift, on=index_cols, how='left').fillna(global_mean)
    train = pd.merge(train, train_shift, on=index_cols, how='left') #@TODO

train.head()
train['MOD_VOLUME_CONSUMPTION_lag_1'] = train['MOD_VOLUME_CONSUMPTION_lag_1'].fillna(((train['ID'].map(ID_mean_map) * id_mean_alpha) + (global_mean * global_mean_alpha)) / (global_mean_alpha + id_mean_alpha))

# 265074.3659514087
train_ = train[-pd.isna(train['MOD_VOLUME_CONSUMPTION_lag_1'])]
evaluate_result(train_['MOD_VOLUME_CONSUMPTION_lag_1'], train_['MOD_VOLUME_CONSUMPTION'])

# 262672.02397025743
train_ = train[-pd.isna(train['MOD_VOLUME_CONSUMPTION_lag_2'])]
evaluate_result(train_['MOD_VOLUME_CONSUMPTION_lag_2'], train_['MOD_VOLUME_CONSUMPTION'])

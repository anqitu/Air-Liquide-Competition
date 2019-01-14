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

# 1. Check data for missing values =============================================
train = pd.read_csv(os.path.join(raw_data_path, 'train.csv'), sep = ';', decimal=',')
test = pd.read_csv(os.path.join(raw_data_path, 'test.csv'))

train.head()
test.head()

train_columns = train.columns.tolist()
test_columns = test.columns.tolist()
target = 'MOD_VOLUME_CONSUMPTION'
train_columns_feature = train.columns.tolist()
train_columns_feature.remove(target)
train_columns
test_columns
for i in range(len(test_columns)):
    train.rename(columns={train_columns_feature[i]: test_columns[i]}, inplace = True)
train.rename(columns={'TIMESTAMP - Year': 'Year', 'TIMESTAMP - Month': 'Month', 'Sum of Sales_CR': 'Sum_of_Sales_CR'}, inplace = True)
test.rename(columns={'TIMESTAMP - Year': 'Year', 'TIMESTAMP - Month': 'Month', 'Sum of Sales_CR': 'Sum_of_Sales_CR'}, inplace = True)

# ['ID',
#  'GAS',
#  'MARKET_DOMAIN_DESCR',
#  'Sum_of_Sales_CR',
#  'ZIPcode',
#  'Year',
#  'Month',
#  'MOD_VOLUME_CONSUMPTION']
train['Month'].unique()

month_map = dict(zip(['December', 'November', 'October', 'September', 'August', 'July', 'June', 'May', 'April', 'March', 'February', 'January'], \
                    [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]))
train['Month'] = train['Month'].map(month_map)
test['Month'] = test['Month'].map(month_map)


check_data(test)
test.info()
test['has_NA'] = test.isnull().any(axis=1)
plt.plot(test['has_NA'], '.') # missing values are clustered
test[test['has_NA'] == True]
test[test['ID'] == 361]
train[train['ID'] == 361]
# Fill ZIPcode NA with most common ZIPcode
test.fillna(test['ZIPcode'].value_counts().reset_index().iloc[0]['index'], inplace = True)
test.drop(columns = ['has_NA'], inplace = True)

check_data(train)
train.info()
train['has_NA'] = train.isnull().any(axis=1)
plt.plot(train['has_NA'], '.') # missing values are clustered
train[train['has_NA'] == True]
train = train[pd.notna(train['ID'])]
train.fillna(test['ZIPcode'].value_counts().reset_index().iloc[0]['index'], inplace = True)
train.drop(columns = ['has_NA'], inplace = True)

for column in ['ID', 'Year', 'Month', 'ZIPcode']:
    train[column] = train[column].astype(int)
    test[column] = test[column].astype(int)

train['Month_cnt'] = (train['Year'] - 2015) * 12 + train['Month'] - 11
test['Month_cnt'] = (test['Year'] - 2015) * 12 + test['Month'] - 11


# Check ID and Split data ======================================================
ID_not_in_test = list(set(train['ID'].unique().tolist()).difference(set(test['ID'].unique().tolist())))
ID_not_in_test.sort()

train_of_test_ID = train[-(train['ID'].isin(ID_not_in_test))].sort_values(['ID', 'Month_cnt']).reset_index().drop(columns = ['index'])
train_not_test_ID = train[(train['ID'].isin(ID_not_in_test))].sort_values(['ID', 'Month_cnt']).reset_index().drop(columns = ['index'])
test = test.sort_values(['ID', 'Month_cnt']).reset_index().drop(columns = ['index'])
test[['Month', 'Year', 'Month_cnt']].drop_duplicates() #19-26
train_of_test_ID[['Month', 'Year', 'Month_cnt']].drop_duplicates() #0-18
train_not_test_ID[['Month', 'Year', 'Month_cnt']].drop_duplicates() #0-26

all_data_of_test_ID  = pd.concat([test, train_of_test_ID])
all_data = pd.concat([all_data_of_test_ID, train_not_test_ID])
# check_data(all_data_of_test_ID)
check_data(all_data)

all_data_of_test_ID = all_data_of_test_ID.sort_values(['ID', 'Month_cnt']).reset_index().drop(columns = ['index'])
all_data = all_data.sort_values(['ID', 'Month_cnt']).reset_index().drop(columns = ['index'])
train = train.sort_values(['ID', 'Month_cnt']).reset_index().drop(columns = ['index'])

datas = [train, test, train_of_test_ID, train_not_test_ID, all_data_of_test_ID, all_data]

# Check MOD_VOLUME_CONSUMPTION =================================================
# @NOTE A few outliers
fig, ax = plt.subplots()
ax.set(xscale="log")
sns.distplot(train['MOD_VOLUME_CONSUMPTION'], ax=ax)
plt.show()

sns.distplot(train['MOD_VOLUME_CONSUMPTION'])
sns.boxplot(train['MOD_VOLUME_CONSUMPTION'].clip(0,1000000))
sns.boxplot(x = train['ID'], y = train['MOD_VOLUME_CONSUMPTION'])
sns.stripplot(x = train['ID'], y = train['MOD_VOLUME_CONSUMPTION'], jitter=True)
sns.boxplot(x = train_of_test_ID['ID'], y = train_of_test_ID['MOD_VOLUME_CONSUMPTION'])
train.head()

all_data['ID'].unique()
plt.plot(all_data[all_data['ID'] == 378]['MOD_VOLUME_CONSUMPTION'])



# Check Month_cnt ====================================================================
# @NOTE Some month are skipped
# train_of_test_ID['ID'].value_counts().reset_index().sort_values(['index']).reset_index()['ID'].plot.bar() # Some ID dont have all 9 month
# test['Month_cnt'].value_counts().reset_index().sort_values(['index']).reset_index()['ID'].plot.bar() # Some ID dont have all 9 month
all_data_of_test_ID.shape
sns.stripplot(x="ID", y="Month_cnt", data=train_of_test_ID)
sns.stripplot(x="ID", y="Month_cnt", data=test)
test.groupby(['ID'])['Month_cnt'].count().reset_index().sort_values(['Month_cnt']).head(10)
train[train['ID'] == 1107]
test[test['ID'] == 1107]
sns.stripplot(x="ID", y="Month_cnt", data=all_data)

data = all_data
ID_month = pd.DataFrame(columns = ['ID'], data = data['ID'].unique())
ID_month['Month_has_comsumption'] = data.groupby(['ID'])['ID'].count().tolist()
ID_month['Month_start'] = data.groupby(['ID'])['Month_cnt'].min().tolist()
ID_month['Month_end'] = data.groupby(['ID'])['Month_cnt'].max().tolist()
ID_month['Month_length'] = ID_month['Month_end'] - ID_month['Month_start'] + 1
ID_month['Month_no_consumption'] = ID_month['Month_length'] - ID_month['Month_has_comsumption']
ID_month['Month_length'].sum()
ID_month['Month_no_consumption'].sum()
ID_month['Month_has_comsumption'].sum()

ID_month[ID_month['Month_has_comsumption'] != ID_month['Month_length']]

# @NOTE Some ID starts and ends at different time
sns.boxplot(ID_month['Month_start'])
sns.boxplot(ID_month['Month_end'])
ID_month[ID_month['Month_start'] != 0]
ID_month[ID_month['Month_end'] != 26]

all_data['Month_cnt'].value_counts().reset_index().sort_values(['index']).reset_index()['Month_cnt'].plot.bar()
train_of_test_ID['Month_cnt'].value_counts().reset_index().sort_values(['index']).reset_index()['Month_cnt'].plot.bar()
train_not_test_ID['Month_cnt'].value_counts().reset_index().sort_values(['index']).reset_index()['Month_cnt'].plot.bar()
train['Month_cnt'].value_counts().reset_index().sort_values(['index']).reset_index()['Month_cnt'].plot.bar()

sns.boxplot(x = all_data['Month_cnt'], y = all_data['MOD_VOLUME_CONSUMPTION'])
sns.boxplot(x = all_data['Month_cnt'], y = all_data['MOD_VOLUME_CONSUMPTION'].clip(0, 2500000))
sns.stripplot(x = all_data['Month_cnt'], y = all_data['MOD_VOLUME_CONSUMPTION'].clip(0, 2500000))
sns.boxplot(x = train_of_test_ID['Month_cnt'], y = train_of_test_ID['MOD_VOLUME_CONSUMPTION'].clip(0, 2500000))
sns.boxplot(x = train_not_test_ID['Month_cnt'], y = train_not_test_ID['MOD_VOLUME_CONSUMPTION'].clip(0, 2500000))
sns.stripplot(x = train_not_test_ID['Month_cnt'], y = train_not_test_ID['MOD_VOLUME_CONSUMPTION'].clip(0, 2500000))

sns.lmplot(x = 'Month_cnt', y = 'MOD_VOLUME_CONSUMPTION', data = all_data)
train_not_test_ID['ID'].unique()


# Check GAS ====================================================================
all_data[['ID', 'GAS']].drop_duplicates().shape  # Same ID with same GAS

all_data['GAS'].value_counts().plot.bar()
sns.boxplot(x = all_data['GAS'], y = all_data['MOD_VOLUME_CONSUMPTION'].clip(0, 1000000))
sns.stripplot(x = all_data['GAS'], y = all_data['MOD_VOLUME_CONSUMPTION'], jitter=True)
sns.stripplot(x = all_data['GAS'], y = all_data['MOD_VOLUME_CONSUMPTION'].clip(0, 1000000), jitter=True)


# Check MARKET_DOMAIN_DESCR ====================================================
all_data[['ID', 'MARKET_DOMAIN_DESCR']].drop_duplicates().shape  # Same ID with same MARKET_DOMAIN_DESCR
all_data['MARKET_DOMAIN_DESCR'].value_counts().plot.bar()
sns.boxplot(x = all_data['MARKET_DOMAIN_DESCR'], y = all_data['MOD_VOLUME_CONSUMPTION'].clip(0, 1000000))



# Check ZIPcode ====================================================
all_data[['ID', 'ZIPcode']].drop_duplicates().shape # Same ID with same ZIPcode
all_data['ZIPcode'].value_counts().plot.bar()
sns.boxplot(x = all_data['ZIPcode'], y = all_data['MOD_VOLUME_CONSUMPTION'])



# Check Year ====================================================
all_data['Year'].value_counts().plot.bar()
sns.boxplot(x = all_data['Year'], y = all_data['MOD_VOLUME_CONSUMPTION'].clip(0,2000000))


# Check Month ====================================================
all_data['Month'].value_counts().plot.bar()
sns.boxplot(x = all_data['Month'], y = all_data['MOD_VOLUME_CONSUMPTION'].clip(0,2000000))



# Check Sum_of_Sales_CR ====================================================
all_data[['ID', 'Sum_of_Sales_CR']].drop_duplicates()  # Same ID with same Sum_of_Sales_CR but different dp
ID_Sales = train[['ID', 'Sum_of_Sales_CR']].drop_duplicates() # Standardize Sum_of_Sales_CR dp for each ID
all_data = all_data.drop(columns = ['Sum_of_Sales_CR'])
all_data = all_data.merge(ID_Sales, on = 'ID', how = 'left')
all_data_of_test_ID = all_data_of_test_ID.drop(columns = ['Sum_of_Sales_CR'])
all_data_of_test_ID = all_data_of_test_ID.merge(ID_Sales, on = 'ID', how = 'left')

sns.boxplot(all_data['Sum_of_Sales_CR'])

sns.regplot(all_data['Sum_of_Sales_CR'], all_data['MOD_VOLUME_CONSUMPTION'])
train_of_test_ID['Sum_of_Sales_CR'].corr(train_of_test_ID['MOD_VOLUME_CONSUMPTION'])
train_not_test_ID['Sum_of_Sales_CR'].corr(train_not_test_ID['MOD_VOLUME_CONSUMPTION'])
train['Sum_of_Sales_CR'].corr(train['MOD_VOLUME_CONSUMPTION'])

train_not_test_ID_sales_sum = train_not_test_ID.groupby(['ID'])['MOD_VOLUME_CONSUMPTION'].sum().reset_index().rename(columns = {'MOD_VOLUME_CONSUMPTION': 'Sum_of_Sales'})
train_not_test_ID_sales_sum_CR = train_not_test_ID[['ID', 'Sum_of_Sales_CR']].drop_duplicates()

train_not_test_ID_sales_sum = train_not_test_ID_sales_sum.merge(train_not_test_ID_sales_sum_CR)
sns.regplot(train_not_test_ID_sales_sum['Sum_of_Sales_CR'], train_not_test_ID_sales_sum['Sum_of_Sales'])





# 2. @TODO Set missing month sales as 0?
from itertools import product
grid = []
for id in train['ID'].unique():
    month_start = train[train['ID']==id]['Month_cnt'].min()
    month_end = train[train['ID']==id]['Month_cnt'].max()
    months = range(month_start, month_end + 1)
    grid.append(np.array(list(product(*[[id], months])),dtype='int32'))
index_cols = ['ID', 'Month_cnt']
grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)
grid.head()

train_fill_missing_month = grid.merge(train[['ID', 'Month_cnt', 'MOD_VOLUME_CONSUMPTION']], how = 'left')
sns.stripplot(x="ID", y="Month_cnt", data=train_fill_missing_month)
train_fill_missing_month = train_fill_missing_month.fillna(0)
train_fill_missing_month = train_fill_missing_month.merge(train[['ID', 'GAS', 'MARKET_DOMAIN_DESCR', 'ZIPcode', 'Sum_of_Sales_CR']].drop_duplicates(), how = 'left')

check_data(train_fill_missing_month)
train_fill_missing_month.head()

train_fill_missing_month['Sum_of_Sales_CR'].corr(train_fill_missing_month['MOD_VOLUME_CONSUMPTION'])
train['Sum_of_Sales_CR'].corr(train['MOD_VOLUME_CONSUMPTION'])

# train_save = train[:]
# test_save = test[:]
# check_data(train_save)
# check_data(test_save)
# train = train_save[:]
# test = test_save[:]

train_fill_missing_month.ID.unique()
train[train['ID']==145]
test['ID'].unique()
plt.plot(train_fill_missing_month[train_fill_missing_month['ID']==145]['MOD_VOLUME_CONSUMPTION'].iloc[:100])

# 3. Mean-encodings ============================================================
# For Trainset
Target = 'MOD_VOLUME_CONSUMPTION'
global_mean = train[Target].mean()
# global_mean = train[Target].median()
# train.describe()
# global_mean = train_fill_missing_month[Target].mean()
y_tr = train[Target].values
global_mean
evaluate_result(pred, train['MOD_VOLUME_CONSUMPTION'])
# ['ID',
#  'GAS',
#  'MARKET_DOMAIN_DESCR',
#  'ZIPcode',
#  'Year',
#  'Month',
#  'Month_cnt',
#  'Sum_of_Sales_CR',
#  'MOD_VOLUME_CONSUMPTION']

mean_encoded_col = ['ID', 'GAS', 'Month_cnt', 'MARKET_DOMAIN_DESCR', 'ZIPcode']
for col in tqdm(mean_encoded_col):

    col_tr = train[[col] + [Target]]
    corrcoefs = pd.DataFrame(columns = ['Cor'])

    # 3.1.1 Mean encodings - KFold scheme
    from sklearn.model_selection import KFold
    kf = KFold(n_splits = 5, shuffle = False, random_state = SEED)

    col_tr[col + '_target_mean_Kfold'] = global_mean
    for tr_ind, val_ind in kf.split(col_tr):
        X_tr, X_val = col_tr.iloc[tr_ind], col_tr.iloc[val_ind]
        means = X_val[col].map(X_tr.groupby(col)[Target].mean())
        X_val[col + '_target_mean_Kfold'] = means
        col_tr.iloc[val_ind] = X_val
        # X_val.head()
    col_tr.fillna(global_mean, inplace = True)
    corrcoefs.loc[col + '_target_mean_Kfold'] = np.corrcoef(y_tr, col_tr[col + '_target_mean_Kfold'])[0][1]

    # 3.1.2 Mean encodings - Leave-one-out scheme
    target_sum = col_tr.groupby(col)[Target].sum()
    target_count = col_tr.groupby(col)[Target].count()
    col_tr[col + '_target_sum'] = col_tr[col].map(target_sum)
    col_tr[col + '_target_count'] = col_tr[col].map(target_count)
    col_tr[col + '_target_mean_LOO'] = (col_tr[col + '_target_sum'] - col_tr[Target]) / (col_tr[col + '_target_count'] - 1)
    col_tr.fillna(global_mean, inplace = True)
    corrcoefs.loc[col + '_target_mean_LOO'] = np.corrcoef(y_tr, col_tr[col + '_target_mean_LOO'])[0][1]


    # 3.1.3 Mean encodings - Smoothing
    target_mean = col_tr.groupby(col)[Target].mean()
    target_count = col_tr.groupby(col)[Target].count()
    col_tr[col + '_target_mean'] = col_tr[col].map(target_mean)
    col_tr[col + '_target_count'] = col_tr[col].map(target_count)
    alpha = 100
    col_tr[col + '_target_mean_Smooth'] = (col_tr[col + '_target_mean'] *  col_tr[col + '_target_count'] + global_mean * alpha) / (alpha + col_tr[col + '_target_count'])
    col_tr[col + '_target_mean_Smooth'].fillna(global_mean, inplace=True)
    corrcoefs.loc[col + '_target_mean_Smooth'] = np.corrcoef(y_tr, col_tr[col + '_target_mean_Smooth'])[0][1]


    # 3.1.4 Mean encodings - Expanding mean scheme
    cumsum = col_tr.groupby(col)[Target].cumsum() - col_tr[Target]
    sumcnt = col_tr.groupby(col).cumcount()
    col_tr[col + '_target_mean_Expanding'] = cumsum / sumcnt
    col_tr[col + '_target_mean_Expanding'].fillna(global_mean, inplace=True)
    corrcoefs.loc[col + '_target_mean_Expanding'] = np.corrcoef(y_tr, col_tr[col + '_target_mean_Expanding'])[0][1]

    train = pd.concat([train, col_tr[corrcoefs['Cor'].idxmax()]], axis = 1)
    # target_mean = col_tr.groupby([col])[Target].mean().reset_index().rename(columns = {Target: corrcoefs['Cor'].idxmax()})
    # test = test.merge(target_mean, how = 'left')

    print(corrcoefs.sort_values('Cor'))
# train_save = train[:]
# test_save = test[:]
# check_data(train_fix)
# check_data(test_fix)
# train = train_save[:]
# test = test_save[:]
test.shape
train.shape

test.to_csv(os.path.join(processed_data_path, 'test.csv'), index = False)
train.to_csv(os.path.join(processed_data_path, 'train.csv'), index = False)

# # Bench mark ===================================================================
# from sklearn.ensemble import RandomForestRegressor
# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score
#
#
# Target = 'MOD_VOLUME_CONSUMPTION'
# y_train = train[Target]
# X_train = train.drop(columns = [Target])
# X_train.head()
# feature_columns = X_train.columns
# feature_columns = list(set(X_train.columns.tolist()).difference(set(['Month', 'Year'])))
# X_train = X_train[feature_columns]
# test = test[feature_columns]
#
# from sklearn.preprocessing import LabelEncoder
# encoder = LabelEncoder()
# cate_columns = ['GAS', 'MARKET_DOMAIN_DESCR']
# for column in cate_columns:
#     X_train[column] = encoder.fit_transform(X_train[column])
#     test[column] = encoder.transform(test[column])
#
# # X_train = pd.get_dummies(X_train, columns = ['GAS', 'MARKET_DOMAIN_DESCR', 'ZIPcode', 'Month_cnt'])
# # test = pd.get_dummies(test, columns = ['GAS', 'MARKET_DOMAIN_DESCR', 'ZIPcode', 'Month_cnt'])
#
# plt.plot(y_train, '.')
# bins = np.linspace(0, 4000000, 8)
# y_binned = np.digitize(y_train, bins)
# train_x, val_x, train_y, val_y = train_test_split(X_train, y_train, train_size = 0.7, test_size = 0.3, stratify = y_binned, random_state = SEED)
# plt.plot(train_y, '.')
# plt.plot(val_y, '.')
#
#
#
# model = RandomForestRegressor()
# model = xgb.XGBRegressor()
#
# model.fit(train_x, train_y)
# evaluate_result(model.predict(train_x), train_y)
# evaluate_result(model.predict(val_x), val_y)
# r2_score(model.predict(train_x), train_y)
# r2_score(model.predict(val_x), val_y)
#
# plt.plot(model.predict(train_x), '.')
# plt.plot(train_y.tolist(), '.')
#
#
#
# #  RFECV -----------------------------------------------------------------------
# from sklearn import tree, linear_model, ensemble, kernel_ridge, neighbors
# import xgboost as xgb
# from sklearn import model_selection # RFE
# from sklearn import feature_selection # RFE
# cv_split = model_selection.KFold(n_splits = 5, random_state = SEED)
#
# estimators = {
#     'lr': linear_model.LinearRegression(),
#     # 'rfr': ensemble.RandomForestRegressor(random_state = SEED),
#     # 'dtree': tree.DecisionTreeRegressor(random_state = SEED),
#     # 'xgb': xgb.XGBRegressor()
#     }
# if 'estimators_RFECV' not in globals(): estimators_RFECV = {}
# for estimatorName, estimator in estimators.items():
#     start = time.perf_counter()
#     rfecv = feature_selection.RFECV(estimator = estimator, step = 1, scoring = scoringMethod, cv = cv_split)
#     rfecv_result = rfecv.fit(X_train, y_train)
#     run = time.perf_counter() - start
#
#     print('{} runs for {:.2f} seconds.'.format(estimator.__class__.__name__, run))
#     estimators_RFECV[estimatorName] = rfecv_result
#
# for selector_name, selector in estimators_RFECV.items():
#     print(selector_name)
#     print('The optimal number of features is {}'.format(selector.n_features_))
#     features = [f for f,s in zip(X_train.columns, selector.support_) if s]
#     print('The selected features are:')
#     print ('{}'.format(features))
#
#     plt.figure()
#     plt.xlabel("Number of features selected")
#     plt.ylabel("Cross validation score (neg_mse)")
#     plt.plot(range(1, len(selector.grid_scores_) + 1), [get_rmse(score) for score in selector.grid_scores_])
#
# model_name = 'lr'
# selector = estimators_RFECV[model_name]
# evaluate_result(selector.predict(X_train), y_train)
# print('The optimal number of features is {}'.format(selector.n_features_))
# features = [f for f,s in zip(X_train.columns, selector.support_) if s]
# print('The selected features are:')
# print ('{}'.format(features))
#
# plt.figure()
# plt.xlabel("Number of features selected")
# plt.ylabel("Cross validation score (neg_mse)")
# plt.plot(range(1, len(selector.grid_scores_) + 1), [get_rmse(score) for score in selector.grid_scores_])
#
#
#
# X_train_s = selector.transform(X_train)
# train_x, val_x, train_y, val_y = train_test_split(X_train_s, y_train, train_size = 0.7, test_size = 0.3, stratify = y_binned, random_state = SEED)
#
# model = tree.DecisionTreeRegressor(random_state = SEED)
# model = ensemble.RandomForestRegressor(random_state = SEED)
# model = xgb.XGBRegressor()
# model = linear_model.LinearRegression()
# model = kernel_ridge.KernelRidge()
# model = svm.SVR()
# model = neighbors.KNeighborsRegressor(n_neighbors=8)
# model = linear_model.SGDRegressor()
#
# model.fit(X_train_s, y_train)
#
# y_train_pred = model.predict(X_train_s)
# evaluate_result(y_train, y_train_pred)
# r2_score(y_train, y_train_pred)
# plt.scatter(y_train, y_train_pred)
# plt.plot(y_train, '.')
# plt.plot(y_train_pred, '.')
#
# model.fit(train_x, train_y)
# evaluate_result(model.predict(train_x), train_y)
# evaluate_result(model.predict(val_x), val_y)
#
# # RandomForestRegressor
# # 95103.49713474252
# # 215306.62625697735
# # 215306.62625697735
# r2_score(model.predict(train_x), train_y)
# r2_score(model.predict(val_x), val_y)
#
# plt.plot(model.predict(train_x), '.')
# plt.plot(train_y.tolist(), '.')
#
#
# X_test_s = selector.transform(test)
# y_test_pred = model.predict(X_test_s)
# plt.plot(y_test_pred, '.')

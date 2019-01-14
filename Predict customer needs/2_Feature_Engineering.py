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


# check_data(train)
# check_data(test)
# train.head()
# test.head()





# 2. Feature Engineering =======================================================
# 2.1 Combine trainset and testset ---------------------------------------------
all_data = pd.concat([train, test], axis = 0)
all_data = all_data.sort_values(['ID', 'Month_cnt']).reset_index().drop(columns = ['index'])

all_data[['ID', 'Sum_of_Sales_CR']].drop_duplicates().shape
ID_Sales = train[['ID', 'Sum_of_Sales_CR']].drop_duplicates() # Standardize Sum_of_Sales_CR dp for each ID
all_data = all_data.drop(columns = ['Sum_of_Sales_CR'])
all_data = all_data.merge(ID_Sales, on = 'ID', how = 'left')

train = train_index.merge(all_data, how = 'left')
test = test_index.merge(all_data, how = 'left').drop(columns = [target])

# check_data(test)


# 2.2 @TODO Deal with skipped month? ---------------------------------------------
from itertools import product
grid = []
for id in all_data['ID'].unique():
    month_start = all_data[all_data['ID'] == id]['Month_cnt'].min()
    month_end = all_data[all_data['ID'] == id]['Month_cnt'].max()
    months = range(month_start, month_end + 1)
    grid.append(np.array(list(product(*[[id], months])),dtype='int32'))
index_cols = ['ID', 'Month_cnt']
grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)
grid.shape

all_data_fill_missing_month = grid.merge(all_data[['ID', 'Month_cnt', 'MOD_VOLUME_CONSUMPTION']], how = 'left')
all_data_fill_missing_month = all_data_fill_missing_month.merge(all_data[['GAS', 'ID', 'MARKET_DOMAIN_DESCR','ZIPcode', 'Sum_of_Sales_CR']].drop_duplicates(), how = 'left', on = ['ID'])
check_data(all_data_fill_missing_month)
sns.stripplot(x="ID", y="Month_cnt", data=all_data)
sns.stripplot(x="ID", y="Month_cnt", data=all_data_fill_missing_month)


# A. Fill NA with ID mean
# ID_mean_map = train.groupby(['ID'])['MOD_VOLUME_CONSUMPTION'].mean()
# id_mean_alpha = 1
# global_mean_alpha = 0
# all_data_fill_missing_month['MOD_VOLUME_CONSUMPTION'] = all_data_fill_missing_month['MOD_VOLUME_CONSUMPTION'].fillna(((all_data_fill_missing_month['ID'].map(ID_mean_map) * id_mean_alpha) + (global_mean * global_mean_alpha)) / (global_mean_alpha + id_mean_alpha))


# B. Fill NA with walkforward rolling
index_cols = ['GAS', 'ID', 'MARKET_DOMAIN_DESCR', 'Month_cnt', 'ZIPcode', 'Sum_of_Sales_CR']
cols_to_shift = list(all_data_fill_missing_month.columns.difference(index_cols))
cols_to_shift

shift_range_max = 20
shift_range = range(1, shift_range_max)
for month_shift in tqdm(shift_range):
    train_shift = all_data_fill_missing_month[index_cols + cols_to_shift].copy()

    train_shift['Month_cnt'] = train_shift['Month_cnt'] + month_shift
    train_shift.head()

    foo = lambda x: '{}_lag_{}'.format(x, month_shift) if x in cols_to_shift else x
    train_shift = train_shift.rename(columns=foo)

    all_data_fill_missing_month = pd.merge(all_data_fill_missing_month, train_shift, on=index_cols, how='left') #@TODO

check_data(all_data_fill_missing_month)
all_data_fill_missing_month['MOD_VOLUME_CONSUMPTION'] = all_data_fill_missing_month.apply(lambda r: \
                                                r['MOD_VOLUME_CONSUMPTION'] if pd.notna(r['MOD_VOLUME_CONSUMPTION']) \
                                                else r['MOD_VOLUME_CONSUMPTION_lag_1'] if pd.notna(r['MOD_VOLUME_CONSUMPTION_lag_1']) \
                                                else r['MOD_VOLUME_CONSUMPTION_lag_2'] if pd.notna(r['MOD_VOLUME_CONSUMPTION_lag_2']) \
                                                else r['MOD_VOLUME_CONSUMPTION_lag_3'] if pd.notna(r['MOD_VOLUME_CONSUMPTION_lag_3']) \
                                                else r['MOD_VOLUME_CONSUMPTION_lag_4'] if pd.notna(r['MOD_VOLUME_CONSUMPTION_lag_4']) \
                                                else r['MOD_VOLUME_CONSUMPTION_lag_5'] if pd.notna(r['MOD_VOLUME_CONSUMPTION_lag_5']) \
                                                else r['MOD_VOLUME_CONSUMPTION_lag_6'] if pd.notna(r['MOD_VOLUME_CONSUMPTION_lag_6']) \
                                                else r['MOD_VOLUME_CONSUMPTION_lag_7'] if pd.notna(r['MOD_VOLUME_CONSUMPTION_lag_7']) \
                                                else r['MOD_VOLUME_CONSUMPTION_lag_8'] if pd.notna(r['MOD_VOLUME_CONSUMPTION_lag_8']) \
                                                else r['MOD_VOLUME_CONSUMPTION_lag_9'] if pd.notna(r['MOD_VOLUME_CONSUMPTION_lag_9']) \
                                                else r['MOD_VOLUME_CONSUMPTION_lag_10'] if pd.notna(r['MOD_VOLUME_CONSUMPTION_lag_10']) \
                                                else r['MOD_VOLUME_CONSUMPTION_lag_11'] if pd.notna(r['MOD_VOLUME_CONSUMPTION_lag_11']) \
                                                else r['MOD_VOLUME_CONSUMPTION_lag_12'] if pd.notna(r['MOD_VOLUME_CONSUMPTION_lag_12']) \
                                                else r['MOD_VOLUME_CONSUMPTION_lag_13'] if pd.notna(r['MOD_VOLUME_CONSUMPTION_lag_13']) \
                                                else r['MOD_VOLUME_CONSUMPTION_lag_14'] if pd.notna(r['MOD_VOLUME_CONSUMPTION_lag_14']) \
                                                else r['MOD_VOLUME_CONSUMPTION_lag_15'] if pd.notna(r['MOD_VOLUME_CONSUMPTION_lag_15']) \
                                                else r['MOD_VOLUME_CONSUMPTION_lag_16'] if pd.notna(r['MOD_VOLUME_CONSUMPTION_lag_16']) \
                                                else r['MOD_VOLUME_CONSUMPTION_lag_17'] if pd.notna(r['MOD_VOLUME_CONSUMPTION_lag_17']) \
                                                else r['MOD_VOLUME_CONSUMPTION_lag_18'] if pd.notna(r['MOD_VOLUME_CONSUMPTION_lag_18']) \
                                                else r['MOD_VOLUME_CONSUMPTION_lag_19'] if pd.notna(r['MOD_VOLUME_CONSUMPTION_lag_19']) \
                                                else np.NaN, axis=1)
all_data_fill_missing_month['Sum_of_Sales_CR'].corr(all_data_fill_missing_month['MOD_VOLUME_CONSUMPTION'])
all_data['Sum_of_Sales_CR'].corr(all_data['MOD_VOLUME_CONSUMPTION'])


train = all_data_fill_missing_month[(all_data_fill_missing_month['Month_cnt'] < 19) | (all_data_fill_missing_month['ID'].isin(only_in_train_ID)) ]
test = all_data_fill_missing_month[(all_data_fill_missing_month['Month_cnt'] >= 19) & (-(all_data_fill_missing_month['ID'].isin(only_in_train_ID))) ].drop(columns = ['MOD_VOLUME_CONSUMPTION'])
check_data(train)
check_data(test)
train = train[['ID', 'Month_cnt', 'GAS', 'MARKET_DOMAIN_DESCR', 'ZIPcode', 'Sum_of_Sales_CR', 'MOD_VOLUME_CONSUMPTION']]
test = test[['ID', 'Month_cnt', 'GAS', 'MARKET_DOMAIN_DESCR', 'ZIPcode', 'Sum_of_Sales_CR']]





# 2.3. Mean-encodings ----------------------------------------------------------
y_tr = train[target].values
global_mean

mean_encoded_col = ['ID', 'GAS', 'Month_cnt', 'MARKET_DOMAIN_DESCR', 'ZIPcode']
for col in tqdm(mean_encoded_col):

    col_tr = train[[col] + [target]]
    corrcoefs = pd.DataFrame(columns = ['Cor'])

    # 3.1.1 Mean encodings - KFold scheme
    from sklearn.model_selection import KFold
    kf = KFold(n_splits = 5, shuffle = False, random_state = SEED)

    col_tr[col + '_target_mean_Kfold'] = global_mean
    for tr_ind, val_ind in kf.split(col_tr):
        X_tr, X_val = col_tr.iloc[tr_ind], col_tr.iloc[val_ind]
        means = X_val[col].map(X_tr.groupby(col)[target].mean())
        X_val[col + '_target_mean_Kfold'] = means
        col_tr.iloc[val_ind] = X_val
        # X_val.head()
    col_tr.fillna(global_mean, inplace = True)
    corrcoefs.loc[col + '_target_mean_Kfold'] = np.corrcoef(y_tr, col_tr[col + '_target_mean_Kfold'])[0][1]

    # 3.1.2 Mean encodings - Leave-one-out scheme
    target_sum = col_tr.groupby(col)[target].sum()
    target_count = col_tr.groupby(col)[target].count()
    col_tr[col + '_target_sum'] = col_tr[col].map(target_sum)
    col_tr[col + '_target_count'] = col_tr[col].map(target_count)
    col_tr[col + '_target_mean_LOO'] = (col_tr[col + '_target_sum'] - col_tr[target]) / (col_tr[col + '_target_count'] - 1)
    col_tr.fillna(global_mean, inplace = True)
    corrcoefs.loc[col + '_target_mean_LOO'] = np.corrcoef(y_tr, col_tr[col + '_target_mean_LOO'])[0][1]


    # 3.1.3 Mean encodings - Smoothing
    target_mean = col_tr.groupby(col)[target].mean()
    target_count = col_tr.groupby(col)[target].count()
    col_tr[col + '_target_mean'] = col_tr[col].map(target_mean)
    col_tr[col + '_target_count'] = col_tr[col].map(target_count)
    alpha = 50
    col_tr[col + '_target_mean_Smooth'] = (col_tr[col + '_target_mean'] *  col_tr[col + '_target_count'] + global_mean * alpha) / (alpha + col_tr[col + '_target_count'])
    col_tr[col + '_target_mean_Smooth'].fillna(global_mean, inplace=True)
    corrcoefs.loc[col + '_target_mean_Smooth'] = np.corrcoef(y_tr, col_tr[col + '_target_mean_Smooth'])[0][1]


    # 3.1.4 Mean encodings - Expanding mean scheme
    cumsum = col_tr.groupby(col)[target].cumsum() - col_tr[target]
    sumcnt = col_tr.groupby(col).cumcount()
    col_tr[col + '_target_mean_Expanding'] = cumsum / sumcnt
    col_tr[col + '_target_mean_Expanding'].fillna(global_mean, inplace=True)
    corrcoefs.loc[col + '_target_mean_Expanding'] = np.corrcoef(y_tr, col_tr[col + '_target_mean_Expanding'])[0][1]

    train = pd.concat([train, col_tr[corrcoefs['Cor'].idxmax()]], axis = 1)
    print(corrcoefs.sort_values('Cor'))


all_data = pd.concat([train, test], axis = 0)
all_data = all_data.sort_values(['ID', 'Month_cnt']).reset_index().drop(columns = ['index'])

# check_data(train)
# check_data(test)
check_data(all_data)



# 2.2 Creating lag-based features -----------------------------------------
index_cols = ['GAS', 'ID', 'MARKET_DOMAIN_DESCR', 'Month_cnt', 'ZIPcode', 'Sum_of_Sales_CR']
cols_to_shift = list(all_data.columns.difference(index_cols))
cols_to_shift
# cols_to_shift = ['GAS_target_mean_Expanding', 'ID_target_mean_Smooth']
# ['GAS_target_mean_Expanding', 'ID_target_mean_Smooth',
# 'MARKET_DOMAIN_DESCR_target_mean_Expanding', 'MOD_VOLUME_CONSUMPTION',
# 'Month_cnt_target_mean_Smooth', 'ZIPcode_target_mean_LOO']

shift_range_max = 10
shift_range = range(1, shift_range_max)
for month_shift in tqdm(shift_range):
    train_shift = all_data[index_cols + cols_to_shift].copy()

    train_shift['Month_cnt'] = train_shift['Month_cnt'] + month_shift
    train_shift.head()

    foo = lambda x: '{}_lag_{}'.format(x, month_shift) if x in cols_to_shift else x
    train_shift = train_shift.rename(columns=foo)
    train_shift

    # all_data = pd.merge(all_data, train_shift, on=index_cols, how='left').fillna(0)
    # all_data = pd.merge(all_data, train_shift, on=index_cols, how='left').fillna(global_mean) #@TODO
    all_data = pd.merge(all_data, train_shift, on=index_cols, how='left') #@TODO

all_data.head(30)
# train = train_index.merge(all_data, how = 'left')
# test = test_index.merge(all_data, how = 'left').drop(columns = ['MOD_VOLUME_CONSUMPTION'])
train = all_data[(all_data['Month_cnt'] < 19) | (all_data['ID'].isin(only_in_train_ID)) ]
test = all_data[(all_data['Month_cnt'] >= 19) & (-(all_data['ID'].isin(only_in_train_ID))) ].drop(columns = ['MOD_VOLUME_CONSUMPTION'])
train = train[train['Month_cnt'] >= shift_range_max - 1]
check_data(train)
check_data(test)

# train.sort_values(['ID', 'Month_cnt'])[['ID','Month_cnt', 'GAS_target_mean_Expanding_lag_6']].head(30)
# train.head(30)




# Bench mark ===================================================================
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


y_train = train[target]
X_train = train.drop(columns = [target])
X_train.head()
feature_columns = X_train.columns
lag_cols = [col for col in all_data.columns if col[-1] in [str(item) for item in shift_range]]
feature_columns = list(set(lag_cols + index_cols))
X_train = X_train[feature_columns]
test = test[feature_columns]

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
cate_columns = ['GAS', 'MARKET_DOMAIN_DESCR']
for column in cate_columns:
    X_train[column] = encoder.fit_transform(X_train[column])
    test[column] = encoder.transform(test[column])

# X_train = pd.get_dummies(X_train, columns = ['GAS', 'MARKET_DOMAIN_DESCR', 'ZIPcode', 'Month_cnt'])
# test = pd.get_dummies(test, columns = ['GAS', 'MARKET_DOMAIN_DESCR', 'ZIPcode', 'Month_cnt'])

# plt.plot(y_train, '.')
bins = np.linspace(0, 4000000, 8)
y_binned = np.digitize(y_train, bins)
train_x, val_x, train_y, val_y = train_test_split(X_train, y_train, train_size = 0.7, test_size = 0.3, stratify = y_binned, random_state = SEED)
# plt.plot(train_y, '.')
# plt.plot(val_y, '.')


model = RandomForestRegressor()
model = xgb.XGBRegressor()

model.fit(train_x, train_y)
evaluate_result(model.predict(train_x), train_y)
evaluate_result(model.predict(val_x), val_y)
r2_score(model.predict(train_x), train_y)
r2_score(model.predict(val_x), val_y)

plt.plot(model.predict(train_x), '.')
plt.plot(train_y.tolist(), '.')

plt.plot(model.predict(val_x), '.')
plt.plot(val_y.tolist(), '.')



#  RFECV -----------------------------------------------------------------------
from sklearn import tree, linear_model, ensemble, kernel_ridge, neighbors, svm
import xgboost as xgb
from sklearn import model_selection # RFE
from sklearn import feature_selection # RFE
cv_split = model_selection.KFold(n_splits = 5, random_state = SEED)

estimators = {
    'lr': linear_model.LinearRegression(),
    'rfr': ensemble.RandomForestRegressor(random_state = SEED),
    'dtree': tree.DecisionTreeRegressor(random_state = SEED),
    'xgb': xgb.XGBRegressor()
    }
if 'estimators_RFECV' not in globals(): estimators_RFECV = {}
for estimatorName, estimator in estimators.items():
    start = time.perf_counter()
    rfecv = feature_selection.RFECV(estimator = estimator, step = 1, scoring = scoringMethod, cv = cv_split)
    rfecv_result = rfecv.fit(X_train, y_train)
    run = time.perf_counter() - start

    print('{} runs for {:.2f} seconds.'.format(estimator.__class__.__name__, run))
    estimators_RFECV[estimatorName] = rfecv_result

for selector_name, selector in estimators_RFECV.items():
    print(selector_name)
    print('The optimal number of features is {}'.format(selector.n_features_))
    features = [f for f,s in zip(X_train.columns, selector.support_) if s]
    print('The selected features are:')
    print ('{}'.format(features))

    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (neg_mse)")
    plt.plot(range(1, len(selector.grid_scores_) + 1), [get_rmse(score) for score in selector.grid_scores_])

model_name = 'lr'
selector = estimators_RFECV[model_name]
evaluate_result(selector.predict(X_train), y_train)
print('The optimal number of features is {}'.format(selector.n_features_))
features = [f for f,s in zip(X_train.columns, selector.support_) if s]
print('The selected features are:')
print ('{}'.format(features))

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (neg_mse)")
plt.plot(range(1, len(selector.grid_scores_) + 1), [get_rmse(score) for score in selector.grid_scores_])



X_train_s = selector.transform(X_train)
train_x, val_x, train_y, val_y = train_test_split(X_train_s, y_train, train_size = 0.7, test_size = 0.3, stratify = y_binned, random_state = SEED)
X_train_s.shape
model = tree.DecisionTreeRegressor(random_state = SEED)
model = ensemble.RandomForestRegressor(random_state = SEED)
model = xgb.XGBRegressor()
model = linear_model.LinearRegression()
model = kernel_ridge.KernelRidge()
model = svm.SVR()
model = neighbors.KNeighborsRegressor(n_neighbors=8)
# model = linear_model.SGDRegressor()

# model.fit(X_train_s, y_train)
#
# y_train_pred = model.predict(X_train_s)
# evaluate_result(y_train, y_train_pred)
# r2_score(y_train, y_train_pred)
# plt.scatter(y_train, y_train_pred)
# plt.plot(y_train, '.')
# plt.plot(y_train_pred, '.')

model.fit(train_x, train_y)
evaluate_result(model.predict(train_x), train_y)
evaluate_result(model.predict(val_x), val_y)

# RandomForestRegressor
# 95103.49713474252
# 215306.62625697735
# 215306.62625697735
r2_score(model.predict(train_x), train_y)
r2_score(model.predict(val_x), val_y)

plt.plot(model.predict(train_x), '.')
plt.plot(train_y.tolist(), '.')


X_test_s = selector.transform(test)
y_test_pred = model.predict(X_test_s)
plt.plot(y_test_pred, '.')







from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

np.random.seed(SEED)

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(shift_range_max - 1, len(cols_to_shift))))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

train_x = train_x.values
val_x = val_x.values
train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
val_x = np.reshape(val_x, (val_x.shape[0], 1, val_x.shape[1]))
model.fit(train_x, train_y, epochs=100, batch_size=32, verbose=2)
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()









train = pd.read_csv(os.path.join(processed_data_path, 'train.csv'))
test = pd.read_csv(os.path.join(processed_data_path, 'test.csv'))

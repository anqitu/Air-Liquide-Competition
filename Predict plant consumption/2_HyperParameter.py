import pandas as pd
import numpy as np
import time
import os
import math
script_start_time = time.time()

import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams["figure.figsize"] = 15,10

import warnings
warnings.filterwarnings('ignore')

# Settings
project_path = os.getcwd()
data_path = os.path.join(project_path, 'DataFile')
raw_data_path = os.path.join(data_path, 'RawData')
processed_data_path = os.path.join(data_path, 'ProcessedData')
meta_feature_dir = os.path.join(data_path, 'MetaData')
submission_path = os.path.join(project_path, 'Submission')

scoringMethod = 'neg_mean_squared_error'
SEED = 2018
Target = 'Power'

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
train = pd.read_csv(os.path.join(processed_data_path, 'train_fix.csv'))
test = pd.read_csv(os.path.join(processed_data_path, 'test_fix.csv'))
train = train.drop(columns = ['TimeStamp'])
test = test.drop(columns = ['TimeStamp'])


outliers = train.iloc[:300][train['Power'] > 11200].index.values
outliers = np.concatenate([outliers,train.iloc[300:2000][train['Power'] > 11000].index.values])
outliers = np.concatenate([outliers,train.iloc[5000:5179][train['Power'] < 11000].index.values])
plt.plot(train[['Power']], '.')

train = train.loc[list(set(train.index).difference(set(outliers)))]
train.describe()
test.describe()


y_train = train[Target]
X_train = train.drop(columns = [Target])
feature_columns = X_train.columns.tolist()

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X_train[feature_columns] = scalar.fit_transform(X_train[feature_columns])
test[feature_columns] = scalar.transform(test[feature_columns])

#  RFECV -----------------------------------------------------------------------
from sklearn import tree, linear_model, ensemble
import xgboost as xgb
from sklearn import model_selection # RFE
from sklearn import feature_selection # RFE
cv_split = model_selection.KFold(n_splits = 5, random_state = SEED)

start_total = time.perf_counter() #https://docs.python.org/3/library/time.html#time.perf_counter

estimators = {
    'lr': linear_model.LinearRegression(),
    'rfr': ensemble.RandomForestRegressor(random_state = SEED),
    'dtree': tree.DecisionTreeRegressor(random_state = SEED),
    'xgb': xgb.XGBRegressor()
    }
estimators_RFECV = {}
for estimatorName, estimator in estimators.items():
    start = time.perf_counter()
    rfecv = feature_selection.RFECV(estimator = estimator, step = 1, scoring = scoringMethod, cv = cv_split)
    rfecv_result = rfecv.fit(X_train, y_train)
    run = time.perf_counter() - start

    print('{} runs for {:.2f} seconds.'.format(estimator.__class__.__name__, run))
    estimators_RFECV[estimatorName] = rfecv_result
run_total = time.perf_counter() - start_total
print('Total running time was {:.2f} minutes.'.format(run_total/60))

for model_name, model in estimators_RFECV.items():
    evaluate_model(model, model_name)

model_name = 'lr'
model = estimators_RFECV[model_name]
evaluate_model(model, model_name)
# y_test_pred = submit_prediction_with_model(model, submission_name = '3_rfc_rfecv'+'.csv')

plt.plot(y_test_pred, '.')
y_test_pred_clip = y_test_pred.clip(10300, 10500)
plt.plot(y_test_pred_clip, '.')
# submit_prediction(y_test_pred_clip, submission_name = '3_rfc_rfecv_clipp'+'.csv')


y_train_pred = model.predict(X_train)
plt.scatter(y_train, y_train_predict)
plt.plot(y_train, '.')
plt.plot(y_train_predict, '.')
evaluate_result(y_train, y_train_pred)

# submission_names: ['2_lr_rfecv', '3_rfc_rfecv', '4_dtree_rfecv', '5_xgb_rfecv']
model_evaluations
# {'dtree': 209.93631292101065,
#  'lr': 122.06448955137155,
#  'rfc': 151.0216363954699,
#  'xgb': 151.38256206976766}

# # Grid Search ------------------------------------------------------------------
# from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process, kernel_ridge
# import xgboost as xgb
# from sklearn import model_selection
# cv_split = model_selection.KFold(n_splits = 5, random_state = SEED)
#
# # cv_split = model_selection.ShuffleSplit(n_splits = 5, test_size = .3, train_size = .6, random_state = SEED)
# grid_loss = ['squared_loss', 'huber']
# grid_penalty = ['l2', 'l1']
# grid_alpha = [0.0001, 0.001, 0.01, 0.1] # [0.0001, 0.001, 0.01]
# grid_iter = [50, 100,500] # [100,300,500,1000]
# grid_n_estimator = [10, 50, 100, 300]
# grid_ratio = [.1, .25, .5, .75, 1.0]
# grid_learn = [.01, .03, .05, .1, .25]
# grid_max_depth = [2, 4, 6, 8, 10, None]
# grid_min_samples = [5, 10, .03, .05, .10]
# grid_bool = [True, False]
# grid_seed = [SEED]
#
#
# estimators = {
#     'sgd': {'model': linear_model.SGDRegressor(random_state = SEED,
#                                                 loss = 'squared_loss',
#                                                 penalty = 'l2',
#                                                 alpha = 0.01,
#                                                 max_iter = 500,
#                                                 warm_start = True), \
#             'param_grid': [{'random_state':grid_seed,
#                             'loss': ['squared_loss'],
#                             'penalty': ['l2'],
#                             'alpha': [0.01] ,
#                             'max_iter':[500],
#                             'warm_start': [True],
#             }]},
#     'knn': {'model': neighbors.KNeighborsRegressor(n_neighbors = 50,
#                                                     weights = 'distance',
#                                                     algorithm = 'auto'), \
#             'param_grid': [{'n_neighbors':[50], # [1,2,3,4,5,6,7]
#                             'weights': ['distance'], # ['uniform', 'distance']
#                             'algorithm': ['auto'], # ['auto', 'ball_tree', 'kd_tree', 'brute']
#             }]},
#     'ElasticNet': {'model': linear_model.ElasticNet(random_state = SEED,
#                                                     alpha = 0.01,
#                                                     max_iter = 100,
#                                                     warm_start = True), \
#             'param_grid': [{'random_state':grid_seed,
#                             'alpha': [0.01],
#                             'max_iter': [100],
#                             'warm_start': [True],
#             }]},
#     'svr': {'model': svm.SVR(C = 1000, # [1,10,100]
#                             gamma = 0.01, # [0.0001, 0.001, 0.01, 0.1]
#                             max_iter = -1), \
#             'param_grid': [{'C': [1000], # [1,10,100]
#                             'gamma': [0.01], # [0.0001, 0.001, 0.01, 0.1]
#                             'max_iter': [-1], # [50, 100,500]
#             }]},
#     'ada': {'model': ensemble.AdaBoostRegressor(random_state = SEED,
#                                                 n_estimators = 300,
#                                                 learning_rate = 0.01,
#                                                 loss = 'square'), \
#             'param_grid': [{'n_estimators': [300], # [10, 50, 100, 300]
#                         'learning_rate': [0.01], # [.01, .03, .05, .1, .25]
#                         'loss': ['square']  # ['linear', 'square', 'exponential']
#             }]},
#     'br': {'model': ensemble.BaggingRegressor(random_state = SEED,
#                                                 n_estimators = 100), \
#             'param_grid': [{'n_estimators': [10, 50, 100, 300], # [10, 50, 100, 300]
#             }]},
#     'etr': {'model': ensemble.ExtraTreesRegressor(random_state = SEED,
#                                                     n_estimators = 6,
#                                                     min_samples_split = 500), \
#             'param_grid': [{'n_estimators': [300, 500], # [10, 50, 100, 300]
#                             'min_samples_split': [4, 6, 10] # [2, 4, 6, 8, 10]
#             }]},
#     'gbr': {'model': ensemble.GradientBoostingRegressor(random_state = SEED,
#                                                         criterion = 'friedman_mse',
#                                                         n_estimators = 2000,
#                                                         learning_rate = 0.01,
#                                                         min_samples_split = 6), \
#             'param_grid': [{'n_estimators': [2000], # [10, 50, 100, 300]
#                         'learning_rate': [0.01], # [.01, .03, .05, .1, .25]
#                         'min_samples_split': [6], # [2, 4, 6, 8, 10]
#                         'criterion': ['friedman_mse'],  # ['friedman_mse', 'mse', 'mae']
#             }]},
#     'rfr': {'model': ensemble.RandomForestRegressor(random_state = SEED,
#                                                     n_estimators = 1000,
#                                                     min_samples_leaf = 6), \
#             'param_grid': [{
#                         'n_estimators': [1300], # [10, 50, 100, 300]
#                         'min_samples_leaf': [6], # [2, 4, 6, 8, 10]
#             }]},
#     # 'xgb': {'model': xgb.XGBRegressor(), \
#     #         'param_grid': [{
#     #         }]},
#     }
#
# # # Tuning
# # if 'estimators_GS' not in globals(): estimators_GS = {}
# # gs_start_time = time.time()
# # model_name = 'rfr'
# # grid = model_selection.GridSearchCV(estimator = estimators[model_name]['model'], param_grid = estimators[model_name]['param_grid'], cv = cv_split, scoring = scoringMethod, verbose=1, n_jobs=-1)
# # grid_result = grid.fit(X_train, y_train)
# # estimators_GS[model_name] = grid_result
# # run_time = time.time() - gs_start_time
# # print("Best: %f using %s" % (get_rmse(grid_result.best_score_), grid_result.best_params_))
# # check_grid_result(grid_result)
# # print('%s took %0.2f min to tune'%(model_name, (time.time() - gs_start_time)/60))
# # model = estimators[model_name]['model'].set_params(**estimators_GS[model_name].best_params_)
# # evaluate_model(model, model_name = model_name)
#
# estimators_GS = {}
# for model_name, model in estimators.items():
#     estimators_GS[model_name] = model['model']
#     evaluate_model(model['model'], model_name)
#
# model_evaluations


# Get meta feature
ntrain = train.shape[0]
ntest = test.shape[0]
NFOLDS = 5 # set folds for out-of-fold prediction
from sklearn.cross_validation import KFold
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)

def get_oof(model, X_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = X_train[train_index]
        y_tr = y_train[train_index]
        x_te = X_train[test_index]

        model.fit(x_tr, y_tr)

        oof_train[test_index] = model.predict(x_te)
        oof_test_skf[i, :] = model.predict(x_test)
        # print('Fold %d done' % (i))

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# Tune keras -------------------------------------------------------------------
np.random.seed(SEED)
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Input, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras import optimizers
# Reference: https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/

def get_callbacks_list(log_label):
    # checkpoint = ModelCheckpoint('%s/model.{epoch:02d}-{val_loss:.2f}.h5'%(model_path), monitor='val_loss', verbose=1,save_best_only=True,save_weights_only=False, mode='auto',period=1)
    tensorboard = TensorBoard(log_dir="logs/{}".format(log_label))
    earlystopping = EarlyStopping(monitor='val_loss', patience=2, verbose=0)
    # callbacks_list = [earlystopping, checkpoint, tensorboard]
    callbacks_list = [earlystopping, tensorboard]
    return callbacks_list

def model():
    model = Sequential()
    model.add(Dense(32, input_dim=X_train.shape[1], activation='relu')) # basic 'relu'
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation = 'relu'))
    model.compile(loss='mse', optimizer='RMSprop', metrics=['mse']) # basic 'RMSprop'
    return model

# optimizer = ['RMSprop', 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
# activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'linear']



bins = np.linspace(10110, 11340, 10)
y_binned = np.digitize(y_train, bins)
train_x, val_x, train_y, val_y = train_test_split(X_train, y_train, train_size = 0.7, test_size = 0.3, stratify = y_binned, random_state = SEED)
plt.plot(train_y, '.')
plt.plot(val_y, '.')

EPOCH = 100
BATCH_SIZE = 32
estimator = model()
estimator.fit(train_x, train_y, epochs = EPOCH, batch_size = BATCH_SIZE, shuffle = False,  \
                verbose = 1, callbacks = get_callbacks_list(log_label = '5_bs32_3dense32161'), validation_data = (val_x, val_y))


evaluate_result(estimator.predict(val_x).clip(10100, 11200), val_y)
evaluate_result(estimator.predict(train_x).clip(10100, 11200), train_y)


y_train_pred = estimator.predict(X_train).clip(10000, 11500)
plt.plot(y_train_pred, '.')
plt.plot(y_train, '.')


y_val_pred = estimator.predict(val_x)
plt.plot(y_val_pred, '.')
plt.plot(val_y.tolist(), '.')
evaluate_result(val_y, y_val_pred)

y_test_pred = submit_prediction_with_model(estimator, '7_knn_ver4_clipXtarget'+'.csv')
plt.plot(y_test_pred, '.')
y_test_pred_clip = y_test_pred.clip(10200, 10600) # Best clip(10200, 10500)
plt.plot(y_test_pred_clip, '.')
submit_prediction(y_test_pred_clip, '7_knn_ver4_clip_10200_10600'+'.csv')



# Tune RNN -------------------------------------------------------------------


# # Assemble =====================================================================
# oof_train_meta_feature = []
# oof_test_meta_feature = []
#
# estimators_GS.keys()
# estimators_RFECV.keys()
#
#
# for modelName, model in estimators_RFECV.items():
#     oof_train, oof_test = get_oof(model, X_train.values, y_train.values, test.values)
#     oof_train_meta_feature.append(oof_train)
#     oof_test_meta_feature.append(oof_test)
#     print('%s score: %d'%(modelName, evaluate_result(oof_train, y_train)))
#
# for modelName, model in estimators_GS.items():
#     oof_train, oof_test = get_oof(model, X_train.values, y_train.values, test.values)
#     oof_train_meta_feature.append(oof_train)
#     oof_test_meta_feature.append(oof_test)
#     print('%s score: %d'%(modelName, evaluate_result(oof_train, y_train)))
#
# oof_train_meta_feature = np.hstack(oof_train_meta_feature)
# oof_test_meta_feature = np.hstack(oof_test_meta_feature)
# np.save(os.path.join(processed_data_path, 'oof_train_meta_feature.npy'), oof_train_meta_feature)
# np.save(os.path.join(processed_data_path, 'oof_test_meta_feature.npy'), oof_test_meta_feature)

KNN oof meta feature
X_train.head()
test.head()
oof_train = estimator.predict(X_train)
oof_test = estimator.predict(test)

plt.plot(oof_train.tolist(), '.')
plt.plot(y_train, '.')


# Get meta feature
ntrain = train.shape[0]
ntest = test.shape[0]
NFOLDS = 10 # set folds for out-of-fold prediction
from sklearn.cross_validation import KFold
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)


def get_oof_knn(X_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = X_train[train_index]
        y_tr = y_train[train_index]
        x_te = X_train[test_index]
        y_te = y_train[test_index]

        estimator = model()
        estimator.fit(x_tr, y_tr, epochs = EPOCH, batch_size = BATCH_SIZE, \
                        verbose = 1, callbacks = get_callbacks_list(log_label = 'oof'), validation_data=(x_te, y_te))
        estimator.predict(x_te)

        oof_train[test_index] = estimator.predict(x_te).reshape(-1)
        oof_test_skf[i, :] = estimator.predict(x_test).reshape(-1)
        # print('Fold %d done' % (i))

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

oof_train, oof_test = get_oof_knn(X_train.values, y_train.values, test.values)
plt.plot(oof_test)

oof_train_meta_feature = np.load(os.path.join(processed_data_path, 'oof_train_meta_feature.npy'))
oof_test_meta_feature = np.load(os.path.join(processed_data_path, 'oof_test_meta_feature.npy'))

oof_train_meta_feature = np.concatenate([oof_train_meta_feature, oof_train], axis = 1)
oof_test_meta_feature = np.concatenate([oof_test_meta_feature, oof_test], axis = 1)
# np.save(os.path.join(processed_data_path, 'oof_train_meta_feature.npy'), oof_train_meta_feature)
# np.save(os.path.join(processed_data_path, 'oof_test_meta_feature.npy'), oof_test_meta_feature)


# 126.88241785442068
# lr score: 126
# 151.38256206976766
# xgb score: 151
# 126.4473099621525
# sgd score: 126
# 178.03969554628864
# knn score: 178
# 126.74839579944036
# ElasticNet score: 126
# 122.15586472016975
# svr score: 122
# 184.54848276351473
# ada score: 184
# 183.7918715073833
# br score: 183
# 174.83611249035002
# etr score: 174
# 143.02430018685985
# gbr score: 143
# 177.53708718773962
# rfr score: 177

train_x, val_x, train_y, val_y = train_test_split(oof_train_meta_feature, y_train, train_size = 0.7, test_size = 0.3, stratify = y_binned, random_state = SEED)

from sklearn import linear_model
import xgboost as xgb
meta_model = linear_model.LinearRegression()
meta_model = xgb.XGBRegressor()


meta_model.fit(train_x, train_y)
y_train_pred = meta_model.predict(train_x)
y_val_pred = meta_model.predict(val_x)
evaluate_result(y_train_pred, train_y)
evaluate_result(y_val_pred, val_y)

from sklearn.metrics import r2_score
r2_score(y_val_pred, val_y)
r2_score(y_train_pred, train_y)

plt.plot(y_train_pred, '.')
plt.plot(train_y.tolist(), '.')

plt.plot(y_val_pred, '.')
plt.plot(val_y.tolist(), '.')

meta_model.fit(oof_train_meta_feature, y_train)
y_train_pred = meta_model.predict(oof_train_meta_feature)
evaluate_result(y_train_pred, y_train)
y_test_pred = meta_model.predict(oof_test_meta_feature)
submit_prediction(y_test_pred, '9_stacking_with_knn_oof10_epoch45_val_xgb.csv')


plt.plot(y_test_pred, '.')
y_test_pred_clip = y_test_pred.clip(10200, 10550)
plt.plot(y_test_pred_clip, '.')
submit_prediction(y_test_pred_clip, submission_name = '8_stacking_with_knn_lr_clip_10200_10550'+'.csv')


best_submission = pd.read_csv(os.path.join(submission_path, '7_stacking_clippp'+'.csv'))
best_submission.describe()
# 10200, 10500, 6_stacking_clippp

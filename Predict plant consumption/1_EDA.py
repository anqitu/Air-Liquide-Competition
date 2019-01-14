import pandas as pd
import numpy as np
import time
script_start_time = time.time()

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
%matplotlib inline
plt.rcParams["figure.figsize"] = 8,6
sns.set(rc={'figure.figsize':(8,6)})
plt.style.use('fivethirtyeight')


# Settings
project_path = os.getcwd()
data_path = os.path.join(project_path, 'DataFile')
raw_data_path = os.path.join(data_path, 'RawData')
processed_data_path = os.path.join(data_path, 'ProcessedData')
meta_feature_dir = os.path.join(data_path, 'MetaData')
submission_path = os.path.join(project_path, 'Submission')

evaluation_method = 'mse'
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

# 1. Check data for missing values =============================================
# @NOTE Change columns of traindata manually
train = pd.read_csv(os.path.join(raw_data_path, 'Train.csv'))
test = pd.read_csv(os.path.join(raw_data_path, 'Test.csv'))

train.head()
test.head()


train = train.iloc[1:]
train = train.reset_index().drop(columns = ['index'])
train_columns = train.columns.tolist()
test_columns = test.columns.tolist()

for i in range(len(test_columns)):
    test.rename(columns={test_columns[i]: train_columns[i]}, inplace = True)

# Missing values in train for Liquid Nitrogen, Liquid Oxygen, Liquid Argon, Cooling Water, total 250
# Missing values in test for 2004 rows
check_data(test)
test['has_NA'] = test.isnull().any(axis=1)
plt.plot(test['has_NA'], '.') # missing values are clustered at the end
test = test[test.has_NA != True].drop(columns = ['has_NA'])

check_data(train)
train['has_NA'] = train.isnull().any(axis=1)
plt.plot(train['has_NA'], '.') # missing values are clustered

train_no_NA = train[train['has_NA'] == False].drop(columns = 'has_NA')
train_has_NA = train[train['has_NA'] == True].drop(columns = 'has_NA')

# Some invalid cells filled with '[-11059] No Good Data For Calculation'
invalid_row = []
for index, row in train_no_NA.iterrows():
    try:
        float(row['Power'])
    except:
        invalid_row.append(index)
len(invalid_row)
train_no_NA.iloc[invalid_row].head()
train_no_NA_no_invalid = train_no_NA[train_no_NA['Power'] != '[-11059] No Good Data For Calculation']
train_no_NA_no_invalid.shape

feature_columns = train_no_NA_no_invalid.columns.tolist()
feature_columns_no_time = train_no_NA_no_invalid.columns.tolist()[1:]
feature_columns
# ['TimeStamp',
#  'Gas Oxygen',
#  'Liquid Nitrogen',
#  'Liquid Oxygen',
#  'Liquid Argon',
#  'Cooling Water',
#  'Power']

train_no_NA_no_invalid[feature_columns_no_time] = train_no_NA_no_invalid[feature_columns_no_time].astype(float)
train_no_NA_no_invalid.info()
train_no_NA_no_invalid.describe()


train_fix = train_no_NA_no_invalid[:]
test_fix = test[:]
train_no_NA_no_invalid.shape


# 2. Distribution of each feature (take out outliersif needed) =================
train_fix.describe()
test_fix.describe()


#  'Gas Oxygen'
sns.distplot(train_fix['Gas Oxygen'], color='Red')
sns.distplot(test_fix['Gas Oxygen'], color='Blue')

train_fix[train_fix['Gas Oxygen'] < 15000].shape
sns.distplot(train_fix[train_fix['Gas Oxygen'] > 18000]['Gas Oxygen'])

train_fix = train_fix[train_fix['Gas Oxygen'] > 15000]


#  'Liquid Nitrogen' @NOTE make outlier in test_fix as clipped
sns.distplot(train_fix['Liquid Nitrogen'], color='Red')
sns.distplot(test_fix['Liquid Nitrogen'], color='Blue')

train_fix[train_fix['Liquid Nitrogen'] > 1.5].shape
test_fix[test_fix['Liquid Nitrogen'] > 1].shape
train_fix[train_fix['Liquid Nitrogen'] > 1.5].head()
train_fix[train_fix['Liquid Nitrogen'] < 1.5].head()

train_fix['Liquid Nitrogen'].corr(train_fix['Power'])
train_fix[train_fix['Liquid Nitrogen'] < 1.5]['Liquid Nitrogen'].corr(train_fix[train_fix['Liquid Nitrogen'] < 1.5]['Power'])
train_fix[train_fix['Liquid Nitrogen'] > 1.5]['Liquid Nitrogen'].corr(train_fix[train_fix['Liquid Nitrogen'] > 1.5]['Power'])
train_fix[(train_fix['Liquid Nitrogen'] > 1.5) & (train_fix['Liquid Nitrogen'] < 600)]['Liquid Nitrogen'].\
    corr(train_fix[(train_fix['Liquid Nitrogen'] > 1.5) & (train_fix['Liquid Nitrogen'] < 600)]['Power'])

plt.scatter(train_fix[(train_fix['Liquid Nitrogen'] > 1.5) & (train_fix['Liquid Nitrogen'] < 100)]['Liquid Nitrogen'], train_fix[(train_fix['Liquid Nitrogen'] > 1.5) & (train_fix['Liquid Nitrogen'] < 100)]['Power'])
sns.distplot(train_fix[train_fix['Liquid Nitrogen'] < 1.5]['Liquid Nitrogen'], color='Red')
sns.distplot(test_fix[test_fix['Liquid Nitrogen'] < 1]['Liquid Nitrogen'], color='Blue')
train_fix = train_fix[train_fix['Liquid Nitrogen'] < 1.5]
test_fix['Liquid Nitrogen'] = test_fix['Liquid Nitrogen'].clip(0,1.5)


#  'Liquid Oxygen'
sns.distplot(train_fix['Liquid Oxygen'], color='Red')
sns.distplot(test_fix['Liquid Oxygen'], color='Blue')

#  'Liquid Argon'
sns.distplot(train_fix['Liquid Argon'], color='Red')
sns.distplot(test_fix['Liquid Argon'], color='Blue')

train_fix[train_fix['Liquid Argon'] < 600].shape
test_fix[test_fix['Liquid Argon'] < 640].shape
sns.distplot(train_fix[train_fix['Liquid Argon'] > 600]['Liquid Argon'], color='Red')
sns.distplot(test_fix['Liquid Argon'], color='Blue')
train_fix = train_fix[train_fix['Liquid Argon'] > 600]


#  'Cooling Water'
sns.distplot(train_fix['Cooling Water'], color='Red')
sns.distplot(test_fix['Cooling Water'], color='Blue')

#  'Power(kWh)'
sns.distplot(train_fix['Power'])


train_fix_fix = train_fix[:]
test_fix_fix = test_fix[:]

# for column in feature_columns_no_time:
#     orginal = train_fix_fix[column]
#     shift = orginal[0:1].tolist() + orginal[:-1].tolist()
#     train_fix_fix[column + ' change'] = orginal - shift
#     print(train_fix_fix[column + ' change'].corr(train_fix_fix['Power']))
# train_fix_fix.head()


train_fix['TimeStamp'] = pd.to_datetime(train_fix['TimeStamp'])
train_fix['Hour'] = train_fix['TimeStamp'].dt.hour
train_fix['dow'] = train_fix['TimeStamp'].dt.dayofweek
sns.boxplot(y = train_fix['Power'], x = train_fix['Hour'])
sns.boxplot(y = train_fix['Power'], x = train_fix['dow'])


# 3. Trend
outliers
outliers = train_fix_fix.iloc[:300][train_fix_fix['Power'] > 11200].index.values
plt.plot(train_fix_fix.iloc[:300][['Power']], '.')


outliers = np.concatenate([outliers,train_fix_fix.iloc[300:2000][train_fix_fix['Power'] > 11000].index.values])
plt.plot(train_fix_fix.iloc[300:2000][['Power']].clip(0,11000), '.')
plt.plot(train_fix_fix.iloc[300:2000][['Power']], '.')

outliers = np.concatenate([outliers,train_fix_fix.iloc[5000:5179][train_fix_fix['Power'] < 11000].index.values])
plt.plot(train_fix_fix.iloc[5000:5179][['Power']], '.')
plt.plot(train_fix_fix[['Power']], '.')


train_fix_fix = train_fix_fix.loc[list(set(train_fix_fix.index).difference(set(outliers)))]
train_fix_fix.index
train_fix_fix = train_fix_fix[train_fix_fix.index.notin(outliers)]
train_fix_fix.loc[list(set(train_fix_fix.index).difference(set(outliers)))][['Power']].iplot()
train_fix_fix.iloc[:2600][['Power']].iplot()
train_fix_fix.iloc[:100][['Power']].iplot()
train_fix_fix.iloc[:100][['Power' + ' change']].iplot()



train_fix_fix[['Gas Oxygen']].iplot()
test.iloc[:2600][['Gas Oxygen']].iplot()
train_fix_fix[['Gas Oxygen']].iplot()
train_fix_fix.iloc[:100][['Gas Oxygen']].iplot()
train_fix_fix.iloc[:100][['Gas Oxygen' + ' change']].iplot()



train_fix_fix[['Liquid Nitrogen']].iplot()
train_fix_fix.iloc[:2600][['Liquid Nitrogen']].iplot()
train_fix_fix.iloc[:100][['Liquid Nitrogen']].iplot()
train_fix_fix.iloc[:100][['Liquid Nitrogen' + ' change']].iplot()



train_fix_fix[['Liquid Oxygen']].iplot()
train_fix_fix.iloc[:2600][['Liquid Oxygen']].iplot()
train_fix_fix.iloc[:100][['Liquid Oxygen']].iplot()
train_fix_fix.iloc[:100][['Liquid Oxygen' + ' change']].iplot()


train_fix_fix[['Liquid Argon']].iplot()
test[['Liquid Argon']].iplot()
train_fix_fix.iloc[:2600][['Liquid Argon']].iplot()
train_fix_fix.iloc[:100][['Liquid Argon']].iplot()
train_fix_fix[['Liquid Argon' + ' change']].iplot()



train_fix_fix[['Cooling Water']].iplot()
test[['Cooling Water']].iplot()
train_fix_fix.iloc[:2600][['Cooling Water']].iplot()
train_fix_fix.iloc[:100][['Cooling Water']].iplot()
train_fix_fix.iloc[:100][['Cooling Water' + ' change']].iplot()



# 4. Correlations ==============================================================
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Correlation Heatmap', y=1.05, size=15)
sns.heatmap(train_fix[feature_columns_no_time].astype(float).corr(),linewidths=0.1,vmax=1.0,
            square=True, cmap=colormap, linecolor='white', annot=True)

corr_column = ['Gas Oxygen', 'Liquid Nitrogen','Liquid Argon', 'Cooling Water']
g = sns.pairplot(train_fix[feature_columns_no_time])

# 5. Benchmark Prediction with train average
test['predictions'] = train['Power'].mean()
test['predictions'].to_csv(os.path.join(submission_path, '0_benchmark_average.csv'), index = False)

# 5. Benchmark Prediction with LR
y_train = train_fix['Power']
X_train = train_fix[feature_columns_no_time].drop(columns = ['Power'])
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
y_train.min()
y_train.max()
bins = np.linspace(10110, 11340, 10)
y_binned = np.digitize(y_train, bins)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size = 0.7, test_size = 0.3, stratify = y_binned, random_state = SEED)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_val_pred = lr.predict(X_val)
y_train_pred = lr.predict(X_train)
evaluate_result(y_val, y_val_pred)
evaluate_result(y_train, y_train_pred)

y_test_pred = lr.predict(test.drop(columns = ['TimeStamp']))
test['predictions'] = y_test_pred
test[['predictions']].to_csv(os.path.join(submission_path, '1_benchmark_lr.csv'), index = False)


train_fix.to_csv(os.path.join(processed_data_path, 'train_fix.csv'), index = False)
test_fix.to_csv(os.path.join(processed_data_path, 'test_fix.csv'), index = False)

train.shape
train_fix.shape
train_fix.isnull().sum()
test.shape

# # Bench mark with all_data mean by ID
# ID_sales_mean = all_data.groupby(['ID'])['MOD_VOLUME_CONSUMPTION'].mean().reset_index()
# test = pd.read_csv(os.path.join(raw_data_path, 'test.csv'))
# test = test.merge(ID_sales_mean, how = 'left')
# test
#
# test['MOD_VOLUME_CONSUMPTION'] = all_data['MOD_VOLUME_CONSUMPTION'].mean()
# all_data['MOD_VOLUME_CONSUMPTION'].mean()
#
# plt.plot(test['MOD_VOLUME_CONSUMPTION'].clip(0,1000000), '.')
# test[['MOD_VOLUME_CONSUMPTION']].to_csv(os.path.join(submission_path, '0_benchmark_all_data_mean_302066.csv'), index = False)
#
#
# train['prediction'] = train['MOD_VOLUME_CONSUMPTION'].mean()
# train = train.merge(ID_sales_mean.rename(columns = {'MOD_VOLUME_CONSUMPTION': 'prediction'}), how = 'left', on = 'ID')
# evaluate_result(train['prediction'].clip(0,5000000), train['MOD_VOLUME_CONSUMPTION'])



# # Deal with NA -----------------------------------------------------------------
# cols_to_shift
# for column in cols_to_shift:
#     col_mean_map = train.groupby([column])[target].mean()
#     print(column)
#     cols_fill_na = [col for col in train.columns if '_'.join((column, 'lag')) in col]
#     print(cols_fill_na)
#     for col_fill_na in cols_fill_na:
#         all_data[col_fill_na] = all_data[col_fill_na].fillna(all_data[column].map(col_mean_map))
#
# ID_mean_map = train.groupby(['ID'])['MOD_VOLUME_CONSUMPTION'].mean()
# all_data_fill_missing_month['MOD_VOLUME_CONSUMPTION'] = all_data_fill_missing_month['MOD_VOLUME_CONSUMPTION'].fillna(all_data_fill_missing_month['ID'].map(ID_mean_map))
# # all_data_fill_missing_month['MOD_VOLUME_CONSUMPTION'] = all_data_fill_missing_month['MOD_VOLUME_CONSUMPTION'].fillna(train[target].mean())
#
# # check_data(all_data_fill_missing_month)
# all_data_fill_missing_month = all_data_fill_missing_month.merge(all_data[['ID', 'GAS', 'MARKET_DOMAIN_DESCR', 'ZIPcode', 'Sum_of_Sales_CR']].drop_duplicates(), how = 'left')
# all_data_fill_missing_month['Sum_of_Sales_CR'].corr(all_data_fill_missing_month['MOD_VOLUME_CONSUMPTION'])
# all_data['Sum_of_Sales_CR'].corr(all_data['MOD_VOLUME_CONSUMPTION'])
#
# train = all_data_fill_missing_month[(all_data_fill_missing_month['Month_cnt'] < 19) | (all_data_fill_missing_month['ID'].isin(only_in_train_ID)) ]
# test = all_data_fill_missing_month[(all_data_fill_missing_month['Month_cnt'] >= 19) & (-(all_data_fill_missing_month['ID'].isin(only_in_train_ID))) ]
# test['MOD_VOLUME_CONSUMPTION'] = np.nan

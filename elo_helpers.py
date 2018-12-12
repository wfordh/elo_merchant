import pandas as pd
import numpy as np

train_dtypes = {
	'feature_1' : np.int32,
	'feature_2' : np.int16,
	'feature_3' : np.int16,
	'target' : np.float64
}

test_dtypes = {
	'feature_1' : np.int16,
	'feature_2' : np.int16,
	'feature_3' : np.int16
}

merch_dtypes = {
	'merchant_group_id' : np.int32,
	'merchant_category_id' : np.int16,
	'subsector_id' : np.int16,
	'active_months_lag3' : np.int16,
	'active_months_lag6' : np.int16,
	'avg_sales_lag3' : np.float64,
	'avg_sales_lag6' : np.float64,
	'avg_sales_lag12' : np.float64,
	'active_months_lag12' : np.int16,
	'city_id' : np.int16,
	'state_id' : np.int16,
	'category_2' : np.float16
}

trans_dtypes = {
	'city_id': np.int16,
	'installments': np.int16,
	'merchant_category_id': np.int16,
	'month_lag': np.int16,
	'category_2': np.float64,
	'state_id': np.int16,
	'subsector_id': np.int16
}


def column_match(df1, df2):
	return df1.columns.intersection(df2.columns).values


from sklearn.metrics import mean_squared_error
def calc_rmse(y_true, y_pred):
	return np.sqrt(mean_squared_error(y_true, y_pred))


# Pandas can actually handle this by default now
# def make_dummies(df, col):
# 	dum = pd.get_dummies(df[col], prefix=col)
# 	return pd.concat([df, dum], axis=1).drop(col, axis=1)


def plot_feature_dists(data, feature):
	fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14,6))
	ftr_str = feature.replace('_', ' ').title()

	sns.boxplot(x=feature, y='target', data=data, ax=ax1)
	ax1.set_title(f'{ftr_str} Boxplots')
	ax1.set_xlabel(f'{ftr_str}', fontsize=12)
	ax2.set_ylabel('Target', fontsize=12)

	sns.violinplot(x=feature, y='target', data=data, ax=ax2)
	ax2.set_xlabel(f'{ftr_str}', fontsize=12)
	ax2.set_ylabel('Target', fontsize=12)
	ax2.set_title(f'{ftr_str} Violin Plots')
	plt.show()


from sklearn.model_selection import KFold

def reg_target_encoding(train, col, targ, func='mean', splits=5):
	""" Computes regularize encoding for a specified function.
	Inputs:
	   train: training dataframe
	   col:
	   targ:
	   func:
	   splits:
	"""
	# YOUR CODE HERE
	kf = KFold(n_splits=5)
	global_enc = train.groupby(col)[targ].agg(func).copy()
	train[col+f'_{func}_enc'] = 0
	for leavein, leaveout in kf.split(train):
		enc = train.iloc[leavein].groupby(col)[targ].agg(func).copy()
		for i in set(train[col]):
			if i not in enc.index:
				enc.loc[i] = global_enc.loc[i].copy()
		train.loc[leaveout,col+f'_{func}_enc'] = train.iloc[leaveout][col].map(enc)
		

def mean_encoding_test(test, train, grp_col, tar_col, func='mean'):
	""" Computes target enconding for test data.
	
	This is similar to how we do validation
	"""
	# YOUR CODE HERE
	func_enc = train.groupby(grp_col)[tar_col].agg(func)
	global_enc = train[tar_col].map(func)

	# wouldn't this just overwrite what I do in reg_target_enc?
	#train[grp_col + "_mean_enc"] = train[grp_col].map(mean_enc) 
	test[grp_col + f'_{func}_enc'] = test[grp_col].map(func_enc)

	#train[grp_col + "_mean_enc"].fillna(global_mean, inplace=True)
	test[grp_col + f'_{func}_enc'].fillna(global_enc, inplace=True)


def aggregate_historical_transactions(history):
    
    history.loc[:, 'purchase_date'] = pd.DatetimeIndex(history['purchase_date']).\
                                      astype(np.int64) * 1e-9
    
    agg_func = {
        'authorized_flag': ['sum', 'mean'],
        'merchant_id': ['nunique'],
        'city_id': ['nunique'],
        'purchase_amount': ['sum', 'median', 'max', 'min', 'std'],
        'installments': ['sum', 'median', 'max', 'min', 'std'],
        'purchase_date': [np.ptp],
        'month_lag': ['min', 'max']
        }
    agg_history = history.groupby(['card_id']).agg(agg_func)
    agg_history.columns = ['hist_' + '_'.join(col).strip() 
                           for col in agg_history.columns.values]
    agg_history.reset_index(inplace=True)
    
    df = (history.groupby('card_id')
          .size()
          .reset_index(name='hist_transactions_count'))
    
    agg_history = pd.merge(df, agg_history, on='card_id', how='left')
    
    return agg_history


def aggregate_new_transactions(new_trans):    
    agg_func = {
        'authorized_flag': ['sum', 'mean'],
        'merchant_id': ['nunique'],
        'city_id': ['nunique'],
        'purchase_amount': ['sum', 'median', 'max', 'min', 'std'],
        'installments': ['sum', 'median', 'max', 'min', 'std'],
        'month_lag': ['min', 'max']
        }
    agg_new_trans = new_trans.groupby(['card_id']).agg(agg_func)
    agg_new_trans.columns = ['new_' + '_'.join(col).strip() 
                           for col in agg_new_trans.columns.values]
    agg_new_trans.reset_index(inplace=True)
    
    df = (new_trans.groupby('card_id')
          .size()
          .reset_index(name='new_transactions_count'))
    
    agg_new_trans = pd.merge(df, agg_new_trans, on='card_id', how='left')
    
    return agg_new_trans


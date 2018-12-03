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
	return mean_squared_error(y_true, y_pred)**0.5


# Pandas can actually handle this by default now
# def make_dummies(df, col):
# 	dum = pd.get_dummies(df[col], prefix=col)
# 	return pd.concat([df, dum], axis=1).drop(col, axis=1)




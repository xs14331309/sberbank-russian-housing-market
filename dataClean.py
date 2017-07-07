import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing
import xgboost as xgb
import matplotlib.pyplot as plt

# load files
train = pd.read_csv('input/fix_train.csv', parse_dates=['timestamp'])
test = pd.read_csv('input/fix_test.csv', parse_dates=['timestamp'])
id_test = test.id

# clean data
bad_index = train[train.life_sq > train.full_sq].index
train.loc[bad_index, "life_sq"] = np.NaN
equal_index = [601, 1896, 2791]
test.loc[equal_index, "life_sq"] = test.loc[equal_index, "full_sq"]
bad_index = test[test.life_sq > test.full_sq].index
test.loc[bad_index, "life_sq"] = np.NaN
bad_index = train[train.life_sq < 5].index
train.loc[bad_index, "life_sq"] = np.NaN
bad_index = test[test.life_sq < 5].index
test.loc[bad_index, "life_sq"] = np.NaN
bad_index = train[train.full_sq < 5].index
train.loc[bad_index, "full_sq"] = np.NaN
bad_index = test[test.full_sq < 5].index
test.loc[bad_index, "full_sq"] = np.NaN
kitch_is_build_year = [13117]
train.loc[kitch_is_build_year, "build_year"] = train.loc[
    kitch_is_build_year, "kitch_sq"]
bad_index = train[train.kitch_sq >= train.life_sq].index
train.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = test[test.kitch_sq >= test.life_sq].index
test.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = train[(train.kitch_sq == 0).values +
                  (train.kitch_sq == 1).values].index
train.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = test[(test.kitch_sq == 0).values +
                 (test.kitch_sq == 1).values].index
test.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = train[(train.full_sq > 210) & (
    train.life_sq / train.full_sq < 0.3)].index
train.loc[bad_index, "full_sq"] = np.NaN
bad_index = test[(test.full_sq > 150) & (
    test.life_sq / test.full_sq < 0.3)].index
test.loc[bad_index, "full_sq"] = np.NaN
bad_index = train[train.life_sq > 300].index
train.loc[bad_index, ["life_sq", "full_sq"]] = np.NaN
bad_index = test[test.life_sq > 200].index
test.loc[bad_index, ["life_sq", "full_sq"]] = np.NaN
train.product_type.value_counts(normalize=True)
test.product_type.value_counts(normalize=True)
bad_index = train[train.build_year < 1500].index
train.loc[bad_index, "build_year"] = np.NaN
bad_index = test[test.build_year < 1500].index
test.loc[bad_index, "build_year"] = np.NaN
bad_index = train[train.build_year > 2017].index
train.loc[bad_index, "build_year"] = np.NaN
bad_index = test[test.build_year > 2017].index
test.loc[bad_index, "build_year"] = np.NaN
bad_index = train[train.num_room == 0].index
train.loc[bad_index, "num_room"] = np.NaN
bad_index = test[test.num_room == 0].index
test.loc[bad_index, "num_room"] = np.NaN
bad_index = [10076, 11621, 17764, 19390, 24007, 26713, 29172]
train.loc[bad_index, "num_room"] = np.NaN
bad_index = [3174, 7313]
test.loc[bad_index, "num_room"] = np.NaN
bad_index = train[(train.floor == 0).values *
                  (train.max_floor == 0).values].index
train.loc[bad_index, ["max_floor", "floor"]] = np.NaN
bad_index = train[train.floor == 0].index
train.loc[bad_index, "floor"] = np.NaN
bad_index = train[train.max_floor == 0].index
train.loc[bad_index, "max_floor"] = np.NaN
bad_index = test[test.max_floor == 0].index
test.loc[bad_index, "max_floor"] = np.NaN
bad_index = train[train.floor > train.max_floor].index
train.loc[bad_index, "max_floor"] = np.NaN
bad_index = test[test.floor > test.max_floor].index
test.loc[bad_index, "max_floor"] = np.NaN
train.floor.describe(percentiles=[0.9999])
bad_index = [23584]
train.loc[bad_index, "floor"] = np.NaN
train.material.value_counts()
test.material.value_counts()
train.state.value_counts()
bad_index = train[train.state == 33].index
train.loc[bad_index, "state"] = np.NaN
test.state.value_counts()

# brings error down a lot by removing extreme price per sqm
train.loc[train.full_sq == 0, 'full_sq'] = 50
train = train[train.price_doc / train.full_sq <= 600000]
train = train[train.price_doc / train.full_sq >= 10000]

# Add month-year
month_year = (train.timestamp.dt.month + train.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
train['month_year_cnt'] = month_year.map(month_year_cnt_map)

month_year = (test.timestamp.dt.month + test.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
test['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (train.timestamp.dt.weekofyear + train.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
train['week_year_cnt'] = week_year.map(week_year_cnt_map)

week_year = (test.timestamp.dt.weekofyear + test.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
test['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month and day-of-week
train['month'] = train.timestamp.dt.month
train['dow'] = train.timestamp.dt.dayofweek

test['month'] = test.timestamp.dt.month
test['dow'] = test.timestamp.dt.dayofweek

# Other feature engineering
train['rel_floor'] = train['floor'] / train['max_floor'].astype(float)
train['rel_kitch_sq'] = train['kitch_sq'] / train['full_sq'].astype(float)

test['rel_floor'] = test['floor'] / test['max_floor'].astype(float)
test['rel_kitch_sq'] = test['kitch_sq'] / test['full_sq'].astype(float)

train.apartment_name = train.sub_area + train['metro_km_avto'].astype(str)
test.apartment_name = test.sub_area + train['metro_km_avto'].astype(str)

train['room_size'] = train['life_sq'] / train['num_room'].astype(float)
test['room_size'] = test['life_sq'] / test['num_room'].astype(float)

rate_2015_q2 = 1
rate_2015_q1 = rate_2015_q2 / .9932
rate_2014_q4 = rate_2015_q1 / 1.0112
rate_2014_q3 = rate_2014_q4 / 1.0169
rate_2014_q2 = rate_2014_q3 / 1.0086
rate_2014_q1 = rate_2014_q2 / 1.0126
rate_2013_q4 = rate_2014_q1 / 0.9902
rate_2013_q3 = rate_2013_q4 / 1.0041
rate_2013_q2 = rate_2013_q3 / 1.0044
# This is 1.002 (relative to mult), close to 1:
rate_2013_q1 = rate_2013_q2 / 1.0104
# maybe use 2013q1 as a base quarter and get rid of mult?
rate_2012_q4 = rate_2013_q1 / 0.9832
rate_2012_q3 = rate_2012_q4 / 1.0277
rate_2012_q2 = rate_2012_q3 / 1.0279
rate_2012_q1 = rate_2012_q2 / 1.0279
rate_2011_q4 = rate_2012_q1 / 1.076
rate_2011_q3 = rate_2011_q4 / 1.0236
rate_2011_q2 = rate_2011_q3 / 1
rate_2011_q1 = rate_2011_q2 / 1.011


# train 2015
train['average_q_price'] = 1

train_2015_q2_index = train.loc[train['timestamp'].dt.year == 2015].loc[
    train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
train.loc[train_2015_q2_index, 'average_q_price'] = rate_2015_q2

train_2015_q1_index = train.loc[train['timestamp'].dt.year == 2015].loc[
    train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
train.loc[train_2015_q1_index, 'average_q_price'] = rate_2015_q1


# train 2014
train_2014_q4_index = train.loc[train['timestamp'].dt.year == 2014].loc[
    train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index
train.loc[train_2014_q4_index, 'average_q_price'] = rate_2014_q4

train_2014_q3_index = train.loc[train['timestamp'].dt.year == 2014].loc[
    train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index
train.loc[train_2014_q3_index, 'average_q_price'] = rate_2014_q3

train_2014_q2_index = train.loc[train['timestamp'].dt.year == 2014].loc[
    train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
train.loc[train_2014_q2_index, 'average_q_price'] = rate_2014_q2

train_2014_q1_index = train.loc[train['timestamp'].dt.year == 2014].loc[
    train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
train.loc[train_2014_q1_index, 'average_q_price'] = rate_2014_q1


# train 2013
train_2013_q4_index = train.loc[train['timestamp'].dt.year == 2013].loc[
    train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index
train.loc[train_2013_q4_index, 'average_q_price'] = rate_2013_q4

train_2013_q3_index = train.loc[train['timestamp'].dt.year == 2013].loc[
    train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index
train.loc[train_2013_q3_index, 'average_q_price'] = rate_2013_q3

train_2013_q2_index = train.loc[train['timestamp'].dt.year == 2013].loc[
    train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
train.loc[train_2013_q2_index, 'average_q_price'] = rate_2013_q2

train_2013_q1_index = train.loc[train['timestamp'].dt.year == 2013].loc[
    train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
train.loc[train_2013_q1_index, 'average_q_price'] = rate_2013_q1


# train 2012
train_2012_q4_index = train.loc[train['timestamp'].dt.year == 2012].loc[
    train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index
train.loc[train_2012_q4_index, 'average_q_price'] = rate_2012_q4

train_2012_q3_index = train.loc[train['timestamp'].dt.year == 2012].loc[
    train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index
train.loc[train_2012_q3_index, 'average_q_price'] = rate_2012_q3

train_2012_q2_index = train.loc[train['timestamp'].dt.year == 2012].loc[
    train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
train.loc[train_2012_q2_index, 'average_q_price'] = rate_2012_q2

train_2012_q1_index = train.loc[train['timestamp'].dt.year == 2012].loc[
    train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
train.loc[train_2012_q1_index, 'average_q_price'] = rate_2012_q1


# train 2011
train_2011_q4_index = train.loc[train['timestamp'].dt.year == 2011].loc[
    train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index
train.loc[train_2011_q4_index, 'average_q_price'] = rate_2011_q4

train_2011_q3_index = train.loc[train['timestamp'].dt.year == 2011].loc[
    train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index
train.loc[train_2011_q3_index, 'average_q_price'] = rate_2011_q3

train_2011_q2_index = train.loc[train['timestamp'].dt.year == 2011].loc[
    train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
train.loc[train_2011_q2_index, 'average_q_price'] = rate_2011_q2

train_2011_q1_index = train.loc[train['timestamp'].dt.year == 2011].loc[
    train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
train.loc[train_2011_q1_index, 'average_q_price'] = rate_2011_q1

train['price_doc'] = train['price_doc'] * train['average_q_price']


mult = 1.054880504
train['price_doc'] = train['price_doc'] * mult
y_train = train["price_doc"]

x_train = train.drop(
    ["id", "timestamp", "price_doc", "average_q_price"], axis=1)
#x_test = test.drop(["id", "timestamp", "average_q_price"], axis=1)
x_test = test.drop(["id", "timestamp"], axis=1)

num_train = len(x_train)
x_all = pd.concat([x_train, x_test])

for c in x_all.columns:
    if x_all[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_all[c].values))
        x_all[c] = lbl.transform(list(x_all[c].values))

x_train = x_all[:num_train]
x_test = x_all[num_train:]


xgb_params = {
    'eta': 0.05,
    'max_depth': 6,
    'subsample': 0.6,
    'colsample_bytree': 1,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)


num_boost_rounds = 422
model = xgb.train(dict(xgb_params, silent=0), dtrain,
                  num_boost_round=num_boost_rounds)

y_predict = model.predict(dtest)
gunja_output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})

###################################################
# train = pd.read_csv('input/train.csv')
# test = pd.read_csv('input/test.csv')
# id_test = test.id

# mult = .969

# y_train = train["price_doc"] * mult + 10
# x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
# x_test = test.drop(["id", "timestamp"], axis=1)

# for c in x_train.columns:
#     if x_train[c].dtype == 'object':
#         lbl = preprocessing.LabelEncoder()
#         lbl.fit(list(x_train[c].values))
#         x_train[c] = lbl.transform(list(x_train[c].values))

# for c in x_test.columns:
#     if x_test[c].dtype == 'object':
#         lbl = preprocessing.LabelEncoder()
#         lbl.fit(list(x_test[c].values))
#         x_test[c] = lbl.transform(list(x_test[c].values))

# xgb_params = {
#     'eta': 0.05,
#     'max_depth': 5,
#     'subsample': 0.7,
#     'colsample_bytree': 0.7,
#     'objective': 'reg:linear',
#     'eval_metric': 'rmse',
#     'silent': 1
# }

# dtrain = xgb.DMatrix(x_train, y_train)
# dtest = xgb.DMatrix(x_test)

# num_boost_rounds = 385  # This was the CV output, as earlier version shows
# model = xgb.train(dict(xgb_params, silent=0), dtrain,
#                   num_boost_round=num_boost_rounds)

# y_predict = model.predict(dtest)
# output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})

train_data = pd.read_csv('input/fix_train.csv')
test_data = pd.read_csv('input/fix_test.csv')
macro_data = pd.read_csv('input/macro.csv')

id_test = test_data['id']
print(id_test)
# Print shapes of training, test, and macro dataframes.
print('Training: ')
print(train_data.shape)
print('Test: ')
print(test_data.shape)
print('Macro: ')
print(macro_data.shape)

"""
# Examine data
train_data.head()
"""

# Downsample by investment type in the training set, as there are differences
# between training/test in that regard (see more details at link below).
# URL:
# https://www.kaggle.com/c/sberbank-russian-housing-market/discussion/32717
train_sub = train_data[train_data.timestamp < '2015-01-01']
train_sub = train_sub[train_sub.product_type == "Investment"]

ind_1m = train_sub[train_sub.price_doc <= 1000000].index
ind_2m = train_sub[train_sub.price_doc == 2000000].index
ind_3m = train_sub[train_sub.price_doc == 3000000].index

train_index = set(train_data.index.copy())

for ind, gap in zip([ind_1m, ind_2m, ind_3m], [10, 3, 2]):
    ind_set = set(ind)
    ind_set_cut = ind.difference(set(ind[::gap]))

    train_index = train_index.difference(ind_set_cut)

train_data = train_data.loc[train_index]

# Split off columns that will be needed later.
train_ids = train_data['id'].values
test_ids = test_data['id'].values
train_prices = train_data['price_doc'].values
train_lprices = np.log1p(train_prices)

train_data.drop(['id', 'price_doc'], axis=1, inplace=True)
test_data.drop(['id'], axis=1, inplace=True)

# Due to issues with multicollinearity, we want to only keep a subset of the
# features from the macro data. This list was generated by calculating/examining
# VIFs through an iterative process, documented below.
# URL: https://www.kaggle.com/robertoruiz/dealing-with-multicollinearity
good_macro_features = ['timestamp', 'balance_trade', 'balance_trade_growth', 'eurrub',
                       'average_provision_of_build_contract', 'micex_rgbi_tr',
                       'micex_cbi_tr', 'deposits_rate', 'mortgage_value',
                       'mortgage_rate', 'income_per_cap', 'rent_price_4.room_bus',
                       'museum_visitis_per_100_cap', 'apartment_build']
good_macro_data = pd.DataFrame(macro_data, columns=good_macro_features)

# Merge good features from macro.csv to training/test data.
n_train = len(train_data.index)
all_tt_data = pd.concat([train_data, test_data])
all_data = pd.merge(all_tt_data, good_macro_data, on='timestamp', how='left')
# print(all_data.shape)

# Fix a couple of values (based on another's work - cited below).
# https://www.kaggle.com/captcalculator/a-very-extensive-sberbank-exploratory-analysis
all_data.loc[all_data.state == 33] = 3
all_data.loc[all_data.build_year == 20052009] = 2007
all_data = all_data[all_data.sub_area != 3]

# FEATURE ENGINEERING
# I. Extracting Features from Timestamps
print('Extracting features from timestamps. . .')

# Extract year, month, day of week, and week of year.
# Taken from: https://www.kaggle.com/bguberfain/naive-xgb-lb-0-317
years = pd.to_datetime(all_data.timestamp, errors='coerce').dt.year
months = pd.to_datetime(all_data.timestamp, errors='coerce').dt.month
dows = pd.to_datetime(all_data.timestamp, errors='coerce').dt.dayofweek
woys = pd.to_datetime(all_data.timestamp, errors='coerce').dt.weekofyear
doys = pd.to_datetime(all_data.timestamp, errors='coerce').dt.dayofyear

# Extract number of sales in month/year combos, week/year combos.
# Taken from: https://www.kaggle.com/bguberfain/naive-xgb-lb-0-317
month_year = (months + years * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
week_year = (woys + years * 100)
week_year_cnt_map = week_year.value_counts().to_dict()

# Add all new features to existing data frame.
all_data['year'] = years
all_data['month'] = months
# all_data['day_of_week'] = dows      # do not see fluctuations on this scale (see below).
# all_data['day_of_year'] = doys      # do not see fluctuations on this
# scale (see below).
all_data['month_year_count'] = month_year.map(month_year_cnt_map)
all_data['week_year_count'] = week_year.map(week_year_cnt_map)

# Drop timestamps.
all_data.drop(['timestamp'], axis=1, inplace=True)

# II. Property-Specific Features
all_data['max_floor'] = all_data[
    'max_floor'].replace(to_replace=0, value=np.nan)
all_data['rel_floor'] = all_data['floor'] / all_data['max_floor'].astype(float)
all_data['rel_kitch_sq'] = all_data[
    'kitch_sq'] / all_data['full_sq'].astype(float)
all_data['rel_life_sq'] = all_data['life_sq'] / \
    all_data['full_sq'].astype(float)
# Corrects for property with zero full_sq.
all_data['rel_life_sq'] = all_data['rel_life_sq'].replace(
    to_replace=np.inf, value=np.nan)
# Does not account for living room, but reasonable enough.
all_data['avg_room_sq'] = all_data['life_sq'] / \
    all_data['num_room'].astype(float)
# Corrects for studios (zero rooms listed).
all_data['avg_room_sq'] = all_data['avg_room_sq'].replace(
    to_replace=np.inf, value=np.nan)


# Replace garbage values in build_year with NaNs, then find average build year
# in each sub_area.
all_data['build_year'] = all_data['build_year'].replace(
    to_replace=[0, 1, 2, 3, 20, 71, 215, 4965], value=np.nan)
mean_by_districts = pd.DataFrame(columns=['district', 'avg_build_year'])
sub_areas_unique = all_data['sub_area'].unique()
for sa in sub_areas_unique:
    temp = all_data.loc[all_data['sub_area'] == sa]
    mean_build_year = temp['build_year'].mean()
    new_df = pd.DataFrame([[sa, mean_build_year]], columns=[
                          'district', 'avg_build_year'])
    mean_by_districts = mean_by_districts.append(new_df, ignore_index=True)

mbd_dis_list = mean_by_districts['district'].tolist()
mbd_dis_full = all_data['sub_area'].tolist()
mbd_aby_np = np.array(mean_by_districts['avg_build_year'])
mbd_aby_full = np.zeros(len(all_data.index))

# (Could find a better way to do this.)
for i in range(len(all_data.index)):
    district = mbd_dis_full[i]
    mbd_aby_full[i] = mbd_aby_np[mbd_dis_list.index(district)]

all_data['avg_build_year'] = mbd_aby_full
all_data['rel_build_year'] = all_data[
    'build_year'] - all_data['avg_build_year']

# III. Categorical Features, Treating NaNs

# Deal with categorical values.
# Adapted from: https://www.kaggle.com/bguberfain/naive-xgb-lb-0-317
df_numeric = all_data.select_dtypes(exclude=['object'])
df_obj = all_data.select_dtypes(include=['object']).copy()
ecology_dict = {'no data': np.nan, 'poor': 1, 'satisfactory': 2, 'good': 3,
                'excellent': 4}


def one_hot_encode(x, n_classes):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
     """
    return np.eye(n_classes)[x]


for c in df_obj:
    factorized = pd.factorize(df_obj[c])
    f_values = factorized[0]
    f_labels = list(factorized[1])
    n_classes = len(f_labels)

    if (n_classes == 2 or n_classes == 3) and c != 'product_type':
        df_obj[c] = factorized[0]
    elif c == 'ecology':
        df_obj[c] = df_obj[c].map(ecology_dict)
    else:
        one_hot_features = one_hot_encode(f_values, n_classes)
        oh_features_df = pd.DataFrame(one_hot_features, columns=f_labels)
        df_obj = df_obj.drop(c, axis=1)
        df_obj = pd.concat([df_obj, oh_features_df], axis=1)

# for c in df_obj:
#    df_obj[c] = pd.factorize(df_obj[c])[0]

all_values = pd.concat([df_numeric, df_obj], axis=1)

# Fill all NaNs with -9999 (hacky fix that should allow Boruta to do it's thing).
#all_values = all_values.fillna(value=-9999)
full_feature_names = list(all_values)


x_train_all = (all_values.values)[:n_train]
y_train_all = train_lprices.ravel()      # Log(price)


x_test = (all_values.values)[n_train:]
x_test_df = pd.DataFrame(x_test, columns=full_feature_names)

lprices_df = pd.Series(train_lprices, name='log_price')

xgb_params = {
    'eta': 0.03,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(x_train_all, y_train_all,
                     feature_names=full_feature_names)
dtest = xgb.DMatrix(x_test, feature_names=full_feature_names)

cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=2000,
                   early_stopping_rounds=20, verbose_eval=50, show_stdv=False)
cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()
#plt.show()


num_boost_rounds = len(cv_output)
model = xgb.train(dict(xgb_params, silent=0), dtrain,
                  num_boost_round=num_boost_rounds)

fig, ax = plt.subplots(1, 1, figsize=(8, 13))
xgb.plot_importance(model, height=0.5, ax=ax)
#plt.show()

y_predict = model.predict(dtest)
y_predict = np.expm1(y_predict)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
# print(output.head())

##################################################################
df_train = pd.read_csv("input/fix_train.csv", parse_dates=['timestamp'])
df_test = pd.read_csv("input/fix_test.csv", parse_dates=['timestamp'])
df_macro = pd.read_csv("input/macro.csv", parse_dates=['timestamp'])

df_train.drop(df_train[df_train["life_sq"] > 7000].index, inplace=True)

mult = 0.969
y_train = df_train['price_doc'].values * mult + 10
id_test = df_test['id']

df_train.drop(['id', 'price_doc'], axis=1, inplace=True)
df_test.drop(['id'], axis=1, inplace=True)

num_train = len(df_train)
df_all = pd.concat([df_train, df_test])
# Next line just adds a lot of NA columns (becuase "join" only works on indexes)
# but somewhow it seems to affect the result
df_all = df_all.join(df_macro, on='timestamp', rsuffix='_macro')
print(df_all.shape)

# Add month-year
month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month and day-of-week
df_all['month'] = df_all.timestamp.dt.month
df_all['dow'] = df_all.timestamp.dt.dayofweek

# Other feature engineering
df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)
df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)

train['building_name'] = pd.factorize(
    train.sub_area + train['metro_km_avto'].astype(str))[0]
test['building_name'] = pd.factorize(
    test.sub_area + test['metro_km_avto'].astype(str))[0]


def add_time_features(col):
    col_month_year = pd.Series(pd.factorize(
        train[col].astype(str) + month_year.astype(str))[0])
    train[
        col + '_month_year_cnt'] = col_month_year.map(col_month_year.value_counts())

    col_week_year = pd.Series(pd.factorize(
        train[col].astype(str) + week_year.astype(str))[0])
    train[
        col + '_week_year_cnt'] = col_week_year.map(col_week_year.value_counts())

add_time_features('building_name')
add_time_features('sub_area')

# Remove timestamp column (may overfit the model in train)
df_all.drop(['timestamp', 'timestamp_macro'], axis=1, inplace=True)


factorize = lambda t: pd.factorize(t[1])[0]

df_obj = df_all.select_dtypes(include=['object'])

X_all = np.c_[
    df_all.select_dtypes(exclude=['object']).values,
    np.array(list(map(factorize, df_obj.iteritems()))).T
]
print(X_all.shape)

X_train = X_all[:num_train]
X_test = X_all[num_train:]


# Deal with categorical values
df_numeric = df_all.select_dtypes(exclude=['object'])
df_obj = df_all.select_dtypes(include=['object']).copy()

for c in df_obj:
    df_obj[c] = pd.factorize(df_obj[c])[0]

df_values = pd.concat([df_numeric, df_obj], axis=1)


# Convert to numpy values
X_all = df_values.values
print(X_all.shape)

X_train = X_all[:num_train]
X_test = X_all[num_train:]

df_columns = df_values.columns


xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(X_train, y_train, feature_names=df_columns)
dtest = xgb.DMatrix(X_test, feature_names=df_columns)

num_boost_rounds = 420  # From Bruno's original CV, I think
model = xgb.train(dict(xgb_params, silent=0), dtrain,
                  num_boost_round=num_boost_rounds)

y_pred = model.predict(dtest)

df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})

############################################################################
first_result = output.merge(df_sub, on="id", suffixes=['_louis', '_bruno'])
first_result["price_doc"] = np.exp(.618 * np.log(first_result.price_doc_louis) +
                                   .382 * np.log(first_result.price_doc_bruno))
result = first_result.merge(gunja_output, on="id",
                            suffixes=['_follow', '_gunja'])

result["price_doc"] = np.exp(.78 * np.log(result.price_doc_follow) +
                             .22 * np.log(result.price_doc_gunja))

result["price_doc"] = result["price_doc"] * 0.9915
result.drop(["price_doc_louis", "price_doc_bruno",
             "price_doc_follow", "price_doc_gunja"], axis=1, inplace=True)
result.head()
result.to_csv('output/result.csv', index=False)

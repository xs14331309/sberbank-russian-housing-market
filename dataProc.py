import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
import xgboost as xgb


color = sns.color_palette()

pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 500)

train_df = pd.read_csv('./input/fix_train.csv')
print(train_df.head())

#x - id y - price
plt.figure(figsize=(8, 6))
plt.scatter(range(train_df.shape[0]), np.sort(train_df.price_doc.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('price', fontsize=12)
plt.show()

#x - price y - num
plt.figure(figsize=(12, 8))
sns.distplot(np.log(train_df.price_doc.values), bins=50, kde=True)
plt.xlabel('price', fontsize=12)
plt.show()

#x - yearmonth y - price
train_df['yearmonth'] = train_df['timestamp'].apply(lambda x: x[:4] + x[5:7])
grouped_df = train_df.groupby(
    'yearmonth')['price_doc'].aggregate(np.median).reset_index()

plt.figure(figsize=(12, 8))
sns.barplot(grouped_df.yearmonth.values,
            grouped_df.price_doc.values, alpha=0.8, color=color[2])
plt.ylabel('Median Price', fontsize=12)
plt.xlabel('Year Month', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


#price and area
ulimit = np.percentile(train_df.price_doc.values, 99.5)
llimit = np.percentile(train_df.price_doc.values, 0.5)
train_df['price_doc'].ix[train_df['price_doc'] > ulimit] = ulimit
train_df['price_doc'].ix[train_df['price_doc'] < llimit] = llimit

col = "full_sq"
ulimit = np.percentile(train_df[col].values, 99.5)
llimit = np.percentile(train_df[col].values, 0.5)
train_df[col].ix[train_df[col] > ulimit] = ulimit
train_df[col].ix[train_df[col] < llimit] = llimit

plt.figure(figsize=(12, 12))
sns.jointplot(x=np.log1p(train_df.full_sq.values),
              y=np.log1p(train_df.price_doc.values), size=10)
plt.ylabel('Log of Price', fontsize=12)
plt.xlabel('Log of Total area in square metre', fontsize=12)
plt.show()

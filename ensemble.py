import pandas as pd
import numpy as np

xgb1 = pd.read_csv('output/xgb1.csv')
xgb2 = pd.read_csv('output/xgb2.csv')
xgb3 = pd.read_csv('output/xgb3.csv')
lgb = pd.read_csv('output/lgb.csv')


first_result = xgb2.merge(xgb3, on="id", suffixes=['_louis', '_bruno'])
first_result["price_doc"] = np.exp(.618 * np.log(first_result.price_doc_louis) +
                                   .382 * np.log(first_result.price_doc_bruno))

# second_result = first_result.merge(lgb, on="id",
#                             suffixes=['_follow', '_gunja'])

# second_result["price_doc"] = np.exp(.78 * np.log(second_result.price_doc_follow) +
#                              .22 * np.log(second_result.price_doc_gunja))

result = first_result.merge(xgb1, on="id",
                            suffixes=['_follow', '_gunja'])

result["price_doc"] = np.exp(.78 * np.log(result.price_doc_follow) +
                             .22 * np.log(result.price_doc_gunja))

result["price_doc"] = result["price_doc"] * 0.9915
result.drop(["price_doc_louis", "price_doc_bruno",
             "price_doc_follow", "price_doc_gunja"], axis=1, inplace=True)
result.head()
result.to_csv('output/result_ensemble.csv', index=False)

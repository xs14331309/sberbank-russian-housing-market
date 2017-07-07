import pandas as pd

tr = pd.read_csv('input/train.csv', index_col='id')
te = pd.read_csv('input/test.csv', index_col='id')
fx = pd.read_excel(
    'input/BAD_ADDRESS_FIX.xlsx').drop_duplicates('id').set_index('id')

tr.update(fx)
te.update(fx)
print('Fix in train: ', tr.index.intersection(fx.index).shape[0])
print('Fix in test : ', te.index.intersection(fx.index).shape[0])

tr.to_csv('input/fix_train.csv')
te.to_csv('input/fix_test.cvs')
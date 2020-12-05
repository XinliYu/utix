import xgboost as xgb
import numpy as np
from xgboost import DMatrix

print(f'xgboost version: {xgb.__version__}')


def get_data(size):
    x = np.random.uniform(0, 5, (size, 10))
    y = np.sum(x ** 2, axis=1) > 100
    return x, y


train_data, train_label = get_data(10000)
dev_data, dev_label = get_data(10000)
test_data, test_label = get_data(10000)

train_dmatrix = DMatrix(train_data, label=train_label)
dev_dmatrix = DMatrix(dev_data, label=dev_label)
test_dmatrix = DMatrix(test_data, label=test_label)

params = {'objective': 'binary:logistic', 'eval_metric': 'error', 'eta': 0.2, 'gamma': 1.5, 'min_child_weight': 1.5, 'max_depth': 5, 'gpu_id': 0, 'tree_method': 'gpu_hist'}

eval_set = [(train_dmatrix, 'train'),
            (test_dmatrix, 'test'),
            (dev_dmatrix, 'dev')]

xgb_model = xgb.train(params, train_dmatrix, num_boost_round=1000, evals=eval_set, early_stopping_rounds=50, verbose_eval=True)

import warnings
from collections import defaultdict
from collections.abc import Iterable
from functools import partial
from itertools import product
from os import path
from random import choice, shuffle, randrange, uniform
from typing import Dict, List, Callable, Tuple, Union, Iterator, Any, Set, Mapping

import joblib
import numpy as np
import pandas as pd
import sklearn.metrics.pairwise as sklearn_pairwise_metrics
import xgboost as xgb
from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from tqdm import tqdm

import utix.argex as argex
import utix.csvex as csvex
import utix.general as gex
import utix.iterex as iterex
import utix.msgex as msgex
import utix.plotex as plotex
from utix.dictex import prioritize_keys
from utix.pathex import ensure_dir_existence, get_main_name
from utix.timex import tic, toc


def incremental_pca(feature_iter, batch_size, n_components=10):
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    for batch in iterex.chunk_iter(it=feature_iter, chunk_size=batch_size, as_list=True):
        ipca.partial_fit(np.array(batch))
    return ipca


def get_avg_rank(score_matrix, weights=None):
    ranks = np.argsort(-np.abs(score_matrix), axis=1)
    if weights is None:
        return np.mean(ranks, axis=0)
    else:
        return np.dot(ranks.T, weights)


# region misc

# endregion

# region sklearn

def get_sklearn_models(**kwargs):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    model_dict = {
        'adaboost': AdaBoostClassifier,
        'logistic': partial(LogisticRegression, n_jobs=-1, max_iter=200, solver='saga'),
        'gp': partial(GaussianProcessClassifier, n_jobs=-1),
        'knn': partial(KNeighborsClassifier, n_jobs=-1),
        'knn_n20': partial(KNeighborsClassifier, n_jobs=-1, n_neighbors=20),
        'random_forest': partial(RandomForestClassifier, n_jobs=-1),
        'random_forest_md03': partial(RandomForestClassifier, n_jobs=-1, max_depth=3),
        'random_forest_md05': partial(RandomForestClassifier, n_jobs=-1, max_depth=5),
        'random_forest_md10': partial(RandomForestClassifier, n_jobs=-1, max_depth=10),
        'random_forest_mss50': partial(RandomForestClassifier, n_jobs=-1, min_samples_split=50),
        "decision_tree_md03": partial(DecisionTreeClassifier, max_depth=3)
    }


class XgBoostSklearnWrapper:
    def __init__(self, params=None, max_rounds=1000, early_stopping_rounds=50, num_parallel_tree=100, max_depth=10, rank_eval_metric='ndcg@10', gpu=None, model_path=None, **kwargs):
        if isinstance(params, str):
            if params == 'rank':
                self._params = {'objective': 'rank:pairwise', 'learning_rate': 0.2, 'min_split_loss': 1.0, 'min_child_weight': 0.1, 'max_depth': max_depth, 'eval_metric': rank_eval_metric}
            elif params == 'rank_ndcg':
                self._params = {'objective': 'rank:ndcg', 'learning_rate': 0.2, 'min_split_loss': 1.0, 'min_child_weight': 0.1, 'max_depth': max_depth, 'eval_metric': rank_eval_metric}
            elif params == 'rank_rf':
                self._params = {'objective': 'binary:logistic', 'learning_rate': 0.2, 'gamma': 1.5, 'min_child_weight': 1.5, 'max_depth': max_depth, 'num_parallel_tree': num_parallel_tree, 'eval_metric': rank_eval_metric}
        elif params is None:
            self._params = params if params else {'objective': 'binary:logistic', 'learning_rate': 0.2, 'gamma': 1.5, 'min_child_weight': 1.5, 'max_depth': max_depth}

        if gpu is not None:
            self._params['gpu_id'] = gpu
            self._params['tree_method'] = 'gpu_hist'
        if kwargs:
            self._params.update(kwargs)

        self._max_rounds = max_rounds
        self._model = None
        self._early_stopping_rounds = early_stopping_rounds
        if model_path is not None:
            self.load_model(model_path)

    def fit(self, X, y, group=None, max_rounds=None, evals: Dict = None, early_stopping_rounds=None, verbose_eval=True):
        if max_rounds is None:
            max_rounds = self._max_rounds
        if early_stopping_rounds is None:
            early_stopping_rounds = self._early_stopping_rounds

        def _proc_input(_X, _y, _group=None):
            if isinstance(_X, (list, tuple)):
                _X = np.array(_X)
            if isinstance(_y, (list, tuple)):
                _y = np.array(_y)
                _y = np.array(_y)
            data = xgb.DMatrix(_X, label=_y)
            if _group is not None:
                if isinstance(_group[0], tuple):
                    _group = [x[1] - x[0] for x in _group]
                data.set_group(_group)
            return data

        train_data = _proc_input(X, y, group)
        if evals is not None:
            evals_has_train = 'train' in evals
            evals = [((_proc_input(*v) if isinstance(v, (list, tuple)) else v), k) for k, v in evals.items()]
            if not evals_has_train:
                evals = [(train_data, 'train')] + evals
        else:
            evals = [(train_data, 'train')]

        self._model = xgb.train(
            params=self._params,
            dtrain=train_data,
            num_boost_round=max_rounds,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval
        )

    def predict_proba(self, data, group=None):
        if isinstance(data, (list, tuple)):
            data = np.array(data)
        data = xgb.DMatrix(data)
        if group is not None:
            data.set_group(group)
        return self._model.predict(data)

    @staticmethod
    def _feature_info_path(model_path):
        return path.join(path.dirname(model_path), f'{get_main_name(model_path)}_features.txt')

    def save_model(self, model_path):
        xgb_model = self._model
        xgb_model.save_model(model_path)
        feature_importance = self._model.get_score(importance_type='gain')
        csvex.write_csv(
            ((name, feat_type, feature_importance.get(name, None)) for name, feat_type in gex.zip__(xgb_model.feature_names, xgb_model.feature_types)),
            output_csv_path=self._feature_info_path(model_path),
            header=('name', 'type', 'importance')
        )

    def load_model(self, model_path):
        self._model = xgb_model = xgb.Booster()
        tic(f'Loading xgboost model from path {model_path}.')
        xgb_model.load_model(model_path)
        feature_info_path = self._feature_info_path(model_path)
        feat_names, feat_types = [], []
        if path.exists(feature_info_path):
            for feat_name, feat_type, _ in csvex.iter_csv(feature_info_path):
                if feat_name == 'None':
                    warnings.warn('invalid feature name as `None`')
                    feat_names = None
                    break

                feat_names.append(feat_name)
                if feat_type != 'None':
                    feat_types.append(feat_type)
        if feat_names:
            xgb_model.feature_names = feat_names
            if feat_types:
                if len(feat_types) == len(feat_names):
                    xgb_model.feature_types = feat_types
                else:
                    warnings.warn(f'expected {len(feat_types)} feature types; got {len(feat_names)}')


# endregion

def find_threshold(pos_sorted: np.array, neg_sorted: np.array, steps=100, use_tqdm=True, above=True):
    max_val = max(pos_sorted[-1], neg_sorted[-1])
    min_val = min(pos_sorted[0], neg_sorted[0])
    ths_to_search = np.linspace(start=min_val + (max_val - min_val) / steps, stop=max_val, num=steps - 1)
    if use_tqdm:
        ths_to_search = tqdm(ths_to_search)

    results = []
    if above:
        for th in ths_to_search:
            num_pos = np.sum(pos_sorted > th)
            num_neg = np.sum(neg_sorted > th)
            results.append((num_pos - num_neg, num_pos, num_neg, th))
    else:
        for th in ths_to_search:
            num_pos = np.sum(pos_sorted < th)
            num_neg = np.sum(neg_sorted < th)
            results.append((num_pos - num_neg, num_pos, num_neg, th))
    results = sorted(results, reverse=True)
    return results[0][-1], results


def normalize_as_prob_dist(nums):
    z = sum(nums)
    return [float(i) / z for i in nums]


def alias_sample_create_lookup(discrete_dist):
    """
    Pre-computes alias sampling lookup table and the thresholds.
    Use this method with the `alias_sample` method for efficient non-uniform discrete sampling.
    IDEA: suppose the size of the discrete distribution is `K`,
        the alias sampling implemented here first tries to scale the probabilities by `K`,
        (i.e. times each probability in the discrete distribution by the size of the distribution),
        and then, geometrically speaking, tries to `flatten` the histogram of the distribution to a `1*K` box.
        After the 'squeeze', on each bar of the histogram at index `i`, we height below `thresholds[i]` corresponds to the original index `i`,
        while the area above the `thresholds[i]` corresponds to another index recorded by `alias_lookup[i]`.
        Refer to https://pandasthumb.org/archives/2012/08/lab-notes-the-a.html for a concrete example.
    :param discrete_dist: Pre-computes alias sampling lookup table and the thresholds for this discrete distribution.
    :return: the alias lookup table and the thresholds.
    """

    # gets the size of the discrete distribution
    dist_size = len(discrete_dist)

    # the `alias_lookup` stores the dimension index
    alias_lookup, thresholds = np.zeros(dist_size, dtype=np.int), np.zeros(dist_size)

    not_full, over_full = [], []
    for i, prob in enumerate(discrete_dist):
        thresholds[i] = dist_size * prob
        (not_full if thresholds[i] < 1.0 else over_full).append(i)

    while len(not_full) > 0 and len(over_full) > 0:
        i, j = over_full.pop(), not_full.pop()
        alias_lookup[j] = i
        thresholds[i] -= 1 - thresholds[j]
        (not_full if thresholds[i] < 1.0 else over_full).append(i)

    return alias_lookup, thresholds


def alias_sample(alias_lookup, thresholds):
    """
    Draws one sample from the discrete distribution represented by the pre-computed alias sampling lookup table and their thresholds.
    :param alias_lookup: the alias lookup table.
    :param thresholds: the alias thresholds.
    :return: one sample from the discrete distribution represented by the pre-computed alias sampling lookup table and their thresholds.
    """
    dist_size = len(alias_lookup)
    i = randrange(dist_size)
    return i if uniform(0, 1) < thresholds[i] else alias_lookup[i]


def sample_index_excluding_one(start: int, end: int, exclusion_index: int, size: int = 1, replace: bool = True):
    choice_list = list(range(start, exclusion_index)) + list(range(exclusion_index + 1, end))
    return np.random.choice(choice_list, size=size, replace=replace)


def sample_index_excluding_range(start: int, end: int, exclusion_range: Tuple[int, int], size=1, replace=True):
    choice_list = list(range(start, exclusion_range[0])) + list(range(exclusion_range[1], end))
    return np.random.choice(choice_list, size=size, replace=replace)


def seq_negative_sample(seq_batch, sample_size, exclusion_index, exclusion_seq, replace=True):
    seq_samples = []
    # print("batch_size:{}".format(len(seq_batch)))
    # print("exclusion_index:{}".format(exclusion_index))
    sampled_seq_idxes = sample_index_excluding_one(length=len(seq_batch),
                                                   exclusion_index=exclusion_index,
                                                   size=sample_size,
                                                   replace=replace)
    # print("sampled_seq_indexes:{}".format(sampled_seq_idxes))
    for sampled_seq_list_idx in sampled_seq_idxes:
        seq_samples.append(choice(seq_batch[sampled_seq_list_idx]))

    return seq_samples


def joint_shuffle_two(a, b):
    combined = list(zip(a, b))
    shuffle(combined)
    a[:], b[:] = zip(*combined)


# region backup code

def _seq_negative_sample(seq_batch, sample_size, exclusion_index, exclusion_seq, replace=True):
    seq_samples = []
    while sample_size != 0:
        sampled_seq_idxes = sample_index_excluding_one(length=len(seq_batch),
                                                       exclusion_index=exclusion_index,
                                                       size=sample_size,
                                                       replace=replace)
        for sampled_seq_list_idx in sampled_seq_idxes:
            sampled_seq = choice(seq_batch[sampled_seq_list_idx])
            if sampled_seq != exclusion_seq:
                seq_samples.append(sampled_seq)
                sample_size -= 1

    return seq_samples


# endregion


# region model evaluation

# region Prediction by Threshold

def binary_predict_pos_by_above_threshold(scores: np.ndarray, score_th, pos_label, neg_label):
    predictions = np.array([neg_label] * len(scores))
    predictions[scores > score_th] = pos_label
    return predictions


def binary_predict_pos_by_above_or_equal_threshold(pos_scores: np.ndarray, threshold, pos_label, neg_label):
    predictions = np.array([neg_label] * len(pos_scores))
    predictions[pos_scores >= threshold] = pos_label
    return predictions


def binary_predict_pos_by_below_threshold(pos_scores: np.ndarray, threshold, pos_label, neg_label):
    predictions = np.array([neg_label] * len(pos_scores))
    predictions[pos_scores < threshold] = pos_label
    return predictions


def binary_predict_pos_by_below_or_equal_threshold(pos_scores: np.ndarray, threshold, pos_label, neg_label):
    predictions = np.array([neg_label] * len(pos_scores))
    predictions[pos_scores <= threshold] = pos_label
    return predictions


# endregion

def get_predefined_model(model_name, return_callable=True, **kwargs):
    common_priority = ['n_jobs', 'criterion']
    xgb_common_priority = ['params', 'gpu', 'objective', 'learning_rate', 'max_depth', 'eval_metric']
    common_defaults = kwargs
    common_defaults.setdefault('n_jobs', -1)
    common_defaults.setdefault('random_state', 0)
    model_dicts = {
        'rf': (
            RandomForestClassifier,
            ['max_depth', 'min_samples_split', 'criterion'] + common_priority,
            common_defaults
        ),
        'dt': (
            DecisionTreeClassifier,
            ['max_depth', 'min_samples_split'] + common_priority,
            common_defaults
        ),
        'gp': (
            GaussianProcessClassifier,
            ['warm_start'] + common_priority,
            common_defaults
        ),
        'knn': (
            KNeighborsClassifier,
            ['n_neighbors'] + common_priority,
            common_defaults
        ),
        'logistic': (
            LogisticRegression,
            ['max_iter', 'solver'] + common_priority,
            common_defaults
        ),
        'xgb': (
            XgBoostSklearnWrapper,
            xgb_common_priority,
            None,
            ['gpu']
        ),
        'xgb_rf': (
            XgBoostSklearnWrapper,
            xgb_common_priority,
            None,
            ['gpu']
        ),
        'xgb_rank_rf': (
            XgBoostSklearnWrapper,
            xgb_common_priority,
            {'params': 'rank_rf'},
            ['gpu']
        ),
        'xgb_rank': (
            XgBoostSklearnWrapper,  # the model class
            xgb_common_priority,  # arguments prioritized for short-naming
            {'params': 'rank'},  # the default argument values
            ['gpu']  # ignore this argument in returned object name
        ),
        'xgb_rank_ndcg': (
            XgBoostSklearnWrapper,
            xgb_common_priority,
            {'params': 'rank_ndcg'},
            ['gpu']
        )
    }
    return argex.fast_init(model_name, callable_dict=model_dicts, return_callable=return_callable, verbose=True)


def get_predefined_eval_funcs(name):
    if name == 'precision':
        return {
            'precision': partial(precision_score, average='binary')
        }
    if name == 'recall':
        return {
            'recall': partial(recall_score, average='binary')
        }
    if name == 'precision_recall':
        return {
            'precision': partial(precision_score, average='binary'),
            'recall': partial(recall_score, average='binary')
        }


def extracts_grouped_data(data: Union[List, Tuple], group_keys, group_sizes, exclusion_keys, group_data: Union[List, Tuple] = None, group_index_start=0):
    if not isinstance(exclusion_keys, set):
        exclusion_keys = set(exclusion_keys)
    out_feature_data = [[] for _ in range(len(data))]
    out_feature_group_data = [[] for _ in range(len(group_data))] if group_data else None
    out_group_keys, out_group_sizes = [], []
    start = group_index_start
    for group_idx, (group_key, group_size) in enumerate(zip(group_keys, group_sizes)):
        end = start + group_size
        if group_key not in exclusion_keys:
            for i, _data in enumerate(data):
                out_feature_data[i].extend(_data[start:end])
            if out_feature_group_data is not None:
                for i, _data in enumerate(group_data):
                    out_feature_group_data[i].append(_data[group_idx])
            out_group_keys.append(group_key)
            out_group_sizes.append(group_size)
        start = end
    return out_feature_data, out_feature_group_data, out_group_keys, out_group_sizes


def bisplit_grouped_data(data: Union[List, Tuple], group_sizes, group_data: Union[List, Tuple] = None, split_ratio=0.8):
    num_groups = len(group_sizes)
    split1_size = int(num_groups * split_ratio)
    group_sizes1 = group_sizes[:split1_size]
    group_sizes2 = group_sizes[split1_size:]

    data_size1 = sum(group_sizes1)
    data_size2 = sum(group_sizes2)
    data1, data2 = [], []
    for i, item in enumerate(data):
        if data_size1 + data_size2 != len(item):
            raise ValueError(f'the {i}th data size is {len(item)}, expected {data_size1 + data_size2}')
        data1.append(item[:data_size1])
        data2.append(item[data_size1:])
    if group_sizes is None:
        gp_data1 = gp_data2 = None
    else:
        gp_data1, gp_data2 = [], []
        for item in group_data:
            gp_data1.append(item[:split1_size])
            gp_data2.append(item[split1_size:])
    return data1, data2, gp_data1, gp_data2, group_sizes1, group_sizes2


def load_model():
    pass


def build_binary_classification_models(models: dict,
                                       data: Union[Callable, Iterator[Tuple[Dict, Dict]]] = None,
                                       eval_funcs: Dict[str, Callable] = None,
                                       overwrite=True,
                                       train_data: Dict = None,
                                       eval_data: Dict = None,
                                       score2pred_func: Callable = binary_predict_pos_by_above_threshold,
                                       pos_label=1,
                                       neg_label=0,
                                       score_th=0.5,
                                       score_th_mode='default',
                                       test_data_filter: Callable = None,
                                       model_save_dir: str = None,
                                       model_name_suffix: str = '',
                                       group_types_to_eval=None,
                                       result_file_path: str = None,
                                       use_existing_models=True,
                                       group_eval_only=False,
                                       group_best_item_eval_funcs=None,
                                       group_label_eval_funcs=None,
                                       print_out=__debug__,
                                       print_ignore=('model', 'train', 'test'),
                                       plot_output_path=None,
                                       is_ranking=False,
                                       runtime_evals=None,
                                       model_loader=None,
                                       replace_nan=None):
    """
    Trains a set of binary classification models on the provided training sets and then test them on the provided test sets.
    :param data:
    :param models: a dictionary; a key is a string as customized model names; a value is a binary classification model class, or a tuple of the model class and their initialization arguments.
                    A model will be initialized for each argument provided
    :param train_data: a dictionary of training data sets keyed by data set names; a model will be trained on each of these training sets.
    :param eval_data: a dictionary of test data sets keyed by names; the values are tuples of features and labels, and optionally the group index;
                        when group index is provided, the evaluation will be also be done on the collection of the top-score predictions from each group.
    :param eval_funcs: provides evaluation functions; each function must take two positional arguments, the first being the labels, and the second being the predictions.
    :param score2pred_func: provides a function that converts scores to a label prediction based given the threshold;
                                taking four arguments, the `scores`, a single score threshold `score_th`, the positive label `pos_label` and the negative label `neg_label`; this function must return predictions based on the given scores and the threshold.
    :param pos_label: the symbol for the positive label; by default it is the integer `1`.
    :param neg_label: the symbol for the negative label; by default it is the integer `0`.
    :param score_th: threshold(s) for the `score` to reach in order to be predicted as positive; if `score_th` is iterable, e.g. tuple, list, or set, then we evaluate for every threshold provided in `score_th`;
                        otherwise, `score_th` is a value we evaluate for just this specified value as the threshold.
    :param score_th_mode: supports 'line_search';
                            in this mode, `score_th` should be a three-tuple specifying the arguments for numpy's `linspace`, i.e. the beginning, the end, and the steps;
                            if `score_th` is specified as a binary tuple, then it is treated as the beginning and the steps for the line search, with the end being the maximum score;
                            if `score_th` is a value, then it is treated as the beginning of the line search, with the end being the maximum score, and the steps being 6.
    :param test_data_filter: a filter applied on the model name, training set name and the test set name to determine if a test should run; returns `False` to skip a test set.
    :param model_save_dir: path to the directory to save the trained models.
    :param result_file_path: path to save the evaluation results.
    :param print_out: `True` if some details of the execution of this method should be printed out on the terminal.
    :param replace_nan: Use one of 'mean', 'median', and 'most_frequent' to replace `np.nan` in the features;
    """
    if not models:
        raise ValueError(msgex.msg_arg_none_or_empty(arg_name='models', extra_msg='no model is specified to build'))
    if not any((data, train_data, eval_data)):
        raise ValueError(msgex.msg_at_least_one_arg_should_avail(arg_names=('data', 'train_data', 'test_data'),
                                                                 extra_msg='no data is provided'))

    if not data:
        data = ((train_data, eval_data),)

    def _fit(_X, _y, _group, _is_ranking, _runtime_evals):
        if _runtime_evals is not None:
            if _is_ranking:
                if _group is None:
                    raise ValueError('groups sizes must be provided for ranking')
                argex.exec__(
                    _model_inst.fit, _X, _y,
                    group=_group, evals=_runtime_evals
                )
            elif _group is None:
                argex.exec__(
                    _model_inst.fit, _X, _y,
                    evals=_runtime_evals
                )
            else:
                _is_ranking = True
                argex.exec__(
                    _model_inst.fit, _X, _y,
                    group=_group, evals=_runtime_evals
                )
        else:
            if _is_ranking:
                if _group is None:
                    raise ValueError('groups sizes must be provided for ranking')
                argex.exec__(
                    _model_inst.fit, _X, _y,
                    group=_group
                )
            elif _group is None:
                _model_inst.fit(_X, _y)
            else:
                _is_ranking = True
                argex.exec__(
                    _model_inst.fit, _X, _y,
                    group=_group
                )
        return _is_ranking

    results, trained_models, data_idx = [], {}, 0
    model_files = {}
    # model_name, model_args = next(iter( models.items()))
    for model_name, model_args in models.items():
        is_threshold = False
        if isinstance(model_args, tuple) and len(model_args) == 2 and isinstance(model_args[0], int):
            is_threshold = True
            model_inst = model_args
        else:
            model_inst = argex.get_obj_from_args(model_args)
            trained_models[model_name] = model_inst

        for train_data, eval_data in (data() if callable(data) else data):
            tic(f'model: {model_name}', verbose=print_out)

            if runtime_evals is not None:
                runtime_evals = {k: eval_data[k] for k in runtime_evals}

            for train_data_name, train_data_tup in train_data.items():

                if len(train_data_tup) == 2:
                    X, y = train_data_tup
                    group = None
                elif len(train_data_tup) == 3:
                    X, y, group = train_data_tup
                else:
                    raise ValueError(f'unexpected train data format for `{train_data_name}`')

                # replaces missing values in the input `X`
                if replace_nan is not None:
                    from sklearn.impute import SimpleImputer
                    if replace_nan in ('mean', 'median', 'most_frequent'):
                        X = SimpleImputer(missing_values=np.nan, strategy=replace_nan).fit(X).transform(X)
                    else:
                        if not isinstance(replace_nan, (float, int)):
                            warnings.warn('replacing np.nan by a non-numeric value; make sure this is intended')
                        X = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=replace_nan).fit(X).transform(X)

                if print_out:
                    gex.hprint_pairs(("train data", train_data_name), ("size", len(train_data_tup[0])))
                _model_inst = model_inst

                if not is_threshold:
                    if model_save_dir:
                        model_save_path = path.join(ensure_dir_existence(model_save_dir),
                                                    f"{model_name}-{train_data_name}{f'-{model_name_suffix}' if model_name_suffix else ''}-{data_idx}.pkl")
                        model_files[(model_name, train_data_name, data_idx)] = model_save_path
                        model_save_path_exists = path.exists(model_save_path)
                        if use_existing_models and model_save_path_exists:
                            if model_loader is not None:
                                _model_inst = model_loader(model_save_path)
                            elif hasattr(_model_inst, 'load_model'):
                                _model_inst.load_model(model_save_path)
                            else:
                                try:
                                    _model_inst = joblib.load(model_save_path)
                                except Exception as err:
                                    gex.eprint_message('unable to load model', model_save_path)
                                    raise err
                        else:
                            if not overwrite and model_save_path_exists:
                                raise ValueError(
                                    f'model file \'{model_save_path}\' already exists; specify the `overwrite` as `True` to overwrite')
                            is_ranking = _fit(X, y, group, is_ranking, runtime_evals)
                            if hasattr(_model_inst, 'save_model'):
                                _model_inst.save_model(model_save_path)
                            else:
                                joblib.dump(_model_inst, model_save_path)

                    else:
                        is_ranking = _fit(X, y, group, is_ranking, runtime_evals)

                eval_binary_classification(
                    model=_model_inst,
                    eval_data=eval_data,
                    eval_funcs=eval_funcs,
                    score2pred_func=score2pred_func,
                    pos_label=pos_label,
                    neg_label=neg_label,
                    score_th=score_th,
                    result_item_base={'model': model_name, 'data_index': data_idx, 'train': train_data_name},
                    result_output_list=results,
                    score_th_mode=score_th_mode,
                    test_data_filter=None if test_data_filter is None else partial(test_data_filter,
                                                                                   model_name=model_name,
                                                                                   train_data_name=train_data_name),
                    group_types_to_eval=group_types_to_eval,
                    print_out=print_out,
                    print_ignore=print_ignore,
                    group_eval_only=group_eval_only,
                    group_best_item_eval_funcs=group_best_item_eval_funcs,
                    group_label_eval_funcs=group_label_eval_funcs,
                    is_ranking=is_ranking
                )
            data_idx += 1
            toc(print_out=print_out)
    if result_file_path:
        csvex.write_dicts_to_csv(row_dicts=results, output_path=result_file_path, append=True)
    if plot_output_path:
        df = pd.DataFrame(results)
        group_types = ('all',)
        if group_types_to_eval is not None:
            group_types += group_types_to_eval
        tests = set(result['test'] for result in results)
        plotex.pd_series_plot(df=df,
                              output_path=plot_output_path,
                              group_cols=('group_type', 'test'),
                              groups=product(group_types, tests),
                              series_col='model',
                              index_col='threshold_pos',
                              value_cols=('precision', 'recall'),
                              xlabel='threshold',
                              plot_args={
                                  'precision': {
                                      'title': 'hist_p@1_est',
                                      'marker': 'o'
                                  },
                                  'recall': {
                                      'title': 'hist_trig_est',
                                      'marker': '^',
                                      'linestyle': 'dashed'
                                  },
                              })
    return model_files


def _parse_test_data_tupe(test_data_name, test_data_tup):
    if len(test_data_tup) == 2:
        test_features, test_labels = test_data_tup
        group_indices = group_types = None
    elif len(test_data_tup) == 3:
        test_features, test_labels, group_indices = test_data_tup
        group_types = None
    elif len(test_data_tup) == 4:
        test_features, test_labels, group_indices, group_types = test_data_tup
    else:
        raise ValueError(f"unexpected test data format for `{test_data_name}`")

    group_sizes = None
    if group_indices is not None and isinstance(group_indices[0], int):  # converts group sizes to group indices
        group_sizes = group_indices
        group_indices = list(tqdm(gex.accumulate_ranges(group_sizes, start=0), desc=''))

    return test_features, test_labels, group_sizes, group_indices, group_types


def eval_binary_classification(
        model,
        eval_data: Dict[str, Tuple],
        eval_funcs: Dict[str, Callable],
        score2pred_func: Callable = binary_predict_pos_by_above_threshold,
        pos_label: int = 1, neg_label: int = 0,
        score_th: Union[float, Tuple, List] = 0.5,
        result_item_base: Dict = None,
        result_output_list: List[Dict] = None,
        score_th_mode='default',
        test_data_filter: Callable = None,
        group_types_to_eval=None,
        group_eval_only=False,
        group_best_item_eval_funcs: Dict[str, Callable] = None,
        group_label_eval_funcs: Dict[str, Callable] = None,
        print_out=True,
        print_ignore=None,
        is_ranking=False,
        score_output_path=None
):
    """
    Evaluates a binary classification model on the provided data sets. Supports threshold line search, and grouped evaluation.
    :param model: the binary classification model to evaluate; must have a method `predict_proba` that takes the features as input, and generate scores for each label.
    :param eval_data: a dictionary of test data sets keyed by names; the values are tuples of features and labels, and optionally the group index;
                        when group index is provided, the evaluation will be also be done on the collection of the top-score predictions from each group.
    :param eval_funcs: provides evaluation functions; each function must take two positional arguments, the first being the labels, and the second being the predictions.
    :param score2pred_func: provides a function that converts scores to a label prediction based given the threshold;
                                taking four arguments, the `scores`, a single score threshold `score_th`, the positive label `pos_label` and the negative label `neg_label`; this function must return predictions based on the given scores and the threshold.
    :param pos_label: the symbol for the positive label; by default it is the integer `1`.
    :param neg_label: the symbol for the negative label; by default it is the integer `0`.
    :param score_th: threshold(s) for the `score` to reach in order to be predicted as positive; if `score_th` is iterable, e.g. tuple, list, or set, then we evaluate for every threshold provided in `score_th`;
                        otherwise, `score_th` is a value we evaluate for just this specified value as the threshold.
    :param result_item_base: if provided, every evaluation result item, which is a dictionary of result names and values, must copy existing fields in this provided dictionary.
    :param result_output_list: if provided, every evaluation result item will be appended to this list.
    :param score_th_mode: supports 'line_search';
                            in this mode, `score_th` should be a three-tuple specifying the arguments for numpy's `linspace`, i.e. the beginning, the end, and the steps;
                            if `score_th` is specified as a binary tuple, then it is treated as the beginning and the steps for the line search, with the end being the maximum score;
                            if `score_th` is a value, then it is treated as the beginning of the line search, with the end being the maximum score, and the steps being 6.
    :param test_data_filter: a filter applied on the test set name; returns `False` to skip a test set given its name.
    :param score_output_path: writes scores of all positive predictions to a file at this path.
    :return: a reference to the `result_output_list` if it provided; or a new result list containing the result items.
    """
    ori_score_th = score_th
    result_output_list = [] if result_output_list is None else result_output_list
    result_item_base = {} if result_item_base is None else result_item_base.copy()

    for test_data_name, test_data_tup in eval_data.items():
        if test_data_tup is None:
            gex.hprint_message("test data not set", test_data_name)
            continue
        if test_data_filter is not None and not test_data_filter(test_data_name):
            if print_out:
                gex.hprint_message("skip test data", test_data_name)
            continue
        if print_out:
            gex.hprint_pairs(("test data", test_data_name), ("size", len(test_data_tup[0])))

        result_item_base['test'] = test_data_name

        test_features, test_labels, group_sizes, group_indices, group_types = _parse_test_data_tupe(test_data_name, test_data_tup)

        if isinstance(model, tuple):
            feature_idx, score_th = model
            scores = np.array([x[feature_idx] for x in test_features])
        else:
            if is_ranking:
                if group_sizes is None:
                    raise ValueError('group sizes must be provided for ranking')
                if hasattr(model, 'predict_proba'):
                    scores = argex.exec__(
                        model.predict_proba,
                        test_features,
                        group=group_sizes
                    )
                else:
                    scores = argex.exec__(
                        model.predict,
                        test_features,
                        group=group_sizes
                    )
            else:
                scores = model.predict_proba(test_features) if hasattr(model, 'predict_proba') else model.predict(test_features)
            if len(scores.shape) > 1:
                scores = scores[:, pos_label]
            if score_th_mode == 'line_search':
                score_th_type = type(ori_score_th)
                max_score = np.max(scores)
                if score_th_type in (tuple, list):
                    score_th = list(ori_score_th)
                    if score_th[0] > max_score:
                        min_score = np.min(scores)
                        score_th[0] = (max_score - min_score) * score_th[0] + min_score
                    if len(score_th) == 1:
                        score_th = np.linspace(score_th[0], max_score, 6)
                    elif len(score_th) == 2:
                        score_th = np.linspace(score_th[0], max_score, score_th[1])
                    else:
                        score_th = np.linspace(score_th[0], score_th[1], score_th[2])
                else:
                    score_th = ori_score_th
                    if score_th > max_score:
                        min_score = np.min(scores)
                        score_th = (max_score - min_score) * score_th + min_score
                    score_th = np.linspace(score_th, max_score, 10)
                score_th = score_th[:-1]

        group_type_eval_enabled = (bool(group_types_to_eval) and group_types is not None)
        result_item_base['grouped'] = False
        # result_item_base['label_type'] = 'item_label'
        if group_type_eval_enabled:
            result_item_base['group_type'] = None

        if not group_eval_only:
            eval_binary_classification_by_score_threshold(
                scores=scores,
                labels=test_labels,
                eval_funcs=eval_funcs,
                score2pred_func=score2pred_func,
                pos_label=pos_label,
                neg_label=neg_label,
                score_th=score_th,
                result_item_base=result_item_base,
                result_output_list=result_output_list,
                print_ignore=print_ignore
            )

        if group_indices is not None:
            best_group_scores, best_group_item_labels, group_labels = [], [], []
            all_group_start = group_indices[0][0]
            result_item_base['grouped'] = True

            if group_type_eval_enabled:
                group_score_labels = {k: ([], [], []) for k in group_types_to_eval}
                group_score_labels['all'] = (best_group_scores, best_group_item_labels, group_labels)
            else:
                group_score_labels = {'all': (best_group_scores, best_group_item_labels, group_labels)}

            for group_idx, (group_start, group_end) in enumerate(group_indices):
                group_start -= all_group_start
                group_end -= all_group_start
                best_score, best_score_label = \
                    sorted(zip(scores[group_start:group_end], test_labels[group_start:group_end]), key=lambda x: x[0],
                           reverse=True)[0]
                best_group_scores.append(best_score)
                best_group_item_labels.append(best_score_label)
                group_label = (best_score_label if best_score_label == pos_label else (
                        pos_label in test_labels[group_start:group_end]))
                group_labels.append(group_label)
                if group_type_eval_enabled:
                    group_type = group_types[group_idx]
                    if group_type in group_score_labels:
                        _best_group_scores, _best_group_item_labels, _group_labels = group_score_labels[group_type]
                        _best_group_scores.append(best_score)
                        _best_group_item_labels.append(best_score_label)
                        _group_labels.append(group_label)

            for k, (best_group_scores, best_group_item_labels, group_labels) in group_score_labels.items():
                if group_type_eval_enabled:
                    result_item_base['group_type'] = k

                if group_best_item_eval_funcs is None and group_label_eval_funcs is None:
                    warnings.warn("no group evaluation functions are specified")
                    group_eval_funcs = {
                        'item_label': eval_funcs,
                        'group_label': eval_funcs
                    }
                    merged_labels = {
                        'item_label': best_group_item_labels,
                        'group_label': group_labels
                    }
                elif group_best_item_eval_funcs is None:
                    group_eval_funcs = {
                        'group_label': group_label_eval_funcs
                    }
                    merged_labels = {
                        'group_label': group_labels
                    }
                elif group_label_eval_funcs is None:
                    group_eval_funcs = {
                        'item_label': group_best_item_eval_funcs
                    }
                    merged_labels = {
                        'item_label': best_group_item_labels
                    }
                else:
                    group_eval_funcs = {
                        'item_label': group_best_item_eval_funcs,
                        'group_label': group_label_eval_funcs
                    }
                    merged_labels = {
                        'item_label': best_group_item_labels,
                        'group_label': group_labels
                    }

                eval_binary_classification_by_score_threshold(
                    scores=np.array(best_group_scores),
                    labels=merged_labels,
                    eval_funcs=group_eval_funcs,
                    score2pred_func=score2pred_func,
                    pos_label=pos_label,
                    neg_label=neg_label,
                    score_th=score_th,
                    result_item_base=result_item_base,
                    result_output_list=result_output_list,
                    print_ignore=print_ignore
                )

                # result_item_base['label_type'] = 'group_label'
                # eval_binary_classification_by_score_threshold(
                #     scores=np.array(best_group_scores),
                #     labels=group_labels,
                #     eval_funcs=eval_funcs,
                #     score2pred_func=score2pred_func,
                #     pos_label=pos_label,
                #     neg_label=neg_label,
                #     score_th=score_th,
                #     result_item_base=result_item_base,
                #     result_output_list=result_output_list,
                #     print_ignore=print_ignore
                # )

    return result_output_list


def eval_binary_classification_by_score_threshold(scores,
                                                  labels: Union[Any, Mapping[str, Any]],
                                                  eval_funcs: Union[
                                                      Mapping[str, Callable], Mapping[str, Mapping[str, Callable]]],
                                                  score2pred_func: Callable = binary_predict_pos_by_above_threshold,
                                                  pos_label: int = 1,
                                                  neg_label: int = 0,
                                                  score_th: Union[float, Tuple, List] = 0.5,
                                                  result_item_base: Dict = None,
                                                  result_with_label_key=False,
                                                  result_output_list: List[Dict] = None,
                                                  print_ignore=None):
    """
    Evaluates the binary classification performance by scores of the positive label and the score threshold(s).
    :param scores: the positive-label scores (e.g. the probability of being positive).
    :param labels: the labels associated with each score in `scores`.
    :param eval_funcs: provides evaluation functions; each function must take two positional arguments, the first being the labels, and the second being the predictions.
    :param score2pred_func: provides a function that converts scores to a label prediction based given the threshold;
                                taking four arguments, the `scores`, a single score threshold `score_th`, the positive label `pos_label` and the negative label `neg_label`; this function must return predictions based on the given scores and the threshold.
    :param pos_label: the symbol for the positive label; by default it is the integer `1`.
    :param neg_label: the symbol for the negative label; by default it is the integer `0`.
    :param score_th: threshold(s) for the `score` to reach in order to be predicted as positive; if `score_th` is iterable, e.g. tuple, list, or set, then we evaluate for every threshold provided in `score_th`;
                        otherwise, `score_th` is a value we evaluate for just this specified value as the threshold.
    :param result_item_base: if provided, every evaluation result item, which is a dictionary of result names and values, must copy existing fields in this provided dictionary.
    :param result_output_list: if provided, every evaluation result item will be appended to this list.
    :return: a reference to the `result_output_list` if it provided; or a new result list containing the result items.
    """
    if result_output_list is None:
        result_output_list = []

    result_item_base = {} if result_item_base is None else result_item_base.copy()

    def _eval_threshold(s_th):
        result_item = result_item_base.copy()
        result_item['threshold_pos'] = s_th
        predictions = score2pred_func(scores=scores, score_th=s_th, pos_label=pos_label, neg_label=neg_label)
        result_item['total'] = len(predictions)
        result_item['trig'] = sum(predictions) / len(predictions)
        if isinstance(labels, Mapping):
            for label_key, _labels in labels.items():
                for eval_name, eval_func in eval_funcs[label_key].items():
                    eval_result = eval_func(_labels, predictions) if pos_label is None else eval_func(_labels,
                                                                                                      predictions,
                                                                                                      pos_label=pos_label)
                    if result_with_label_key:
                        result_item[f'{label_key}_{eval_name}'] = eval_result
                    else:
                        result_item[eval_name] = eval_result
        else:
            for eval_name, eval_func in eval_funcs.items():
                result_item[eval_name] = eval_func(labels, predictions) if pos_label is None else eval_func(labels,
                                                                                                            predictions,
                                                                                                            pos_label=pos_label)

        if print_ignore:
            gex.hprint_pairs(*((k, v) for k, v in result_item.items() if k not in print_ignore))
        else:
            gex.hprint_pairs(*result_item.items())
        result_output_list.append(result_item)

    if isinstance(score_th, Iterable):
        score_ths = score_th
        for score_th in score_ths:
            _eval_threshold(score_th)
    else:
        _eval_threshold(score_th)

    return result_output_list


# endregion


# region accumulative


# endregion

# region improved counter


# endregion

# region avg tracker

class AvgInfo:
    __slots__ = ('sum', 'count')

    def __init__(self, init_value, weight=None):
        if weight is None:
            if isinstance(init_value, AvgInfo):
                self.sum = init_value.sum
                self.count = init_value.count
            else:
                self.sum = init_value
                self.count = 1
        else:
            if isinstance(init_value, AvgInfo):
                self.sum = init_value.sum * weight
                self.count = init_value.count * weight
            else:
                self.sum = init_value * weight
                self.count = weight

    def __add__(self, other):
        if isinstance(other, AvgInfo):
            self.sum += other.sum
            self.count += other.count
        else:
            self.sum += other
            self.count += 1
        return self

    def __sub__(self, other):
        if isinstance(other, AvgInfo):
            self.sum -= other.sum
            self.count -= other.count
        else:
            self.sum -= other
            self.count -= 1
        return self

    def __call__(self):
        return self.sum / self.count

    def add_weighted(self, value, weight):
        if isinstance(value, AvgInfo):
            self.count += value.count * weight
            self.sum += value.sum * weight
        else:
            self.count += weight
            self.sum += value * weight

    def __repr__(self):
        return f'sum: {self.sum}, count: {self.count}, avg: {self()}'

    def __str__(self):
        return str(self())


class AvgTracker(dict):

    def __add__(self, other: Union[dict, Iterator[Union[Tuple[Any, Any], List]]]):
        """
        Adds the value of each key/value pair from the provided iterable to the corresponding value of the same key in this average tracker.
        :param other: must be a dictionary, or an iterable of key/value pairs.
        :return: the current average tracker.
        """
        if isinstance(other, dict):
            for k, v in other.items():
                if k in self:
                    self[k] += v
                else:
                    self[k] = AvgInfo(v)
        else:
            for k, v in other:
                if k in self:
                    self[k] += v
                else:
                    self[k] = AvgInfo(v)

        return self

    def __sub__(self, other: Union[dict, Iterator[Union[Tuple[Any, Any], List]]]):
        """
        For each key/value from the provided iterable,
        if the key exists in this average tracker,
        then subtracts the value of the same key in this average tracker by the value in the key/value pair.
        :param other: must be a dictionary, or an iterable of key/value pairs.
        :return: the current average tracker.
        """
        if isinstance(other, dict):
            for k, v in other.items():
                if k in self:
                    self[k] -= v
        else:
            for k, v in other:
                if k in self:
                    self[k] -= v

        return self

    def __call__(self):
        return {k: v() for k, v in self.items()}


# endregion

# region all-in-one stat

class _Stat(defaultdict):
    def __init__(self, default_factory=int):
        super().__init__(default_factory)

    def count(self, stat_name: str, increase: Any = 1):
        if stat_name in self:
            if isinstance(increase, list):
                cnt_list = self[stat_name]
                for i in range(len(increase)):
                    cnt_list[i] += increase[i]
            elif isinstance(increase, tuple):
                self[stat_name] = tuple(x + y for x, y in zip(self[stat_name], increase))
            else:
                self[stat_name] += increase
        else:
            if isinstance(increase, list):
                self[stat_name] = increase.copy()
            else:
                self[stat_name] = increase

    def average(self, stat_name: str, value: Any, weight: Any = None):
        if stat_name in self:
            if weight is None:
                self[stat_name] += value
            else:
                self[stat_name].add_weighted(value, weight)
        else:
            if weight is None:
                self[stat_name] = AvgInfo(value, 1)
            else:
                self[stat_name] = AvgInfo(value, weight)

    def aggregate(self, stat_name: str, value: Any):
        if stat_name not in self:
            self[stat_name] = []
        self[stat_name].append(value)

    def aggregate_many(self, stat_name: str, values: Iterator):
        if stat_name not in self:
            self[stat_name] = list(values)
        else:
            self[stat_name].extend(values)

    def aggregate_unique(self, stat_name: str, value: Any):
        if stat_name not in self:
            self[stat_name] = set()
        self[stat_name].add(value)

    def aggregate_unique_many(self, stat_name: str, values: Iterator):
        if stat_name not in self:
            self[stat_name] = set(values)
        else:
            self[stat_name].update(values)

    def aggregate_count(self, stat_name: str, value: Any, increase: Any = 1):
        if stat_name not in self:
            self[stat_name] = {}
        d = self[stat_name]
        if value in d:
            d[value] += increase
        else:
            d[value] = increase

    def aggregate_count_many(self, stat_name: str, values: Iterator, increase: Any = 1, increases: Iterator = None):
        if stat_name not in self:
            self[stat_name] = {}
        d = self[stat_name]

        if isinstance(values, Mapping):
            for val_key in values.keys():
                if val_key in d:
                    d[val_key] += values[val_key]
                else:
                    d[val_key] = values[val_key]
        elif increases is None:
            for val_key in values:
                if val_key in d:
                    d[val_key] += increase
                else:
                    d[val_key] = increase
        else:
            for val_key, increase in zip(values, increases):
                if val_key in d:
                    d[val_key] += increase
                else:
                    d[val_key] = increase


class _StatTypes:
    def __init__(self):
        self._names_for_cnt, self._names_for_avg, self._names_for_agg = set(), set(), {}


def _get_stat(self, stat_types: _StatTypes, name: str):
    if name in stat_types._names_for_cnt:
        return self[name]
    elif name in stat_types._names_for_avg:
        return self[name]()
    elif name in stat_types._names_for_agg:
        agg_func = stat_types._names_for_agg[name]
        return agg_func(self[name]) if agg_func else self[name]


def _get_all_stats(self, stat_types: _StatTypes, top_names: list = None, output: dict = None):
    if output is None:
        output = {}
    for k, v in self.items():
        output[k] = _get_stat(self, stat_types, k)

    if top_names is not None:
        prioritize_keys(output, top_names)

    return output


class Stat(_Stat, _StatTypes):
    """
    A convenient all-in-one class for simple statistics including counting, averaging, aggregation and unique aggregation.
    """

    def __init__(self):
        _Stat.__init__(self, int)
        _StatTypes.__init__(self)

    def count(self, stat_name: str, increase: Any = 1) -> None:
        """
        Adds the `increase` to the statistic of the specified `stat_name`.
        See also `aggregate_count` for counting multiple values in one statistic entry.
        :param stat_name: provides a name for this statistic.
        :param increase: adds this value to the statistic; works as long as the `+=` operator is well-defined for the type of `increase`.
        """
        self._names_for_cnt.add(stat_name)
        super().count(stat_name, increase)

    def average(self, stat_name: str, value: Any, weight: Any = None):
        """
        Adds the `value` to the average statistic of the specified `stat_name`.
        :param stat_name: provides a name for this statistic.
        :param value: adds this value to the average statistic; works as long as the `+=` operator is well-defined for the type of `value`.
        """
        self._names_for_avg.add(stat_name)
        super().average(stat_name, value, weight)

    def aggregate(self, stat_name: str, value: Any, agg_func: Callable[[List], Any] = None):
        """
        Adds the `value` to the aggregation statistic of the specified `stat_name`.
        :param stat_name: provides a name for this statistic.
        :param value: adds this value to the average statistic.
        :param agg_func: provides the aggregation function, e.g. Python build-in functions like len, max, min, sum, etc., or any callable that accepts an iterable as the input;
                         this aggregation function will be called when retrieving statistic values via functions `get_stat` and `get_all_stats`;
                         if `None` is provided, then no aggregation will be done, and a list of all previously added values will be returned.
        """
        self._names_for_agg[stat_name] = agg_func
        super().aggregate(stat_name, value)

    def aggregate_many(self, stat_name: str, values: Iterator, agg_func: Callable[[List], Any] = None):
        """
        Aggregate multiple values to the statistic of the specified `stat_name`.
        See `aggregate` for parameter meanings.
        """
        self._names_for_agg[stat_name] = agg_func
        super().aggregate_many(stat_name, values)

    def aggregate_unique(self, stat_name: str, value: Any, agg_func: Callable[[Set], Any] = None):
        """
        Adds the `value` to the aggregation statistic of the specified `stat_name` if it does not currently exist in the aggregation.
        If `agg_func` is `None`, then then no aggregation will be done while retrieving this statistic by `get_stat` and `get_all_stats`, and a set of all previously added values will be returned.
        See `aggregate` for parameter meanings.
        """
        self._names_for_agg[stat_name] = agg_func
        super().aggregate_unique(stat_name, value)

    def aggregate_unique_many(self, stat_name: str, values: Iterator, agg_func: Callable[[Set], Any] = None):
        """
        Aggregate each of the multiple values to the statistic of the specified `stat_name` if they do not currently exist in the aggregation.
        See `aggregate` and `aggregate_unique` for parameter meanings.
        """
        self._names_for_agg[stat_name] = agg_func
        super().aggregate_unique_many(stat_name, values)

    def aggregate_count(self, stat_name: str, value: Any, increase: Any = 1, agg_func: Callable[[Dict], Any] = None):
        """
        Adds the `increase` the 'count' of the `value` in the statistic of the specified `stat_name`.
        :param stat_name: provides a name for this statistic.
        :param value: adds `increase` to the 'count' of this `value`; can be viewed as a secondary key for the counting.
        :param increase: adds this value to 'count' of the `value` in the statistic; works as long as the `+=` operator is well-defined for the type of `increase`.
        """
        self._names_for_agg[stat_name] = agg_func
        super().aggregate_count(stat_name=stat_name, value=value, increase=increase)

    def aggregate_count_many(self, stat_name: str, values: Iterator, agg_func: Callable[[Dict], Any] = None,
                             increase: Any = 1, increases: Iterator = None):
        """
        If `increases` is `None`, then adds the `increase` the 'count' of each of the `values` in the statistic of the specified `stat_name`;
        otherwise, `increases` is used, to zip with `values`, and for each value/increase pair, adds the 'increase' to the 'count' of the 'value'.
        See `aggregate_count` for parameter meanings.
        """
        self._names_for_agg[stat_name] = agg_func
        super().aggregate_count_many(stat_name=stat_name, values=values, increase=increase, increases=increases)

    def get_stat(self, stat_name: str):
        """
        Retrieves the statistic for the specified `stat_name`.
        :param stat_name: provides the statistic name.
        """
        return _get_stat(self, self, stat_name)

    def get_all_stats(self, top_names: list = None, output: dict = None) -> dict:
        """
        Retrieves all statistics in a dictionary, with statistic names as the keys.
        :param top_names: moves statistics of these names to the front of the dictionary (assuming the dictionary is ordered),
                            so that when the result dictionary is printed, these statistics are on the top.
        :param output: if a dictionary is provided, then saves the statistics in this dictionary.
        :return: the dictionary of retrieved statistics; if `output` is not None, then `output` is returned with the retrieved statistics.
        """
        return _get_all_stats(self, self, top_names, output)


class CategorizedStat(dict, _StatTypes):
    """
    A tool for categorized counting; implemented by a two-layer dictionary.
    """

    def __add__(self, other):
        if isinstance(other, CategorizedStat):
            for category, stats in other.items():
                if category not in self:
                    self[category] = stats
                else:
                    for stat_name, value in stats.items():
                        if other.is_count(stat_name):
                            if category == self._overall_count_name:
                                self.count(category=category, stat_name=stat_name, value=value, overall_only=True, update_overall=True)
                            else:
                                self.count(category=category, stat_name=stat_name, value=value, overall_only=False, update_overall=False)
                        elif other.is_avg(stat_name):
                            if category == self._overall_count_name:
                                self.average(category=category, stat_name=stat_name, value=value, overall_only=True, update_overall=True)
                            else:
                                self.average(category=category, stat_name=stat_name, value=value, overall_only=False, update_overall=False)
                        elif other.is_agg(stat_name):
                            if isinstance(value, list):
                                if category == self._overall_count_name:
                                    self.aggregate_many(category=category, stat_name=stat_name, values=value, overall_only=True, update_overall=True)
                                else:
                                    self.aggregate_many(category=category, stat_name=stat_name, values=value, overall_only=False, update_overall=False)
                            elif isinstance(value, set):
                                if category == self._overall_count_name:
                                    self.aggregate_unique_many(category=category, stat_name=stat_name, values=value, overall_only=True, update_overall=True)
                                else:
                                    self.aggregate_unique_many(category=category, stat_name=stat_name, values=value, overall_only=False, update_overall=False)
                            elif isinstance(value, dict):
                                if category == self._overall_count_name:
                                    self.aggregate_count_many(category=category, stat_name=stat_name, values=value, overall_only=True, update_overall=True)
                                else:
                                    self.aggregate_count_many(category=category, stat_name=stat_name, values=value, overall_only=False, update_overall=False)
        return self

    def is_count(self, stat_name):
        return stat_name in self._names_for_cnt

    def is_avg(self, stat_name):
        return stat_name in self._names_for_avg

    def is_agg(self, stat_name):
        return stat_name in self._names_for_agg

    def __init__(self, overall_count_name='all', category_filter=None):
        dict.__init__(self)
        _StatTypes.__init__(self)
        self._overall_count_name = overall_count_name
        self._category_filter = category_filter
        if overall_count_name:
            self[self._overall_count_name] = _Stat(int)

    def __missing__(self, key):
        if key not in self:
            self[key] = _Stat(int)
        return self[key]

    def _count(self, category: str, name: str, value: Any):
        if category not in self:
            d: _Stat = _Stat(int)
            self[category] = d
        else:
            d = self[category]
        d.count(name, value)

    def count(self, category: str, stat_name: str, value: Any = 1, overall_only: bool = False, update_overall=True):
        self._names_for_cnt.add(stat_name)
        if (not overall_only) and category != self._overall_count_name and (self._category_filter is None or category in self._category_filter):
            self[category].count(stat_name, value)

        if update_overall and self._overall_count_name:
            self[self._overall_count_name].count(stat_name, value)

    def count_multi_categories(self, categories: Union[str, Tuple[str, ...], List[str]], stat_name: str, value=1, overall_only=False, update_overall=True):
        self._names_for_cnt.add(stat_name)

        if isinstance(categories, str):
            categories = (categories,)

        if not overall_only:
            for category in categories:
                if category != self._overall_count_name or self._category_filter is None or category in self._category_filter:
                    self[category].count(stat_name, value)

        if update_overall and self._overall_count_name:
            self[self._overall_count_name].count(stat_name, value)

    def average(self, category: str, stat_name: str, value=1, weight=None, overall_only=False, update_overall=True):
        self._names_for_avg.add(stat_name)

        if (not overall_only) and category != self._overall_count_name and (self._category_filter is None or category in self._category_filter):
            self[category].average(stat_name, value, weight)

        if update_overall and self._overall_count_name:
            self[self._overall_count_name].average(stat_name, value, weight)

    def average_multi_categories(self, categories: Union[str, Tuple[str, ...], List[str]], stat_name: str, value=1, overall_only=False, update_overall=True):
        self._names_for_avg.add(stat_name)

        if isinstance(categories, str):
            categories = (categories,)

        if not overall_only:
            for category in categories:
                if category != self._overall_count_name or self._category_filter is None or category in self._category_filter:
                    self[category].average(stat_name, value)

        if update_overall and self._overall_count_name:
            self[self._overall_count_name].average(stat_name, value)

    def aggregate(self, category: str, stat_name: str, value: Any, agg_func: Callable[[List], Any] = None, overall_only=False, update_overall=True):
        self._names_for_agg[stat_name] = agg_func

        if (not overall_only) and category != self._overall_count_name and (self._category_filter is None or category in self._category_filter):
            self[category].aggregate(stat_name, value)

        if update_overall and self._overall_count_name:
            self[self._overall_count_name].aggregate(stat_name, value)

    def aggregate_many(self, category: str, stat_name: str, values: Any, agg_func: Callable[[List], Any] = None, overall_only=False, update_overall=True):
        self._names_for_agg[stat_name] = agg_func

        if (not overall_only) and category != self._overall_count_name and (self._category_filter is None or category in self._category_filter):
            self[category].aggregate_many(stat_name, values)

        if update_overall and self._overall_count_name:
            self[self._overall_count_name].aggregate_many(stat_name, values)

    def aggregate_unique(self, category: str, stat_name: str, value: Any, agg_func: Callable[[Set], Any] = None, overall_only=False, update_overall=True):
        self._names_for_agg[stat_name] = agg_func

        if (not overall_only) and category != self._overall_count_name and (
                self._category_filter is None or category in self._category_filter):
            self[category].aggregate_unique(stat_name, value)

        if update_overall and self._overall_count_name:
            self[self._overall_count_name].aggregate_unique(stat_name, value)

    def aggregate_unique_many(self, category: str, stat_name: str, values: Any, agg_func: Callable[[Set], Any] = None, overall_only=False, update_overall=True):
        self._names_for_agg[stat_name] = agg_func

        if (not overall_only) and category != self._overall_count_name and (self._category_filter is None or category in self._category_filter):
            self[category].aggregate_unique_many(stat_name, values)

        if update_overall and self._overall_count_name:
            self[self._overall_count_name].aggregate_unique_many(stat_name, values)

    def aggregate_count(self, category: str, stat_name: str, value: Any, increase: Any = 1, agg_func: Callable[[Dict], Any] = None, overall_only=False, update_overall=True):
        self._names_for_agg[stat_name] = agg_func

        if (not overall_only) and category != self._overall_count_name and (
                self._category_filter is None or category in self._category_filter):
            self[category].aggregate_count(stat_name=stat_name, value=value, increase=increase)

        if update_overall and self._overall_count_name:
            self[self._overall_count_name].aggregate_count(stat_name=stat_name, value=value, increase=increase)

    def aggregate_count_many(self, category: str, stat_name: str, values: Any, increase: Any = 1, increases: Iterator = None, agg_func: Callable[[Dict], Any] = None, overall_only=False, update_overall=True):
        self._names_for_agg[stat_name] = agg_func

        if (not overall_only) and category != self._overall_count_name and (
                self._category_filter is None or category in self._category_filter):
            self[category].aggregate_count_many(stat_name=stat_name, values=values, increase=increase, increases=increases)

        if update_overall and self._overall_count_name:
            self[self._overall_count_name].aggregate_count_many(stat_name=stat_name, values=values, increase=increase, increases=increases)

    def count_unique(self, category: str, name: str, value, overall_only=False):
        self.aggregate_unique(category=category, stat_name=name, value=value, agg_func=len, overall_only=overall_only)

    def get_all_stats(self, top_names: list = None):
        results = {}
        for category, stat in self.items():
            results[category] = _get_all_stats(self=self[category], stat_types=self, top_names=top_names)

        return results

    def get_stats(self, name: str, categories=None):
        count_dict = {}
        if categories is None:
            for category, counts in self.items():
                count_dict[category] = _get_stat(self[category], self, name)
        else:
            for category in categories:
                count_dict[category] = _get_stat(self[category], self, name)
        return count_dict


# endregion

PAIRWISE_METRICS_MAP = {
    'cos': sklearn_pairwise_metrics.cosine_similarity,
    'dot': lambda x, y: x @ y.T,
    'l2': sklearn_pairwise_metrics.euclidean_distances,
    'l1': sklearn_pairwise_metrics.manhattan_distances,
    'lk': sklearn_pairwise_metrics.laplacian_kernel,
    'pk': sklearn_pairwise_metrics.polynomial_kernel,
    'rbfk': sklearn_pairwise_metrics.rbf_kernel,
    'sk': sklearn_pairwise_metrics.sigmoid_kernel,
    'l1min': lambda x, y: np.min(np.abs(x - y), axis=1),
    'l1max': lambda x, y: np.max(np.abs(x - y), axis=1),
}


def get_pairwise_metrics(X, Y, metric_names, flatten=False, unpack_single=False, decimals=6):
    out = {}
    for metric_name in metric_names:
        if metric_name in PAIRWISE_METRICS_MAP:
            metric_val = PAIRWISE_METRICS_MAP[metric_name](X, Y)
            if unpack_single and not isinstance(metric_val, float) and (
                    metric_val.shape == (1, 1) or metric_val.shape == (1,)):
                metric_val = round(float(metric_val), decimals) if decimals is not None else float(metric_val)
            elif flatten:
                metric_val = [round(float(x), decimals) for x in metric_val.flatten()] if decimals is not None else [
                    float(x) for x in metric_val.flatten()]
            elif decimals is not None:
                metric_val = np.round(metric_val, decimals=decimals)
            out[metric_name] = metric_val
    return out


# region feature generation and management

def get_features(
        data_item: Any,
        feature_gen: Callable,
        feature_key_gen: Callable,
        label_gen: Callable,
        flags: Union[List, Tuple, Callable] = None,
        num_flags=None,
        num_active_flags=None,
        feature_deduplication=False,
        multiple_features=True,
        label_reference=None,
        feat_dim_tracking: set = None,
        label_tracking: set = None,
        **kwargs):
    if feature_deduplication is True:
        feat_key_dd = set()
    elif isinstance(feature_deduplication, set):
        feat_key_dd = feature_deduplication
        feature_deduplication = True

    out = []
    if multiple_features:
        feat_iter = feature_gen(data_item, **kwargs)
    else:
        feat_iter = (feature_gen(data_item, **kwargs),)
    for i, (meta_data, feat) in enumerate(feat_iter):
        if isinstance(meta_data, (list, tuple)):
            feat_key = feature_key_gen(*meta_data) if feature_key_gen else tuple(meta_data)
            if feature_deduplication and (feat_key in feat_key_dd):
                continue
            label = label_gen(label_reference, *meta_data)
        else:
            feat_key = feature_key_gen(meta_data) if feature_key_gen else meta_data
            if feature_deduplication and (feat_key in feat_key_dd):
                continue
            label = label_gen(label_reference, meta_data)
        if flags is None:
            if num_flags is not None:
                _flags = [0] * num_flags
                if num_active_flags is None:
                    _flags[i % num_flags] = 1
                else:
                    _flags[i % num_active_flags] = 1
            else:
                _flags = []
        elif callable(flags):
            _flags = flags(i)
        else:
            _flags = flags
        feat = _flags + feat
        out.append((feat_key, label, feat))
        if label_tracking is not None:
            label_tracking.add(label)
        if feat_dim_tracking is not None:
            feat_dim_tracking.add(len(feat))
            if len(feat_dim_tracking) > 1:
                raise ValueError(f"Got different dimensions '{feat_dim_tracking}'; feature '{feat}' of key {feat_key} has dimension {len(feat)}.")
    return out


def get_grouped_features(
        data_iter: Iterator,
        group_key_gen: Callable,
        label_reference_gen: Callable,
        feature_iter: Callable,
        feature_key_gen: Callable,
        label_gen: Callable,
        flags: Union[List, Tuple, Callable] = None,
        num_flags=None,
        num_active_flags=None,
        feature_deduplication=False,
        feat_dim_tracking=True,
        num_expected_labels=None,
        in_group_shuffle=False,
        **kwargs
):
    all_feat_tups, keyed_group_sizes = [], []
    feat_dim_tracking = set() if feat_dim_tracking else None
    label_tracking = set() if num_expected_labels else None
    for data_item in data_iter:
        group_key = group_key_gen(data_item)
        if isinstance(group_key, (tuple, list)):
            label_reference = label_reference_gen(*group_key)
        else:
            label_reference = label_reference_gen(group_key)
        feat_tups = get_features(
            data_item=data_item,
            feature_gen=feature_iter,
            feature_key_gen=feature_key_gen,
            label_gen=label_gen,
            flags=flags,
            num_flags=num_flags,
            num_active_flags=num_active_flags,
            feature_deduplication=feature_deduplication,
            label_reference=label_reference,
            feat_dim_tracking=feat_dim_tracking,
            label_tracking=label_tracking,
            **kwargs
        )
        keyed_group_sizes.append((group_key, len(feat_tups)))
        if in_group_shuffle:
            feat_tups = shuffle(feat_tups)
        all_feat_tups.extend(feat_tups)
    if len(label_tracking) != num_expected_labels:
        raise ValueError(f'expected number of distinct labels must be {num_expected_labels}; got {label_tracking}')
    return all_feat_tups, keyed_group_sizes

# endregion

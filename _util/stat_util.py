import os
from collections.abc import Iterable
from functools import partial
from os import path
from random import choice, shuffle, randrange, uniform
from typing import Dict, List, Callable, Tuple, Union, Iterator, Any, Set

import numpy as np
from sklearn.externals import joblib
from tqdm import tqdm

import _util.arg_ext as argex
import _util.csv_ext as csvex
import _util.general_ext as gex
from _util.dict_ext import dict_try_div, dict_try_floor_div, prioritize_keys
from _util.general_ext import nonstr_iterable
from _util.time_ext import tic, toc
import _util.msg_ext as msgex

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

def build_binary_classification_models(models: dict, data: Union[Callable, Iterator[Tuple[Dict, Dict]]], eval_funcs: Dict[str, Callable],
                                       overwrite=True, train_data: Dict = None, test_data: Dict = None,
                                       score2pred_func: Callable = binary_predict_pos_by_above_threshold,
                                       pos_label=1, neg_label=0,
                                       score_th=0.5, score_th_mode='default', test_data_filter: Callable = None,
                                       model_saving_dir_path: str = None, model_name_suffix: str = '', result_file_path: str = None,
                                       print_out=__debug__):
    """
    Trains a set of binary classification models on the provided training sets and then test them on the provided test sets.
    :param data:
    :param models: a dictionary; a key is a string as customized model names; a value is a binary classification model class, or a tuple of the model class and their initialization arguments.
                    A model will be initialized for each argument provided
    :param train_data: a dictionary of training data sets keyed by data set names; a model will be trained on each of these training sets.
    :param test_data: a dictionary of test data sets keyed by names; the values are tuples of features and labels, and optionally the group index;
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
    :param model_saving_dir_path: path to the directory to save the trained models.
    :param result_file_path: path to save the evaluation results.
    :param print_out: `True` if some details of the execution of this method should be printed out on the terminal.
    """
    if not models:
        raise ValueError(msgex.msg_arg_none_or_empty(arg_name='models', extra_msg='no model is specified to build'))
    if not any((data, train_data, test_data)):
        raise ValueError(msgex.msg_at_least_one_arg_should_avail(arg_names=('data', 'train_data', 'test_data'), extra_msg='no data is provided'))

    if not data:
        data = ((train_data, test_data),)

    results = []
    data_idx = 0
    for model_name, model_args in models.items():
        for train_data, test_data in (data() if callable(data) else data):
            tic(f'model: {model_name}', verbose=print_out)
            model_inst = argex.get_obj_from_args(model_args)
            for train_data_name, train_data_tup in train_data.items():
                if print_out:
                    gex.hprint_pairs(("train data", train_data_name), ("size", len(train_data_tup[0])))
                model_inst.fit(*train_data_tup)
                if model_saving_dir_path:
                    if not path.exists(model_saving_dir_path):
                        os.makedirs(model_saving_dir_path)
                    joblib.dump(model_inst, path.join(model_saving_dir_path, f"{model_name}-{train_data_name}{f'-{model_name_suffix}' if model_name_suffix else ''}.pkl"))
                eval_binary_classification(
                    model=model_inst,
                    test_data=test_data,
                    eval_funcs=eval_funcs,
                    score2pred_func=score2pred_func,
                    pos_label=pos_label,
                    neg_label=neg_label,
                    score_th=score_th,
                    result_item_base={'model': model_name, 'data_index': data_idx, 'train': train_data_name},
                    result_output_list=results,
                    score_th_mode=score_th_mode,
                    test_data_filter=None if test_data_filter is None else partial(test_data_filter, model_name=model_name, train_data_name=train_data_name),
                    print_out=print_out
                )
            data_idx += 1
            toc(print_out=print_out)
    if result_file_path:
        csvex.write_dicts_to_csv(row_dicts=results, output_path=result_file_path, append=True)


def eval_binary_classification(model, test_data: Dict[str, Tuple],
                               eval_funcs: Dict[str, Callable], score2pred_func: Callable = binary_predict_pos_by_above_threshold,
                               pos_label: int = 1, neg_label: int = 0,
                               score_th: Union[float, Tuple, List] = 0.5,
                               result_item_base: Dict = None, result_output_list: List[Dict] = None, score_th_mode='default', test_data_filter: Callable = None, print_out=True):
    """
    Evaluates a binary classification model on the provided data sets. Supports threshold line search, and grouped evaluation.
    :param model: the binary classification model to evaluate; must have a method `predict_proba` that takes the features as input, and generate scores for each label.
    :param test_data: a dictionary of test data sets keyed by names; the values are tuples of features and labels, and optionally the group index;
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
    :return: a reference to the `result_output_list` if it provided; or a new result list containing the result items.
    """
    ori_score_th = score_th
    result_output_list = [] if result_output_list is None else result_output_list
    result_item_base = {} if result_item_base is None else result_item_base.copy()

    for test_data_name, test_data_tup in test_data.items():
        if test_data_filter is not None and not test_data_filter(test_data_name):
            if print_out:
                gex.hprint_message("skip test data", test_data_name)
            continue
        if print_out:
            gex.hprint_pairs(("test data", test_data_name), ("size", len(test_data_tup[0])))

        result_item_base['test'] = test_data_name

        if len(test_data_tup) == 2:
            test_features, test_labels = test_data_tup
            group_indices = None
        elif len(test_data_tup) == 3:
            test_features, test_labels, group_indices = test_data_tup
        else:
            raise ValueError("The format of `test_data` is not recognized.")

        scores = model.predict_proba(test_features)[:, pos_label]
        if score_th_mode == 'line_search':
            score_th_type = type(ori_score_th)
            max_score = np.max(scores)
            if score_th_type in (tuple, list):
                score_th = ori_score_th.copy()
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

        result_item_base['grouped'] = False
        eval_binary_classification_by_score_threshold(
            scores=scores,
            labels=test_labels,
            eval_funcs=eval_funcs,
            score2pred_func=score2pred_func,
            pos_label=pos_label,
            neg_label=neg_label,
            score_th=score_th,
            result_item_base=result_item_base,
            result_output_list=result_output_list
        )

        if group_indices is not None:
            grouped_pos_scores, grouped_labels = [], []
            all_group_start = group_indices[0][0]
            result_item_base['grouped'] = True
            for group_start, group_end in group_indices:
                group_start -= all_group_start
                group_end -= all_group_start
                best_score, best_score_label = sorted(zip(scores[group_start:group_end], test_labels[group_start:group_end]), reverse=True)[0]
                grouped_pos_scores.append(best_score)
                grouped_labels.append(best_score_label)

            eval_binary_classification_by_score_threshold(
                scores=np.array(grouped_pos_scores),
                labels=grouped_labels,
                eval_funcs=eval_funcs,
                score2pred_func=score2pred_func,
                pos_label=pos_label,
                neg_label=neg_label,
                score_th=score_th,
                result_item_base=result_item_base,
                result_output_list=result_output_list
            )

    return result_output_list


def eval_binary_classification_by_score_threshold(scores, labels,
                                                  eval_funcs: Dict[str, Callable],
                                                  score2pred_func: Callable = binary_predict_pos_by_above_threshold,
                                                  pos_label: int = 1,
                                                  neg_label: int = 0,
                                                  score_th: Union[float, Tuple, List] = 0.5,
                                                  result_item_base: Dict = None,
                                                  result_output_list: List[Dict] = None):
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
        for eval_name, eval_func in eval_funcs.items():
            result_item[eval_name] = eval_func(labels, predictions) if pos_label is None else eval_func(labels, predictions, pos_label=pos_label)
        gex.hprint_pairs(*list(result_item.items()))
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

def count_or_accumulate(count_dict: dict, items: Union[dict, Iterator[Any], Any]):
    if isinstance(items, dict):
        for k, v in items.items():
            if k in count_dict:
                if hasattr(v, '__add__') or hasattr(v, '__iadd__'):
                    count_dict[k] += v
                else:
                    count_dict[k] |= v
            else:
                count_dict[k] = v
    elif nonstr_iterable(items):
        for item in items:
            if item in count_dict:
                count_dict[item] += 1
            else:
                count_dict[item] = 1
    elif items in count_dict:
        count_dict[items] += 1
    else:
        count_dict[items] = 1


def count_reduce(count_dict, items: Union[dict, Iterator[Any], Any]):
    if isinstance(items, dict):
        for k, v in items.items():
            if k in count_dict:
                count_dict[k] -= v
    elif nonstr_iterable(items):
        for item in items:
            if item in count_dict:
                count_dict[item] -= 1
    elif items in count_dict:
        count_dict[items] -= 1

    return count_dict


class XCounter(dict):
    """
    A subclass of :class:`dict` designed for counting/accumulating items.
    It is similar to :class:`~collections.Counter`, while designed to be compatible with tuples, lists, sets and dictionaries.
    This class also runs faster than the build-in counter class.
    """

    def __init__(self, init: dict):
        super().__init__()
        self.update(init)

    def __abs__(self):
        """
        Makes every value in this counter be their absolute values if possible. Error will not be thrown if the absolute value is not valid for some values in this counter.
        :return: the current counter.
        """
        for k, v in self.items():
            if hasattr(v, '__abs__'):
                self[k] = abs(v)
        return self

    def __add__(self, other: Union[dict, Iterator[Any], Any]):
        """
        If `other` is a dictionary, then merges the `other` dictionary into this counter.
        For each key in the dictionary `other` to add,
        if the key exists in this counter,
        then the value associated with the key in the other dictionary is first added to this counter's value of the same key,
        by trying to invoke the `+=` operator, or the `|=` operator.
        Error will be thrown if `+=` operator and `|=` operators are both not valid for a value.

        Otherwise, if `other` is a non-string iterable, then treat all items in this iterable as keys, and adds count 1 for each item in this iterable.

        Otherwise, treat `other` as a key, and adds count 1 for `other` if it is in this counter.

        :param other: another dictionary, whose values are to be merged into this counter; or an iterable of keys to count; or just the key to count.
        :return: the current counter.
        """
        return count_or_accumulate(self.copy(), other)

    def __iadd__(self, other):
        return count_or_accumulate(self, other)

    def __sub__(self, other: Union[dict, Iterator[Any], Any]):
        """
        If `other` is a dictionary, then subtracts this counter by the `other` dictionary.
        For each key in the dictionary `other` to subtract,
        if the key exists in this counter,
        then the value associated with the key in this counter is subtracted by the corresponding value in the other dictionary of the same key,
        by trying to invoke the `-=` operator. Error will be thrown if `-=` operator is not valid for a value.

        Otherwise, if `other` is a non-string iterable, then treat all items in this iterable as keys, and reduces count 1 for each item in this iterable; the count after eduction can be negative.

        Otherwise, treat `other` as a key, and reduces count 1 for `other` if it is in this counter.

        :return: the current counter.
        """
        return count_reduce(self.copy(), other)

    def __isub__(self, other):
        return count_reduce(self, other)

    def __truediv__(self, divisor):
        """
        Tries to fields_div each value in this counter by the provided divisor by invoking the `/=` operator. Error will not be thrown if this operator is not valid.
        :param divisor: the divisor.
        :return: the current counter.
        """
        return dict_try_div(self.copy(), divisor)

    def __itruediv__(self, divisor):
        return dict_try_div(self, divisor)

    def __floordiv__(self, divisor):
        """
        Tries to floor-fields_div each value in this counter by the provided divisor by invoking the `//=` operator. Error will not be thrown if this operator is not valid.
        :param divisor: the divisor.
        :return: the current counter.
        """
        return dict_try_floor_div(self.copy(), divisor)

    def __ifloordiv__(self, divisor):
        return dict_try_floor_div(self, divisor)

    def __missing__(self, key):
        return None


# endregion

# region avg tracker

class AvgInfo:
    __slots__ = ('sum', 'count')

    def __init__(self, init_value):
        self.sum = init_value
        self.count = 1

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

class _Stat(dict):
    def count(self, stat_name: str, increase: Any = 1):
        if stat_name in self:
            self[stat_name] += increase
        else:
            self[stat_name] = increase

    def average(self, stat_name: str, value: Any):
        if stat_name in self:
            self[stat_name] += value
        else:
            self[stat_name] = AvgInfo(value)

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

        if increases is None:
            for value in values:
                if value in d:
                    d[value] += increase
                else:
                    d[value] = increase
        else:
            for value, increase in zip(values, increases):
                if value in d:
                    d[value] += increase
                else:
                    d[value] = increase


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
        _Stat.__init__(self)
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

    def average(self, stat_name: str, value: Any):
        """
        Adds the `value` to the average statistic of the specified `stat_name`.
        :param stat_name: provides a name for this statistic.
        :param value: adds this value to the average statistic; works as long as the `+=` operator is well-defined for the type of `value`.
        """
        self._names_for_avg.add(stat_name)
        super().average(stat_name, value)

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

    def aggregate_count_many(self, stat_name: str, values: Iterator, agg_func: Callable[[Dict], Any] = None, increase: Any = 1, increases: Iterator = None):
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
    def __init__(self, overall_count_name='all', category_filter=None):
        dict.__init__(self)
        _StatTypes.__init__(self)
        self._overall_count_name = overall_count_name
        self._category_filter = category_filter
        if overall_count_name:
            self[self._overall_count_name] = _Stat()

    def __missing__(self, key):
        if key not in self:
            self[key] = _Stat()
        return self[key]

    def _count(self, category: str, name: str, value: Any):
        if category not in self:
            d: _Stat = _Stat()
            self[category] = d
        else:
            d = self[category]
        d.count(name, value)

    def count(self, category: str, stat_name: str, value: Any = 1, overall_only: bool = False):
        self._names_for_cnt.add(stat_name)
        if (not overall_only) and category != self._overall_count_name and (self._category_filter is None or category in self._category_filter):
            self[category].count(stat_name, value)

        if self._overall_count_name:
            self[self._overall_count_name].count(stat_name, value)

    def count_multi_categories(self, categories: Union[str, Tuple[str, ...], List[str]], stat_name: str, value=1, overall_only=False):
        self._names_for_cnt.add(stat_name)

        if isinstance(categories, str):
            categories = (categories,)

        if not overall_only:
            for category in categories:
                if category != self._overall_count_name or self._category_filter is None or category in self._category_filter:
                    self[category].count(stat_name, value)

        if self._overall_count_name:
            self[self._overall_count_name].count(stat_name, value)

    def average(self, category: str, stat_name: str, value=1, overall_only=False):
        self._names_for_avg.add(stat_name)

        if (not overall_only) and category != self._overall_count_name and (self._category_filter is None or category in self._category_filter):
            self[category].average(stat_name, value)

        if self._overall_count_name:
            self[self._overall_count_name].average(stat_name, value)

    def average_multi_categories(self, categories: Union[str, Tuple[str, ...], List[str]], stat_name: str, value=1, overall_only=False):
        self._names_for_avg.add(stat_name)

        if isinstance(categories, str):
            categories = (categories,)

        if not overall_only:
            for category in categories:
                if category != self._overall_count_name or self._category_filter is None or category in self._category_filter:
                    self[category].average(stat_name, value)

        if self._overall_count_name:
            self[self._overall_count_name].average(stat_name, value)

    def aggregate(self, category: str, stat_name: str, value: Any, agg_func: Callable[[List], Any] = None, overall_only=False):
        self._names_for_agg[stat_name] = agg_func

        if (not overall_only) and category != self._overall_count_name and (self._category_filter is None or category in self._category_filter):
            self[category].aggregate(stat_name, value)

        if self._overall_count_name:
            self[self._overall_count_name].aggregate(stat_name, value)

    def aggregate_unique(self, category: str, stat_name: str, value: Any, agg_func: Callable[[Set], Any] = None, overall_only=False):
        self._names_for_agg[stat_name] = agg_func

        if (not overall_only) and category != self._overall_count_name and (self._category_filter is None or category in self._category_filter):
            self[category].aggregate_unique(stat_name, value)

        if self._overall_count_name:
            self[self._overall_count_name].aggregate_unique(stat_name, value)

    def aggregate_count(self, category: str, stat_name: str, value: Any, increase: Any = 1, agg_func: Callable[[Dict], Any] = None, overall_only=False):
        self._names_for_agg[stat_name] = agg_func

        if (not overall_only) and category != self._overall_count_name and (self._category_filter is None or category in self._category_filter):
            self[category].aggregate_count(stat_name=stat_name, value=value, increase=increase)

        if self._overall_count_name:
            self[self._overall_count_name].aggregate_count(stat_name=stat_name, value=value, increase=increase)

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

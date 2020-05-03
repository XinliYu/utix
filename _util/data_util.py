import codecs
import pickle as pkl
from os import path
from typing import List, Tuple, Dict, Union, Iterable, Any

import numpy as np
import scipy.sparse as sp
from numpy.random import binomial, normal, randn, randint, choice
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from _util.io_ext import read_all_lines, batch_copy, batch_move
from _util.iter_ext import random_split_iter_by_ratios
from _util.list_ext import split_list, split_list_by_ratios
from _util.msg_ext import ensure_sum_to_one_arg, msg_invalid_arg_value
from _util.np_ext import nums_to_prob, numpy_local_seed
import _util.path_ext as paex
from _util.time_ext import tic, toc
from typing import Callable


class SlotObj:
    def set_none_for_non_existing_slots(self):
        for field_name in self.__slots__:
            if not hasattr(self, field_name):
                setattr(self, field_name, None)

    def set_slots_by_dict(self, d: dict, converters: Dict[str, Tuple[str, Callable]] = None) -> bool:
        value_set_flag = False
        has_converters = bool(converters)
        if has_converters:
            for field_name in converters:
                if field_name in d:
                    converter = converters[field_name]
                    if isinstance(converter, tuple):
                        field_name, converter = converter
                    if field_name in self.__slots__:
                        setattr(self, field_name, converter(d[field_name]))
                        value_set_flag = True

        for field_name in self.__slots__:
            if (not has_converters or field_name not in converters) and field_name in d:
                setattr(self, field_name, d[field_name])
                value_set_flag = True

        if value_set_flag:
            self.set_none_for_non_existing_slots()

        return value_set_flag

    def set_slots_by_tuples(self, tuples: Iterable[Tuple[str, Any]], converters: Dict[str, Tuple[str, Callable]] = None) -> bool:
        value_set_flag = False
        if converters:
            for field_name, field_val in tuples:
                if field_name in converters:
                    converter = converters[field_name]
                    if isinstance(converter, tuple):
                        field_name, converter = converter
                    if field_name in self.__slots__:
                        setattr(self, field_name, converter(field_val))
                        value_set_flag = True
                elif field_name in self.__slots__:
                    setattr(self, field_name, field_val)
                    value_set_flag = True
        else:
            for field_name, field_val in tuples:
                if field_name in self.__slots__:
                    setattr(self, field_name, field_val)
                    value_set_flag = True

        if value_set_flag:
            self.set_none_for_non_existing_slots()

        return value_set_flag


def random_onehot_embeddings(batch_size=1000, embed_size=1, embed_dim=128, one_prob=0.004, min_num_ones: int = 2, noise_para=(0.1, 0.5), noise_dist=normal):
    shape = (batch_size, embed_size, embed_dim)
    a = binomial(n=1, p=one_prob, size=shape).astype(dtype=np.float32)
    a[:, :, range(min_num_ones)] = 1
    a = np.take_along_axis(a, randn(*a.shape).argsort(axis=2), axis=2)  # randomly permute the embedding vector dimensions
    if noise_para is not None:
        a += noise_dist(*noise_para, size=shape)
    a[:, :, 0:2] = 0
    return a


def synthetic_onehot_similarity_dataset(batch_size=1000,
                                        embedding_dim=128,
                                        one_prob=0.004,
                                        min_num_ones: int = 2,
                                        num_negative_samples=4,
                                        pivot_noise_para: tuple = (0, 0.5),
                                        candidate_noise_para: tuple = (0, 0.5),
                                        noise_dist=normal,
                                        label_type=np.int):
    pivot_embeddings = random_onehot_embeddings(batch_size=batch_size,
                                                embed_size=1,
                                                embed_dim=embedding_dim,
                                                one_prob=one_prob,
                                                min_num_ones=min_num_ones,
                                                noise_para=pivot_noise_para,
                                                noise_dist=noise_dist)
    sample_size = num_negative_samples + 1
    candidate_embeddings = pivot_embeddings[choice(batch_size, size=batch_size * sample_size, replace=True)]
    labels = randint(sample_size, size=batch_size, dtype=label_type)
    candidate_embeddings[labels + np.arange(batch_size) * sample_size] = pivot_embeddings
    candidate_embeddings = candidate_embeddings.squeeze().reshape(batch_size, sample_size, embedding_dim)

    if __debug__:
        assert all(pivot_embeddings[i] in candidate_embeddings[i] for i in range(batch_size)), \
            "some pivot embeddings are not among the candidates"

    if candidate_noise_para is not None:
        candidate_embeddings += noise_dist(*candidate_noise_para, size=candidate_embeddings.shape)

    return pivot_embeddings, candidate_embeddings, labels


def batch_pairwise_score(batch1, batch2, score_func=cosine_similarity):
    return [score_func(x, y) for x, y in zip(batch1, batch2)]


def batch_pairwise_score_labels(batch1, batch2, score_func=cosine_similarity):
    return [np.argmax(score_func(x, y), axis=1) for x, y in zip(batch1, batch2)]


def mask1d_by_index(mask_idxes, size: int, ref_idxes=None):
    if type(mask_idxes[0]) in (list, np.ndarray):
        mask_count = len(mask_idxes)
        masks = [np.zeros(size, dtype=bool) for _ in range(mask_count)]
        if ref_idxes is None:
            for i in range(mask_count):
                masks[i][mask_idxes] = True
        else:
            for j, ref_idx in enumerate(ref_idxes):
                for i in range(mask_count):
                    if ref_idx in mask_idxes[i]:
                        masks[i][j] = True
        return tuple(masks)

    else:
        mask = np.zeros(size, dtype=bool)
        if ref_idxes is None:
            mask[mask_idxes] = True
        else:
            for i, ref_idx in enumerate(ref_idxes):
                if ref_idx in mask_idxes:
                    mask[i] = True
        return mask


def labels_to_onehot(labels):
    min_label = min(labels)
    max_label = max(labels)
    if type(labels) is not np.ndarray:
        labels = np.array(labels)
    labels = labels - min_label
    label_count = len(labels)
    one_hots = np.zeros((label_count, max_label - min_label + 1))
    one_hots[np.arange(label_count), labels] = 1
    return one_hots


def sample_index_by_percentage(size: int, percents: Tuple[float]) -> List[np.ndarray]:
    permuted_indices = np.random.permutation(size)
    split_count = len(percents)
    index_samples = [None] * split_count
    start = 0

    for i in range(split_count):
        end_p = percents[i]
        if end_p == 0:
            continue
        end = start + int(end_p * size) if i != split_count - 1 else size
        index_samples[i] = permuted_indices[start:end]
        start = end
    return index_samples


def mask_by_percentage(size: int, percents: Tuple):
    index_samples = sample_index_by_percentage(size, percents)
    mask_count = len(percents)
    masks = [None] * mask_count
    for i in range(mask_count):
        indices = index_samples[i]
        if indices is None:
            continue
        masks[i] = np.zeros(size, dtype=bool)
        masks[i][index_samples[i]] = True

    return masks


def save_lists(file_path: str, lists: List, delimiter=' ', encoding=None):
    list_count = len(lists[0])
    if file_path.endswith('.txt'):
        with open(file_path, 'w') if encoding is None else codecs.open(file_path, 'w', encoding) as f:
            for i in range(list_count):
                f.write(delimiter.join([str(l[i]) for l in lists if l is not None]))
                f.write('\n')
            f.flush()
    else:
        with open(file_path, 'wb') as f:
            pkl.dump(lists, f)


def sparse_to_tuple(sparse_mat):
    if not sp.isspmatrix_coo(sparse_mat):
        sparse_mat = sparse_mat.tocoo()
    coords = np.vstack((sparse_mat.row, sparse_mat.col)).transpose()
    values = sparse_mat.data
    shape = sparse_mat.shape
    return coords, values, shape


def sparse_row_normalize(a):
    row_sum = np.array(a.sum(1))
    r_inv = np.power(row_sum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    a = r_mat_inv.dot(a)
    return sparse_to_tuple(a)


def row_normalize(a: np.ndarray):
    row_sums = a.sum(axis=1)
    return a / row_sums[:, np.newaxis]


def train_test_val_split(input_iter, train_ratio, test_ratio, val_ratio, id_gen=None, train_ids=None, test_ids=None, val_ids=None, exclusion_id_list=None, item_process=None, item_filter=None, shuffle=True, rnd_seed: int = -1):
    """
    Performs train/test/validation set split over an iterator of data items.

    :param input_iter: the iterator that returns one data entry at a time.
    :param train_ratio: the split ratio for the training set.
    :param test_ratio: the split ratio for the test set.
    :param val_ratio: the split ratio for the validation (development) set.
    :param id_gen: a function to retrieve an id from each data entry, or to generate an id for each data entry.
    :param train_ids: data entries of these ids should be put in the training set.
    :param test_ids: data entries of these ids should be put in the test set.
    :param val_ids: data entries of these ids should be put in the validation set.
    :param item_filter: provides a function used to filter data entries; it this function returns `True` when applied to an data entry, then that data entry will be discarded.
    :return: a three-tuple, the training set, the test set and the validation set.
    """
    splits, num_excluded = random_split_iter_by_ratios(
        it=input_iter,
        split_ratios=(train_ratio, test_ratio, val_ratio),
        id_gen=id_gen,
        pre_split_id_lists=(train_ids, test_ids, val_ids),
        exclusion_id_list=exclusion_id_list,
        item_process=item_process,
        item_filter=item_filter,
        shuffle=shuffle,
        rnd_seed=rnd_seed
    )
    return splits[0], splits[1], splits[2], num_excluded


def train_test_val_split_for_lines(input_iter):
    pass


def train_test_val_split_for_files(file_paths: List,
                                   train_test_val_ratios: Tuple[float, float, float],
                                   output_path: Union[str, Tuple[str, str, str]],
                                   copy_files=True,
                                   overwrite=False,
                                   sort=False,
                                   shuffle=True,
                                   rnd_seed=-1,
                                   verbose=__debug__,
                                   num_p=1):
    if verbose:
        tic(f"Splitting {len(file_paths)} files into train/test/val sets with split ratios {train_test_val_ratios}", newline=True)
    if len(train_test_val_ratios) != 3:
        raise ValueError(f"must specify three ratios for the train/test/validation set splits; got {len(train_test_val_ratios)} ratios '{','.join((str(x) for x in train_test_val_ratios))}'")
    if sort:
        file_paths.sort()
    elif shuffle:
        with numpy_local_seed(rnd_seed) as _:
            if rnd_seed >= 0:
                file_paths.sort()  # NOTE reproducibility needs this sort
            np.random.shuffle(file_paths)

    if isinstance(output_path, str):
        train_dir = path.join(output_path, 'train')
        test_dir = path.join(output_path, 'test')
        val_dir = path.join(output_path, 'val')
    elif len(output_path) == 3:
        train_dir, test_dir, val_dir = output_path
    else:
        raise ValueError(msg_invalid_arg_value(arg_val=output_path, arg_name='output_path'))

    ensure_sum_to_one_arg(arg_val=train_test_val_ratios, arg_name='train_test_val_ratios', warning=True)
    paex.ensure_dir_existence(train_dir, clear_dir=overwrite, verbose=verbose)
    paex.ensure_dir_existence(test_dir, clear_dir=overwrite, verbose=verbose)
    paex.ensure_dir_existence(val_dir, clear_dir=overwrite, verbose=verbose)
    splits = split_list_by_ratios(list_to_split=file_paths, split_ratios=train_test_val_ratios, check_ratio_sum_to_one=False)
    for cur_path_list, cur_output_dir in zip(splits, (train_dir, test_dir, val_dir)):
        if copy_files:
            batch_copy(src_paths=cur_path_list, dst_dir=cur_output_dir, solve_conflict=True, use_tqdm=verbose, tqdm_msg=f"copy files to {path.basename(cur_output_dir)}" if verbose else None, num_p=num_p)
        else:
            batch_move(src_paths=cur_path_list, dst_dir=cur_output_dir, solve_conflict=True, undo_move_on_failure=verbose, use_tqdm=True, tqdm_msg=f"move files to {path.basename(cur_output_dir)}" if verbose else None)
    if verbose:
        toc()


def loads_labeled_numpy_vectors(dir_path: str, np_file_pattern: str, label_file_pattern: str, recursive: bool = False, use_tqdm: bool = False, tqdm_msg: str = None, filter=None) -> Dict[str, np.ndarray]:
    """
    Loading labeled numpy vectors from paired files in a directory (and its sub-directories if `recursive` is set `True`).
    You must specify the regex pattern for the numpy files in argument `np_file_pattern`,
    and the format pattern for the label files in argument `label_file_pattern`.
    For example, if a numpy file name is in the format like 'batch_1.npy', and its label file is 'batch_1_label.txt',
    then you can specify `np_file_pattern` as `(batch_[0-9]+).npy` and the `label_file_pattern` as `{}_label.txt`.
    The `i`th line in a label file is the text label for the corresponding `i`th row vector in the paired numpy file.

    :param dir_path: the directory to search for the paired files of the numpy vectors and their labels.
    :param np_file_pattern: specify a regex pattern for the numpy file names. This pattern will be used to exactly match the file name in the specified directory.
    :param label_file_pattern: the format pattern for the label files with one slot; the first regex group matched on the numpy file name by `np_file_pattern` will fill in this slot.
    :param recursive: if `True`, this method will search the subfolders in `dir_path`.
    :param use_tqdm: if `True`, tqdm will be used to display the loading progress.
    :param tqdm_msg: sets the message for tqdm to display, and it may contain a format slot to display the current file being loaded.
    :return: a dictionary of vectors, keyed by labels with values being their corresponding vectors.
    """
    embeds_dict = {}
    queue = [dir_path]

    while len(queue) != 0:
        dir_path = queue.pop()
        paired_file_iter = paex.iter_paired_files(dir_path=dir_path,
                                                  main_file_reg_pattern=np_file_pattern,
                                                  paired_file_format_pattern=label_file_pattern)
        if use_tqdm:
            paired_file_iter = tqdm(paired_file_iter)
            if tqdm_msg:
                paired_file_iter.set_description(tqdm_msg.format(dir_path))
            else:
                paired_file_iter.set_description(f"reading numpy vectors and labels from {dir_path}")

        for np_file, label_file in paired_file_iter:
            embeds = np.load(np_file)
            if filter is None:
                labels = read_all_lines(label_file)  # ! NOTE the white spaces at the end of each line are removed; this method `read_all_lines` does that.
                for i in range(len(labels)):
                    embeds_dict[labels[i].strip()] = embeds[i].copy()
            else:
                with open(label_file, 'r') as fin:
                    for i, line in enumerate(fin):
                        line = line.rstrip()  # ! NOTE the white spaces at the end of each line are removed
                        if line in filter:
                            embeds_dict[line] = embeds[i].copy()
        if recursive:
            queue.extend(list(paex.iter_all_immediate_sub_dirs(dir_path)))
    del embeds
    return embeds_dict

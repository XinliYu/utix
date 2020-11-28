from itertools import chain, islice
from typing import Callable, Iterator, Iterable, Union, List, Tuple, Any

from utix.general import tqdm_wrap, sliceable
from utix.listex import split_list, beam_find_first
import numpy as np
import uuid


def with_uuid(it, prefix='', suffix=''):
    yield from ((prefix + str(uuid.uuid4()) + suffix, x) for x in it)


def with_names(it, name_format: str = None, name_prefix='', name_suffix=''):
    if name_format is None or name_format == 'uuid':
        return with_uuid(it=it, prefix=name_prefix, suffix=name_suffix)
    else:
        for i, x in enumerate(it):
            yield name_prefix + name_format.format(i) + name_suffix, x


def next__(it, n):
    """
    Returns the next n items from an iterator.
    :param it: the iterator.
    :return: the next n items from the iterator.
    """
    return tuple(islice(it, n))


def islice__(it, n):
    return it if n is None else islice(it, n)


def chunk_iter(it: Union[Iterator, Iterable], chunk_size: int, as_list=False) -> Union[Iterator[Iterator], Iterator[List]]:
    """
    Returns an iterator that iterates through chunks of the provided iterator; each chunk is also represented by an iterator that can iterate through the elements in it.
    :param it: the iterator to chunk.
    :param chunk_size: the size of each chunk.
    :return: an iterator that iterates through chunks of the provided iterator.
    """
    it = iter(it)
    if as_list:
        while True:
            cur = list(islice(it, chunk_size))
            if not cur:
                break
            yield cur
    else:
        while True:
            cur_it = islice(it, chunk_size)
            try:
                first = next(cur_it)
            except StopIteration:
                break
            yield chain((first,), cur_it)


def split_iter(it: Union[Iterator, Iterable, List], num_splits: int, use_tqdm=False, tqdm_msg=None) -> List[List]:
    """
    Splits the items read from an iterator into a list of lists, where each inner list is a split.
    :param it: the iterator to read from.
    :param num_splits: the number of splits to generate.
    :return: a list of lists, where each inner list is a split from the iterator.
    """
    return split_list(list_to_split=it if sliceable(it) else list(tqdm_wrap(it, use_tqdm, tqdm_msg)), num_splits=num_splits)


def random_split_iter_by_ratios(it: Union[Iterator, Iterable, List],
                                split_ratios: Union[List, Tuple],
                                id_gen: Callable = None,
                                pre_split_id_lists: Union[List[List], Tuple] = None,
                                exclusion_id_list: List = None,
                                item_process: Callable = None,
                                item_filter: Callable = None,
                                return_splits=True,
                                shuffle=True,
                                rnd_seed: int = -1):
    """
    Splits the items read from an iterator into a list of lists by pre-defined ratios and pre-defined assignment.
    This method is intended for data splits like train/test/validation set splits.
    :param it: the iterator to read from.
    :param split_ratios: the split ratios for each split; make sure the sum of the ratios is 1; the number of split ratios define the number of splits.
    :param id_gen: a function to generate an item id for each item read from the iterator.
    :param pre_split_id_lists: a list of lists, where the number of internal lists equals the number of splits.
                            Each internal list corresponds to a split and contains item ids. If the id of one item appears in one list, then that item is added to its corresponding split.
                            Not effective if `id_gen` is not provided.
    :param exclusion_id_list: a list of item ids; an item will be discarded if its id is this list.Not effective if `id_gen` is not provided.
    :param item_process: a function that applies on each item read from the iterator, and returns a processed item.
    :param item_filter: provides a function used to filter the items; it this function returns `True` when applied to an item, then that item will be discarded.
    :param return_splits: `True` if the splits are returned; otherwise, `False`.
    :param shuffle: shuffle the items in each split.
    :param rnd_seed: the random seed for the random split.
    :return: the splits and the number of excluded items if `return_splits` is `True`; otherwise `None` and the number of excluded items.
    """

    if rnd_seed >= 0:
        np.random.seed(rnd_seed)

    num_splits = len(split_ratios)
    if item_process is None or return_splits:
        splits = [[] for _ in range(num_splits)]
        if item_process is None:
            def item_process(item, split_idx_):
                splits[split_idx_].append(item)

        elif return_splits:
            _item_process = item_process

            def item_process(item, split_idx_):
                splits[split_idx_].append(_item_process(item))

    num_excluded = 0
    if id_gen:
        for data_entry in it:
            if item_filter is None or not item_filter(data_entry):  # <- check if we should skip the current data entry
                entry_id = id_gen(data_entry)
                if entry_id not in exclusion_id_list:
                    pre_assign_flag = False
                    for split_idx, pre_split_ids in enumerate(pre_split_id_lists):
                        if entry_id is not None and pre_split_ids is not None and entry_id in pre_split_ids:  # <- check if the customer id is in the `train_customer_id`
                            item_process(data_entry, split_idx)
                            pre_assign_flag = True
                            break
                    if not pre_assign_flag:
                        item_process(data_entry, int(np.random.choice(num_splits, 1, p=split_ratios)))  # <- otherwise, randomly place the data entry in a data split according to the specified `ratios`
                else:
                    num_excluded += 1
            else:
                num_excluded += 1
    else:
        for data_entry in it:
            if item_filter is None or not item_filter(data_entry):
                item_process(data_entry, int(np.random.choice(num_splits, 1, p=split_ratios)))

    if return_splits:
        for split in splits:
            np.random.shuffle(split)
        return splits, num_excluded
    else:
        return None, num_excluded


def apply_on_slices(func: Callable, iterable, slice_size: int):
    return map(func, iter(lambda: list(islice(iterable, slice_size)), []))


# region slicing

def slices(iterable: Iterator, slice_size: int) -> Iterator[Tuple]:
    """
    An iterator through slices of the provided iterable, where each slice is a tuple of the size `slice_size` except for the last slice can be of a smaller size.

    >>> import utix.iter_ext as itx
    >>> print(list(itx.slices(range(11), slice_size=2)) == [(0,1),(2,3),(4,5),(6,7),(8,9),(10,)])

    If the `slice_size` is `None`, this method yields a single tuple with all elements in `iterable`.
    >>>  print(list(itx.slices(range(11), None)) == [(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)])

    :param iterable: the iterable to slice.
    :param slice_size: the size of each slice.
    :return: an iterator that yields one slice of the `iterable` at the time; the slice is of size `slice_size` except for that the last slice can be of a smaller size.
    """
    iterable = iter(iterable)
    return iter(lambda: tuple(islice(iterable, slice_size)), ())


def sliceable_slices(sliceable, slice_size: int):
    """
    The same as `slices`, but only takes a sliceable object in order for speed up.

    >>> import utix.iter_ext as itx
    >>> # note this result is different from `itx.slices`, because the normal slicing of `range` object still returns a `range` object.
    >>> print(list(itx.sliceable_slices(range(11), slice_size=2)) == [range(0, 2), range(2, 4), range(4, 6), range(6, 8), range(8, 10), range(10, 11)])
    >>> # this will return the same result as `itx.slices` on the range object
    >>> print(list(itx.sliceable_slices(tuple(range(11)), slice_size=2)) == [(0,1),(2,3),(4,5),(6,7),(8,9),(10,)])

    >>> # performance experiments; 3x better performance against `slices`, and 2x against `slices__`
    # >>> from timeit import timeit
    # >>> a = list(range(512))
    # >>> def target1():
    # >>>     list(itx.slices(a, 32))
    # >>> def target2():
    # >>>     list(itx.sliceable_slices(a, 32))
    # >>> def target3():
    # >>>     list(itx.slices__(a, 32))
    # >>> print(timeit(target1))
    # >>> print(timeit(target2))
    # >>> print(timeit(target3))
    """
    start, end, len_list = 0, slice_size, len(sliceable)
    while start < len_list:
        yield sliceable[start:end]
        start = end
        end += slice_size


def slices__(iterable, slice_size: int) -> Iterator[Tuple]:
    """
    The same as `slices`, except it tests if the `iterable` is sliceable first, and calls `sliceable_slices` if so.
    """
    if sliceable(iterable):
        return sliceable_slices(sliceable=iterable, slice_size=slice_size)
    else:
        return slices(iterable=iterable, slice_size=slice_size)


def slices_by_break_criterion(iterable: Iterator, slice_size: int, break_criterion: Callable, max_slice_size=None):
    """
    An iterator yielding slices of the provided iterable, where each slice is a tuple intended to be of size about `slice_size`.
    However, the last value in the slice must satisfy the `break_criterion` (except for the last slice);
        if this is not the case,
        the actual slice size might be larger than `slice_size` but not exceeding the `max_slice_size`,
        and might also be smaller than the `slice_size` if a longer slice violates the `max_slice_size`.

    >>> import utix.iter_ext as itx
    >>> a = list(range(100))
    >>> print(list(itx.slices_by_break_criterion(iterable=a,
    >>>                                          slice_size=32,
    >>>                                          break_criterion=lambda x: x % 5 == 0,
    >>>                                          max_slice_size=48)) == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
    >>>                                                                  [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70],
    >>>                                                                  [71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]])
    >>> print(list(itx.slices_by_break_criterion(iterable=a,
    >>>                                          slice_size=16,
    >>>                                          break_criterion=lambda x: x % 5 == 0,
    >>>                                          max_slice_size=48)) == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    >>>                                                                  [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
    >>>                                                                  [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55],
    >>>                                                                  [56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75],
    >>>                                                                  [76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
    >>>                                                                  [96, 97, 98, 99]])
    >>> print(list(itx.slices_by_break_criterion(iterable=a,
    >>>                                          slice_size=32,
    >>>                                          break_criterion=lambda x: x % 7 == 0,
    >>>                                          max_slice_size=32)) == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
    >>>                                                                  [29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56],
    >>>                                                                  [57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84],
    >>>                                                                  [85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]])

    :param iterable: this iterator yields consecutive slices of this iterable.
    :param slice_size: the intended slice size.
    :param break_criterion: a condition represented by a function that accepts a value in `iterable` and returns a boolean value,
                            with `True` meaning the value from the iterable satisfying this condition;
                            the last element of each slice (except for the last slice) must satisfy this condition.
    :param max_slice_size: the maximum slice size; every yielded slice has a size smaller than this.
    :return: an iterator yielding slices of the provided iterable as discussed above.
    """
    if max_slice_size < slice_size:
        raise ValueError('`max_slice_size` must be smaller than the `slice_size`')

    iterable = iter(iterable)
    prev = []
    while True:
        len_prev = len(prev)
        cur = list(islice(iterable, max_slice_size - len_prev))
        if not cur:
            break
        len_cur = len(cur)
        start_idx = break_idx = 0
        _slice_size = slice_size - len_prev
        while len_cur >= _slice_size:
            find_start_idx = start_idx + _slice_size - 1  # `start_idx + _slice_size` is the index of the first value after the 'slice', thus it must be -1 in order to check if we should break on the last value of the slice
            break_idx = beam_find_first(src=cur,
                                        start_idx=find_start_idx,
                                        condition=break_criterion,
                                        max_backward=_slice_size - 1)
            if break_idx == -1:
                if start_idx == 0:
                    raise ValueError('unable to find value satisfying the break criterion without violating the maximum slice size')
                cur = cur[start_idx:] + list(islice(iterable, start_idx))
                start_idx = 0
            else:
                break_idx += 1
                yield prev + cur[start_idx:break_idx]
                len_cur -= (break_idx - start_idx)
                prev = []
                start_idx = break_idx
                _slice_size = slice_size
        prev += cur[break_idx:]

    if prev:
        yield prev


def apply_on_slices_but_yield_first(func: Callable, iterable, slice_size: int):
    """
    The same as `slices`, function; in addition, apply a function `func` on each yielded slice; the function is applied AFTER the slice is yielded.
    The `func` here is usually for post-processing. Any returned value of that function will be discarded.

    >>> import utix.iter_ext as itx
    >>> sums = []
    >>> print(list(itx.apply_on_slices_but_yield_first(lambda x: sums.append(sum(x)), range(11), slice_size=2)) == [(0,1),(2,3),(4,5),(6,7),(8,9),(10,)])
    >>> print(sums == [1, 5, 9, 13, 17, 10])

    >>> # compare with the following
    >>> print(list(map(lambda x: sum(x), itx.slices(range(11), 2)))) # [1, 5, 9, 13, 17, 10]
    >>> sums = []
    >>> print(list(map(lambda x: sums.append(sum(x)), itx.slices(range(11), 2)))) # [None, None, None, None, None, None]
    """
    for s in slices(iterable=iterable, slice_size=slice_size):
        yield from s
        func(s)


# endregion

class CachedIterable:
    """
    Defines a general save mechanism for iterables.
    """

    def __init__(self, iterable):
        """
        Initializes this CachedIterable, wrapping around a provided `iterable`.
        :param iterable: the iterable to wrap.
        """
        self.iter = iter(iterable)
        self.cache_done = False
        self.cache = []

    def __iter__(self):
        if self.cache_done:
            return iter(self.cache)
        # Chain values cached so far and then generate the rest;
        # this is in case the iteration is not complete in the previous use.
        return chain(self.cache, self._inner_iter())

    def _inner_iter(self):
        for new_val in self.iter:
            self.cache.append(new_val)
            yield new_val
        self.cache_done = True
        del self.iter

    def cache_all(self):
        self.cache = list(self.iter)
        self.cache_done = True
        del self.iter


def slice_iter(sliceable, slice_size, start=0, end=0):
    if end <= 0:
        end += len(sliceable)
    while start < end:
        slice_end = min(start + slice_size, end)
        yield sliceable[start:slice_end]
        start = slice_end


def slice_iter_multi(sliceables, slice_size, start=0, end=0):
    if end <= 0:
        end += len(sliceables[0])
    while start < end:
        slice_end = min(start + slice_size, end)
        yield [sliceable[start:slice_end] for sliceable in sliceables]
        start = slice_end


def labelled_slice_iter(sliceables_and_labels, slice_size, start=0, end=0):
    if end <= 0:
        end += len(sliceables_and_labels[0])
    while start < end:
        slice_end = min(start + slice_size, end)
        yield tuple(sliceable[start:slice_end] for sliceable in sliceables_and_labels[:-1]), sliceables_and_labels[-1][start:slice_end]
        start = slice_end

import itertools
import random
from itertools import cycle, islice
from os import path
from typing import List, Union, Tuple, Iterator, Callable, Mapping
import numpy as np
import utix.pathex as paex
from utix.general import iterable__, iterable, zip__, count_and_rank, count
from utix.msgex import ensure_positive_arg, ensure_sum_to_one_arg


# region shuffle

def shuffle_together(*lists):
    """
    Randomly shuffles multiple lists altogether, so that elements at the same position of each list still have the same indices after shuffle.

    >>> import utix.listex as lix
    >>> lix.shuffle_together([1,2,3,4],['i', 'ii', 'iii', 'iv']) # one possible result: ((2, 1, 3, 4), ('ii', 'i', 'iii', 'iv'))

    :param lists: the lists to shuffle together.
    :return: the lists after shuffle.
    """
    tmp = list(zip(*lists))
    random.shuffle(tmp)
    return tuple(zip(*tmp))


def shuffle_list_iter(ls: List, num_shuffles_to_generate: int = 1, random_seed: int = 0, index_file_dir: str = None, index_file_name_pattern: str = 'shuffle_idx_{}.idx', verbose: bool = __debug__) -> Iterator[List]:
    ensure_positive_arg(arg_val=num_shuffles_to_generate, arg_name='num_shuffles_to_generate')
    ls_len = len(ls)
    if index_file_dir:
        if not path.exists(index_file_dir):
            paex.ensure_dir_existence(index_file_dir, verbose=verbose)
        else:
            from utix.ioex import pickle_load
            for i in range(num_shuffles_to_generate):
                idx_file_path = path.join(index_file_dir, index_file_name_pattern.format(i))
                if path.exists(idx_file_path):
                    idxes = pickle_load(idx_file_path)
                elif i == 0:
                    break
                else:
                    raise ValueError(f"the index file {idx_file_path} does not exist.")
                if len(idxes) != ls_len:
                    raise ValueError(f"the number of the loaded list indexes ({len(idxes)}) is different from the size of the list to shuffle ({ls_len}).")
                yield [ls[idx] for idx in idxes]
            if i != 0:
                return

    rng = random if random_seed < 0 else random.Random(random_seed)
    idxes = list(range(ls_len))
    from utix.ioex import pickle_save__
    for i in range(num_shuffles_to_generate):
        idx_file_path = path.join(index_file_dir, index_file_name_pattern.format(i))
        rng.shuffle(idxes)
        if index_file_dir:
            pickle_save__(data=idxes, file_or_dir_path=idx_file_path, auto_timestamp=False)
        yield [ls[idx] for idx in idxes]


def shuffle_lists_iter(lists: List, num_shuffles_to_generate: int = 1, random_seed: int = 0, index_file_dir: str = None, index_file_name_pattern: str = 'shuffle_idx_{}.idx', verbose: bool = __debug__) -> Iterator[List]:
    zipped_lists = list(zip(*lists))
    for tmp in shuffle_list_iter(ls=zipped_lists,
                                 random_seed=random_seed,
                                 index_file_dir=index_file_dir,
                                 num_shuffles_to_generate=num_shuffles_to_generate,
                                 index_file_name_pattern=index_file_name_pattern,
                                 verbose=verbose):
        yield zip(*tmp)


# endregion

# region split

def iter_split_list(list_to_split: List, num_splits) -> Iterator[List]:
    """
    Returns an iterator that iterates through even splits of the provided `list_to_split`. If the size of `list_to_split` is not dividable ty `num_chunks`, then the last split will be longer or shorter than others
    :param list_to_split: the list to split.
    :param num_splits: the number of splits to yield; if the length of the `list_to_split` is not larger than this `num_splits`, then singleton lists will be yielded and the total number of yielded splits is the same as the length of the `list_to_split`.
    :return: an iterator that iterates through even splits of the provided list.
    """
    list_len = len(list_to_split)
    if list_len <= num_splits:
        for item in list_to_split:
            yield [item]
    else:
        list_len = len(list_to_split)
        chunk_size = int(list_len / num_splits)
        remainder = int(list_len - chunk_size * num_splits)

        if remainder > 1:
            begin, end = 0, chunk_size + 1
            for i in range(0, remainder - 1):
                yield list_to_split[begin:end]
                begin, end = end, end + chunk_size + 1
        else:
            begin, end = 0, chunk_size

        for i in range(remainder - 1, num_splits - 1):
            yield list_to_split[begin:end]
            begin, end = end, end + chunk_size
        if begin < list_len:
            yield list_to_split[begin:]


def iter_split_list_by_size(list_to_split: List, split_size: int) -> Iterator[List]:
    """
    Returns an iterator that iterates through splits of the provided `list_to_split`, where each split is of a fixed size specified by `split_size`, except that the last chunk could be of a smaller size.
    :param list_to_split: the list to split.
    :param split_size: the size of each split.
    :return: an iterator that iterates through splits of the provided list.
    """
    for i in range(0, len(list_to_split), split_size):
        yield split_size[i:i + split_size]


def iter_split_list_by_ratios(list_to_split: List, *split_ratios, check_ratio_sum_to_one: bool = True):
    """
    Returns an iterator that iterates through splits of the provided `list_to_split`. The split is done according to the specified `split_ratios`.
    :param list_to_split: the list to split.
    :param split_ratios: the split ratios, where each ratio is a float specifying the ratio of the size of the split over the total size of the list to split.
    :param check_ratio_sum_to_one: `True` if checking if the provided `split_ratios` sum to 1; otherwise `False`.
    :return: an iterator that iterates through splits of the provided list.
    """
    if check_ratio_sum_to_one:
        ensure_sum_to_one_arg(arg_val=split_ratios, arg_name='split_ratios')
    list_len = len(list_to_split)
    cnt_splits = len(split_ratios)
    start = end = 0
    for i in range(cnt_splits - 1):
        ratio = split_ratios[i]
        cnt = round(list_len * ratio)
        end += cnt
        yield list_to_split[start: end]
        start = end
    yield list_to_split[start:]


def split_list(list_to_split: List, num_splits: int) -> List[List]:
    """
    Returns a list of even splits of the provided `list_to_split`. If the size of `list_to_split` is not dividable ty `num_chunks`, then the last split will be longer or shorter than others
    :param list_to_split: the list to split.
    :param num_splits: the number of splits to yield; if the length of the `list_to_split` is not larger than this `num_splits`, then singleton lists will be returned and the total number of returned splits is the same as the length of the `list_to_split`.
    :return: a list of even splits of the provided list.
    """
    return list(iter_split_list(list_to_split, num_splits))


def split_list_by_size(list_to_split: List, split_size: int) -> Iterator[List]:
    """
    Returns a list of splits of the provided `list_to_split`, where each split is of a fixed size specified by `chunk_size`, except that the last chunk could be of a smaller size.
    :param list_to_split: the list to split.
    :param split_size: the size of each split.
    :return: a list of even splits of the provided list.
    """
    return list(iter_split_list_by_size(list_to_split, split_size))


def split_list_by_ratios(list_to_split: List, split_ratios, check_ratio_sum_to_one: bool = True):
    """
    Returns a list of splits of the provided `list_to_split`. The split is done according to the specified `split_ratios`.
    :param list_to_split: the list to split.
    :param split_ratios: the split ratios, where each ratio is a float specifying the ratio of the size of the split over the total size of the list to split.
    :param check_ratio_sum_to_one: `True` if checking if the provided `split_ratios` sum to 1; otherwise `False`.
    :return: a list of splits of the provided list according to the `split_ratios`.
    """
    return list(iter_split_list_by_ratios(list_to_split, *split_ratios, check_ratio_sum_to_one=check_ratio_sum_to_one))


def random_split_lists_iter(lists: List, split_ratios, num_generations: int = 1, random_seed: int = 0, check_ratio_sum_to_one: bool = True, index_file_dir: str = None, index_file_name_pattern: str = 'shuffle_idx_{}.idx', verbose: bool = __debug__):
    if check_ratio_sum_to_one:
        ensure_sum_to_one_arg(arg_val=split_ratios, arg_name='split_ratios')
    zipped_lists = list(zip(*lists))
    for tmp in shuffle_list_iter(ls=zipped_lists,
                                 random_seed=random_seed,
                                 index_file_dir=index_file_dir,
                                 num_shuffles_to_generate=num_generations,
                                 index_file_name_pattern=index_file_name_pattern,
                                 verbose=verbose):
        yield [zip(*tmp2) for tmp2 in split_list_by_ratios(list_to_split=tmp, split_ratios=split_ratios, check_ratio_sum_to_one=False)]


# endregion

# region concatenation

def join_list(connector, lists: Iterator[List]):
    """
    Connects a sequence of lists into a single list, with an element `connector` inserted between every two adjacent lists.

    >>> import utix.listex as lix
    >>> print(lix.join_list(0, [[1, 2], [3, 4], [5, 6, 7]]) == [1, 2, 0, 3, 4, 0, 5, 6, 7])
    >>> print(lix.join_list(0, [[1, 2]]) == [1, 2])

    :param connector: the element to insert between two adjacent lists; set this to `None` to indicate there is no connector.
    :param lists: the sequence of lists to join.
    :return: a single list derived by connecting each list in `lists` by the `connector`.
    """
    if connector is None:
        return sum(lists, [])

    lists = iter(lists)
    first = next(lists)
    return sum((first,) + sum((([connector], ll) for ll in lists), ()), [])


def join_chain(connector, *iterables):
    """
    Chains a sequence of iterables, and in addition yield the `connector` object between every two iterables if it is not `None`.

    >>> import utix.listex as lix
    >>> from itertools import product
    >>> print(list(lix.join_chain(0, [1, 2], 'abc', product((3, 4), (5, 6)))) == [1, 2, 0, 'a', 'b', 'c', 0, (3, 5), (3, 6), (4, 5), (4, 6)])
    >>> print(list(lix.join_chain(0, [1, 2])) == [1, 2])

    :param connector: the element to insert between two adjacent iterables; set this to `None` to indicate there is no connector.
    :param iterables: the iterables to chain.
    :return: an iterator through all `iterables`, and in addition it yields an extra element `connector` between every two iterables, if the `connector` is not `None`.
    """
    if connector is None:
        yield from itertools.chain(*iterables)
    elif iterables:
        yield from iterables[0]
        for it in iterables[1:]:
            yield connector
            yield from it


def join_chain__(connector, *iterables, atom_types=(str,)):
    """
    The same as `join_chain`, and supports atom types (i.e. a type that is iterable by Python, but should be treated as atoms in terms of data processing, for example, a string).

    >>> import utix.listex as lix
    >>> from itertools import product
    >>> print(list(lix.join_chain__(0, [1, 2], 'abc', product((3, 4), (5, 6)))) == [1, 2, 0, 'abc', 0, (3, 5), (3, 6), (4, 5), (4, 6)])

    """
    if not atom_types:
        return join_chain(connector, *iterables)
    if iterable__(iterables[0], atom_types=atom_types):
        yield from iterables[0]
    else:
        yield iterables[0]

    if connector is not None:
        for item in iterables[1:]:
            yield connector
            if iterable__(item, atom_types=atom_types):
                yield from item
            else:
                yield item
    else:
        for item in iterables[1:]:
            if iterable__(item, atom_types=atom_types):
                yield from item
            else:
                yield item


# endregion

# region search

def find_sub_list(src_list, sub, start: int = 0, end: int = None, return_sub_end: bool = False):
    src_len, sub_len = len(src_list), len(sub)
    if end is None:
        end = src_len - sub_len
    else:
        end -= sub_len
    if return_sub_end:
        while start < end:
            if src_list[start:(start + sub_len)] == sub:
                start += sub_len
                yield start
            else:
                start += 1
    else:
        while start < end:
            if src_list[start:(start + sub_len)] == sub:
                yield start
                start += sub_len
            else:
                start += 1


def find_first_larger_than(_iterable, target):
    """
    Searches the provided sequence for the index of the first element larger than the specified `target`.
    For example, the following returns 3, because the element `5` at index `3` is the first element in the list larger than `4`
    >>> import utix.listex as lx
    >>> lx.find_first_larger_than([1,3,1,5,5,6], 4)

    :param _iterable: the sequence to search
    :param target: the target value.
    :return: the index of the first element larger than the `target`.
    """
    try:
        return next(x[0] for x in enumerate(_iterable) if x[1] > target)
    except StopIteration:
        return None


def find_first_larger_than_or_equal_to(_iterable, target):
    """
    Searches the provided sequence for the index of the first element larger than or equal to the specified `target`.
    """
    try:
        return next(x[0] for x in enumerate(_iterable) if x[1] >= target)
    except StopIteration:
        return None


def find_first_smaller_than(_iterable, target):
    """
    Searches the provided sequence for the index of the first element smaller than the specified `target`.
    """
    try:
        return next(x[0] for x in enumerate(_iterable) if x[1] < target)
    except StopIteration:
        return None


def find_first_smaller_than_or_equal_to(_iterable, target):
    """
    Searches the provided sequence for the index of the first element smaller than or equal to the specified `target`.
    """
    try:
        return next(x[0] for x in enumerate(_iterable) if x[1] <= target)
    except StopIteration:
        return None


def find_first(_iterable, condition: Callable):
    """
    Searches the provided sequence for the index of the first element satisfying the provided `condition`.
    """
    try:
        return next(x[0] for x in enumerate(_iterable) if condition(x[1]))
    except StopIteration:
        return None


def beam_find_first(src, start_idx: int, condition: Callable, max_forward: int = None, max_backward: int = None, search_forward_then_backward: bool = True):
    """
    Returns the index of the first item satisfying the `condition`;
        the search starts in the middle at index `start_idx`, and the searches forward up to the position `start_idx + max_forward` (inclusive);
        if the search is not successful, then it looks backward down to the position `start_idx - max_backward` (inclusive).
    If `first_forward_then_backward` is set `False`, then it searches backward first, and then searches forward.

    >>> import utix.listex as lix
    >>> condition = lambda x: x % 5 == 0
    >>> print(lix.beam_find_first(list(range(20)), start_idx=7, condition=condition, max_forward=5, max_backward=5) == 10)  # forward search succeeds
    >>> print(lix.beam_find_first(list(range(20)), start_idx=7, condition=condition, max_forward=5, max_backward=5, search_forward_then_backward=False) == 5)  # backward search runs first and succeeds
    >>> print(lix.beam_find_first(list(range(20)), start_idx=7, condition=condition, max_forward=2, max_backward=5) == 5)  # forward search fails, but then backward search succeeds
    >>> print(lix.beam_find_first(list(range(20)), start_idx=7, condition=condition, max_forward=2, max_backward=2) == 5)  # forward search fails, but then backward search succeeds
    >>> print(lix.beam_find_first(list(range(20)), start_idx=8, condition=condition, max_forward=1, max_backward=2) == -1)
    >>> print(lix.beam_find_first(list(range(20)), start_idx=5, condition=condition, max_forward=2, max_backward=2) == 5)

    :param src: the list to search.
    :param start_idx: the beam search start position.
    :param condition: the condition that returns `True` to indicate a desired value is found.
    :param max_forward: the maximum number of values to check on the `condition` in front of the `start_idx`.
    :param max_backward: the maximum number of values to check on the `condition` before the `start_idx`.
    :param search_forward_then_backward: `True` to search forward first; if no item is hit, then search backward; `False` if to search backward first, and then search forward.
    :return: the index of the first item satisfying the `condition` inside the search window `start_idx - max_backward` to `start_idx + max_forward` (both inclusive).
    """
    _start_idx = start_idx
    if search_forward_then_backward:
        max_forward = len(src) if max_forward is None else min(max_forward + start_idx + 1, len(src))
        while start_idx < max_forward:
            if condition(src[start_idx]):
                return start_idx
            start_idx += 1
        start_idx = _start_idx - 1
        max_backward = -1 if max_backward is None else max(0, start_idx - max_backward)
        while start_idx > max_backward:
            if condition(src[start_idx]):
                return start_idx
            start_idx -= 1
    else:
        max_backward = -1 if max_backward is None else max(0, start_idx - max_backward - 1)
        while start_idx > max_backward:
            if condition(src[start_idx]):
                return start_idx
            start_idx -= 1
        start_idx = _start_idx + 1
        max_forward = len(src) if max_forward is None else min(max_forward + start_idx, len(src))
        while start_idx < max_forward:
            if condition(src[start_idx]):
                return start_idx
            start_idx += 1
    return -1


def find_sub_list_overlap(src_list, sub, start: int = 0, end: int = None):
    src_len, sub_len = len(src_list), len(sub)
    if end is None:
        end = src_len - sub_len
    else:
        end -= sub_len
    while start < end:
        if src_list[start:(start + sub_len)] == sub:
            yield start
            start += 1
        else:
            start += 1


def obsolete_find_sub_list(src_list, tgt_list, tag=None):
    results = []
    sll = len(tgt_list)
    for ind in (i for i, e in enumerate(src_list) if e == tgt_list[0]):
        if src_list[ind:ind + sll] == tgt_list:
            results.append((ind, ind + sll, tag) if tag is not None else (ind, ind + sll))

    return results


# endregion

# region misc

def list__(it, max_length: int = None, padding=0, dtype: Callable = None):
    """
    Converting a sequence of objects to a list, like the build-in function `list`, with extra options.
    :param it: the sequence of objects to convert to a list.
    :param max_length: specifies a positive integer to limit the length of the list; otherwise no list length limit.
    :param padding: effective if `max_length` is a positive integer; adds extra values at the end to ensure the list
    :param dtype: the objects are converted to this type in the returned list.
    :return: a list objects from `it`.
    """
    if max_length > 0:
        if isinstance(it, (list, tuple)):
            it = list(it[:max_length]) if dtype is None else list(islice(map(dtype, it), max_length))
        else:
            it = list(islice(it, max_length)) if dtype is None else list(islice(map(dtype, it), max_length))

        return it if padding == 'no padding' else it + [padding] * (max_length - len(it))
    elif dtype is None:
        return list(it)
    else:
        return [dtype(_x) for _x in it]


def onehot_list(dim, hots):
    l = [0] * dim
    if isinstance(hots, int):
        l[hots] = 1
    else:
        for hot in hots:
            l[hot] = 1
    return l


def ensure_list(obj, atom_types=(str,)):
    """
    A convenient function that returns a possible list equivalent of `obj`. This function is usually applied to process parameters, allowing inputs being either an iterable or a non-iterable element.
    Returns a singleton `[obj]` if the type of `obj` is in `atom_types`;
    otherwise, returns `obj` itself if `obj` is a list, and returns `list(obj)` if `obj` is a tuple;
    otherwise, returns a list with all elements from `obj` if `obj` is an iterable and the type of obj is not one of the `atom_types`.
    Returns `None` if `obj` is `None`.
    Otherwise, returns a singleton list with `obj` as the only element.

    For example,
    >>> import utix.listex as lx
    >>> a = [1,2,3,4]
    >>> print(lx.ensure_list(a)) # [1,2,3,4]
    >>> print(lx.ensure_list(a) is a) # True
    >>> a = (1,2,3,4)
    >>> print(lx.ensure_list(a)) # [1,2,3,4]
    >>> a = (x for x in range(4))
    >>> print(lx.ensure_list(a)) # [0,1,2,3]
    >>> a = 1
    >>> print(lx.ensure_list(a)) # [1]
    >>> a = (1,2,3,4)
    >>> print(lx.ensure_list(a, atom_types=(str, tuple))) # [(1,2,3,4)]
    """
    if atom_types and isinstance(obj, atom_types):
        return [obj]
    elif isinstance(obj, list):
        return obj
    elif isinstance(obj, tuple):
        return list(obj)
    elif obj is None:
        return None
    elif iterable(obj):
        return list(obj)
    else:
        return [obj]


def ensure_list_or_tuple(obj, atom_types=(str,)):
    """
    The same as `ensure_list`; the only difference is that it directly returns a tuple object without converting it to a list.
    """
    if atom_types and isinstance(obj, atom_types):
        return [obj]
    elif isinstance(obj, (list, tuple)):
        return obj
    elif obj is None:
        return None
    elif iterable(obj):
        return list(obj)
    else:
        return [obj]


def ensure_list__(obj, atom_types=(str,)):
    """
    The same as `ensure_list`, except for that an error will be thrown if `obj` cannot be converted to a non-singleton list.

    For example,
    >>> from utix.listex import ensure_list__
    >>> a = 1
    >>> print(ensure_list__(a)) # ! error

    For another example,
    >>> from utix.listex import ensure_list__
    >>> a = (1,2,3,4)
    >>> print(ensure_list__(a, atom_types=(str, tuple))) # ! error
    """
    if atom_types and isinstance(obj, atom_types):
        raise TypeError(f'the provided object `{obj}` is of one of the atom types `{atom_types}` and hence cannot be converted to a list')
    elif isinstance(obj, list):
        return obj
    elif isinstance(obj, tuple):
        return list(obj)
    elif obj is None:
        return None
    elif iterable(obj):
        return list(obj)
    else:
        raise TypeError(f'the provided object `{obj}` cannot be converted to a list equivalent; it must be a Python list, tuple or an iterable not of types in `{atom_types}`')


def ensure_list_or_tuple__(obj, atom_types=(str,)):
    """
    The same as `ensure_list__`; the only difference is that it directly returns a tuple object without converting it to a list.
    """
    if atom_types and isinstance(obj, atom_types):
        raise TypeError(f'the provided object `{obj}` is of one of the atom types `{atom_types}` and hence cannot be converted to a list')
    elif isinstance(obj, (list, tuple)):
        return obj
    elif obj is None:
        return None
    elif iterable(obj):
        return list(obj)
    else:
        raise TypeError(f'the provided object `{obj}` cannot be converted to a list equivalent; it must be a Python list, tuple or an iterable not of types in `{atom_types}`')


def make_list_of_type(objs, type):
    return (
        None if objs is None
        else ([objs] if isinstance(objs, type)
              else ([type(**objs)] if isinstance(objs, Mapping)
                    else list(x if isinstance(x, type)
                              else (type(**x) if isinstance(x, Mapping)
                                    else (type(*x) if isinstance(x, (tuple, list))
                                          else type(x))) for x in objs
                              )))
    )


def make_tuple_of_type(objs, type, atom_types=(str,)):
    """
    This function is used to create a tuple of objects ob type `type` from the provided `objs`, after applying intuitive conversions.

    Makes a tuple of the specified `type` out of the provided `objs`. The rule is
    1) if `objs` is of type `type`, then returns a one-element tuple `(objs,)`;
    2) otherwise, if `objs` is a mapping, then returns a one-element tuple `(type(**objs),)`,
    3) otherwise, if `objs` is iterable, then for each `obj` in `objs`,
        3a) if `obj` is of type `type`, then keeps what it is;
        3b) if `type` is a mapping, then converts `obj` to `type(**obj)`;
        3c) if `type` is a list or tuple, then converts `obj` to `type(*obj)`;
       then returns a tuple of `obj` in `objs` after applying rule 3a), 3b) and 3c).
    4) otherwise, returns a single-element tuple `(type(objs),)`.

    Straightforward Creation.
    -------------------------
    >>> import utix.listex as lx
    >>> print(lx.make_tuple_of_type(33, int) == (33, ))
    >>> print(lx.make_tuple_of_type('33', int) == (33, ))
    >>> print(lx.make_tuple_of_type(('33', '34', '35'), int) == (33, 34, 35))

    Each in `objs` is the initalizer parameters or `type`.
    ------------------------------------------------------
    >>> class A:
    >>>     def __init__(self, para1, para2):
    >>>         self.a = para1
    >>>         self.b = para2
    >>> tups = lx.make_tuple_of_type(({ 'para1': 1, 'para2': 2}, { 'para1': 3, 'para2': 4}), A)
    >>> print(tups[0].a == 1 and tups[0].b == 2 and tups[1].a == 3 and tups[1].b == 4)
    >>> tups = lx.make_tuple_of_type(([5, 6], [7, 8]), A)
    >>> print(tups[0].a == 5 and tups[0].b == 6 and tups[1].a == 7 and tups[1].b == 8)

    :param objs: the objects to be converted as a tuple.
    :param type: the object type for elements in the returned tuple.
    :param atom_types: used to determine if `objs` is an iterable; see also `utix.general.iterable__`.
    :return: a tuple of objects of type `type`; the objects are from `objs`, after applying rule-based conversions as detailed above.
    """
    return (
        None if objs is None
        else ((objs,) if isinstance(objs, type)
              else ((type(**objs),) if isinstance(objs, Mapping)
                    else tuple(x if isinstance(x, type)
                               else (type(**x) if isinstance(x, Mapping)
                                     else (type(*x) if isinstance(x, (tuple, list))
                                           else type(x))) for x in objs
                               )) if iterable__(objs, atom_types=atom_types) else (type(objs),))
    )


def seg_tags(seg_end_idxes, seq_len: int):
    """
    Consider the list has several consecutive segments, where the indices of the end of the segments are provided by `seg_end_idxes`,
    this method returns a list of integers that can be viewed as segmentation labels for the list.

    For example, the following list `a` can be viewed as a list with several sub-lists as its 'segments', and the delimiter is two consecutive 0s,
    >>> import utix.listex as lix
    >>> a = [2, 3, 6, 0, 0, 2, 5, 1, 5, 0, 0, 3, 1, 3, 4, 0, 0, 2, 5]
    >>> needle = [0, 0]
    >>> needle_idxes = lix.find_sub_list(a, needle, return_sub_end=True) # yield the positions of the segmentation needles
    >>> print(lix.seg_tags(seg_end_idxes=needle_idxes, seq_len=len(a)) == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3])

    :param seg_end_idxes: the indexes of the end of each segment (plus 1, because for Python the end index of a slice is exclusive).
    :param seq_len: the sequence length; the returned tag list also has this length.
    :return: a segment tag list.
    """
    if seg_end_idxes:
        output = np.empty(seq_len, dtype=int)
        start, i = 0, -1
        for i, end in enumerate(seg_end_idxes):
            if start >= seq_len:
                return output.tolist()
            output[start:end] = i
            start = end
        if i != -1:
            output[start:seq_len] = i + 1
        return output.tolist()


def slide_window(seq: Union[List, Tuple], window_len, step_size, offsets=None, must_not_exceed_window_size=True):
    """
    Iterates through sub lists by a sliding window.

    For example,
    >>> import utix.listex as lx
    >>> seq = list(range(0, 21))
    >>> offsets = [0, 3, 4, 8, 15, 17]
    >>> sub_lists = list(lx.slide_window(seq, window_len=5, step_size=2, offsets=offsets))
    >>> print(sub_lists == [[0, 1, 2, 3], [4, 5, 6, 7], [15, 16, 17, 18, 19], [17, 18, 19, 20]])
    >>> sub_lists = list(lx.slide_window(seq, window_len=5, step_size=2, offsets=offsets, must_not_exceed_window_size=False))
    >>> print(sub_lists == [[0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [15, 16, 17, 18, 19, 20]])
    >>> sub_lists = list(lx.slide_window(seq, window_len=5, step_size=2, offsets=None))
    >>> print(sub_lists == [[0, 1, 2, 3, 4], [2, 3, 4, 5, 6], [4, 5, 6, 7, 8], [6, 7, 8, 9, 10], [8, 9, 10, 11, 12], [10, 11, 12, 13, 14], [12, 13, 14, 15, 16], [14, 15, 16, 17, 18]])

    :param seq: the list to apply the sliding window.
    :param window_len: the window size.
    :param step_size: the sliding step size.
    :param offsets: contains positions in the list that segment the whole list into 'pieces'; each sub list should contain entire pieces (i.e. elements in piece should be yieled into two sub lists) except for the ending pieces.
    :param must_not_exceed_window_size: `True` if each sub list must have a length less than or equal to `window_len`; otherwise the length of a sub list could exceed the window size.
                                        This parameter is only effective when `offsets` is provided; otherwise, each sub list will always have the same length as the `window_len`.
    :return: sub lists by sliding a window through the list `seq`.
    """

    if offsets:
        start_idx = 0
        while start_idx < len(offsets):
            start = offsets[start_idx]
            end_idx = find_first_larger_than(offsets[start_idx:], start + window_len)
            if end_idx is None:
                if len(seq) - start > window_len and must_not_exceed_window_size:
                    yield from (seq[_start:(_start + window_len)] for _start in range(start,
                                                                                      len(seq) - window_len + step_size + 1,  # see the discussion in `slide_window__`
                                                                                      step_size))
                else:
                    yield seq[start:]
                break
            else:
                end_idx += start_idx - must_not_exceed_window_size
                if end_idx == start_idx:
                    break
                end = offsets[end_idx]
                yield seq[start:end]
                start_idx += step_size
    else:
        yield from (seq[start:(start + window_len)] for start in range(0,
                                                                       len(seq) - window_len + step_size + 1,  # see the discussion in `slide_window__`
                                                                       step_size))


def slide_window__(seq: Union[List, Tuple], window_len, step_size, offsets=None, must_not_exceed_window_size=True):
    """
    This same as `slide_window`, but yields the `start` and the `end` of the windows.

    For example,
    >>> from utix.listex import slide_window__
    >>> seq = list(range(0, 21))
    >>> offsets = [0, 3, 4, 8, 15, 17]
    >>> sub_lists = list(slide_window__(seq, window_len=5, step_size=2, offsets=offsets))
    >>> print(sub_lists == [(0, 4), (4, 8), (15, 20), (17, 21)])
    >>> sub_lists = list(slide_window__(seq, window_len=5, step_size=2, offsets=offsets, must_not_exceed_window_size=False))
    >>> print(sub_lists == [(0, 8), (4, 15), (15, 21)])
    >>> sub_lists = list(slide_window__(seq, window_len=5, step_size=2, offsets=None))
    >>> print(sub_lists == [(0, 5), (2, 7), (4, 9), (6, 11), (8, 13), (10, 15), (12, 17), (14, 19)])
    """

    # NOTE the range end for the `start` pointer should be 'len(seq) - window_len + step_size + 1';
    # this is to ensure all elements in the list are covered by the sliding window.
    # First, if the range for the `start` pointer is `range(0, len(seq), step_size)`, then it causes the ending windows being sub windows of a previous window;
    #   consider `len(seq) == 12`, `window_size == 6` and `step_size == 3`, then it yields `(0, 6), (3, 9), (6, 12), (9, 12)`, where the last window `(9, 12)` is a sub window of the previous one `(6, 12)`.
    # Then `len(seq) - window_len` won't work, and consider the case of `len(seq) == 12`, `window_size == 6`, `step_size==3`,
    #   then `start < len(seq) - window_len` (which is `start < 6` in this case) make it only yield one window `(0, 6), (3, 9)`.
    # Also `len(seq) - window_len + 1` won't work either, and consider the case of `len(seq) == 13`, `window_size == 6` and `step_size == 2`;
    #   then `start < len(seq) - window_len + 1` (which is `start < 8` in this case) make it will yield `(0, 6), (2, 8), (4, 10), (6, 12)`, missing the final window `(8, 13)`.
    # We can mathematically prove that if we set `start < len(seq) - window_len + 1`, then it will only miss a window of size not wider than the step size (proof by contradiction),
    #   then it is easy to see `start < len(seq) - window_len + step_size + 1` is the correct range end for `start` pointer.

    if offsets:
        start_idx = 0
        while start_idx < len(offsets):
            start = offsets[start_idx]
            end_idx = find_first_larger_than(offsets[start_idx:], start + window_len)
            if end_idx is None:
                if len(seq) - start > window_len and must_not_exceed_window_size:
                    yield from ((_start, min(_start + window_len, len(seq))) for _start in range(start,
                                                                                                 len(seq) - window_len + step_size + 1,  # see above discussion above
                                                                                                 step_size))
                else:
                    yield start, len(seq)
                break
            else:
                end_idx += start_idx - must_not_exceed_window_size
                if end_idx == start_idx:
                    break
                end = offsets[end_idx]
                yield start, end
                start_idx += step_size
    else:
        yield from ((start, min(start + window_len, len(seq))) for start in range(0,
                                                                                  len(seq) - window_len + step_size + 1,  # see the discussion in `slide_window__`
                                                                                  step_size))


def arrange(ls, order):
    return [ls[x] for x in order]


# endregion


def alternating_merge(*iterables):
    pending = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))


def fold_list(list_to_fold: List, group_sizes: List, shuffle_each_group=True, shuffle_output=True):
    grouped = []
    begin = 0
    for group_size in group_sizes:
        end = begin + group_size
        if shuffle_each_group:
            group = list_to_fold[begin:end]
            random.shuffle(group)
            grouped.append(group)
        else:
            grouped.append(list_to_fold[begin:end])
        begin = end
    if shuffle_output:
        random.shuffle(grouped)
    return grouped


def iter_list_of_lists(list_of_lists: List[List]):
    for l in list_of_lists:
        for x in l:
            yield x


def _iter_nested_lists_sizes(lists: list, idx=0, max_idx=0):
    yield idx, len(lists)
    if idx < max_idx:
        for row in lists:
            yield from _iter_nested_lists_sizes(row, idx + 1, max_idx)


def nested_lists_dim(nl: list, _top_level_dim=0):
    """
    Gets the dimension of a nested list.

    For example,
    >>> import utix.listex as lx
    >>> a = [[1,2,3], [4, 5], [6,7]]
    >>> lx.nested_lists_dim(a)
    2
    >>> a = [[[1],[2,3],[4,5,6]], [[7], [8, 9, 10]], [[11],[12, 13]]] # can be a ragged list
    >>> lx.nested_lists_dim(a)
    3
    >>> lx.nested_lists_dim(a, _top_level_dim=1) # add `1` to the dimension of the input nested list
    4

    :param nl: the nested list.
    :param _top_level_dim: the default top level dimension number is 0; otherwise, this number will add to the dimension of the input nested list.
    :return: the dimension of the given nested list.
    """
    return nested_lists_dim(nl[0], _top_level_dim + 1) if isinstance(nl, list) else _top_level_dim


def nested_lists_regular_shape(nl: list):
    """
    Gets the regular shape of a nested list, i.e. the shape where each dimension size is the maximum length of the nested list in that dimension.

    >>> import utix.listex as lx
    >>> lx.nested_lists_regular_shape([[1,2], [3]])
    (2, 2)
    >>> lx.nested_lists_regular_shape([[[1],[2,3],[4,5,6]], [[7], [8, 9, 10]], [[11],[12, 13]]])
    (3, 3, 3)

    :param nl: get the regular shape of this nested list.
    :return: the regular shape of the provided nested list.
    """
    dim = nested_lists_dim(nl)
    shape = [0] * dim
    for level, length in _iter_nested_lists_sizes(nl, max_idx=dim - 1):
        shape[level] = max(shape[level], length)

    return tuple(shape)


def nested_lists_regularize(nl: list, padding=0, dtype=None, **kwargs):
    """
    Make a nested list regular-shaped.
    Internally, this function is a simple wrap around the `utix.npex.array__` function.

    >>> import utix.listex as lx
    >>> lx.nested_lists_regularize([[1,2], [3]])
    [[1, 2], [3, 0]]
    >>> lx.nested_lists_regularize([[[1],[2,3],[4,5,6]], [[7], [8, 9, 10]], [[11],[12, 13]]])
    [[[1, 0, 0], [2, 3, 0], [4, 5, 6]], [[7, 0, 0], [8, 9, 10], [0, 0, 0]], [[11, 0, 0], [12, 13, 0], [0, 0, 0]]]

    :param nl: a nested list.
    :param padding: the padding value.
    :param dtype: the numpy data type; the nested list is first converted to a numpy array and then converted back to a nested list; therefore we may specify the `dtype` to convert value types in the nested list.
    :param kwargs: other arguments for `numpy.array` function.
    :return: a regular-shaped list with all the values from `nl` by padding it with the `padding`.
    """
    from utix.npex import array__
    return array__(nl, padding=padding, dtype=dtype, **kwargs).tolist()


def nested_lists_get(lists: list, index: Union[List, Tuple]):
    """
    Recursive retrieval from a nested list. The `index` parameter specifies a 'path' to retrieve the target data.

    >>> import utix.listex as lx
    >>> lx.nested_lists_get([[1, 2, [3, 4]], [5, 6, [7, [8, 9]]]], index=[0, 2, 0]) == 3
    True
    >>> lx.nested_lists_get([[1, 2, [3, 4]], [5, 6, [7, [8, 9]]]], index=[1, 2, 1]) == [8, 9]
    True

    """
    for i in index:
        if i < len(lists):
            lists = lists[i]
        else:
            return None
    return lists


def agg(key_list, value_list, agg_func=None, include_rank=None):
    d = {}
    if include_rank is None:
        for key, val in zip(key_list, value_list):
            rec = d.get(key, None)
            if rec:
                rec[0] += 1
                rec[1].append(val)
            else:
                d[key] = [1, [val]]
    elif include_rank is min or include_rank == 'min':
        for i, (key, val) in enumerate(zip(key_list, value_list)):
            rec = d.get(key, None)
            if rec:
                rec[0] += 1
                rec[2].append(val)
            else:
                d[key] = [1, i, [val]]
    elif include_rank is max or include_rank == 'max':
        for i, (key, val) in enumerate(zip(key_list, value_list)):
            rec = d.get(key, None)
            if rec:
                rec[0] += 1
                rec[1] = i
                rec[2].append(val)
            else:
                d[key] = [1, i, [val]]
    else:
        raise ValueError('include_rank can only be None, min or max')
    if agg_func is None:
        for key in d:
            d[key] = tuple(d[key])
    else:
        if include_rank is None:
            for key in d:
                rec = d[key]
                d[key] = tuple((rec[0], agg_func(rec[1])))
        else:
            for key in d:
                rec = d[key]
                d[key] = tuple((rec[0], rec[1], agg_func(rec[2])))
    return d


def agg_multiple(key_list, values, agg_funcs=None):
    d = {}

    for key, vals in zip(key_list, zip__(*values)):
        rec = d.get(key, None)
        if rec:
            rec[0] += 1
            for i, val in enumerate(vals):
                if callable(val):
                    rec[i + 1].append(val(key))
                else:
                    rec[i + 1].append(val)
        else:
            d[key] = [1, *([val(key)] if callable(val) else [val] for val in vals)]
    if agg_funcs is None:
        for key in d:
            d[key] = tuple(d[key])
    else:
        for key in d:
            rec = d[key]
            d[key] = tuple((rec[0], *((agg_func(rec[i + 1]) if agg_func else rec[i + 1]) for i, agg_func in enumerate(agg_funcs))))
    return d

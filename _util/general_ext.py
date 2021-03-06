import code
import copy
import importlib
import inspect
import json
import os
import random
import sys
import traceback
from ast import literal_eval
from collections import Counter, OrderedDict
from contextlib import contextmanager
from functools import partial, reduce
from itertools import chain
from typing import Tuple, Callable, Iterator, Union, Iterable, Mapping, List

import tqdm
from IPython import get_ipython
from colorama import Fore
from colorama import init as colorama_init

"""
This file contains commonly and frequently used utility functions.
"""


# region misc

class ref:
    """
    A simple convenient wrap around a value, so that function is capable of modifying its value if a `ref` object is passed into it.
    Analogous to a pointer or 'pass by reference' in other programming languages.
    NOTE this wrap supports most arithmetic operations and string related operations, but not all magic functions.

    >>> from utix.general import ref
    >>> a = ref(1)
    >>> print(a + 2 == 3)
    >>> print(a == 1) # `a` is still 1
    >>> a += 2
    >>> print(a == 3) # `a` is now updated

    >>> # the following returns different types
    >>> print(type(2 + a) is int)
    >>> print(type(a + 2) is ref)
    >>> print(a.value is int)

    >>> # pass the value by reference
    >>> def func(x):
    >>>     x += 2
    >>> func(a)
    >>> print(a == 5)
    >>> b = 3
    >>> func(b)
    >>> print(b == 3) # `b` is not updated

    """
    __slots__ = ('value',)

    def __init__(self, _v):
        self.value = _v

    def __add__(self, other):
        return ref(self.value + other)

    def __iadd__(self, other):
        self.value += other
        return self

    def __radd__(self, other):
        return other + self.value

    def __sub__(self, other):
        return ref(self.value - other)

    def __isub__(self, other):
        self.value -= other
        return self

    def __rsub__(self, other):
        return other - self.value

    def __mul__(self, other):
        return ref(self.value * other)

    def __imul__(self, other):
        self.value *= other
        return self

    def __rmul__(self, other):
        return other * self.value

    def __matmul__(self, other):
        return ref(self.value @ other)

    def __imatmul__(self, other):
        self.value @= other
        return self

    def __rmatmul__(self, other):
        return other @ self.value

    def __truediv__(self, other):
        return ref(self.value / other)

    def __itruediv__(self, other):
        self.value /= other
        return self

    def __rtruediv__(self, other):
        return other / self.value

    def __floordiv__(self, other):
        return ref(self.value // other)

    def __ifloordiv__(self, other):
        self.value //= other
        return self

    def __rfloordiv__(self, other):
        return other // self.value

    def __mod__(self, other):
        return ref(self.value % other)

    def __imod__(self, other):
        self.value %= other
        return self

    def __rmod__(self, other):
        return other % self.value

    def __divmod__(self, other):
        return ref(divmod(self.value, other))

    def __rdivmod__(self, other):
        return divmod(other, self.value)

    def __pow__(self, other):
        return ref(self.value ** other)

    def __ipow__(self, other):
        self.value **= other
        return self

    def __rpow__(self, other):
        return other ** self.value

    def __or__(self, other):
        return ref(self.value | other)

    def __ior__(self, other):
        self.value |= other
        return self

    def __ror__(self, other):
        return other | self.value

    def __and__(self, other):
        return ref(self.value & other)

    def __iand__(self, other):
        self.value &= other
        return self

    def __rand__(self, other):
        return other % self.value

    def __xor__(self, other):
        return ref(self.value ^ other)

    def __ixor__(self, other):
        self.value ^= other
        return self

    def __rxor__(self, other):
        return other ^ self.value

    def __lshift__(self, other):
        return ref(self.value << other)

    def __ilshift__(self, other):
        self.value <<= other
        return self

    def __rlshift__(self, other):
        return other << self.value

    def __rshift__(self, other):
        return ref(self.value >> other)

    def __irshift__(self, other):
        self.value >>= other
        return self

    def __rrshift__(self, other):
        return other >> self.value

    def __eq__(self, other):
        return self.value == other

    def __ne__(self, other):
        return self.value != other

    def __lt__(self, other):
        return self.value < other

    def __le__(self, other):
        return self.value <= other

    def __gt__(self, other):
        return self.value > other

    def __ge__(self, other):
        return self.value >= other

    def __neg__(self):
        self.value = -self.value
        return self

    def __pos__(self):
        self.value = +self.value
        return self

    def __abs__(self):
        self.value = abs(self.value)
        return self

    def __invert__(self):
        self.value = ~self.value
        return self

    def __floor__(self):
        self.value = self.value.__floor__()
        return self

    def __ceil__(self):
        self.value = self.value.__ceil__()
        return self

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)

    def __complex__(self):
        return complex(self.value)

    def __str__(self):
        return self.value.__str__()

    def __bool__(self):
        return self.value.__bool__()

    def __repr__(self):
        return self.value.__repr__()

    def __hash__(self):
        return self.value.__hash__()

    def __format__(self, format_spec):
        return self.value.__format__(format_spec)

    def copy(self):
        return ref(self.value)


def equals_by_elements(x, y):
    """
    Uses `numpy.testing.assert_equal` to test if two objects are exactly equal in terms of their element data.
    This ignores the objects' original definition of identity by the magic function `__eq__`; for example, two `numpy.nan` objects are always non-equal by their definition, but will be equal by this function since their members are identical.
    Very expensive operation, typically only use this for debugging.
    """
    import numpy as np
    try:
        np.testing.assert_equal(x, y)
        return True
    except:
        return False


def equals_by_str(x, y, use_tqdm=False):
    if iterable__(x):
        if iterable__(y):
            it = zip(x, y)
            if use_tqdm:
                it = tqdm.tqdm(it)
            return all(str(x) == str(y) for x, y in it)
        else:
            return False
    else:
        return str(x) == str(y)


def tuple__(x, defaults: Union[List, Tuple, int] = None, raise_err_when_length_exceeds_defaults=False):
    x = tuple(x)
    if isinstance(defaults, int):
        if len(x) == defaults:
            return x
        elif len(x) > defaults:
            if raise_err_when_length_exceeds_defaults is True:
                raise ValueError(f'expected maximum tuple length {defaults}; got {len(x)}')
            elif isinstance(raise_err_when_length_exceeds_defaults, str):
                raise ValueError(raise_err_when_length_exceeds_defaults)
            else:
                return x
        else:
            return x + (None,) * (defaults - len(x))
    elif defaults is None:
        return x
    else:
        len_defaults = len(defaults)
        if len(x) == len_defaults:
            return x
        elif len(x) > len_defaults:
            if raise_err_when_length_exceeds_defaults is True:
                raise ValueError(f'expected maximum tuple length {len_defaults}; got {len(x)}')
            elif isinstance(raise_err_when_length_exceeds_defaults, str):
                raise ValueError(raise_err_when_length_exceeds_defaults)
            else:
                return x
        else:
            return x + tuple(defaults[len(x):])


def cpu_count():
    """
    Returns the number of cpus on this machine.
    """
    import multiprocessing
    return multiprocessing.cpu_count()


def divide__(x, y, default=0):
    try:
        return x / y
    except:
        return default


def num_divide__(x, y, default=0):
    if y == 0:
        return default

    return x / y


def get__(d: Mapping, *keys, default=None, raise_key_error=False):
    for key in keys:
        if key in d:
            return d[key]
    if raise_key_error:
        raise KeyError(keys)
    return default


# endregion

# region type utilities

@contextmanager
def add_to_path(p):
    import sys
    old_path = sys.path
    sys.path = sys.path[:]
    sys.path.insert(0, p)
    try:
        yield
    finally:
        sys.path = old_path


def path_import(absolute_path):
    '''implementation taken from https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly'''
    with add_to_path(os.path.dirname(absolute_path)):
        spec = importlib.util.spec_from_file_location(absolute_path, absolute_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def iterable(_obj) -> bool:
    """
    Check whether or not an object can be iterated over.
    >>> import utix.general as gx
    >>> print(gx.iterable([1, 2, 3, 4]))
    >>> print(gx.iterable(iter(range(5))))
    >>> print(gx.iterable('123'))
    >>> print(gx.iterable(123) == False)
    """
    try:
        iter(_obj)
    except TypeError:
        return False
    return True


def iterable__(_obj, atom_types=(str,)):
    """
    A variant of `iterable`. Returns `True` if the type of `obj` is not in the `atom_types`, and it is iterable.
    By default, the `atom_types` has a single type, the string.

    >>> import utix.general as gx
    >>> print(gx.iterable__('123') == False)
    >>> print(gx.iterable__((1, 2 ,3)))
    >>> print(gx.iterable__((1, 2, 3), atom_types=(tuple,str)) == False)
    """
    return not isinstance(_obj, atom_types) and iterable(_obj)


def iter__(it: Union[Iterator, Iterable], filter: Callable = None, sentinel: Callable = None, atom_types=(str,)):
    """
    A convenient function to create an iterator with rich options.
    Can apply to a function parameter to allow passing a value as if it is an iterable.
    Can specify callable `filter` to skip certain elements in `it`.
    Can specify `sentinel`, either a callable or an object, so that the iteration stops if this `sentinal` returns `True` on an element in the `it`, or it equals an element in `it`; the sentinel will not be yielded.

    Parameter Convenience
    ----------------------
    >>> from utix.general import iter__
    >>> def foo(arr):
    >>>     for x in iter__(arr):
    >>>         print(x)
    >>> foo('a') # treated as if it is `['a']`
    >>> foo(['a', 'b', 'c'])

    Sentinel
    --------
    >>> print(list(iter__(['a', 'b', 'c', 'd'], sentinel='c')) == ['a', 'b'])
    >>> print(list(iter__([1, 2, 3, 4], sentinel=lambda x: x % 3 == 0)) == [1, 2])

    Filter
    ------
    >>> print(list(iter__([1, 2, 3, 4], sentinel=lambda x: x % 3 == 0, filter=lambda x: x % 2 == 1)) == [1])

    :param it: an iterable or iterator; if a non-iterable object (including those of type in `atom_types`) is provided, then that object is treated as a singelton.
    :param filter: a function that returns `False` to skip an element in `it`.
    :param sentinel: a sentinal function or object; the iteration stops if this `sentinal` returns `True` on an element in the `it`, or it equals an element in `it`; the sentinel will not be yielded.
    :param atom_types: an object of of any oft these types will not be treated as an iterable; the default is `(str, )`, meaning the string type is not treated as an iterable.
    :return: an iterator that wraps around the original iterator `it`.
    """
    if iterable__(it, atom_types):
        if sentinel:
            if filter:
                for x in it:
                    if sentinel(x) if callable(sentinel) else (x == sentinel):
                        return
                    elif filter(x):
                        yield x
            else:
                for x in it:
                    if sentinel(x) if callable(sentinel) else (x == sentinel):
                        return
                    yield x
        elif filter:
            yield from (x for x in it if filter(x))
        else:
            yield from it
    else:
        if sentinel:
            if filter:
                if sentinel(it) if callable(sentinel) else (it == sentinel):
                    return
                elif filter(it):
                    yield it
            else:
                if sentinel(it) if callable(sentinel) else (it == sentinel):
                    return
                yield it
        elif not filter or filter(it):
            yield it


def count(it, sort_by_count_desc=False):
    d = Counter(it)
    return OrderedDict(sorted(d.items(), key=lambda x: x[1], reverse=True)) if sort_by_count_desc else d


def count_and_rank(it, min_rank=True, make_tuple=True):
    d = {}
    if min_rank:
        for i, key in enumerate(it):
            rec = d.get(key, None)
            if rec:
                rec[0] += 1
            else:
                d[key] = [1, i]
    else:
        for i, key in enumerate(it):
            rec = d.get(key, None)
            if rec:
                rec[0] += 1
                rec[1] = i
            else:
                d[key] = [1, i]

    if make_tuple:
        for key in d:
            d[key] = tuple(d[key])

    return d


def distinct(it, sort_by_count_desc=False):
    d = Counter(it)
    return sorted(d.keys(), key=lambda x: d[x], reverse=True) if sort_by_count_desc else list(d)


def accumulate(it):
    it = iter(it)
    x = next(it)
    yield x
    for y in it:
        x += y
        yield x


def accumulate_ranges(it, start=None):
    if start is None:
        it = iter(it)
        start = next(it)
    for x in it:
        end = start + x
        yield start, end
        start = end


def exclude_none(it):
    return (x for x in it if x is not None)


def sampled_iter(it, sample_ratio):
    r = sample_ratio % 1
    i = int(sample_ratio - r)
    for item in it:
        for i in range(i):
            yield item
        if r != 0:
            if random.uniform(0, 1) < r:
                yield item


def nonstr_iterable(obj) -> bool:
    """
    Checks whether or not the object is a non-string iterable.
    This is a function equivalent to `iterable__` with the default `atom_types`, with a more expressive name and slightly better performance.

    >>> from timeit import timeit
    >>> import utix.general as gx
    >>> def target1():
    >>>     gx.iterable__('1234')
    >>> def target2():
    >>>     gx.nonstr_iterable('1234')
    >>> print(timeit(target1))
    >>> print(timeit(target2)) # slightly faster

    """
    return (not is_str(obj)) and iterable(obj)


def sliceable(_obj):
    """
    Checks whether or not the object can be sliced.
    >>> import utix.general as gx
    >>> print(gx.sliceable(2) == False)
    >>> print(gx.sliceable((1, 2, 3)) == True)
    >>> print(gx.sliceable('abc') == True)
    >>> print(gx.sliceable([]) == True)
    """
    if not hasattr(_obj, '__getitem__'):
        return False
    try:
        _obj[0:1]
    except:
        return False
    return True


def make_range(_x):
    """
    A convenient function that make a range from the given object.
    The object can be an integer, a list, a tuple, a generator of no more than three items, or a mapping with one required keys 'stop' and two optional keys 'start' and 'step'.
    Mainly used for parameter parsing.

    >>> import utix.general as gx
    >>> print(gx.make_range(5))
    >>> print(gx.make_range((1, 6, 2)))
    >>> print(gx.make_range({ 'stop': 10, 'step': 2 }))

    """
    _t = type(_x)
    if _t is range:
        return _x
    if _t is int:
        return range(_x)
    if isinstance(_x, Mapping):
        return range(_x.get('start', 0), _x['stop'], _x.get('step', 1))
    return range(*_x)


def broadcastable(shape1, shape2) -> bool:
    """
    Checks if two shapes are broadcastable.
    Broadcasting is a common operation in many modern Python packages.
    See https://docs.scipy.org/doc/numpy/user/theory.broadcasting.html#array-broadcasting-in-numpy.

    >>> import utix.general as gx
    >>> print(gx.broadcastable(3, 1) == True)
    >>> print(gx.broadcastable((3, 4, 5), (3, 1, 5)) == True)
    >>> print(gx.broadcastable((3, 4, 5), (4, 1)) == True)
    >>> print(gx.broadcastable((4, 1), (3, 4, 5)) == True)
    >>> print(gx.broadcastable((3, 4), (4, 3)) == False)

    """
    if type(shape1) is int:
        if type(shape2) is int:
            return shape1 == 1 or shape2 == 1 or shape1 == shape2
        else:
            return shape1 == 1 or shape1 == shape2[-1]
    elif type(shape2) is int:
        return shape2 == 1 or shape1[:-1] == shape2
    else:
        for a, b in zip(shape1[::-1], shape2[::-1]):
            if a != 1 and b != 1 and a != b:
                return False
    return True


def broadcastable__(obj1, obj2):
    """
    The same as `broadcastable`, but the inputs are now two objects with the `shape` attribute.
    """
    shape1 = obj1.shape if hasattr(obj1, 'shape') else None
    shape2 = obj2.shape if hasattr(obj2, 'shape') else None
    return broadcastable(shape1, shape2)


def shape_after_broadcast(shape1, shape2):
    """
    Gets the out shape of after broadcasting for two input shapes.
    Broadcasting is a common operation in many modern Python packages.

    >>> import utix.general as gx
    >>> print(gx.shape_after_broadcast(3, 1) == (3,))
    >>> print(gx.shape_after_broadcast((3, 4, 5), (3, 1, 5)) == (3,4,5))
    >>> print(gx.shape_after_broadcast((3, 4, 5), (4, 1)) == (3, 4, 5))
    >>> print(gx.shape_after_broadcast((4, 1), (3, 4, 5)) == (3, 4, 5))
    >>> print(gx.shape_after_broadcast((3, 4), (4, 3)) == None)
    """
    if shape1 is None:
        return shape2
    elif shape2 is None:
        return shape1
    elif type(shape1) is int:
        shape1 = (shape1,)
    if type(shape2) is int:
        shape2 = (shape2,)

    out = list(shape1 if len(shape1) > len(shape2) else shape2)

    for i, (a, b) in enumerate(zip(shape1[::-1], shape2[::-1])):
        if a == 1:
            out[-(i + 1)] = b
        elif b == 1 or a == b:
            out[-(i + 1)] = a
        else:
            return None
    return tuple(out)


def shape_after_broadcast__(obj1, obj2):
    """
    The same as `shape_after_broadcast`, but the inputs are now two objects with the `shape` attribute.
    """
    shape1 = obj1.shape if hasattr(obj1, 'shape') else None
    shape2 = obj2.shape if hasattr(obj2, 'shape') else None
    return shape_after_broadcast(shape1, shape2)


def is_dict(_obj) -> bool:
    """
    A convenient function for `isinstance(_obj, dict)`.
    """
    return isinstance(_obj, dict)


def is_mapping(_obj) -> bool:
    """
    A convenient function for `isinstance(_obj, Mapping)`.
    """
    return isinstance(_obj, Mapping)


def can_dict_like_read(obj):
    """
    Check if the object support dict-like item reading.
    """
    return (
            hasattr(obj, '__getitem__') and hasattr(obj, 'items')
            and not isinstance(obj, type)
    )


def is_dict_like(obj):
    """
    Check if the object is dict-like.
    """
    dict_like_attrs = ("__getitem__", "keys", "__contains__")
    return (
            all(hasattr(obj, attr) for attr in dict_like_attrs)
            and not isinstance(obj, type)
    )


def is_list(obj) -> bool:
    return isinstance(obj, list)


def is_tuple(obj) -> bool:
    return isinstance(obj, tuple)


def is_list_or_tuple(obj) -> bool:
    return isinstance(obj, (tuple, list))


def is_num(obj) -> bool:
    return isinstance(obj, (int, float))


def is_class(variable):
    return isinstance(variable, type)


def is_basic_type(variable):
    return isinstance(variable, (int, float, str, bool))


def is_str(variable):
    return isinstance(variable, str)


def is_named_tuple(obj) -> bool:
    return isinstance(obj, tuple) and hasattr(obj, '_fields')


def has_slots(obj) -> bool:
    return hasattr(obj, '__slots__') and obj.__slots__


def has_fixed_fields(obj):
    return has_slots(obj) or (is_named_tuple(obj) and obj._fields)


def convert_values(_container, _converter, atom_types=(str,)):
    try:
        if isinstance(_container, atom_types):
            return _container(_container)
        else:
            return type(_container)(convert_values(x, _converter, atom_types=atom_types) for x in _container)
    except:
        pass
    return _converter(_container)


def value_type(_container, atom_types=(str,), key=0):
    """
    Gets the type of an atomic value in the provided object container. The container may be nested, and we use `key` to recursively retrieve the inside container.
    NOTE this method generally assumes the values in the container are of the same type.

    >>> import numpy as np
    >>> import utix.general as gx
    >>> 
    >>> print(gx.value_type(np.array([[1, 2], [3, 4]])) is np.int32)
    >>> print(gx.value_type(((True, False), (True, False))) is bool)
    >>> print(gx.value_type([[['a', [0, 1, 2]], 'c']]) is str)

    :param _container: the container holding values.
    :param atom_types: the types that should be treated as an atomic object.
    :param key: the key used to recursively retrieve the inside containers.
    :return: the type of an atomic value.
    """
    try:
        while not isinstance(_container, atom_types):
            _container = _container[key]
    except:
        pass
    return type(_container)


def _default_filter(x, filter_obj):
    return x in filter_obj


def make_filter(filter_obj: Union[Callable, Iterable]):
    return filter_obj if filter_obj is None or callable(filter_obj) else partial(_default_filter, filter_obj=filter_obj)


def compose2(func2, func1):
    """
    A composition of two functions.

    >>> import utix.general as gx
    >>> def f1(x):
    >>>     return x + 2
    >>> def f2(x):
    >>>     return x * 2
    >>> print(gx.compose2(f2, f1)(5) == 14)

    :param func2: the outside function.
    :param func1: the inside function.
    :return: the composed function.
    """

    def _composed(*args, **kwargs):
        return func2(func1(*args, **kwargs))

    return _composed


def compose(*funcs):
    """
    A composition of n functions.
    >>> from timeit import timeit
    >>> import utix.general as gx
    >>> def f1(x):
    >>>     return x + 2
    >>> def f2(x):
    >>>     return x * 2
    >>> print(gx.compose(f2, f1)(5) == 14)

    # `compose2` is faster to compose two functions
    >>> def target1():
    >>>     gx.compose2(f2, f1)(5)
    >>> def target2():
    >>>     gx.compose(f2, f1)(5)
    >>> print(timeit(target1)) # about 30% faster
    >>> print(timeit(target2))

    :param funcs: the functions to compose.
    :return: the composed function
    """
    return reduce(compose2, funcs)


_STRS_TRUE = {'true', 'yes', 'y', 'ok', '1'}
_STRS_FALSE = {'false', 'no', 'n', '0'}


def str2bool(s: str):
    return s.lower() in _STRS_TRUE


def str2int(s: str):
    _s = s.lower()
    return 1 if _s in _STRS_TRUE else (0 if _s in _STRS_FALSE else int(s))


def str2float(s: str, replace_nan=None):
    _s = s.lower()
    if replace_nan is None:
        return 1.0 if _s in _STRS_TRUE else (0.0 if _s in _STRS_FALSE else float(s))
    else:
        return replace_nan if (s == 'nan') else (1.0 if _s in _STRS_TRUE else (0.0 if _s in _STRS_FALSE else float(s)))


def str2bool__(s: str):
    s = s.lower()
    return True if s in _STRS_TRUE else (False if s in _STRS_FALSE else None)


def str2num(s: str):
    try:
        return int(s)
    except:
        try:
            return float(s)
        except:
            return None


def str2numbool(s: str):
    try:
        return int(s)
    except:
        try:
            return float(s)
        except:
            return str2bool__(s)


def str2val(s: str):
    ss = s.strip()

    def _literal_eval():
        try:
            return literal_eval(ss)
        except:
            return s

    if ss[0] == '{':
        try:
            return json.loads(ss)
        except:
            return _literal_eval()
    elif ss[0] == '[' or ss[0] == '(':
        return _literal_eval()
    else:
        try:
            return int(ss)
        except:
            try:
                return float(ss)
            except:
                sl = ss.lower()
                if sl in _STRS_TRUE:
                    return True
                elif sl in _STRS_FALSE:
                    return False
                else:
                    try:
                        return literal_eval(ss)
                    except:
                        return s


def str2val__(s: str, success_label=False):
    """
    Parses a string as its likely equivalent value.
    Typically trie sto convert to integers, floats, bools, lists, tuples, dictionaries.

    >>> import utix.general as gx
    >>> print(gx.str2val__('1') == 1)
    >>> print(gx.str2val__('2.554') == 2.554)
    >>> print(gx.str2val__("[1, 2, 'a', 'b', False]") == [1, 2, 'a', 'b', False])

    :param s: the string to parse as a value.
    :param success_label: returns a tuple, with the first being the parsed value, and the second being a boolean value indicating if the the parse is successful.
    :return: the parsed value if `success_label` is `False`, or a tuple with the second being the parse success flag if `success_label` is `True`.
    """
    ss = s.strip()

    if success_label:
        def _literal_eval():
            try:
                return literal_eval(ss), True
            except:
                return s, False

        if ss[0] == '{':
            try:
                return json.loads(ss), True
            except:
                return _literal_eval()
        elif ss[0] == '[' or ss[0] == '(':
            return _literal_eval()
        else:
            try:
                return int(ss), True
            except:
                try:
                    return float(ss), True
                except:
                    sl = ss.lower()
                    if sl in _STRS_TRUE:
                        return True, True
                    elif sl in _STRS_FALSE:
                        return False, True
                    else:
                        try:
                            return literal_eval(ss), True
                        except:
                            return s, False
    else:
        def _literal_eval():
            try:
                return literal_eval(ss)
            except:
                return s

        if ss[0] == '{':
            try:
                return json.loads(ss)
            except:
                return _literal_eval()
        elif ss[0] == '[' or ss[0] == '(':
            return _literal_eval()
        else:
            try:
                return int(ss)
            except:
                try:
                    return float(ss)
                except:
                    sl = ss.lower()
                    if sl in _STRS_TRUE:
                        return True
                    elif sl in _STRS_FALSE:
                        return False
                    else:
                        try:
                            return literal_eval(ss)
                        except:
                            return s


def list2d(row_count: int, col_count: int, init_val=None):
    return [[init_val] * col_count for _ in range(row_count)]


def get_fields(obj, include_protected=False) -> Iterator[str]:
    if include_protected:
        fields = getattr(obj, '__slots__', getattr(obj, '_fields', ()))
        if hasattr(obj, '__dict__'):
            fields = chain(fields, obj.__dict__.keys())
        return fields
    else:
        fields = (x for x in getattr(obj, '__slots__', getattr(obj, '_fields', ())) if x[0] != '_')
        if hasattr(obj, '__dict__'):
            fields = chain(fields, (x for x in obj.__dict__ if x[0] != '_'))
        return fields


def has_varkw(func: Callable) -> bool:
    return inspect.getfullargspec(func).varkw is not None


def import__(name: str):
    _name = name
    while True:
        try:
            m = __import__(_name)
            break
        except Exception as err:
            try:
                ridx = _name.rindex('.')
                _name = _name[:ridx]
                continue
            except:
                raise extra_msg(err, f'unable to import `{name}`')

    if len(m.__name__) == name:
        return m
    for p in name[len(m.__name__) + 1:].split('.'):
        try:
            m = getattr(m, p)
        except Exception as err:
            raise extra_msg(err, f'unable to import `{name}`')
    return m


class Importer:
    __slots__ = ('_cache',)

    def __init__(self):
        self._cache = {}

    def __call__(self, name: str):
        c = self._cache.get(name, None)
        if not c:
            c = import__(name)
            self._cache[name] = c
        return c


def full_cls_name(obj):
    return f'{obj.__module__}.{obj.__class__.__name__}'


class JSerializable:

    def __to_dict__(self, type_str: str = None) -> dict:
        d = self.__dict__.copy()
        if hasattr(self, '__slots__'):
            d.update({k: getattr(self, k) for k in self.__slots__})
        if hasattr(self, '_fields'):
            d.update({k: getattr(self, k) for k in self._fields})
        d['__type__'] = type_str if type_str else full_cls_name(self)
        return d

    @classmethod
    def __from_dict__(cls, d: dict, factory: Callable = None):
        d.pop('__type__', None)
        factory = factory or cls
        return factory(**d) if has_varkw(factory) else factory(d)


# endregion

# region sequence merge

"""
Provides sequence merge utilities.
Typically used by other utilities; for example, merging leave values from multiple dictionaries.
"""


def first__(iterables):
    it = iter(iterables)
    if it is iterables:
        first = next(it)
        return first, chain((first,), it)
    else:
        return next(it), iterables


def default_merge_func(first, iterables):
    if isinstance(first, list):
        return sum(iterables, [])
    elif isinstance(first, tuple):
        return sum(iterables, ())
    elif isinstance(first, dict):
        for d in iterables:
            first.update(d)
        return first


def iterable_merge(iterables, merge_funcs: Tuple[Callable, ...] = (default_merge_func,)):
    first, iterables = first__(iterables)
    for merge_func in merge_funcs:
        out = merge_func(first, iterables)
        if out is not None:
            return out
    return tuple(iterables)


# endregion


# region misc


class SlotsTuple:

    def __init__(self):
        self._tup = tuple(getattr(self, x) for x in self.__slots__ if x != '_tup')

    def __getitem__(self, item):
        return self._tup[item]

    def __len__(self):
        return len(self._tup)


class Callables:
    """
    A sequence o callables for the convenience to apply them all at once.

    >>> from utix.general import Callables
    >>> foo = Callables((int, float, str, bool), no_return=False)
    >>> print(foo(3) == (3, 3.0, '3', True))

    """
    __slots__ = ('callables', 'no_return')

    def __init__(self, callables, no_return=True):
        self.callables = tuple(x for x in callables if x is not None)
        self.no_return = no_return

    def __call__(self, *args, **kwargs):
        if self.no_return:
            for x in self.callables:
                x(*args, **kwargs)
        else:
            return tuple(x(*args, **kwargs) for x in self.callables)


def value__(_obj):
    return _obj.value if hasattr(_obj, 'value') else (tuple(map(value__, _obj)) if isinstance(_obj, (tuple, list)) else _obj)


def bool2obj(obj, obj_for_true):
    if obj is True:
        return obj_for_true
    elif obj is False:
        return None
    else:
        return obj


def hasattr_or_exec(_obj, attr_to_see: str, attr_type=None, *args, **kwargs):
    if hasattr(_obj, attr_to_see) and (attr_type is None or isinstance(getattr(_obj, attr_to_see), attr_type)):
        return _obj
    elif callable(_obj):
        __obj = _obj(*args, **kwargs)
        if hasattr(__obj, attr_to_see) and (attr_type is None or isinstance(getattr(__obj, attr_to_see), attr_type)):
            return __obj
        else:
            return _obj


def extra_msg(err: Exception, extra_message: str):
    return type(err)(str(err) + '\n' + extra_message)


def in_interactive_console():
    try:
        return get_ipython() is not None
    except NameError:
        return False


def tqdm_wrap(it, use_tqdm, tqdm_msg, verbose=__debug__):
    if use_tqdm:
        it = tqdm.tqdm(it)
        if tqdm_msg:
            it.set_description(tqdm_msg)
    elif verbose and tqdm_msg is not None:
        print(tqdm_msg)
    return it


def apply_tqdm(func):
    def _wrap(*args, **kwargs):
        use_tqdm = kwargs.pop('use_tqdm', True)
        tqdm_msg = kwargs.pop('tqdm_msg', kwargs.pop('display_msg', None))
        verbose = kwargs.pop('verbose', __debug__)

        if len(args) > 0:
            first_arg = args[0]
            args = args[1:]
        else:
            first_arg = next(iter(kwargs.items()))
            del kwargs[first_arg[0]]
            first_arg = first_arg[1]
        if callable(tqdm_msg):
            tqdm_msg = tqdm_msg()
        if isinstance(first_arg, tqdm):
            first_arg.set_description(tqdm_msg)
        else:
            first_arg = tqdm_wrap(first_arg, use_tqdm=use_tqdm, tqdm_msg=tqdm_msg, verbose=verbose)

        return func(first_arg, *args, **kwargs)

    return _wrap


def setattr_if_none_or_empty(obj, attr: str, val) -> None:
    """
    The same as the build-in function `setattr`, with the difference that this function only set the attribute if the it is `None` or does not currently exist in the object.

    >>> import utix.general as gx
    >>> from collections import namedtuple
    >>> from math import factorial
    >>> from functools import partial

    >>> class Point:
    >>>     def __init__(self, x, y):
    >>>         self.x, self.y = x, y

    >>> p = Point(1, None)
    >>> gx.setattr_if_none_or_empty(p, 'x', 10)
    >>> gx.setattr_if_none_or_empty(p, 'y', 10)
    >>> print(p.x == 1)
    >>> print(p.y == 10)

    :param obj: the object to set the attribute.
    :param attr: the name of the attribute to set.
    :param val: the value to set for the specified attribute.
    """
    if not getattr(obj, attr, None):
        setattr(obj, attr, val)


def setattr_if_none_or_empty__(obj, attr: str, get_val: Callable) -> None:
    """
    The same as `setattr_if_none_or_empty`, but the "value" is a callable `get_val`. If the attribute to set is `None` or does not currently exist in the object, then the callable `get_val` is executed to compute the actual value to set.
    The purpose is avoid unnecessary computation of the value. Sometime the value is expensive to compute, and here we only compute the value if the attribute to set does not currently exists or has a `None` value.

    >>> import utix.general as gx
    >>> from collections import namedtuple
    >>> from math import factorial
    >>> from functools import partial

    >>> class Point:
    >>>     def __init__(self, x, y):
    >>>         self.x, self.y = x, y

    >>> p = Point(1, None)
    >>> get_val = lambda: factorial(10)
    >>> gx.setattr_if_none_or_empty__(p, 'x', get_val)  # get_val will NOT be computed in this line
    >>> gx.setattr_if_none_or_empty__(p, 'y', get_val)  # get_val will be computed in this line
    >>> print(p.x == 1)
    >>> print(p.y == get_val())
    """

    if not getattr(obj, attr, None):
        setattr(obj, attr, get_val())


def getattr2(obj, attr_name: str, attr_type: Callable = None, default_obj=None, raise_error=False):
    """
    A specialized function to get an attribute from `obj` or `obj()`.
    The intention of this function is to provide flexible argument parsing, so that the argument can be either tha class, or an instance.


    :param obj:
    :param attr_name:
    :param default_obj:
    :return:
    """
    if obj is not None:
        if obj is True:
            if default_obj is not None:
                return getattr2(default_obj, attr_name, None)
        elif obj is not False:
            if hasattr(obj, attr_name):
                attrval = getattr(obj, attr_name)
                if attr_type is None or isinstance(attrval, attr_type):
                    return attrval
                elif raise_error:
                    raise ValueError(f'unable to retrieve attribute `{attr_name}` from `{obj}` satisfying the specified `{attr_type}` function')
            elif callable(obj):
                try:
                    obj = obj()
                    attrval = getattr(obj, attr_name)
                    if attr_type is None or isinstance(attrval, attr_type):
                        return attrval
                    elif raise_error:
                        raise ValueError(f'unable to retrieve attribute `{attr_name}` from `{obj}` satisfying the specified `{attr_type}` function')
                except:
                    if default_obj is not None:
                        return getattr2(default_obj, attr_name, None)


def getattr__(obj, name: str, other_name: str, default=None):
    """
    The same as the build-in function `getattr`, with an additional parameter `other_name` so that if the `name` does not exist, then as the alternative, this function tries to retrieve the attribute of `other_name`.

    >>> import utix.general as gx
    >>> from collections import namedtuple
    >>> Point = namedtuple('Point', ['x1', 'x2'])
    >>> Pair = namedtuple('Pair', ['value1', 'value2'])
    >>> p1 = Point(1, 2)
    >>> p2 = Pair(1, 2)
    >>> print(gx.getattr__(p1, 'value1', 'x1') == gx.getattr__(p2, 'value1', 'x1'))
    >>> print(gx.getattr__(p1, 'value2', 'x2') == gx.getattr__(p2, 'value2', 'x2'))

    :param obj: gets an attribute from this object.
    :param name: the name of the attribute to retrieve.
    :param other_name: the alternative attribute name to retrieve.
    :param default: the default value if the object does not have an attribute of `name` or `other_name`.
    :return: the retrieved attribute value.
    """
    return getattr(obj, name) if hasattr(obj, name) else getattr(obj, other_name, default)


def getattr____(obj, *attrs: str, default=None):
    """
    The same as the build-in function `getattr` or the `getattr__` function, with the ability to specify multiple alternative attribute names.
    """
    if len(attrs) == 1:
        return getattr(obj, attrs[0], default)
    elif len(attrs) == 2:
        return getattr__(obj, attrs[0], attrs[1], default)
    else:
        for attr in attrs:
            if hasattr(obj, attr):
                return getattr(obj, attr)
        return default


def exec_callable(obj, callable_names: Iterator[str], arguments, error_tag: str = None, error_msg: str = None):
    """
    Checks if the `obj` has any callable member of a name specified in `method_names`. The first found method will be executed.
    :param obj: the object whose callable member of a name first found in `method_names` will be executed.
    :param callable_names: the callable names to check.
    :param arguments: the arguments for the callable.
    :param error_tag: the error tag to display if none of `callable_names` is found as a callable member of `obj`.
    :param error_msg: the error message to display.
    :return: whatever the found callable returns.
    """
    for method_name in callable_names:
        op = getattr(obj, method_name, None)
        if callable(op):
            return op(*arguments)
    eprint_message(error_tag if error_tag is not None else exec_callable.__name__,
                   error_msg if error_msg is not None else "no method '" + "' or '".join(callable_names) + "' is found for type '" + type(obj).__name__ + "'")


def debug_on_error_wrap(func):
    def _wrap(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as err:
            print(err)
            type, value, tb = sys.exc_info()
            traceback.print_exc()
            last_frame = lambda tb=tb: last_frame(tb.tb_next) if tb.tb_next else tb
            frame = last_frame().tb_frame
            ns = dict(frame.f_globals)
            ns.update(frame.f_locals)
            code.interact(local=ns)

    return _wrap


def try_apply(func, arg):
    try:
        return func(arg)
    except:
        return arg


def try__(func: Callable, afunc: Callable, *args, post_error_raise_check: Callable = None, extra_msg: str = None, **kwargs):
    """
    Tries to run a function. If an error occurs, then
    1) tries to apply the optional `raise_when` function on the arguments, then raise the error with the optional `extra_msg` if `raise_when` returns `True`;
    2) otherwise, runs the alternative function `afunc`.

    >>> import utix.general as gx
    >>> def func1(x):
    >>>     return x[0]
    >>> def func2(x):
    >>>     return x['0']
    >>> print(gx.try__(func1, func2, { '0' : 1 }))

    :param func: the function to try.
    :param afunc: the alternative function to try.
    :param post_error_raise_check: when error occurs, apply this function to check if it returns `True`; if so, raise the error and do not execute `afunc`.
    :param extra_msg: the extra message to display should any error occurs.
    :return: the returned object from the function `func`.
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if not afunc or (post_error_raise_check and post_error_raise_check(*args, **kwargs)):
            raise (type(e)(str(e) + '\n' + extra_msg) if extra_msg else e).with_traceback(sys.exc_info()[2])
        else:
            return afunc(*args, **kwargs)


def try____(func: Callable, afunc: Callable = None, post_error_raise_check: Callable = None, extra_msg: str = None, *args, **kwargs):
    """
    The same as `try__`, and returns an additional bool value indicating if the execution of `func` was successful.
    """
    try:
        return func(*args, **kwargs), True
    except Exception as e:
        if not afunc or (post_error_raise_check and post_error_raise_check(*args, **kwargs)):
            raise (type(e)(str(e) + '\n' + extra_msg) if extra_msg else e).with_traceback(sys.exc_info()[2])
        else:
            return afunc(*args, **kwargs), False


def zip__(*objs, atom_types=(str,)):
    """
    A variant of zip function that allows non-iterables.

    >>> from utix.general import zip__
    >>> z = zip__(0, [1,2,3,4], [5, 6, 7, 8])
    >>> print(list(z) == [(0, 1, 5), (0, 2, 6), (0, 3, 7), (0, 4, 8)])

    """
    it_idxes, its = zip(*((i, obj) for i, obj in enumerate(objs) if iterable__(obj, atom_types=atom_types)))
    objs = list(objs)
    len_its = len(its)
    for items in zip(*its):
        for i in range(len_its):
            objs[it_idxes[i]] = items[i]
        yield tuple(objs)


def unzip(tuples, idx=None):
    if idx is None:
        return zip(*tuples)
    elif idx == 0:
        return next(zip(*tuples))
    elif type(idx) is int:
        return tuple(zip(*tuples))[idx]
    else:
        unzips = tuple(zip(*tuples))
        return (unzips[_idx] for _idx in idx)


def sorted__(_iterable, key, reverse: bool = False, return_tuple=False, return_indexes=False):
    """
    An alternative for the build-in `sorted` function. Allows the `key` be a sequence of values (as the sorting keys) for the `_iterable`.
    There is an extra parameter `return_tuple` for convenience; if it is set `True`, then the return is a sorted tuple; otherwise it is a sorted list.

    >>> import utix.general as gx
    >>> class A:
    >>>     def __init__(self, x):
    >>>         self._x = x
    >>>
    >>>     def __repr__(self):
    >>>         return str(self._x)

    >>> # the following prints out '[1, 3, 5, 7, 9, 0, 2, 4, 6, 8]'
    >>> print(gx.sorted__(list(map(A, range(10))),
    >>>         # we allow the key be a sequence of values, serving as the sorting keys; here the sorting keys are 1,0,1,0,1,0,1,0,1,0, so the odd numbers a promoted to the front because their sorting keys are smaller
    >>>         key=(x % 2 == 0 for x in range(10))))

    :param _iterable: a sequence of objects to sort.
    :param key: the sorting key; can be a function such like the `key` parameter for the build-in sorted function, a sequence of values as the sorting keys.
    :param reverse: `True` to sort descendingly; `False` to sort ascendingly.
    :param return_tuple: `True` to return a tuple; `False` to return a list.
    :return: a list or a tuple of sorted values from the `_iterable`.
    """

    if return_indexes is True:
        return sorted__(((x, i) for i, x in enumerate(_iterable)), key=key, reverse=reverse, return_tuple=return_tuple, return_indexes=False)
    elif return_indexes == 'labels':
        sorted_tups = sorted__(((x, i) for i, x in enumerate(_iterable)), key=key, reverse=reverse, return_tuple=True, return_indexes=False)
        labels = [0] * len(sorted_tups)
        for j, (x, i) in enumerate(sorted_tups):
            labels[i] = j
        out = ((x, l) for (x, i), l in zip(sorted_tups, labels))
        return tuple(out) if return_tuple else list(out)

    if callable(key):
        s = sorted(_iterable, key=key, reverse=reverse)
        return tuple(s) if return_tuple else s
    else:
        s = unzip(unzip(sorted(zip(key, enumerate(_iterable)), reverse=reverse), 1), 1)  # `enumerate(_iterable)` ensures the original order of the `_iterable` when keys are the same
        return s if return_tuple else list(s)


def is_os_windows():
    return os.name == 'nt'


# endregion


# region primitive types

# region field repr

def fields2str(obj, fields=None):
    if fields is None:
        fields = get_fields(obj)

    if fields:
        d = {field: getattr(obj, field) for field in fields}
        if d:
            return d.__repr__()
        else:
            return str(obj.__class__)
    else:
        return str(obj.__class__)


class FieldsRepr:
    """
    Inherits this class to for a `__repr__` function that generates a dict-like string representation, given the names of the fields for this representation;
    if no representation fields is specified, then the field names in `_slots__` or `_fields` (e.g. :class:`~collections.namedtuple`) and in addition the `__dict__` will be considered as the fields for representation.
    """

    def __init__(self, repr_fields: Iterator[str] = None):
        self._repr_fields = tuple(repr_fields) if repr_fields else None

    def __repr__(self):
        return fields2str(self, self._repr_fields)


# endregion


# region accumulative

class list_(list):

    def __add__(self, other):
        l = copy.copy(self)
        l.__iadd__(other)
        return l

    def __iadd__(self, other):
        self.append(other)
        return self

    def __or__(self, other):
        l = copy.copy(self)
        l.extend(other)
        return l

    def __ior__(self, other):
        self.extend(other)
        return self


class list__(list):
    """
    A simple variant of list that allows the '+' and '+=' operator to work with non-iterable objects.

    >>> from utix.general import list__
    >>> x = list__([1, 2, 3, 4])
    >>> x += 5
    >>> x += [6, 7]
    >>> x += 8
    >>> print(x == [1, 2, 3, 4, 5, 6, 7, 8])

    """

    def __add__(self, other):
        try:
            return self + other
        except TypeError:
            return self + [other]

    def __iadd__(self, other):
        try:
            self.extend(other)
        except TypeError:
            self.append(other)
        return self

    def __radd__(self, other):
        try:
            return other + self
        except TypeError:
            return [other] + self


def fields_accu(target, item, fields=None):
    def _accu():
        for field in fields:
            if hasattr(item, field):
                val1, val2 = getattr(target, field), getattr(item, field)
                if type(val1) is list_:
                    val1 += val2
                else:
                    setattr(target, field, list_([val1, val2]))

    if fields is None:
        fields = set(get_fields(target))
        _accu()
        for field in get_fields(item):
            if field[0] != '_' and field not in fields:
                val2 = getattr(item, field)
                if type(val2) is list_:
                    setattr(target, field, copy.copy(val2))
                else:
                    setattr(target, field, list_([val2]))
    else:
        _accu()
    return target


def fields_add(target, item, fields=None):
    def _add():
        for field in fields:
            if hasattr(item, field):
                val1, val2 = getattr(target, field), getattr(item, field)
                if hasattr(val1, '__iadd__') or hasattr(val1, '__iconcat__'):
                    val1 += val2
                elif hasattr(val1, '__add__') or hasattr(val1, '__concat__'):
                    setattr(target, field, val1 + val2)
                elif hasattr(val1, '__ior__'):
                    val1 |= val2
                elif hasattr(val1, '__or__'):
                    setattr(target, field, val1 | val2)
                else:
                    raise TypeError(f"{type(val1)} must define one of '__iadd__', '__add__', '__ior__' and '__or__' in order to be accumulative")

    if fields is None:
        fields = set(get_fields(target))
        _add()
        for field in get_fields(item):
            if field[0] != '_' and field not in fields:
                setattr(target, field, getattr(item, field))
    else:
        _add()
    return target


def fields_max(target, item, fields=None):
    def _max():
        for field in fields:
            if hasattr(item, field):
                val1, val2 = getattr(target, field), getattr(item, field)
                try:
                    if val1 < val2:
                        setattr(target, field, val2)
                except:
                    if hasattr(val1, '__imax__'):
                        val1.__imax__(val2)
                    elif hasattr(val1, '__max__'):
                        setattr(target, field, val1.__max__(val2))
                    else:
                        raise TypeError(f"{type(val1)} must define the '<' operator compatible with {type(val2)},  or {type(val1)} must define one of the '__imax__', '__max__' functions.")

    if fields is None:
        fields = set(get_fields(target))
        _max()
        for field in get_fields(item):
            if field[0] != '_' and field not in fields:
                setattr(target, field, getattr(item, field))
    else:
        _max()
    return target


def fields_min(target, item, fields=None):
    def _min():
        for field in fields:
            if hasattr(item, field):
                val1, val2 = getattr(target, field), getattr(item, field)
                try:
                    if val1 > val2:
                        setattr(target, field, val2)
                except:
                    if hasattr(val1, '__imin__'):
                        val1.__imin__(val2)
                    elif hasattr(val1, '__min__'):
                        setattr(target, field, val1.__min__(val2))
                    else:
                        raise TypeError(f"{type(val1)} must define the '>' operator compatible with {type(val2)},  or {type(val1)} must define one of the '__imin__', '__min__' functions.")

    if fields is None:
        fields = set(get_fields(target))
        _min()
        for field in get_fields(item):
            if field[0] != '_' and field not in fields:
                setattr(target, field, getattr(item, field))
    else:
        _min()
    return target


def fields_sub(target, item, fields=None):
    if fields is None:
        fields = get_fields(target)

    for field in fields:
        if hasattr(item, field):
            val1, val2 = getattr(target, field), getattr(item, field)
            if hasattr(val1, '__isub__'):
                val1 -= val2
            elif hasattr(val1, '__sub__'):
                setattr(target, field, val1 - val2)
            else:
                raise TypeError(f"{type(val1)} must define either '__isub__' or '__sub__' in order to be reduced")
    return target


def fields_div(target, divisor, fields=None):
    if fields is None:
        fields = get_fields(target)

    for field in fields:
        val = getattr(target, field)
        if hasattr(val, '__itruediv__'):
            val /= divisor
        elif hasattr(val, '__truediv__'):
            setattr(target, field, val / divisor)
        else:
            raise TypeError(f"{type(val)} must define either '__itruediv__' or '__truediv__' in order to be divided")
    return target


def fields_floor_div(target, divisor, fields=None):
    if fields is None:
        fields = get_fields(target)

    for field in fields:
        val = getattr(target, field)
        if hasattr(val, '__ifloordiv__'):
            val //= divisor
        elif hasattr(val, '__floordiv__'):
            setattr(target, field, val // divisor)
        else:
            raise TypeError(f"{type(val)} must define either '__ifloordiv__' or '__floordiv__' in order to be floor-divided")
    return target


def fields_multiply(target, multiplier, fields=None):
    if fields is None:
        fields = get_fields(target)

    for field in fields:
        val = getattr(target, field)
        if hasattr(val, '__imul__'):
            val *= multiplier
        elif hasattr(val, '__mul__'):
            setattr(target, field, val * multiplier)
        else:
            raise TypeError(f"{type(val)} must define either '__imul__' or '__mul__' in order to be multiplied")
    return target


class Accumulative:
    """
    A general-purpose base class that equips fixed-size classes (those with non-empty `__slots__`) and named tuples (see :class:`~collections.namedtuple`) with basic operations including '+', '+=', '-', '-=', '/', '/=', '//', '//=', '*', '*='.
    This class is typically used for formulating statistic classes and metric classes.
    """

    def __init__(self, accu_fields: Iterator[str] = None):
        self._accu_fields = tuple(accu_fields) if accu_fields else None

    def __or__(self, other):
        return fields_accu(copy.deepcopy(self), other, self._accu_fields)

    def __ior__(self, other):
        return fields_accu(self, other, self._accu_fields)

    def __add__(self, other):
        return fields_add(copy.deepcopy(self), other, self._accu_fields)

    def __iadd__(self, other):
        return fields_add(self, other, self._accu_fields)

    def __sub__(self, other):
        return fields_sub(copy.deepcopy(self), other, self._accu_fields)

    def __isub__(self, other):
        return fields_sub(self, other, self._accu_fields)

    def __truediv__(self, other):
        return fields_div(copy.deepcopy(self), other, self._accu_fields)

    def __itruediv__(self, other):
        return fields_div(self, other, self._accu_fields)

    def __floordiv__(self, other):
        return fields_floor_div(copy.deepcopy(self), other, self._accu_fields)

    def __ifloordiv__(self, other):
        return fields_floor_div(self, other, self._accu_fields)

    def __mul__(self, other):
        return fields_multiply(copy.deepcopy(self), other, self._accu_fields)

    def __imul__(self, other):
        return fields_multiply(self, other, self._accu_fields)

    def __max__(self, other):
        return fields_max(copy.deepcopy(self), other, self._accu_fields)

    def __imax__(self, other):
        return fields_max(self, other, self._accu_fields)

    def __min__(self, other):
        return fields_min(copy.deepcopy(self), other, self._accu_fields)

    def __imin__(self, other):
        return fields_min(self, other, self._accu_fields)


def xsum(it: Iterator):
    """
    Sums over the iterable like the build-in `sum` function, but compatible with all objects with defined `+` or `+=` operators.
    For example, `xsum([1,2,3,4,5])` gives the same result as `sum([1,2,3,4,5])`.
    For another example, `xsum([[1,2], [3,4], [5,6]])` gives the same result as `sum([[1,2], [3,4], [5,6]], [])`.
    More efficient for summing over non-number objects, e.g. combining lists, or :class:`Accumulative` objects.
    :param it:
    :return:
    """

    # ! this line is necessary to ensure `it` is an iterator
    # `it` might be list, tuple, etc., which might not be an iterator
    it = iter(it)

    first = copy.deepcopy(next(it))
    for x in it:
        first += x
    return first


def xsum_(it: Iterator):
    it = iter(it)
    first = next(it)
    for x in it:
        first += x
    return first


def xmean(it: Iterator):
    it = iter(it)
    first = copy.deepcopy(next(it))
    cnt = 1
    for x in it:
        first += x
        cnt += 1
    first /= cnt
    return first


def xmean_(it: Iterator):
    """
    Computes the mean of a sequence of objects. The result is written into the first object of the sequence.
    :param it: the object iterator.
    :return: the first object in `it` which saves the mean of the whole sequence.
    """
    it = iter(it)
    first = next(it)
    cnt = 1
    for x in it:
        first += x
        cnt += 1
    first /= cnt
    return first


def xmin(it: Iterator):
    _it = iter(it)
    first = next(_it)
    if hasattr(first, '__imin__'):
        first = copy.deepcopy(first)
        for item in _it:
            first.__imin__(item)
        return first
    elif hasattr(first, '__min__'):
        for item in _it:
            first = first.__min__(item)
        return first
    else:
        return min(it)


def xmax(it: Iterator):
    _it = iter(it)
    first = next(_it)
    if hasattr(first, '__imax__'):
        first = copy.deepcopy(first)
        for item in _it:
            first.__imax__(item)
        return first
    elif hasattr(first, '__max__'):
        for item in _it:
            first = first.__max__(item)
        return first
    else:
        return min(it)


# endregion

# endregion


# region colorful print
colorama_init()


def get_cprint_str(text: str, color_quote: str = '`', color: str = Fore.CYAN, bk_color: str = Fore.WHITE, end: str = '\n'):
    output = [Fore.WHITE]
    color_start: bool = True
    prev_color_quote: bool = False
    for c in text:
        if c == color_quote:
            if prev_color_quote:
                output.append('`')
                prev_color_quote = False
                color_start = True
            elif color_start:
                prev_color_quote = True
                color_start = False
            else:
                output.append(bk_color)
                color_start = True
        else:
            if prev_color_quote:
                output.append(color)
            output.append(c)
            prev_color_quote = False
    if end is not None:
        output.append(end)
    output.append(Fore.WHITE)
    return ''.join(output)


def cprint(text, color_place_holder='`', color=Fore.CYAN, bk_color=Fore.WHITE, end='\n'):
    print(get_cprint_str(text=text, color_quote=color_place_holder, color=color, bk_color=bk_color, end=end))


def get_cprint_message_str(title, content='', title_color=Fore.CYAN, content_color=Fore.WHITE, start='', end='\n'):
    return f'{start}{title_color}{title}{Fore.WHITE}{end}' if content == '' \
        else f'{start}{title_color}{title}: {content_color}{content}{Fore.WHITE}{end}'


def cprint_message(title, content='', title_color=Fore.CYAN, content_color=Fore.WHITE, start='', end='\n'):
    print(get_cprint_message_str(title, content, title_color, content_color, start, end))


def get_cprint_pairs_str(*args, first_color=Fore.CYAN, second_color=Fore.WHITE, sep='\t', end='\n'):
    return sep.join((get_cprint_message_str(title=arg[0], content=arg[1], title_color=first_color, content_color=second_color, end='') for arg in args)) + end


def get_pair_strs_for_color_print_and_regular_print(*args, first_color=Fore.CYAN, second_color=Fore.WHITE, sep: str = ' ', end: str = '\n') -> Tuple[str, str]:
    colored_strs, uncolored_strs = [], []
    for arg_idx, arg in enumerate(args):
        colored_strs.append(get_cprint_message_str(title=arg[0], content=arg[1], title_color=first_color, content_color=second_color, end=''))
        uncolored_strs.append(f'{arg[0]}: {arg[1]},' if arg_idx != len(args) - 1 else f'{arg[0]}: {arg[1]}')

    return sep.join(colored_strs) + end, sep.join(uncolored_strs) + end


def cprint_pairs(*args, first_color=Fore.CYAN, second_color=Fore.WHITE, sep=' ', end='\n'):
    print(get_cprint_pairs_str(*args, first_color=first_color, second_color=second_color, sep=sep, end=end))


# region highlight print

def get_hprint_str(msg, color_place_holder='`', end='') -> str:
    return get_cprint_str(text=msg, color_quote=color_place_holder, color=Fore.CYAN, bk_color=Fore.WHITE, end=end)


def get_hprint_pairs_str(*args, sep=' ', end='\n'):
    return get_cprint_pairs_str(*args, first_color=Fore.CYAN, second_color=Fore.WHITE, sep=sep, end=end)


def get_pairs_str_for_hprint_and_regular_print(*args, sep: str = ' ', end: str = '\n') -> Tuple[str, str]:
    return get_pair_strs_for_color_print_and_regular_print(*args, first_color=Fore.CYAN, second_color=Fore.WHITE, sep=sep, end=end)


def hprint(msg, color_quote='`', end=''):
    """
    Print the message `msg`, highlighting texts enclosed by a pair of `color_quote`s (by default the backtick `) with the cyan color.
    Use two backticks '``' to escape the color quote.
    :param msg: the message to print.
    :param color_quote: the character used to mark the beginning and the end of each piece of texts to highlight.
    :param end: string appended at the end of the message, newline by default.
    """
    cprint(text=msg, color_place_holder=color_quote, color=Fore.CYAN, bk_color=Fore.WHITE, end=end)


def get_hprint_message_str(title, content='', start='', end=''):
    return get_cprint_message_str(title=title, content=content, title_color=Fore.CYAN, content_color=Fore.WHITE, start=start, end=end)


def hprint_message(title, content='', start='', end=''):
    print(get_hprint_message_str(title=title, content=content, start=start, end=end))


hmsg = hprint_message


def hprint_pairs(*args, sep=' ', end=''):
    cprint_pairs(*args, first_color=Fore.CYAN, second_color=Fore.WHITE, sep=sep, end=end)


# endregion

# region error print


def eprint(text, color_quote='`', end='\n'):
    """
    Print the message `msg` with the , highlighting texts enclosed by a pair of `color_quote`s (by default the backtick `) with the red color.
    :param msg: the message to print.
    :param color_quote: the character used to mark the beginning and the end of each piece of texts to highlight.
    :param end: string appended at the end of the message, newline by default.
    """
    cprint(text=text, color_place_holder=color_quote, color=Fore.RED, bk_color=Fore.MAGENTA, end=end)


def eprint_message(title, content='', start='', end=''):
    cprint_message(title, content, title_color=Fore.RED, content_color=Fore.MAGENTA, start=start, end=end)


# endregion

# region warning print

def wprint_message(title, content='', start='', end='\n'):
    cprint_message(title, content, title_color=Fore.MAGENTA, content_color=Fore.YELLOW, start=start, end=end)


# endregion


# endregion


class flogger(object):

    def __init__(self, path, print_terminal=True):
        self.terminal = sys.stdout
        self.log = open(path, "w")
        self.print_to_terminal = print_terminal

    def write(self, message):
        if self.print_to_terminal:
            self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()
        pass

    def reset(self):
        self.flush()
        self.log.close()
        sys.stdout = self.terminal


def color_print_pair_str(pair_str: str, pair_delimiter=',', kv_delimiter=':', key_color=Fore.CYAN, value_color=Fore.WHITE, end='\n'):
    pairs = pair_str.split(pair_delimiter)
    pair_count = len(pairs)
    for i in range(pair_count):
        kv_str = pairs[i].strip()
        if len(kv_str) > 0:
            k, v = kv_str.split(kv_delimiter, maxsplit=2)
            cprint_message(k, v, key_color, value_color, end=', ' if i != pair_count - 1 else end)


def _get_print_tag_str(tag):
    if is_class(tag):
        return tag.__module__ + '.' + tag.__name__
    elif is_basic_type(tag):
        return str(tag)
    else:
        return tag.__class__


def retrieve_and_print_attrs(obj, *attr_names):
    num_attr_names = len(attr_names)
    attr_vals = [None] * num_attr_names
    for i in range(num_attr_names):
        attr_name = attr_names[i]
        attr_val = getattr(obj, attr_name)
        hprint_message(attr_name, attr_val)
        attr_vals[i] = attr_val
    return tuple(attr_vals)


def print_attrs(obj):
    for attr in dir(obj):
        if attr[0] != '_':
            attr_val = getattr(obj, attr)
            if not callable(attr_val):
                hprint_message(attr, attr_val)


def hprint_message_pair_str(pair_str, pair_delimiter=',', kv_delimiter=':'):
    color_print_pair_str(pair_str, pair_delimiter=pair_delimiter, kv_delimiter=kv_delimiter, key_color=Fore.YELLOW, value_color=Fore.WHITE)


def log_pairs(logging_fun, *args):
    msg = ' '.join(str(arg_tup[0]) + ' ' + str(arg_tup[1]) for arg_tup in args)
    logging_fun(msg)


def info_print(tag, content):
    if not hasattr(tag, '_verbose') or getattr(tag, '_verbose') is True:
        cprint_message(_get_print_tag_str(tag), content, title_color=Fore.CYAN)


def debug_print(tag, content):
    if not hasattr(tag, '_verbose') or getattr(tag, '_verbose') is True:
        cprint_message(_get_print_tag_str(tag), content, title_color=Fore.YELLOW)


def kv_list_format(keys, values, kv_delimiter=':', pair_delimiter: str = ', ', value_format='{}', value_transform=None, value_idx=-1):
    return pair_delimiter.join([keys[i] + kv_delimiter +
                                ('{}' if value_format is None else (value_format if type(value_format) is str else value_format[i]))
                               .format((values[i][value_idx] if value_transform is None else value_transform(values[i][value_idx])) if type(values[i]) in (list, tuple)
                                       else (values[i]) if value_transform is None else value_transform(values[i])) for i in range(len(keys))])


def kv_tuple_format(kv_tuples, kv_delimiter, pair_delimiter, value_idx=-1):
    return pair_delimiter.join([tup[0] + kv_delimiter +
                                ('{}' if len(tup) == 2 else tup[2]).format((tup[1][value_idx] if len(tup) < 4 else tup[3](tup[1][value_idx])) if type(tup[1]) in (list, tuple)
                                                                           else tup[1] if len(tup) < 4 else tup[3](tup[1]))
                                for tup in kv_tuples])


def take_element_if_list(potential_list, i: int):
    return potential_list[i] if isinstance(potential_list, list) else potential_list


def mapping_or_namespace_items(x):
    if isinstance(x, Mapping):
        return x.items()
    elif hasattr(x, '__dict__'):
        return x.__dict__.items()


def mapping_or_namespace_keys(x):
    if isinstance(x, Mapping):
        return x.keys()
    elif hasattr(x, '__dict__'):
        return x.__dict__.keys()


def mapping_or_namespace_values(x, filter=None):
    if filter is None:
        if isinstance(x, Mapping):
            return x.values()
        elif hasattr(x, '__dict__'):
            return x.__dict__.values()
    else:
        d = x if isinstance(x, Mapping) else x.__dict__
        return [d[k] for k in filter]


def deduplicate_sum_lists(_lists: List[List], keep_order=False):
    if keep_order:
        if not isinstance(_lists, (list, tuple)):
            _lists = list(_lists)
        dd_sum = set(sum(_lists, []))
        for _list in _lists:
            if dd_sum == set(_list) and len(dd_sum) == len(_list):
                return _list
        return list(dd_sum)
    else:
        return list(set(sum(_lists, [])))


def iter_pairs(_x):
    """
    A convenient function to iterate through an object that is conceptually a sequence of pairs.
    This functions allows the `inputs` be:
    1) a single pair, e.g. `(2.3, 5.14)`; in this case this function will just yield this pair;
    2) a sequence of pairs, represented by an iterable of tuples or lists;
    3) a dictionary, or any object with an `items()` method that iterate through a sequence of pairs.
    NOTE this method is for convenience and does not check if everything in the `inputs` are actually pairs.
    """
    if _x:
        if is_list_or_tuple(_x):
            if len(_x) == 2 and (is_str(_x[0]) or is_str(_x[1]) or not is_list_or_tuple(_x[0]) or len(_x[0]) != 2 or not is_list_or_tuple(_x[1]) or len(_x[1]) != 2):
                yield _x
            else:
                yield from _x
        else:
            items = getattr(_x, 'items', None)
            yield from items() if callable(items) else _x

# region internal standardized print outs


# endregion

# region parameter handling


# endregion

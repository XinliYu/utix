import copy
import sys
from collections import defaultdict
from collections.abc import Mapping
from itertools import chain
from typing import Dict, Any, List, Iterator, Counter, Callable, Tuple

from utilx.general import FieldsRepr, Accumulative, get_fields, JSerializable, list__

# region frozen & hybrid dict

DICT_PROTECTION_PREFIX = '_'


class _FrozenDict(Mapping, FieldsRepr, JSerializable):

    def __len__(self):
        return sum(x[0] != DICT_PROTECTION_PREFIX for x in chain(self.__slots__, self.__dict__))

    def __init__(self, d, repr_fields=None, deepcopy=False):
        if isinstance(d, dict):
            if deepcopy:
                for k, v in d.items():
                    setattr(self, k, copy.deepcopy(v))
            else:
                for k, v in d.items():
                    setattr(self, k, v)
        elif deepcopy:
            for field in get_fields(d, include_protected=True):
                setattr(self, field, copy.deepcopy(getattr(d, field)))
        else:
            for field in get_fields(d, include_protected=True):
                setattr(self, field, getattr(d, field))
        FieldsRepr.__init__(self, repr_fields=repr_fields)

    def __iter__(self):
        yield from (x for x in chain(self.__slots__, self.__dict__) if x[0] != DICT_PROTECTION_PREFIX)

    def to_dict(self):
        return {k: v for k, v in self.items()}

    def __getitem__(self, item: str):
        if item[0] == DICT_PROTECTION_PREFIX:
            raise KeyError('trying to access protected keys')
        return getattr(self, item, None)

    def keys(self):
        yield from self

    def items(self):
        for k in self:
            yield k, getattr(self, k)

    def __contains__(self, item: str):
        if item[0] == DICT_PROTECTION_PREFIX:
            return False
        return item in self.__slots__ or item in self.__dict__

    def __copy__(self):
        return self.__class__(d=self, repr_fields=self._repr_fields, deepcopy=False)

    def __deepcopy__(self, memodict={}):
        return self.__class__(d=self, repr_fields=self._repr_fields, deepcopy=True)

    def __to_dict__(self) -> dict:
        return JSerializable.__to_dict__(self, type_str='utilx.dict_ext.fdict')

    @classmethod
    def __from_dict__(cls, d: dict):
        return JSerializable.__from_dict__(d, fdict)


class _HybridDict(_FrozenDict):
    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __delitem__(self, key):
        self.__dict__.pop(key, None)


class _AccumulativeFrozenDict(_FrozenDict, Accumulative):
    def __init__(self, d, repr_fields=None, accu_fields=None, deepcopy=False):
        _FrozenDict.__init__(self, d=d, repr_fields=repr_fields, deepcopy=deepcopy)
        Accumulative.__init__(self, accu_fields=accu_fields)

    def __copy__(self):
        return self.__class__(d=self, repr_fields=self._repr_fields, accu_fields=self._accu_fields, deepcopy=False)

    def __deepcopy__(self, memodict={}):
        return self.__class__(d=self, repr_fields=self._repr_fields, accu_fields=self._accu_fields, deepcopy=True)


class _AccumulativeHybridDict(_AccumulativeFrozenDict):
    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __delitem__(self, key):
        self.__dict__.pop(key, None)


def _frozen_dict_type(d, base, extra_slots):
    if isinstance(d, dict):
        __slots__ = tuple(d.keys())
    else:
        __slots__ = tuple(get_fields(d, include_protected=True))
        if not __slots__:
            raise TypeError(f"the frozen dict accepts a dict, or an object with at least one of non-empty `__slots__` or `_fields` or `__dict__`; got {type(d)} which is not supported")

    typename = f'{base.__name__}_{abs(hash(__slots__))}'
    module = sys.modules['__main__']
    t = getattr(module, typename, None)
    if t is None:
        t = type(typename, (base,), {'__slots__': __slots__ + tuple(x for x in extra_slots if x not in __slots__)})
        t.__module__ = '__main__'
        setattr(module, typename, t)
    return t


def FrozenDict(__init=None, repr_fields: Iterator[str] = None, deepcopy: bool = False, **kwargs):
    """
    Constructs a dict-like instance with read-only key/value pairs. When the dict size is less than 10, the frozen dict can roughly save 60% memory.
    A frozen dict does not inherit from the :class:`dict` class, therefore `isinstance(x, dict)` will return `False` if `x` is a frozen dict.
    Instead, a frozen dict is using `__slots__` and in addiction simulates a dict's reading by implementing methods `__getitem__`, `__len__`, `__iter__`, `__repr__` and `items`.

    :param __init: a dict, an object with `__slots__`, or a namedtuple.
              If an object with `__slots__` is provided, these 'slots' and their values will be the key/value pairs saved in this frozen dict.
              If a namedtuple is provided, its fields and their values will be the key/value pairs saved in this frozen dict.
    :param repr_fields: the names of the representation fields.
    :param deepcopy: `True` if deepcopying values in `d` to the frozen dict; otherwise making shallow copies.
    """
    if kwargs:
        if __init:
            try:
                kwargs.update(__init)
            except:
                raise ValueError(f"the provided frozen dict init object of type {type(__init)} is not compatible with `dict.update` method in order to incorporate updates in `kwargs`")
        __init = kwargs

    t = _frozen_dict_type(__init, _FrozenDict, extra_slots=('_repr_fields',))
    return t(d=__init, repr_fields=repr_fields, deepcopy=deepcopy)


# d = FrozenDict(a=1,b=2,c=3,d=4)
# import  json
# s = json.dumps(d)

def AccumulativeFrozenDict(__init=None, repr_fields=None, accu_fields=None, deepcopy=False, **kwargs):
    """
    The same as `FrozenDict`, and in addiction the frozen dict is made accumulative (i.e. works with operators like '+=', '/=', etc., see :class:`utilx.general_ext.Accumulative`).
    """
    if kwargs:
        if __init:
            try:
                kwargs.update(__init)
            except:
                raise ValueError(f"the provided frozen dict init object of type {type(__init)} is not compatible with `dict.update` method in order to incorporate updates in `kwargs`")
        __init = kwargs
    t = _frozen_dict_type(__init, _AccumulativeFrozenDict, extra_slots=('_repr_fields', '_accu_fields'))
    return t(d=__init, repr_fields=repr_fields, accu_fields=accu_fields, deepcopy=deepcopy)


def HybridDict(__init=None, repr_fields: Iterator[str] = None, deepcopy: bool = False, extra_keys: Iterator[str] = None, **kwargs):
    if kwargs:
        if __init:
            try:
                kwargs.update(__init)
            except:
                raise ValueError(f"the provided frozen dict init object of type {type(__init)} is not compatible with `dict.update` method in order to incorporate updates in `kwargs`")
        __init = kwargs
    if extra_keys is not None:
        for key in extra_keys:
            __init[key] = None

    t = _frozen_dict_type(__init, _HybridDict, extra_slots=('_repr_fields',))
    return t(d=__init, repr_fields=repr_fields, deepcopy=deepcopy)


def AccumulativeHybridDict(__init=None, repr_fields=None, accu_fields=None, deepcopy=False, extra_keys: Iterator[str] = None, **kwargs):
    if kwargs:
        if __init:
            try:
                kwargs.update(__init)
            except:
                raise ValueError(f"the provided frozen dict init object of type {type(__init)} is not compatible with `dict.update` method in order to incorporate updates in `kwargs`")
        __init = kwargs
    if extra_keys is not None:
        for key in extra_keys:
            __init[key] = None
    t = _frozen_dict_type(__init, _AccumulativeHybridDict, extra_slots=('_repr_fields', '_accu_fields'))
    return t(d=__init, repr_fields=repr_fields, accu_fields=accu_fields, deepcopy=deepcopy)


def is_frozen_dict(obj):
    return obj.__class__.__name__.startswith('_FrozenDict_')


def is_hybrid_dict(obj):
    return obj.__class__.__name__.startswith('_HybridDict_')


def is_accumulative_frozen_dict(obj):
    return obj.__class__.__name__.startswith('_AccumulativeFrozenDict_')


def is_accumulative_hybrid_dict(obj):
    return obj.__class__.__name__.startswith('_AccumulativeHybridDict_')


def is_dict__(obj):
    return isinstance(obj, dict) or obj.__class__.__name__.startswith(('_FrozenDict_', '_HybridDict_', '_AccumulativeFrozenDict_', '_AccumulativeHybridDict_'))


fdict = FrozenDict
xfdict = AccumulativeFrozenDict
hdict = HybridDict
xhdict = AccumulativeHybridDict
is_fdict = is_frozen_dict
is_xfdict = is_accumulative_frozen_dict
is_hdict = is_hybrid_dict
is_xhdict = is_accumulative_hybrid_dict


# endregion

# region dict wrap

class _DictWrap1(Mapping):
    __slots__ = ('_obj',)

    def __init__(self, obj):
        self._obj = obj

    def __len__(self) -> int:
        return sum(x[0] != DICT_PROTECTION_PREFIX for x in chain(self._obj.__slots__, self._obj.__dict__))

    def __iter__(self):
        yield from (x for x in chain(self._obj.__slots__, self._obj.__dict__) if x[0] != DICT_PROTECTION_PREFIX)

    def __getitem__(self, item: str):
        if item[0] == DICT_PROTECTION_PREFIX:
            raise KeyError('trying to access protected keys')
        return getattr(self._obj, item, None)

    def __contains__(self, item: str):
        if item[0] == DICT_PROTECTION_PREFIX:
            return False
        return item in self._obj.__slots__ or item in self._obj.__dict__

    def keys(self):
        yield from self

    def items(self):
        for k in self:
            yield k, getattr(self._obj, k)

    def to_dict(self):
        return {k: v for k, v in self.items()}

    def __repr__(self):
        return self.to_dict().__repr__()


class _DictWrap2(Mapping):
    __slots__ = ('_obj',)

    def __init__(self, obj):
        self._obj = obj

    def __len__(self) -> int:
        return sum(x[0] != DICT_PROTECTION_PREFIX for x in self._obj.__slots__)

    def __iter__(self):
        yield from (x for x in self._obj.__slots__ if x[0] != DICT_PROTECTION_PREFIX)

    def __getitem__(self, item: str):
        if item[0] == DICT_PROTECTION_PREFIX:
            raise KeyError('trying to access protected keys')
        return getattr(self._obj, item, None)

    def __contains__(self, item: str):
        if item[0] == DICT_PROTECTION_PREFIX:
            return False
        return item in self._obj.__slots__

    def keys(self):
        yield from self

    def items(self):
        for k in self:
            yield k, getattr(self._obj, k)

    def to_dict(self):
        return {k: v for k, v in self.items()}

    def __repr__(self):
        return self.to_dict().__repr__()


class _DictWrap3(Mapping):
    __slots__ = ('_obj',)

    def __init__(self, obj):
        self._obj = obj

    def __len__(self) -> int:
        return sum(x[0] != DICT_PROTECTION_PREFIX for x in self._obj.__dict__)

    def __iter__(self):
        yield from (x for x in self._obj.__dict__ if x[0] != DICT_PROTECTION_PREFIX)

    def __getitem__(self, item: str):
        if item[0] == DICT_PROTECTION_PREFIX:
            raise KeyError('trying to access protected keys')
        return getattr(self._obj, item, None)

    def __contains__(self, item: str):
        if item[0] == DICT_PROTECTION_PREFIX:
            return False
        return item in self._obj.__dict__

    def keys(self):
        yield from self

    def items(self):
        for k in self:
            yield k, getattr(self._obj, k)

    def to_dict(self):
        return {k: v for k, v in self.items()}

    def __repr__(self):
        return self.to_dict().__repr__()


def dict_wrap(_obj):
    """
    Wraps any object with non-empty `__slots__` or `__dict__` as a read-only dictionary, with keys being the non-protective fields (with a name not starting with '_') defined in `__slots__` and `__dict__`.
    The short name for this wrap is `dwrap`.

    For example,
    >>> from utilx.dict_ext import dwrap
    >>> @dwrap
    >>> class A:
    >>>     __slots__ = ('field_a', 'field_b', 'field_c')
    >>>     def __init__(self, a, b, c):
    >>>         self.field_a, self.field_b, self.field_c = a, b, c
    >>> a = A(1,2,3)
    >>> print(a['field_a'], a['field_b'], a['field_c']) # 1 2 3
    >>> print(a) # {'field_a': 1, 'field_b': 2, 'field_c': 3}

    """

    def _wrap(*args, **kwargs):
        obj = _obj(*args, **kwargs)
        has_slots = hasattr(obj, '__slots__')
        has_dict = hasattr(obj, '__dict__')
        if has_slots and has_dict:
            return _DictWrap1(obj)
        elif has_slots:
            return _DictWrap2(obj)
        else:
            return _DictWrap3(obj)

    return _wrap


dwrap = dict_wrap


class SlotsDict(Mapping):
    def __len__(self) -> int:
        return sum(x[0] != DICT_PROTECTION_PREFIX for x in self.__slots__)

    def __iter__(self):
        yield from (x for x in self.__slots__ if x[0] != DICT_PROTECTION_PREFIX)

    def __getitem__(self, item: str):
        if item[0] == DICT_PROTECTION_PREFIX:
            raise KeyError('trying to access protected keys')
        return getattr(self, item, None)

    def __contains__(self, item: str):
        if item[0] == DICT_PROTECTION_PREFIX:
            return False
        return item in self.__slots__

    def keys(self):
        yield from self

    def items(self):
        for k in self:
            yield k, getattr(self, k)

    def to_dict(self):
        return {k: v for k, v in self.items()}

    def __repr__(self):
        return self.to_dict().__repr__()


class FieldDict(Mapping):

    def __len__(self) -> int:
        return sum(x[0] != DICT_PROTECTION_PREFIX for x in chain(self.__slots__, self.__dict__))

    def __iter__(self):
        yield from (x for x in chain(self.__slots__, self.__dict__) if x[0] != DICT_PROTECTION_PREFIX)

    def __getitem__(self, item: str):
        if item[0] == DICT_PROTECTION_PREFIX:
            raise KeyError('trying to access protected keys')
        return getattr(self, item, None)

    def __contains__(self, item: str):
        if item[0] == DICT_PROTECTION_PREFIX:
            return False
        return item in self.__slots__ or item in self.__dict__

    def keys(self):
        yield from self

    def items(self):
        for k in self:
            yield k, getattr(self, k)

    def to_dict(self):
        return {k: v for k, v in self.items()}

    def __repr__(self):
        return self.to_dict().__repr__()


# endregion


# region specialized dicts

class ListDict(defaultdict):
    """
    A dictionary of lists. Implements the `+` operator for convenience to merge two or more list dictionaries.
    NOTE this class is not intended to be used as a `defaultdict(list)`.

    >>> from utilx.dict_ext import ListDict
    >>> d = ListDict()
    >>> d += { 'a': 1, 'b': 2, 'c': 3 }
    >>> d += { 'a': 4, 'b': 5, 'c': 6 }
    >>> print(d) # {'a': [1, 4], 'b': [2, 5], 'c': [3, 6]}

    """

    def __init__(self, init: dict = None):
        """
        :param init: an initial dictionary; all values in this `init` dictionary will be treated as an atom even it is iterable.
        """
        super().__init__(list)
        if init:
            for k, v in init.items():
                self[k] = [v]

    def __add__(self, other: dict):
        """
        Adds values in the `other` dictionary to the set of the same key in this :class:`listdict`.
        :param other: the other dictionary.
        :return: the current list dictionary.
        """
        for k, v in other.items():
            self[k].append(v)
        return self


class PaddedListDict(defaultdict):
    """
    A dictionary of lists. Implements the `+` operator for convenience to merge two or more list dictionaries.
    The same as `ListDict`, except for it uses a placeholder like `None` to keep track of missing values.

    >>> from utilx.dict_ext import PaddedListDict
    >>> d = PaddedListDict()
    >>> d += { 'b': 2, 'c': 3 }
    >>> d += { 'a': 4, 'c': 6 }
    >>> print(d) # {'b': [2], 'c': [3, 6], 'a': [None, 4]}

    For above example, 'a' is missing in the first dictionary, so a `None` is inserted to represent that it is missing. 'b' is missing in the second dictionary, however there is no added `None` at this moment, because the place-holder is "lazy"-added.
    Now add another dictionary; the `None` will now be inserted.

    >>> d += { 'b': 7 }
    >>> print(d) # {'b': [2, None, 7], 'c': [3, 6], 'a': [None, 4]}

    """

    def __init__(self, init: dict = None, placeholder=None):
        """
        :param init: an initial dictionary; all values in this `init` dictionary will be treated as an atom even it is iterable.
        :param placeholder: the place-holder used to track the missing values.
        """
        super().__init__(list)
        if init:
            for k, v in init.items():
                self[k] = [v]
            self._add_num = 1
        else:
            self._add_num = 0
        self._ph = placeholder

    def __add__(self, other: dict):
        for k, v in other.items():
            l: list = self[k]
            pad_len = self._add_num - len(l)
            if pad_len > 0:
                l.extend([self._ph] * pad_len)
            elif pad_len < 0:
                raise RuntimeError("some data must have been added to the `PaddedListDict` by a method other than the '+' or '+=' operator")
            l.append(v)
        self._add_num += 1

        return self


class SetDict(dict):
    """
    A dictionary of sets. Implements the `+` operator for convenience to merge two set dictionaries.
    """

    def __add__(self, other: Dict[Any, Iterator]):
        """
        Adds values in the `other` dictionary to the set of the same key in this :class:`setdict`.
        :param other: the other dictionary.
        :return: the current set dictionary.
        """
        for k, v in other.items():
            if k in self:
                self[k].add(v)
            else:
                self[k] = {v}

        return self


class TupleDict(dict):
    """
    A dictionary of tuples. Implements the `+` operator for convenience to merge two tuple dictionaries.
    """

    def __add__(self, other: Dict[Any, Iterator]):
        """
        Adds values in the `other` dictionary to the tuple of the same key in this :class:`tupledict`.
        :param other: the other dictionary.
        :return: the current tuple dictionary.
        """
        for k, v in other.items():
            if k in self:
                self[k] += (v,)
            else:
                self[k] = (v,)
        return self


class IndexDict:
    __slots__ = ('_d', '_r')

    def __init__(self, reverse_lookup=False):
        self._d = defaultdict(lambda: len(self._d))
        self._r = {} if reverse_lookup else None

    def index(self, x):
        if self._r is None:
            return self._d[x]
        else:
            idx = self._d[x]
            self._r[idx] = x
            return idx

    def index_all(self, it: Iterator):
        if self._r is None:
            return [self._d[x] for x in it]
        else:
            def _index(x):
                if x not in self._d:
                    idx = len(self._d)
                    self._d[x] = idx
                    self._r[idx] = x
                    return idx
                else:
                    return self._d[x]

            return [_index(x) for x in it]

    def add(self, x):
        if self._r is None:
            if x not in self._d:
                self._d[x] = len(self._d)
        else:
            idx = self._d[x]
            self._r[idx] = x

    def add_all(self, it: Iterator):
        if self._r is None:
            for x in it:
                if x not in self._d:
                    self._d[x] = len(self._d)
        else:
            for x in it:
                if x not in self._d:
                    idx = len(self._d)
                    self._d[x] = idx
                    self._r[idx] = x

    def get_normalized_index_dict(self):
        l = len(self._d)
        return {k: v / l for k, v in self._d.items()}

    def get(self, index: int):
        return self._r.get(index, None) if self._r is not None else None


# endregion


# region dict merge
def tup2dict(_tups) -> dict:
    """
    Converts a sequence of tuples into a dictionary.

    >>> import utilx.dict_ext as dx
    >>> print(dx.tup2dict((('a', 1), ('b', 2), ('c', 3))) == {'a': 1, 'b': 2, 'c': 3})

    :param _tups: an iterable of tuples.
    :return: a dictionary, with keys being the first values in the tuples, and the values being the second values in the tuples.
    """
    return {k: v for k, v in _tups}


def tup2listdict(_tups) -> defaultdict:
    """
    Converts a sequence of tuples into a dictionary of lists. The first value of the tuple is the dictionary key, and the second value is the dictionary value.
    Multiple values associated with the same key will be grouped into a list associated with the key.
    """
    d = defaultdict(list)
    for k, v in _tups:
        d[k].append(v)
    return d


def tup2setdict(_tups, dual=False) -> defaultdict:
    """
    Converts a sequence of tuples into a dictionary of sets. The first value of the tuple is the dictionary key, and the second value is the dictionary value.
    Multiple values associated with the same key will be grouped into a set associated with the key.
    """
    d = defaultdict(set)
    for k, v in _tups:
        d[k].add(v)
        if dual:
            d[v].add(k)
    return d

def tup2dict__(tuples, key_prefix=None, _key_connector='_') -> dict:
    """
    The same as `tup2dict`, with an option for key prefix.
    If the key prefix is specified, then the prefix adds to the start of each key unless the key already starts with the prefix.

    >>> import utilx.dict_ext as dx
    >>> print(dx.tup2dict__((('a', 1), ('b', 2), ('prefix_c', 3)), key_prefix='prefix_') == {'prefix_a': 1, 'prefix_b': 2, 'prefix_c': 3})

    :param tuples: a sequence of tuples to convert to a dictionary.
    :param key_prefix: the prefix to add in front of each key.
    :param _key_connector: TODO: not ready; DO NOT USE.
    :return: a dictionary where keys are the first elements from the tuples, and values are the second elements from the tuples.
    """
    output = {}
    if key_prefix:
        for k, v in tuples:
            try:
                expand = k.startswith('**')
            except:
                expand = False
            if expand:
                if len(k) > 3 and k[2] == '*':
                    k = k[3:]
                    k = k if k.startswith(key_prefix) else f'{key_prefix}{k}'
                else:
                    k = key_prefix
                for kk, vv in v.items():
                    output[f'{k}{_key_connector}{kk}'] = vv
            else:
                output[k if k.startswith(key_prefix) else f'{key_prefix}{k}'] = v
    else:
        for k, v in tuples:
            try:
                expand = k.startswith('**')
            except:
                expand = False
            if expand:
                if len(k) > 3 and k[2] == '*':
                    k = k[3:]
                    for kk, vv in v.items():
                        output[f'{k}{_key_connector}{kk}'] = vv
                else:
                    output.update(v)
            else:
                output[k] = v
    return output


def merge_list_dicts(dicts: Iterator[Dict[Any, List]], in_place: bool = False):
    """
    Merges dictionaries of lists. Lists with the same key will be merged as a single list.
    :param dicts: the dictionaries to merge.
    :param in_place: `True` if the merge results are saved in-place in the first dictionary of `dicts` and returned; `False` if creating a new dictionary to store the merge results.
    :return: a dictionary of merged lists; either the first dictionary of `dicts` if `in_place` is `True`, or otherwise a new dictionary object.
    """
    output_dict = None if in_place else defaultdict(list)
    for d in dicts:
        if output_dict is None:
            output_dict = defaultdict(list, d)
        else:
            assert d is not output_dict, "the first dictionary appears twice and `in_place` is set True; in this case we have lost the original data in the first dictionary and hence the merge cannot proceed"
            for k, v in d.items():
                output_dict[k].extend(v)
    return output_dict


def merge_set_dicts(dicts: Iterator[Dict[Any, set]], in_place: bool = False):
    """
    Merges dictionaries of sets. Sets with the same key will be merged as a single set.
    :param dicts: the dictionaries to merge.
    :param in_place: `True` if the merge results are saved in-place in the first dictionary of `dicts` and returned; `False` if creating a new dictionary to store the merge results.
    :return: a dictionary of merged sets; either the first dictionary of `dicts` if `in_place` is `True`, or otherwise a new dictionary object.
    """
    output_dict = None if in_place else defaultdict(set)
    for d in dicts:
        if output_dict is None:
            output_dict = defaultdict(set, d)
        else:
            assert d is not output_dict, "the first dictionary appears twice and `in_place` is set True; in this case we have lost the original data in the first dictionary and hence the merge cannot proceed"
            for k, v in d.items():
                output_dict[k] = output_dict[k].union(v)
    return output_dict


def merge_counter_dicts(dicts: Iterator[Dict[Any, Counter]], in_place: bool = False):
    """
    Merges dictionaries of counts (represented by :class:`~collections.Counter` objects). Counts with the same key will be merged as a single set.
    :param dicts: the dictionaries to merge.
    :param in_place: `True` if the merge results are saved in-place in the first dictionary of `dicts` and returned; `False` if creating a new dictionary to store the merge results.
    :return: a dictionary of merged counts; either the first dictionary of `dicts` if `in_place` is `True`, or otherwise a new dictionary object.
    """
    output_dict = None if in_place else {}
    for d in dicts:
        if output_dict is None:
            output_dict = d
        else:
            assert d is not output_dict, "the first dictionary appears twice and `in_place` is set True; in this case we have lost the original data in the first dictionary and hence the merge cannot proceed"
            for k, v in d.items():
                if k in output_dict:
                    output_dict[k] += v
                else:
                    output_dict[k] = v
    return output_dict


# endregion

# region nested dicts


def iter_leaves(tree: Mapping, init_key=(), yield_combo_key=False, key_filter: Callable[[tuple], bool] = None):
    """
    Iterates through the leaves of the tree (represented by nested mappings).
    Each time, it yields a 2-tuple, the first value is the parent node,
        and the second value is 1) a combo-key consisting of keys form the root to the leaf if `yield_combo_key` is `True`,
                             or 2) the key of the leaf if `yield_combo_key` is `False`.

    >>> import utilx.dict_ext as dx
    >>> tree = {
    >>>     'a': 1,
    >>>     'b': {'b1': 2,
    >>>           'b2': {'b21': 3,
    >>>                  'b22': 4,
    >>>                  'b23': {'b231': 5}}},
    >>>     'c': 6,
    >>>     'd': {'d1': 7,
    >>>           'd2': 8,
    >>>           'd3': {},
    >>>           'd4': {'d41': 9,
    >>>                  'd42': 10}}
    >>> }

    >>> # {'a': 0.5, 'b': {'b1': 1.0, 'b2': {'b21': 1.5, 'b22': 2.0, 'b23': {'b231': 2.5}}}, 'c': 3.0, 'd': {'d1': 3.5, 'd2': 4.0, 'd3': {}, 'd4': {'d41': 4.5, 'd42': 5.0}}}
    >>> for p, k in dx.iter_leaves(tree):
    >>>     p[k] /= 2
    >>> print(tree)

    >>> # the following code prints out these combo-keys:
    >>> #   ('a',)
    >>> #   ('b', 'b1')
    >>> #   ('b', 'b2', 'b21')
    >>> #   ('b', 'b2', 'b22')
    >>> #   ('b', 'b2', 'b23', 'b231')
    >>> #   ('c',)
    >>> #   ('d', 'd1')
    >>> #   ('d', 'd2')
    >>> #   ('d', 'd4', 'd41')
    >>> #   ('d', 'd4', 'd42')
    >>> for p, k in dx.iter_leaves(tree, yield_combo_key=True): # `k` here will consist of all keys from the root to the leaf
    >>>     print(k)
    >>>     p[k[-1]] *= 2
    >>> print(tree == {'a': 1.0, 'b': {'b1': 2.0, 'b2': {'b21': 3.0, 'b22': 4.0, 'b23': {'b231': 5.0}}}, 'c': 6.0, 'd': {'d1': 7.0, 'd2': 8.0, 'd3': {}, 'd4': {'d41': 9.0, 'd42': 10.0}}})

    >>> # the following skips multiplying the 'd' branch
    >>> for p, k in dx.iter_leaves(tree, key_filter=lambda x: 'd' not in x):
    >>>     p[k] *= 2
    >>> print(tree == {'a': 2.0, 'b': {'b1': 4.0, 'b2': {'b21': 6.0, 'b22': 8.0, 'b23': {'b231': 10.0}}}, 'c': 12.0, 'd': {'d1': 7.0, 'd2': 8.0, 'd3': {}, 'd4': {'d41': 9.0, 'd42': 10.0}}})

    :param tree: represented by a nested mapping.
    :param init_key: a tuple of keys prefixing each combo-key if `yield_combo_key` is `True`.
    :param yield_combo_key: `True` if yielding all keys from the root to the leaf as a tuple; otherwise, just the key of the leaf.
    :param key_filter: a function that accepts the combo-key (a tuple of keys from the root) as the input, and returns a boolean value such that `False` indicates the branch should be skipped.
    :return: an iterator as discussed above.
    """

    if yield_combo_key:
        for k, v in tree.items():
            k = (*init_key, k)
            if not key_filter or key_filter(k):
                if isinstance(v, Mapping):
                    yield from iter_leaves(v, init_key=k, yield_combo_key=True, key_filter=key_filter)
                else:
                    yield tree, k
    elif key_filter:
        for k, v in tree.items():
            _k = (*init_key, k)
            if key_filter(_k):
                if isinstance(v, Mapping):
                    yield from iter_leaves(v, init_key=_k, yield_combo_key=False, key_filter=key_filter)
                else:
                    yield tree, k
    else:
        for k, v in tree.items():
            if isinstance(v, Mapping):
                yield from iter_leaves(v, init_key=None, yield_combo_key=False, key_filter=None)
            else:
                yield tree, k


def iter_leaves_combo_key(tree: Mapping, init_key: str = None, combo_key_joint: str = '-', key_filter: Callable[[str], bool] = None):
    """
    Iterates through the leaves of the tree (represented by nested mappings).
    Each time, it yields a 2-tuple, the first value is a combo-key consisting of keys form the root to the leaf, and the value of the leaf.
    NOTE: this function only applies to a mapping whose keys are string-friendly, because the combo-key is a string; if not, use `iter_leaves_combo_key__` instead, whose combo-key is a tuple of all keys from the root.
    This function can be applied to flatten a tree.

    >>> import utilx.dict_ext as dx
    >>> tree = {
    >>>     'a': 1,
    >>>     'b': {'b1': 2,
    >>>           'b2': {'b21': 3,
    >>>                  'b22': 4,
    >>>                  'b23': {'b231': 5}}},
    >>>     'c': 6,
    >>>     'd': {'d1': 7,
    >>>           'd2': 8,
    >>>           'd3': {},
    >>>           'd4': {'d41': 9,
    >>>                  'd42': 10}}
    >>> }

    >>> # the following gives a flattened tree:
    >>> # {'a': 1, 'b-b1': 2, 'b-b2-b21': 3, 'b-b2-b22': 4, 'b-b2-b23-b231': 5, 'c': 6, 'd-d1': 7, 'd-d2': 8, 'd-d4-d41': 9, 'd-d4-d42': 10}
    >>> print({k: v for k, v in (dx.iter_leaves_combo_key(tree))})

    >>> # add a prefix to all keys; we may also set the `combo_key_joint` parameter
    >>> # {'root>a': 1, 'root>b>b1': 2, 'root>b>b2>b21': 3, 'root>b>b2>b22': 4, 'root>b>b2>b23>b231': 5, 'root>c': 6, 'root>d>d1': 7, 'root>d>d2': 8, 'root>d>d4>d41': 9, 'root>d>d4>d42': 10}
    >>> print({k: v for k, v in (dx.iter_leaves_combo_key(tree, init_key='root', combo_key_joint='>'))})

    >>> # the following filters the 'd' branch
    >>> # {'a': 1, 'b-b1': 2, 'b-b2-b21': 3, 'b-b2-b22': 4, 'b-b2-b23-b231': 5, 'c': 6}
    >>> print({k: v for k, v in (dx.iter_leaves_combo_key(tree, key_filter=lambda x: 'd' not in x))})

    :param tree: represented by a nested mapping.
    :param init_key: a string prefixing every yielded key.
    :param combo_key_joint: the separator between two levels of keys in each yielded combo-key.
    :param key_filter: a function that accepts the combo-key (a string consisting of keys from the root, joint by `combo_key_joint`) as the input, and returns a boolean value; return `False` to indicate the branch should be filtered.
    :return: an iterator through the leaves of the tree as discussed above.
    """
    for k, v in tree.items():
        combo_key = f'{init_key}{combo_key_joint}{k}' if init_key else k
        if not key_filter or key_filter(combo_key):
            if isinstance(v, Mapping):
                yield from iter_leaves_combo_key(v, init_key=combo_key, combo_key_joint=combo_key_joint, key_filter=key_filter)
            else:
                yield combo_key, v


def iter_leaves_combo_key__(tree: Mapping, init_key=(), key_filter: Callable[[tuple], bool] = None):
    """
    The same as `iter_leaves_combo_key`; the difference is that the combo-key is now a tuple of keys from the root to the leaf, rather than a string joint of the keys.
    The `key_filter` will now accept a tuple as the input.

    >>> import utilx.dict_ext as dx
    >>> tree = {
    >>>     'a': 1,
    >>>     'b': {'b1': 2,
    >>>           'b2': {'b21': 3,
    >>>                  'b22': 4,
    >>>                  'b23': {'b231': 5}}},
    >>>     'c': 6,
    >>>     'd': {'d1': 7,
    >>>           'd2': 8,
    >>>           'd3': {},
    >>>           'd4': {'d41': 9,
    >>>                  'd42': 10}}
    >>> }

    >>> # {('a',): 1, ('b', 'b1'): 2, ('b', 'b2', 'b21'): 3, ('b', 'b2', 'b22'): 4, ('b', 'b2', 'b23', 'b231'): 5, ('c',): 6, ('d', 'd1'): 7, ('d', 'd2'): 8, ('d', 'd4', 'd41'): 9, ('d', 'd4', 'd42'): 10}
    >>> print({k: v for k, v in (dx.iter_leaves_combo_key__(tree))})

    >>> # {(0, 'a'): 1, (0, 'b', 'b1'): 2, (0, 'b', 'b2', 'b21'): 3, (0, 'b', 'b2', 'b22'): 4, (0, 'b', 'b2', 'b23', 'b231'): 5, (0, 'c'): 6, (0, 'd', 'd1'): 7, (0, 'd', 'd2'): 8, (0, 'd', 'd4', 'd41'): 9, (0, 'd', 'd4', 'd42'): 10}
    >>> print({k: v for k, v in (dx.iter_leaves_combo_key__(tree, init_key=(0,)))})

    >>> # the following filters the 'd' branch
    >>> # {('a',): 1, ('b', 'b1'): 2, ('b', 'b2', 'b21'): 3, ('b', 'b2', 'b22'): 4, ('b', 'b2', 'b23', 'b231'): 5, ('c',): 6}
    >>> print({k: v for k, v in (dx.iter_leaves_combo_key__(tree, key_filter=lambda x: 'd' not in x))})

    """

    for k, v in tree.items():
        combo_key = (*init_key, k)
        if not key_filter or key_filter(combo_key):
            if isinstance(v, Mapping):
                yield from iter_leaves_combo_key__(v, init_key=combo_key, key_filter=key_filter)
            else:
                yield combo_key, v


def iter_leaves_with_another(tree: Mapping, iter_with: Mapping, ttype: Callable = dict, skip_empty_branches=False, yield_combo_key=False, init_key=(), key_formatter: Callable[[tuple], Any] = None):
    """
    Iterates through the leaves of the tree (represented by nested mappings), together with another mapping.
    This method can be applied to construct a dictionary with the same structure as the `tree`, with some transformations on the values.

    For example,
    >>> import utilx.dict_ext as dx

    >>> tree = {'a': 1,
    >>>         'b': {'b1': 2,
    >>>               'b2': {'b21': 3,
    >>>                      'b22': 4,
    >>>                      'b23': {'b231': 5}}},
    >>>         'c': 6,
    >>>         'd': {'d1': 7,
    >>>               'd2': 8,
    >>>               'd3': {},
    >>>               'd4': {'d41': 9,
    >>>                      'd42': 10}}}

    >>> reconstruct = {}
    >>> for d, k, v in dx.iter_leaves_with_another(tree, iter_with=reconstruct, ttype=dict):
    >>>     d[k] = v
    >>> print(reconstruct == tree)

    >>> # the following prints out the same tree structure, with all values being added by 1.
    >>> reconstruct = {}
    >>> for d, k, v in dx.iter_leaves_with_another(tree, iter_with=reconstruct, ttype=dict):
    >>>     d[k] = v + 1
    >>> print(reconstruct == {'a': 2,
    >>>                       'b': {'b1': 3,
    >>>                             'b2': {'b21': 4,
    >>>                                    'b22': 5,
    >>>                                    'b23': {'b231': 6}}},
    >>>                       'c': 7,
    >>>                       'd': {'d1': 8,
    >>>                             'd2': 9,
    >>>                             'd3': {},
    >>>                             'd4': {'d41': 10,
    >>>                                    'd42': 11}}})


    >>> # the empty branch 'd3' is skipped
    >>> reconstruct = {}
    >>> for d, k, v in dx.iter_leaves_with_another(tree, iter_with=reconstruct, ttype=dict, skip_empty_branches=True):
    >>>     d[k] = v + 1
    >>> print(reconstruct == {'a': 2,
    >>>                       'b': {'b1': 3,
    >>>                             'b2': {'b21': 4,
    >>>                                    'b22': 5,
    >>>                                    'b23': {'b231': 6}}},
    >>>                       'c': 7,
    >>>                       'd': {'d1': 8,
    >>>                             'd2': 9,
    >>>                             'd4': {'d41': 10,
    >>>                                    'd42': 11}}})

    # the following formats the key by `key_filter`
    >>> reconstruct = {}
    >>> for d, k, v in dx.iter_leaves_with_another(tree, iter_with=reconstruct, ttype=dict, key_formatter=lambda x: f'{x[-1]}+1', skip_empty_branches=True):
    >>>     d[k] = v + 1
    >>> print(reconstruct == {'a+1': 2,
    >>>                       'b+1': {'b1+1': 3,
    >>>                               'b2+1': {'b21+1': 4,
    >>>                                        'b22+1': 5,
    >>>                                        'b23+1': {'b231+1': 6}}},
    >>>                       'c+1': 7,
    >>>                       'd+1': {'d1+1': 8,
    >>>                               'd2+1': 9,
    >>>                               'd4+1': {'d41+1': 10,
    >>>                                        'd42+1': 11}}})

    >>> # yield combo-keys from the root to the leaf, rather than just the key of the leaf.
    >>> reconstruct = {}
    >>> for d, k, v in dx.iter_leaves_with_another(tree, iter_with=reconstruct, ttype=dict, yield_combo_key=True):
    >>>     d[k] = v
    >>> print(reconstruct == {('a',): 1,
    >>>                       ('b',): {('b', 'b1'): 2,
    >>>                                ('b', 'b2'): {('b', 'b2', 'b21'): 3,
    >>>                                              ('b', 'b2', 'b22'): 4,
    >>>                                              ('b', 'b2', 'b23'): {('b', 'b2', 'b23', 'b231'): 5}}},
    >>>                       ('c',): 6,
    >>>                       ('d',): {('d', 'd1'): 7,
    >>>                                ('d', 'd2'): 8,
    >>>                                ('d', 'd3'): {},
    >>>                                ('d', 'd4'): {('d', 'd4', 'd41'): 9,
    >>>                                              ('d', 'd4', 'd42'): 10}}})

    :param tree: represented by a nested mapping.
    :param iter_with: iterate through the leaves of the `tree` together with this mapping.
    :param ttype: when the key is not found in the `iter_with`, this function is called to generate a new mapping object which is then assigned to the key.
    :param skip_empty_branches: `True` if to skip an empty branch in the input tree (i.e. the ouptut tree will not have that empty branch); otherwise `False`.
    :param yield_combo_key: `True` if this iterator should yield a tuple combo-key consisting of all keys from the root to the leave; `False` if it only yields the key of the leaf.
    :param init_key: a tuple of keys prefixing each combo-key if `yield_combo_key` is `True`; see also examples in `iter_leaves_combo_key__`
    :param key_formatter: a function that accepts the combo-key (a tuple of keys from the root) as the input, and returns a formatted key; SPECIALLY, return `None` to indicate the branch should be skipped.
    :return: an iterator; yielding a three-tuple at a time: 1) a sub-mapping in `iter_with` that corresponds to the current level in the `tree` being iterated through; 2) the key; 3) the value.
    """
    ttype = ttype or dict
    for k, v in tree.items():

        if yield_combo_key:
            k = (*init_key, k)
            _k = key_formatter(k) if key_formatter else k
        elif key_formatter:
            k = (*init_key, k)
            _k = key_formatter(k)
        else:
            _k = k

        if _k is not None:
            if isinstance(v, Mapping):
                if len(v) == 0:
                    if not skip_empty_branches and _k not in iter_with:
                        iter_with[_k] = ttype()
                else:
                    if _k not in iter_with:
                        iter_with[_k] = ttype()
                        iter_with_key_created = True
                    else:
                        iter_with_key_created = False
                    yield from iter_leaves_with_another(tree=v, iter_with=iter_with[_k], ttype=ttype, skip_empty_branches=skip_empty_branches, yield_combo_key=yield_combo_key, init_key=k, key_formatter=key_formatter)
                    if iter_with_key_created and skip_empty_branches and len(iter_with[_k]) == 0:
                        del iter_with[_k]
            else:
                yield iter_with, _k, v


def merge_tree_leaves_padded(trees, padding=None, ttype=None, leaf_filter: Callable[[Tuple, Any], Tuple[Any, Any]] = None, skip_empty_branches=True, key_formatter: Callable[[tuple], Any] = None):
    """
    Merges the leaves of several trees into a single tree, where each leaf of the merged tree is a list of the values from the corresponding leaves of the original trees.
    Missing values will be filled by the `padding`.
    This function can be applied to turn a sequence of keyed data entries into a batch.

    >>> import utilx.dict_ext as dx
    >>> trees = (
    >>>         [{'a': 1, 'b': {'b1': 2, 'b2': 3, 'b3': {'b31': 4, 'b32': 5}}, 'b4': {'b5': {}}}] * 2  # missing 'b33'; 'b4' branch is empty
    >>>         + [{'a': 1, 'b': {'b1': 2, 'b3': {'b31': 4, 'b33': 6}}, 'b4': {'b5': {}}}] * 2  # missing 'b2', 'b32'
    >>>         + [{'a': 1, 'c': 7}] * 2  # missing entire 'b' branch
    >>> )
    >>> print(dx.merge_tree_leaves_padded(trees=trees) == {'a': [1, 1, 1, 1, 1, 1],
    >>>                                                    'b': {'b1': [2, 2, 2, 2, None, None],
    >>>                                                          'b2': [3, 3, None, None, None, None],
    >>>                                                          'b3': {'b31': [4, 4, 4, 4, None, None],
    >>>                                                                 'b32': [5, 5, None, None, None, None],
    >>>                                                                 'b33': [None, None, 6, 6, None, None]}},
    >>>                                                    'c': [None, None, None, None, 7, 7]})

    >>> # the filter removes the 'b2' branch, and format the keys
    >>> print(merge_tree_leaves_padded(trees=trees,
    >>>                                padding=0,
    >>>                                key_formatter=lambda k: None if 'b2' in k else '>'.join(k)) == {'a': [1, 1, 1, 1, 1, 1],
    >>>                                                                                                'b': {'b>b1': [2, 2, 2, 2, 0, 0],
    >>>                                                                                                      'b>b3': {'b>b3>b31': [4, 4, 4, 4, 0, 0],
    >>>                                                                                                               'b>b3>b32': [5, 5, 0, 0, 0, 0],
    >>>                                                                                                               'b>b3>b33': [0, 0, 6, 6, 0, 0]}},
    >>>                                                                                                'c': [0, 0, 0, 0, 7, 7]})

    :param trees: the input trees.
    :param padding: the padding for missing values.
    :param ttype: the type for the mapping that represents each node in the output tree.
    :param leaf_filter: a function that accepts the tuple combo-key and and value of each leaf (a tuple of keys from the root),
                            and returns the transformed key/value pair. Return a `None` key to skip that leaf.
                        NOTE the key passed in to this `leaf_filter` function is a tuple of all keys from the root to the leaf;
                            it is your responsibility to turn the key into an appropriate format in the function.
    :param skip_empty_branches: `True` if to skip an empty branches in the input trees; otherwise `False`.
    :param key_formatter: a function that accepts the combo-key (a tuple of keys from the root) as the input, and returns a formatted key; SPECIALLY, return `None` to indicate the branch should be skipped.
    :return: the merged tree as discussed above.
    """
    out = ttype() if ttype else {}

    def _add():
        if k in d:
            d_k = d[k]
            if len(d_k) == tree_idx:
                d[k].append(v)
            else:
                d[k].extend([padding] * (tree_idx - len(d_k)) + [v])
        else:
            d[k] = [padding] * tree_idx + [v]

    if leaf_filter is not None:
        for tree_idx, tree in enumerate(trees):
            num_trees = tree_idx
            for d, k, v in iter_leaves_with_another(tree=tree, iter_with=out, ttype=ttype, yield_combo_key=True, skip_empty_branches=skip_empty_branches, key_formatter=key_formatter):
                k, v = leaf_filter(k, v)
                if k is not None:
                    _add()
    else:
        for tree_idx, tree in enumerate(trees):
            num_trees = tree_idx
            for d, k, v in iter_leaves_with_another(tree=tree, iter_with=out, ttype=ttype, yield_combo_key=False, skip_empty_branches=skip_empty_branches, key_formatter=key_formatter):
                _add()

    if len(out) != 0:
        num_trees += 1
        for d, k in iter_leaves(tree=out):
            v = d[k]
            diff: int = num_trees - len(v)
            if diff != 0:
                v.extend([padding] * diff)
    return out


def merge_tree_leaves(trees, ttype=None, leaf_filter: Callable[[Tuple, Any], Tuple[Any, Any]] = None, skip_empty_branches=True, key_formatter: Callable[[tuple], Any] = None):
    """
    The same as `merge_tree_leaves_padded`, except for that this function do not fill in missing values.
    This function can be applied to aggregate data from multiple dictionaries.
    If missing values need to be filled, use `merge_tree_leaves_padded` instead.

    >>> import utilx.dict_ext as dx

    >>> trees = (
    >>>         [{'a': 1, 'b': {'b1': 2, 'b2': 3, 'b3': {'b31': 4, 'b32': 5}}, 'b4': {'b5': {}}}] * 2  # missing 'b33'; 'b4' branch is empty
    >>>         + [{'a': 1, 'b': {'b1': 2, 'b3': {'b31': 4, 'b33': 6}}, 'b4': {'b5': {}}}] * 2  # missing 'b2', 'b32'
    >>>         + [{'a': 1, 'c': 7}] * 2  # missing entire 'b' branch
    >>> )
    >>> print(dx.merge_tree_leaves(trees=trees) == {'a': [1, 1, 1, 1, 1, 1],
    >>>                                             'b': {'b1': [2, 2, 2, 2],
    >>>                                                   'b2': [3, 3],
    >>>                                                   'b3': {'b31': [4, 4, 4, 4],
    >>>                                                          'b32': [5, 5],
    >>>                                                          'b33': [6, 6]}},
    >>>                                             'c': [7, 7]})

    >>> # the filter removes the 'b2' branch, and format the keys
    >>> print(dx.merge_tree_leaves(trees=trees, key_formatter=lambda k: None if 'b2' in k else '>'.join(k)) == {'a': [1, 1, 1, 1, 1, 1],
    >>>                                                                                                         'b': {'b>b1': [2, 2, 2, 2],
    >>>                                                                                                               'b>b3': {'b>b3>b31': [4, 4, 4, 4],
    >>>                                                                                                                        'b>b3>b32': [5, 5],
    >>>                                                                                                                        'b>b3>b33': [6, 6]}},
    >>>                                                                                                         'c': [7, 7]})

    """
    out = ttype() if ttype else {}

    def _add():
        if k in d:
            d[k].append(v)
        else:
            d[k] = [v]

    if leaf_filter is not None:
        for tree in trees:
            for d, k, v in iter_leaves_with_another(tree=tree, iter_with=out, ttype=ttype, yield_combo_key=True, skip_empty_branches=skip_empty_branches, key_formatter=key_formatter):
                k, v = leaf_filter(k, v)
                if k is not None:
                    _add()
    else:
        for tree in trees:
            for d, k, v in iter_leaves_with_another(tree=tree, iter_with=out, ttype=ttype, yield_combo_key=False, skip_empty_branches=skip_empty_branches, key_formatter=key_formatter):
                _add()

    return out


def merge_tree_leaves_flat_padded(trees, padding=None, ttype=None, leaf_filter: Callable[[Tuple, Any], Tuple[Any, Any]] = None, combo_key_joint='-', key_filter: Callable[[tuple], bool] = None):
    """
    Merges the leaves of several trees into a single flattened mapping, where each key/value pair of the merged mapping is a list of the values from the corresponding leaves of the original trees.
    This function can be applied to turn a sequence of keyed data entries into a batch, with missing values being filled by the `padding`.

    >>> import utilx.dict_ext as dx
    >>> trees = (
    >>>         [{'a': 1, 'b': {'b1': 2, 'b2': 3, 'b3': {'b31': 4, 'b32': 5}}, 'b4': {'b5': {}}}] * 2  # missing 'b33'; 'b4' branch is empty
    >>>         + [{'a': 1, 'b': {'b1': 2, 'b3': {'b31': 4, 'b33': 6}}, 'b4': {'b5': {}}}] * 2  # missing 'b2', 'b32'
    >>>         + [{'a': 1, 'c': 7}] * 2  # missing entire 'b' branch
    >>> )

    >>> print(dx.merge_tree_leaves_flat_padded(trees) == {'a': [1, 1, 1, 1, 1, 1],
    >>>                                                   'b-b1': [2, 2, 2, 2, None, None],
    >>>                                                   'b-b2': [3, 3, None, None, None, None],
    >>>                                                   'b-b3-b31': [4, 4, 4, 4, None, None],
    >>>                                                   'b-b3-b32': [5, 5, None, None, None, None],
    >>>                                                   'b-b3-b33': [None, None, 6, 6, None, None],
    >>>                                                   'c': [None, None, None, None, 7, 7]})

    >>> # the `key_filter` removes the 'b3' branch
    >>> print(dx.merge_tree_leaves_flat_padded(
    >>>     trees=trees,
    >>>     padding=0,
    >>>     combo_key_joint='>',
    >>>     key_filter=lambda k: 'b3' not in k) == {'a': [1, 1, 1, 1, 1, 1],
    >>>                                             'b>b1': [2, 2, 2, 2, 0, 0],
    >>>                                             'b>b2': [3, 3, 0, 0, 0, 0],
    >>>                                             'c': [0, 0, 0, 0, 7, 7]})
    
    >>> # `leaf_filter` overwrites the key format (in this case the `combo_key_joint` is now ignored), and can change the values
    >>> print(dx.merge_tree_leaves_flat_padded(
    >>>     trees=trees,
    >>>     padding=0,
    >>>     combo_key_joint='>',
    >>>     key_filter=lambda k: 'b2' not in k,
    >>>     leaf_filter=lambda k, v: ('|'.join(k), v + 1)) == {'a': [2, 2, 2, 2, 2, 2],
    >>>                                                        'b|b1': [3, 3, 3, 3, 0, 0],
    >>>                                                        'b|b3|b31': [5, 5, 5, 5, 0, 0],
    >>>                                                        'b|b3|b32': [6, 6, 0, 0, 0, 0],
    >>>                                                        'b|b3|b33': [0, 0, 7, 7, 0, 0],
    >>>                                                        'c': [0, 0, 0, 0, 8, 8]})

    :param trees: the input trees.
    :param padding: the padding for missing values.
    :param ttype: the type for the output mapping.
    :param leaf_filter: a function that accepts the tuple combo-key and and value of each leaf (a tuple of keys from the root),
                            and returns the transformed key/value pair. Return a `None` key to skip that leaf.
                        NOTE the key passed in to this `leaf_filter` function is a tuple of all keys from the root to the leaf;
                            it is your responsibility to turn the key into an appropriate format in the function.
    :param combo_key_joint: an optional parameter effective only if `leaf_filter` is not set; in the output mapping, the keys will be a string consisting of all string representations of keys from the root to the leaf, combined by this `combo_key_joint`
    :param key_filter: a function that accepts the combo-key (a tuple of keys from the root) as the input, and returns a boolean value; return `False` to indicate the branch should be filtered.
    :return: a merged mapping from merged leaf keys to lists of corresponding leaf values.
    """
    out = ttype() if ttype else {}

    def _add():
        if k in out:
            out_k = out[k]
            if len(out_k) == tree_idx:
                out[k].append(v)
            else:
                out[k].extend([padding] * (tree_idx - len(out_k)) + [v])
        else:
            out[k] = [padding] * tree_idx + [v]

    if leaf_filter:
        for tree_idx, tree in enumerate(trees):
            num_trees = tree_idx
            for k, v in iter_leaves_combo_key__(tree=tree, key_filter=key_filter):
                k, v = leaf_filter(k, v)
                if k is not None:
                    _add()
    elif key_filter:
        for tree_idx, tree in enumerate(trees):
            num_trees = tree_idx
            for k, v in iter_leaves_combo_key__(tree=tree, key_filter=key_filter):
                k = combo_key_joint.join(k)
                _add()
    else:
        for tree_idx, tree in enumerate(trees):
            num_trees = tree_idx
            for k, v in iter_leaves_combo_key(tree=tree, combo_key_joint=combo_key_joint):
                _add()

    if len(out) != 0:
        num_trees += 1
        for v in out.values():
            diff: int = num_trees - len(v)
            if diff != 0:
                v.extend([padding] * diff)
    return out


def merge_tree_leaves_flat(trees, ttype=None, leaf_filter: Callable[[Tuple, Any], Tuple[Any, Any]] = None, combo_key_joint='-', key_filter: Callable[[tuple], bool] = None):
    """
    The same as `merge_tree_leaves_flat_padded`, except for that this function do not fill in missing values.
    This function can be applied to aggregate data from multiple dictionaries.
    If missing values need to be filled, use `merge_tree_leaves_flat_padded` instead.

    >>> import utilx.dict_ext as dx
    >>> 
    >>> trees = (
    >>>         [{'a': 1, 'b': {'b1': 2, 'b2': 3, 'b3': {'b31': 4, 'b32': 5}}, 'b4': {'b5': {}}}] * 2  # missing 'b33'; 'b4' branch is empty
    >>>         + [{'a': 1, 'b': {'b1': 2, 'b3': {'b31': 4, 'b33': 6}}, 'b4': {'b5': {}}}] * 2  # missing 'b2', 'b32'
    >>>         + [{'a': 1, 'c': 7}] * 2  # missing entire 'b' branch
    >>> )

    >>> print(dx.merge_tree_leaves_flat(trees) == {'a': [1, 1, 1, 1, 1, 1],
    >>>                                            'b-b1': [2, 2, 2, 2],
    >>>                                            'b-b2': [3, 3],
    >>>                                            'b-b3-b31': [4, 4, 4, 4],
    >>>                                            'b-b3-b32': [5, 5],
    >>>                                            'b-b3-b33': [6, 6],
    >>>                                            'c': [7, 7]})

    >>> # the `key_filter` removes the 'b3' branch
    >>> print(dx.merge_tree_leaves_flat(
    >>>     trees=trees,
    >>>     combo_key_joint='>',
    >>>     key_filter=lambda k: 'b3' not in k) == {'a': [1, 1, 1, 1, 1, 1],
    >>>                                             'b>b1': [2, 2, 2, 2],
    >>>                                             'b>b2': [3, 3],
    >>>                                             'c': [7, 7]})

    >>> # `leaf_filter` overwrites the key format (in this case the `combo_key_joint` is now ignored), and can change the values
    >>> print(dx.merge_tree_leaves_flat(
    >>>     trees=trees,
    >>>     combo_key_joint='>',
    >>>     key_filter=lambda k: 'b2' not in k,
    >>>     leaf_filter=lambda k, v: ('|'.join(k), v + 1)) == {'a': [2, 2, 2, 2, 2, 2],
    >>>                                                        'b|b1': [3, 3, 3, 3],
    >>>                                                        'b|b3|b31': [5, 5, 5, 5],
    >>>                                                        'b|b3|b32': [6, 6],
    >>>                                                        'b|b3|b33': [7, 7],
    >>>                                                        'c': [8, 8]})

    """
    out = ttype() if ttype else {}

    def _add():
        if k in out:
            out[k].append(v)
        else:
            out[k] = [v]

    if leaf_filter:
        for tree in trees:
            for k, v in iter_leaves_combo_key__(tree=tree, key_filter=key_filter):
                k, v = leaf_filter(k, v)
                if k is not None:
                    _add()
    elif key_filter:
        for tree in trees:
            for k, v in iter_leaves_combo_key__(tree=tree, key_filter=key_filter):
                k = combo_key_joint.join(k)
                _add()
    else:
        for tree in trees:
            for k, v in iter_leaves_combo_key(tree=tree, combo_key_joint=combo_key_joint):
                _add()

    return out


def recursive_getitem(tree, keys):
    for key in keys:
        tree = tree[key]
    return tree


# endregion


def index_dict(enumerable):
    idx_dict = {}
    for idx, item in enumerate(enumerable):
        if item not in idx_dict:
            idx_dict[item] = idx
    return idx_dict


def update_dict_by_addition(src_dict, update_dict):
    for k, v in update_dict.items():
        if k in src_dict:
            src_dict[k] += v
        else:
            src_dict[k] = v


def rename_keys(d: dict, rename_pairs):
    """
    Renames the keys in a dictionary. Useful for renaming keys in JSON objects.
    :param d: the dictionary.
    :param rename_pairs: a list of renaming pairs; for each pair, the first is the old key, and the second is the new key.
    """
    for old_key, new_key in rename_pairs:
        if old_key in d:
            d[new_key] = d[old_key]
            del d[old_key]


def select_by_keys(d: dict, keys):
    return {k: d[k] for k in keys if k in d}


def select_values_by_keys(d: dict, keys):
    return [d[k] for k in keys if k in d]


def prioritize_keys(d: dict, keys_to_promote, in_place=True):
    keys_to_demote = set(d.keys()).difference(keys_to_promote)
    if in_place:
        for k in keys_to_promote:
            if k in d:
                v = d[k]
                del d[k]
                d[k] = v
        for k in keys_to_demote:
            v = d[k]
            del d[k]
            d[k] = v
        return d
    else:
        new_dict = {}
        for k in keys_to_promote:
            if k in d:
                new_dict[k] = d[k]
        for k in keys_to_demote:
            new_dict[k] = d[k]
        return new_dict


# region misc

def same_key_same_type(*mappings):
    """
    Checks if every mapping has the same keys, and the objects associated with the same key are of the same type.

    >>> from utilx.dict_ext import same_key_same_type, fdict
    >>> d1 = {'a': 1, 'b':2, 'c':3}
    >>> d2 = fdict(a=1,b=2,c=3)
    >>> d3 = {'a': 1, 'b': True, 'c': 3 }
    >>> print(same_key_same_type(d1, d2) == True)
    >>> print(same_key_same_type(d1, d3) == False)

    :param mappings: the mappings to check.
    :return: `True` if the check is success; otherwise `False`.
    """
    key_types = [{k: type(v) for k, v in mapping.items()} for mapping in mappings]
    first = key_types[0]
    return all(first == x for x in key_types[1:])


def dict_try_div(d: dict, divisor):
    for k in d:
        try:
            d[k] /= divisor
        except:
            continue
    return d


def dict_try_floor_div(d: dict, divisor):
    for k in d:
        try:
            d[k] //= divisor
        except:
            continue
    return d

# endregion

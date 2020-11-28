# In comparison to AllenNLP, our conversion of data to numerical numbers is simpler, flexible and more efficient.
# For our design,
#   1) Data entries are designed to only hold data; AllenNLP's fields can hold things other than data (e.g. text field holds indexers), which makes it harder to save processed data on disk.
#   2) The tokenizer should only split text into a list of strings as tokens; AllenNLP's tokenizer is alllowed to do the job of indexers, which makes simple things much more complicated.
from enum import Enum
from itertools import chain
from typing import Union, Iterator, List, Any, Tuple, Dict, Callable, Mapping

import numpy as np
import torch
from tqdm import tqdm

import utix.general as gx
from utix.timex import tic, toc
from utix.listex import ensure_list_or_tuple__, make_tuple_of_type
from utix.npex import array__
from utix.rndex import make_rnd_gen
from utix.torchu import tensor__
from utix.dictex import IndexDict, is_dict__, ListDict, same_key_same_type, merge_tree_leaves_flat_padded, hdict, recursive_getitem
from utix.listex import nested_lists_regularize


# region data entries

class Data:
    """
    Represents one data item.
    """
    DATA_KEY = None
    def __init__(self, data):
        self.data = data.data if isinstance(data, Data) else self._preprocess(data)

    def _preprocess(self, data):
        return data

    def __call__(self, **kwargs):
        raise NotImplementedError


class IndexableData(Data, Mapping):
    """
    Represents one indexable data item.
    """

    tensorizable = True
    INDEX_KEY = 'index'

    def __init__(self, data: Union[List, Any]):
        super(IndexableData, self).__init__(data)
        self.indices = None

    def __len__(self):
        if self.indices:
            return {k: self._len(v) for k, v in self.indices.items()} if is_dict__(self.indices) else self._len(self.indices)
        else:
            raise RuntimeError('index the data first; otherwise the length is not available')

    def __call__(self, *args, **kwargs):
        return self.index(*args, **kwargs)

    def __getitem__(self, item):
        if self.indices is None:
            raise RuntimeError('index the data first; otherwise the indices are not available')
        if item == self.INDEX_KEY:
            return self.indices
        elif item == self.DATA_KEY:
            return self.data
        elif gx.is_mapping(self.indices):
            if gx.is_str(item):
                return self.indices[item]
            else:
                return recursive_getitem(self.indices, item)

        raise KeyError

    def __iter__(self):
        yield self.DATA_KEY
        yield self.INDEX_KEY
        if gx.is_mapping(self.indices):
            yield from self.indices

    def items(self):
        yield self.DATA_KEY, self.data
        yield self.INDEX_KEY, self.indices
        if gx.is_mapping(self.indices):
            yield from self.indices.items()

    @staticmethod
    def _len(indices):
        return len(indices)

    def index(self, indexer):
        """
        Index this data with the given `indexer`.
        :param indexer: can be a single indexer or a dictionary of named indexers; an indexer must be either a callable, or has an `index` method.
        :return: the indexed data; if named indexers are provided, the indexed data is a dictionary with indices associated with the corresponding names.
        """
        if self.indices is None:
            if isinstance(indexer, Mapping):
                results = {}
                for k, v in indexer.items():
                    cur_index = self._index(v)
                    if cur_index is not None and (not isinstance(cur_index, (list, tuple)) or (len(cur_index) != 0 and cur_index[0] is not None)):
                        if isinstance(cur_index, Mapping):
                            for kk, vv in cur_index.items():
                                results[f'{k}_{kk}'] = vv
                        else:
                            results[k] = cur_index
                if results:
                    self.indices = results
            else:
                self.indices = self._index(indexer)
                if isinstance(self.indices, (list, tuple)) and self.indices[0] is None:
                    self.indices = None
        return self.indices

    def _index(self, indexer):
        raise NotImplementedError


class TextData(IndexableData):
    __slots__ = ('data', 'indices')
    DATA_KEY = 'text'

    def _index(self, indexer):
        if callable(indexer):
            return indexer(self.data)
        else:
            return indexer.index(self.data)


class CategoricalData(IndexableData):
    __slots__ = ('data', 'indices')
    DATA_KEY = 'ids'

    def _preprocess(self, data, **kwargs):
        return data if type(data) is list else [data]

    def _index(self, indexer: IndexDict):
        if callable(indexer):
            return [indexer(x) for x in self.data]
        else:
            return [indexer.index(x) for x in self.data]

    def __iter__(self):
        return self.data.__iter__()


class ListTextData(IndexableData):
    __slots__ = ('data', 'indices')
    DATA_KEY = 'texts'

    @staticmethod
    def _len(indices):
        return max(len(x) for x in indices)

    def _index(self, indexer):
        indices = [indexer.index(x) for x in self.data]
        return sum(indices, ListDict()) if is_dict__(indices[0]) else indices


class ListCategoricalData(IndexableData):
    DATA_KEY = 'id_list'

    @staticmethod
    def _len(indices):
        return max(len(x) for x in indices)

    def _preprocess(self, data):
        self.data = [x if type(x) is list else [x] for x in data]

    def _index(self, indexer: IndexDict):
        first = indexer.index_all(self.data[0])
        if is_dict__(first):
            return sum((indexer.index_all(x) for x in self.data[1:]), first)
        else:
            return [first] + [indexer.index_all(x) for x in self.data[1:]]


class NumDataTypes(Enum):
    NumpyArray = 'numpy'
    TorchTensor = 'tensor'
    RaggedNestedListsToNumpyArray = 'list2numpy'
    RaggedNestedListsToTorchTensor = 'list2tensor'


class NumData(Data, Mapping):
    tensorizable = True
    DATA_KEY = 'features'

    def __init__(self, data):
        Data.__init__(self, data=data)
        self._data = None

    def __getitem__(self, item):
        if item == self.DATA_KEY:
            return self.data
        elif gx.is_mapping(self._data):
            if gx.is_str(item):
                return self._data[item]
            else:
                return recursive_getitem(self._data, item)

        raise KeyError

    def __len__(self):
        if self.data:
            return {k: len(v) for k, v in self.data.items()} if is_dict__(self.data) else len(self.data)
        else:
            return 0

    def __iter__(self):
        yield self.DATA_KEY
        if gx.is_mapping(self.data):
            yield from self.data

    def items(self):
        yield self.DATA_KEY, self.data
        if gx.is_mapping(self.data):
            yield from self.data.items()

    def __call__(self, converter: NumDataTypes = NumDataTypes.NumpyArray, **kwargs):
        if self._data is None:
            if callable(converter):
                self._data = converter(self.data, **kwargs)
            elif converter == NumDataTypes.NumpyArray:
                self._data = np.array(self.data, **kwargs)
            elif converter == NumDataTypes.TorchTensor:
                self._data = torch.tensor(self.data, **kwargs)
            elif converter == NumDataTypes.RaggedNestedListsToNumpyArray:
                self._data = array__(self.data, **kwargs)
            elif converter == NumDataTypes.RaggedNestedListsToTorchTensor:
                self._data = tensor__(self.data, **kwargs)
            else:
                raise ValueError(f"the converter `{converter}` is not supported")
        return self._data


class DataList(Mapping):
    __slots__ = ('list',)

    def __init__(self, init=None):
        if isinstance(init, list):
            if all((isinstance(x, Mapping) for x in init)):
                self.list = init
            else:
                raise ValueError('objects in a data list must be mappings')
        else:
            if isinstance(init, Mapping):
                self.list = [init]
            else:
                raise ValueError('objects in a data list must be mappings')

    def append(self, item):
        if isinstance(item, Mapping):
            self.list.append(item)
        else:
            raise ValueError('objects in a data list must be mappings')

    def __len__(self):
        return [len(x) for x in self.list]

    def __iter__(self):
        yield from chain(*self.list)

    def items(self):
        yield from chain(*(x.items() for x in self.list))

    def __getitem__(self, item):
        if isinstance(item, str):
            return next(x for x in self.list if isinstance(x, Data) and x.DATA_KEY == item)
        elif isinstance(item, type):
            return next(x for x in self.list if isinstance(x, item))
        else:
            raise ValueError(f"expected the data key be a string or a type; got {type(item)}")


def make_data_(data_dict: dict, data_typing: Union[Mapping, Callable, None] = None):
    """
    A function that turns values in a dictionary to `Data` objects according to the `data_typing` information.
    The conversion is in-place.
    :param data_dict: the data dictionary.
    :param data_typing: the data typing information; can be 1) a callable (e.g. one of the build-in `Data` types); in this case all values in `data_dict` will be converted using this callable;
                                                            2) a string of one of the following for build-in `Data` types: 'text' (or 'txt'), 'categorical' (or 'cat'), 'text-list' (or 'txtlist'), 'categorical-list' (or 'catlist'), 'num'; in this case all values in `data_dict` will be converted to the specified `Data` type.
                                                            3) a mapping from field names to a callable or a build-in `Data` type; fields in `data_dict` not included in this dict will not be converted.
    :return: the same dictionary as the input `data_dict`, with values converted to `Data` objects according to `data_typing`.
    """
    if data_typing is None:
        return data_dict

    def _convert():
        if data_type:
            if callable(data_type):
                data_dict[k] = data_type(v)
            elif data_type == 'text' or data_type == 'txt':
                data_dict[k] = TextData(v)
            elif data_type == 'categorical' or data_type == 'cat':
                data_dict[k] = CategoricalData(v)
            elif data_type == 'text-list' or data_type == 'txtlist':
                data_dict[k] = ListTextData(v)
            elif data_type == 'categorical-list' or data_type == 'catlist':
                data_dict[k] = ListCategoricalData(v)
            elif data_type == 'num':
                data_dict[k] = NumData(v)
            else:
                raise ValueError(f"the data type `{data_type}` is not supported")

    if callable(data_typing) or isinstance(data_typing, str):
        data_type = data_typing
        for k, v in data_dict.items():
            _convert()
    elif isinstance(data_typing, Mapping):
        for k, v in data_dict.items():
            data_type = data_typing.get(k, None)
            _convert()
    else:
        raise ValueError(f'the data typing object {data_typing} is not recognized')
    return data_dict


class DataEntry(Mapping):
    """
    Represents a single data entry、
    A data entry is analogous to one row in a table, or one data item in a training batch.

    A data entry itself supports up to 2-level structure. This 2-level structure is defined by this `DataEntry` class and the `DataList` class.
    Every non-meta-data object in this class must be a `Data` or a `DataList`. A `DataList` may contain multiple `Data` objects, but we expect there is one object for each type; a `DataList` can be logically viewed as a data namespace.
    All other objects not of type `Data` or `DataList` are treated as meta data.
    Higher levels of data structures are supported by each type `Data`; for example, a `TextData` may have multiple indices, and a `NumData` may contain multiple features.

    This class assumes all data are passed in through it constructor. Only adding meta data objects is supported afterwards.

    """

    __slots__ = ('data',)

    def _add_data(self, k, d):
        if k in self.data:
            if isinstance(self.data[k], DataList):
                self.data[k].append(d)
            else:
                self.data[k] = DataList([self.data[k], d])
        else:
            self.data[k] = d

    def __init__(self,
                 as_text=None,
                 as_text_list=None,
                 as_meta=None,
                 as_categoricals=None,
                 as_categorical_list=None,
                 as_nums=None,
                 data=None,
                 data_typing=None):
        """
        Sets data in this data entry.
        :param as_text: accepts the following formats:
                        1) a list of key/string tuples; each string will be converted to a single `TextField` that hosts the text tokens, and the given key as the field name for the `TextField`;
                        2) a list of key/lines tuples ('lines' is a list of strings); the lines in each list will be tokenized first and then concatenated by a special separation token (e.g. the '[SEP]' as in BERT); the concatenated tokens will be a single `TextField`.
                      The original strings can be preserved as `MetaField`s.
        :param as_text_list: a list of key/text-list tuples. Each text list will be converted to a `ListField` of `TextField`s that host the text tokens, and with the given key as the field name for the `ListField`.
                            Each 'text' in the text list can be a string or a list of strings, and they are converted to tokens in the same way as the `texts`.
                            The original texts can be preserved as `MetaField`s.
        :param as_meta: a list of key/metadata-object tuples. Each metadata object will be converted to a single `MetaField` with data object as its value, and the given key as the field name.
        :param as_meta_list: a list of key/metadata-list tuples. Each metadata list will be converted to a `ListField` of `MetaField`s with the metadata objects as their values, and the given key as the field name for the `ListField`.
        """

        if data is not None:
            self.data = make_data_(data, data_typing)
        else:
            self.data = {}

        for k, v in gx.iter_pairs(as_text):
            self._add_data(k, TextData(v))

        for k, v in gx.iter_pairs(as_text_list):
            self._add_data(k, ListTextData(v))

        for k, v in gx.iter_pairs(as_categoricals):
            self._add_data(k, CategoricalData(v))

        for k, v in gx.iter_pairs(as_categorical_list):
            self._add_data(k, ListCategoricalData(v))

        for k, v in gx.iter_pairs(as_nums):
            self._add_data(k, NumData(v))

        for k, v in gx.iter_pairs(as_meta):
            self._add_data(k, v)

    def add_meta(self, data_key, meta_data):
        """
        Adds a meta data object associated with the specified data key.
        A data entry only supports adding meta data after initialization.
        """
        if data_key in self.data:
            raise ValueError('a data object associated with the key already exists')
        self.data[data_key] = meta_data

    def index(self, indexers: Dict[Union[str, type], Any]):
        for k, v in self.data.items():
            if isinstance(v, IndexableData):
                indexer = indexers.get(k, indexers.get(type(v), None))
                if indexer is None:
                    raise ValueError(f'the indexer for data entry `{k}` cannot be found')
                v.index(indexer)

    def __len__(self) -> int:
        return len(self.data) if self.data else 0

    def __iter__(self):
        yield from self.data.items()

    def __contains__(self, item):
        return bool(self.__getitem__(item))

    def __getitem__(self, item):
        """
        Gets a callable object, which is one `Data` object in this data entry, or multiple `Data` objects of the same type, in order for us to "call" these objects for data processing.
        Or gets a meta data object from this data entry.

        If the key `item` refers to non-meta-data objects, this method assumes the next step is to call these objects; the intention is not for data accessing.
        If the key `item` refers to a meta-data object, then we can access what is inside the data object, since a meta data object is typically just a Python object.
        This method only recognizes the two-level data-entry structure, and cannot retrieve data inside the `Data` object.

        The `item` can be a string, which directly retrieves the object associated with this key;
        or a type which is a subclass of `Data`, which retrieves all `Data` objects in this data entry of that type;
        or a tuple of primary key and secondary key, where the primary key is a string, and the secondary key is type, and this is intended to retrieve the `Data` object of the specified type from a `DataList`.
        """
        if isinstance(item, tuple):  # a tuple of keys; retrieves `Data` objects of the same type from a `DataList`; if the primary key points to a `Data` object, then this object must be of type specified by the secondary key.
            primary_key, secondary_key = item
            data = self.data.get(primary_key, None)
            if data is not None:
                if isinstance(data, DataList):
                    return data.get(secondary_key, None)
                elif isinstance(secondary_key, type):
                    return data if isinstance(data, secondary_key) else None
        if isinstance(item, type):  # retrieves `Data` objects of this type, including those inside a `DataList`
            callables = []
            for x in self.data.values():
                if isinstance(x, DataList):
                    x = x[item]
                    if x is not None:
                        callables.append(x)
                elif isinstance(x, item):
                    callables.append(x)
            if callables:
                return gx.Callables(callables)
        else:
            return self.data.get(item, None)

    def __setitem__(self, key, value):
        if key in self.data:
            self.data[key] = value
        else:
            raise ValueError("a data entry does not support adding new non-meta-data object after initialization")

    def items(self):
        return self.data.items()


# endregion

# region DataInfo, the metadata for conversion of data to the format required by the models
_INFO_NAME_DATA_KEY = 'data_key'
_INFO_NAME_PADDING = 'padding'


class DataArgInfoItem:
    __slots__ = ('data_key', 'data_arg', 'vocab_build')

    def __init__(self, data_key: str, data_arg, vocab_build=None):
        """
        Data argument.
        :param data_key:
        :param data_arg:
        """
        self.data_key, self.data_arg, self.vocab_build = data_key, data_arg, vocab_build


def _default_sort_key_gen(x):
    return -len(x)


class DataSortInfoItem:
    """
    Stores data sorting information.
    With this information, we are able to extract values from the data entries from the specified data and the specified field as the sorting keys (with the option to mix noise),
        and then the data entries can be sorted by the extracted sorting keys.

    Each data entry can be viewed as 2-level structure. A data entry may have multiple data items indexed by the `data_key`, and each data item may have several fields;
    therefore we need `data_key` and `sort_field` to retrieve a value from the data entry as the sorting key associated with the data value, with the option to apply the `sort_key_gen`, and mix with the `sort_noise`.

    """
    __slots__ = ('data_key', 'sort_field', 'sort_key_gen', 'sort_noise')

    def __init__(self, data_key, sort_field: str = None, sort_key_gen: Callable = _default_sort_key_gen, sort_noise=None):
        """
        :param data_key: provides the key of the data item in a data entry; a data entry is a two-level structure
        :param sort_field: provides the key for a field in a data item.
        :param sort_key_gen: a callable to transform the retrieved value from a data entry; for example, the default is the negative length `-len`.
        :param sort_noise: a random number generator that generates noise values; each noise value will multiply the original sorting key to create a noisy sorting key;
                                1) if an integer `a` is provided, then it is uniform noise between 0 and `a`, e.g. sort_noise=1, then it is a uniform noise between 0 and 1;
                                2) if two integers `a` and `b` are provided, then it is uniform noise between `a` and `b`, e.g. `sort_noise=(0.8, 1)`, then it is uniform noise between 0.8 and 1;
                                3) otherwise, a `utix.randex.RndGen` object that can provide any type of random noise.
        """

        self.data_key = data_key
        self.sort_field = sort_field
        self.sort_key_gen = sort_key_gen
        self.sort_noise = make_rnd_gen(sort_noise)

    def get_sorting_keys(self, data_entries):
        if self.sort_field is None:
            if self.sort_noise:
                noise = self.sort_noise(size=len(data_entries))
                if self.sort_key_gen:
                    return (self.sort_key_gen(x[self.data_key]) * noise[i] for i, x in enumerate(data_entries))
                else:
                    return (x[self.data_key] * noise[i] for i, x in enumerate(data_entries))
            else:
                if self.sort_key_gen:
                    return (self.sort_key_gen(x[self.data_key]) for x in data_entries)
                else:
                    return (x[self.data_key] for x in data_entries)
        else:
            if self.sort_noise:
                noise = self.sort_noise(size=len(data_entries))
                if self.sort_key_gen:
                    return (self.sort_key_gen(x[self.data_key][self.sort_field]) * noise[i] for i, x in enumerate(data_entries))
                else:
                    return (x[self.data_key][self.sort_field] * noise[i] for i, x in enumerate(data_entries))
            else:
                if self.sort_key_gen:
                    return (self.sort_key_gen(x[self.data_key][self.sort_field]) for x in data_entries)
                else:
                    return (x[self.data_key][self.sort_field] for x in data_entries)


class TensorizationInfoItem:
    __slots__ = (_INFO_NAME_DATA_KEY, 'tensor_type', 'tensor_args')

    def __init__(self, data_key: str, tensor_type='torch', **tensor_args):
        self.data_key, self.tensor_type, self.tensor_args = data_key, tensor_type, tensor_args


class TensoriztionMethodInfo:
    __slots__ = ('no_padding_method', 'with_padding_method', 'type_map')

    def __init__(self, no_padding_method: Callable, with_padding_method: Callable, type_map: Mapping):
        self.no_padding_method = no_padding_method
        self.with_padding_method = with_padding_method
        self.type_map = type_map


class DataInfo:
    """
    Saves meta information for data processing.
    """
    __slots__ = ('_data_args', '_data_sort', '_tensorization', '_other')

    def __init__(self, data_args=None, data_sort=None, tensorization=None, **kwargs):
        self._data_args: Tuple[DataArgInfoItem, ...] = make_tuple_of_type(data_args, DataArgInfoItem) if data_args else None
        self._data_sort: Tuple[DataSortInfoItem, ...] = make_tuple_of_type(data_sort, DataSortInfoItem) if data_sort else None

        # region for convenience of simple tensorization configuration
        if isinstance(tensorization, dict):
            tensorization2 = []
            for data_key in tensorization.get('data_keys', []):
                tensorization2.append({'data_key': data_key, 'padding': 0})

            for data_key in tensorization.get('data_keys_no_padding', []):
                tensorization2.append({'data_key': data_key})

            for data_key in tensorization.get('meta_data_keys', []):
                tensorization2.append({'data_key': data_key, 'tensor_type': 'list'})
            tensorization = tensorization2
        # endregion

        self._tensorization: Tuple[TensorizationInfoItem, ...] = make_tuple_of_type(tensorization, TensorizationInfoItem) if tensorization else None

    TensorizationMethods = hdict({
        'list': TensoriztionMethodInfo(no_padding_method=lambda x, **kwargs: list(x), with_padding_method=nested_lists_regularize, type_map={int: np.int, float: np.float, bool: np.bool}),
        'numpy': TensoriztionMethodInfo(no_padding_method=np.array, with_padding_method=array__, type_map={int: np.int, float: np.float, bool: np.bool}),
        'torch': TensoriztionMethodInfo(no_padding_method=torch.tensor, with_padding_method=tensor__, type_map={int: torch.int64, float: torch.float32, bool: torch.bool})
    })

    @classmethod
    def add_tensorization_method(cls, key: str, no_padding_method: Callable, with_padding_method: Callable, type_map: Mapping):
        cls.TensorizationMethods[key] = TensoriztionMethodInfo(no_padding_method=no_padding_method, with_padding_method=with_padding_method, type_map=type_map)

    def tensorize(self, data_dict: Mapping):
        if self._tensorization:
            for item in self._tensorization:
                method_info = DataInfo.TensorizationMethods[item.tensor_type]
                tensorize_method = method_info.with_padding_method if _INFO_NAME_PADDING in item.tensor_args else method_info.no_padding_method
                k = item.data_key
                if item.tensor_args.get('dtype', None) is None:
                    v = data_dict[k]
                    if isinstance(v, np.ndarray):
                        data_dict[k] = tensorize_method(v)
                    else:
                        data_dict[k] = tensorize_method(data_dict[k], dtype=method_info.type_map.get(gx.value_type(v), None), **item.tensor_args)
                else:
                    data_dict[k] = tensorize_method(data_dict[k], **item.tensor_args)
        else:
            for k, v in data_dict.items():
                if k[0] != '_':
                    if isinstance(v, np.ndarray):
                        data_dict[k] = torch.tensor(v)
                    else:
                        _type = gx.value_type(v)
                        if _type is int:
                            data_dict[k] = tensor__(v, padding=0, dtype=torch.int64)
                        elif _type is float:
                            data_dict[k] = tensor__(v, padding=0, dtype=torch.float32)
                        elif _type is bool:
                            data_dict[k] = tensor__(v, padding=False, dtype=torch.bool)
                        else:
                            raise ValueError(f"the default tensorization cannot be applied on values of type `{_type}`")

    def process_data(self, data_entries):
        for data_entry in data_entries:
            for item in self._data_args:
                if item.data_arg is None:
                    data_entry[item.data_key]()
                else:
                    data_entry[item.data_key](item.data_arg)

    def sort(self, data_entries):
        if self._data_sort:
            return gx.sorted__(data_entries, key=zip(*(item.get_sorting_keys(data_entries) for item in self._data_sort)), return_tuple=True)
        else:
            return data_entries

    def to_tensors(self, data_entries, data_key_filter=None, data_key_joint='-'):
        # region 1 - merge the indexable data
        if data_key_filter is None:
            if self._tensorization:
                # if tensorization info is available, the default filter is just the data keys of all tensorization info items
                data_key_filter = set(item.data_key for item in self._tensorization)

                def leaf_filter(_k, _v):
                    _k = data_key_joint.join(_k)
                    return (_k if _k in data_key_filter else None), _v
            else:
                # when tensorization info is not available, the default filter only tensorizes non-protective non-meta fields
                def leaf_filter(_k, _v):
                    return (None if len(_k) == 1 and not isinstance(_v, Data) else data_key_joint.join(_k)), _v
        elif isinstance(data_key_filter, str):
            # the `data_key_filter` can be a string with comma-separated key names
            data_key_filter = set(x.strip() for x in data_key_filter.split(','))

            def leaf_filter(_k, _v):
                _k = data_key_joint.join(_k)
                return (_k if _k in data_key_filter else None), _v
        elif isinstance(data_key_filter, (set, list, tuple)):
            item = next(iter(data_key_filter))
            if isinstance(item, str):
                # the data `data_key_filter` can be a set/list/tuple of key names
                def leaf_filter(_k, _v):
                    _k = data_key_joint.join(_k)
                    return (_k if _k in data_key_filter else None), _v
            elif isinstance(item, tuple):
                # the data `data_key_filter` can be a set/list/tuple of tuples of sub-key names
                def leaf_filter(_k, _v):
                    return (data_key_joint.join(_k) if _k in data_key_filter else None), _v
            else:
                raise TypeError('the data key filter is not of a supported type')
        elif callable(data_key_filter):
            # the data `data_key_filter` can be just a callable
            def leaf_filter(_k, _v):
                return data_key_filter(_k), _v
        else:
            raise TypeError('the data key filter is not of a supported type')

        out = merge_tree_leaves_flat_padded(
            trees=data_entries,
            leaf_filter=leaf_filter,
            combo_key_joint=data_key_joint)
        # endregion

        # region 2 - tensorization
        self.tensorize(out)
        # endregion

        return out

    def build_vocab(self, data_entries):
        """
        Build vocabularies. The vocabularies are
        :param data_entries:
        :return:
        """
        vocab_build_data_args = []
        for item in self._data_args:
            if item.data_arg is not None and item.vocab_build:
                for vocab in item.vocab_build:
                    if vocab.build_mode:
                        vocab_build_data_args.append(item)
                        break

        if vocab_build_data_args:
            data_entries = tqdm(data_entries)
            data_entries.set_description('build vocabulary')
            vocabs = set()
            for data_entry in data_entries:
                for item in self._data_args:
                    if item.data_arg is not None and item.vocab_build:
                        vocabs.update(item.vocab_build)
                        data_entry[item.data_key](item.data_arg)  # it is assumed this `item.data_arg` has a vocabulary-based indexer for this to work
            for vocab in vocabs:
                vocab.save()
        else:
            gx.hprint_message('all vocabularies are ready')


# endregion


class DataBatch:
    __slots__ = ('data_entries', 'tensors', 'cache_tensors')

    def __init__(self, data_entries: List[DataEntry], cache_tensors=True) -> None:
        super().__init__()
        self.data_entries: List[DataEntry] = ensure_list_or_tuple__(data_entries)
        self.cache_tensors = cache_tensors
        self.tensors = None
        if not same_key_same_type(*self.data_entries):
            raise ValueError('the data input to the batch must be consistent: with exactly the same keys, and the data associated with same key must have the same type')

    def to_tensors(self, data_info: DataInfo, data_key_filter=None, data_key_joint='-'):
        if self.cache_tensors:
            if self.tensors is None:
                self.tensors = data_info.to_tensors(self.data_entries, data_key_filter=data_key_filter, data_key_joint=data_key_joint)
            return self.tensors
        else:
            return data_info.to_tensors(self.data_entries, data_key_filter=data_key_filter, data_key_joint=data_key_joint)

    def __iter__(self) -> Iterator[DataEntry]:
        return iter(self.data_entries)

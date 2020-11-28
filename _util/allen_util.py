"""
Utility functions for AllenNLP.
"""
import warnings
from functools import partial
from typing import Tuple, List, Union, Iterator

from numpy import iterable
from os import path

from utix._util.general_ext import can_dict_like_read
from allennlp.modules.token_embedders import PretrainedBertEmbedder
from pytorch_transformers import AutoModel

import allennlp
import allennlp.data.token_indexers as indexers
from allennlp.data import DatasetReader
from allennlp.data.token_indexers import PosTagIndexer, TokenCharactersIndexer, NerTagIndexer, SingleIdTokenIndexer, PretrainedBertIndexer, PretrainedTransformerIndexer
from utix.pathex import solve_multi_path, get_files_by_pattern
from utix.strex import bisplit, split_by_cap_letters
import utix.msgex as msgex

_AVAILABLE_INDEXERS = None

DIR_ALLENNLP_CACHE = '_file_cache'
PATH_ALLENNLP_CACHE = path.join(allennlp.__path__[0], DIR_ALLENNLP_CACHE)


def ls_pretrained_transformers():
    pattern = '*-config.json'
    config_files = get_files_by_pattern(dir_or_dirs=PATH_ALLENNLP_CACHE, pattern=pattern)
    return [path.basename(x)[:-len(pattern) + 1] for x in config_files]


def get_pretrained_transformer_indexer(model_name: str = 'bert-base-uncased', cache_dir: str = PATH_ALLENNLP_CACHE, lowercase=True, **kwargs):
    return PretrainedTransformerIndexer(model_name=model_name, cache_dir=cache_dir, do_lowercase=lowercase, **kwargs)


def get_pretrained_transformer(model_name: str = 'bert-base-uncased', cache_dir: str = PATH_ALLENNLP_CACHE, *model_args, **kwargs):
    try:
        model = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name, *model_args, cache_dir=cache_dir, **kwargs)
    except Exception as err:
        transformer_names = ls_pretrained_transformers()
        if model_name not in transformer_names:
            raise ValueError("the specified transformer name is not in the model cache; cached models include " + str(transformer_names))
        else:
            raise err
    return model, model.config.hidden_size


def get_pretrained_bert_indexer(model_name: str = 'bert-base-uncased', cache_dir: str = PATH_ALLENNLP_CACHE, max_length=512, lowercase=True, **kwargs):
    model_path = path.join(cache_dir, 'bert', f'{model_name}-vocab.txt')
    msgex.assert_path_exist(path_str=model_path, arg_name='model_path', extra_msg=f"the specified BERT model '{model_name}' is not found")
    return PretrainedBertIndexer(
        pretrained_model=model_path,
        max_pieces=max_length,
        do_lowercase=lowercase, **kwargs)


def get_pretrained_bert(model_name: str = 'bert-base-uncased', cache_dir: str = PATH_ALLENNLP_CACHE, top_layer_only=True, requires_grad=True, **kwargs):
    model_path = path.join(cache_dir, 'bert', f'{model_name}.tar.gz')
    msgex.assert_path_exist(path_str=model_path, arg_name='model_path', extra_msg=f"the specified BERT model '{model_name}' is not found")
    model = PretrainedBertEmbedder(
        pretrained_model=model_path,
        top_layer_only=top_layer_only,
        requires_grad=requires_grad,
        **kwargs)
    return model, model.output_dim


def get_sorting_key(field_name: str, is_field_list: bool, indexer_key: str, sort_key: str):
    return field_name, f'list_{indexer_key}_{sort_key}' if is_field_list else f'{indexer_key}_{sort_key}'


def check_sorting_keys_against_indexers(sorting_keys: Union[Tuple[Tuple[str, str]], List[Tuple[str, str]]], indexers: dict):
    for field_name, sort_key in sorting_keys:
        ori_sort_key = sort_key
        if sort_key.startswith('list_'):
            sort_key: str = sort_key[5:]
        if sort_key.startswith('num_'):
            continue
        check_passed = False
        for indexer_key in indexers:
            if sort_key.startswith(indexer_key):
                check_passed = True
                break
        if not check_passed:
            warnings.warn(f"The sort key `{ori_sort_key}` might not be valid since no indexer key is found as part of it."
                          "An indexer-dependent AllenNLP sort key is typically of format `[list]_$indexerkey_length`, where 'list' is specified when the field is a list field.")


def get_indexer(indexer_name: str, index_name_model_name_sep='/'):
    global _AVAILABLE_INDEXERS
    if not _AVAILABLE_INDEXERS:
        _AVAILABLE_INDEXERS = {'_'.join(split_by_cap_letters(member_name, skip_consecutive_cap_letters=True)).lower(): member_name for member_name in dir(indexers) if not member_name.startswith('__') and member_name.endswith('Indexer')}
    indexer_name, model_name = bisplit(indexer_name, index_name_model_name_sep)
    indexer_name = indexer_name.lower()

    for indexer_full_name in _AVAILABLE_INDEXERS:
        if indexer_name in indexer_full_name:
            indexer = getattr(indexers, _AVAILABLE_INDEXERS[indexer_full_name])
            return partial(indexer, model_name=model_name) if model_name else indexer


def solve_indexers(indexers_obj, index_name_model_name_sep='/',
                   default_indexer_key_map=None,
                   *args, **kwargs) -> dict:
    if indexers_obj is None:
        return indexers_obj

    if default_indexer_key_map is None:
        default_indexer_key_map = {
            PosTagIndexer: 'pos_tag',
            TokenCharactersIndexer: 'characters',
            NerTagIndexer: 'ner_tag',
            'others': 'tokens'
        }

    def _solve_single_indexer(indexer_obj):
        if isinstance(indexer_obj, str):
            return get_indexer(indexer_obj, index_name_model_name_sep)(*args, **kwargs)
        elif callable(indexer_obj):
            return indexer_obj(*args, **kwargs)
        else:
            return indexer_obj

    if isinstance(indexers_obj, dict):
        for indexer_key in list(indexers_obj.keys()):
            indexers_obj[indexer_key] = _solve_single_indexer(indexers_obj[indexer_key])
        return indexers_obj
    elif not isinstance(indexers_obj, str) and iterable(indexers_obj):
        indexers_dict = {}
        for indexer in indexers_obj:
            indexers_dict[default_indexer_key_map.get(type(indexer), default_indexer_key_map['others'])] = _solve_single_indexer(indexer)
            return indexers_dict
    else:
        indexer = _solve_single_indexer(indexers_obj)
        return {default_indexer_key_map.get(type(indexer), default_indexer_key_map['others']): indexer}


def get_tokenizer_from_indexers(indexers: dict, return_first=False):
    tokenizers = {}
    for indexer_key, indexer in indexers.items():
        if hasattr(indexer, 'tokenizer'):
            if return_first:
                return getattr(indexer, 'tokenizer')
            tokenizers[indexer_key] = getattr(indexer, 'tokenizer')
    return tokenizers


def read_multi_path_as_multiple_iters(reader: DatasetReader, multi_path_str: Union[str, Iterator[str]], file_pattern: str):
    input_paths, path_exists, has_available_path = solve_multi_path(multi_path_str, file_pattern=file_pattern)
    if has_available_path:
        return [(reader.read(input_path), input_path) for input_path, path_exist in zip(input_paths, path_exists) if path_exist]


# class embedding_extraction:
#     def __init__(self, model):
#         self._model = model
#
#     def __enter__(self):
#         self._model.


def whitespace_tokenizer(text: str):
    return text.split()


@property
def default_indexers() -> dict:
    return {'tokens': SingleIdTokenIndexer()}


def get_metric_val_dict(metrics, reset: bool = False):
    d = {}
    for name, metric in metrics.items():
        value = metric.get_metric(reset)
        if can_dict_like_read(value):
            for name2, value2 in value.items():
                if name2.startswith(name):
                    d[name2] = value2
                else:
                    d['{}-{}'.format(name, name2)] = value2
        elif isinstance(value, tuple):
            for i, value2 in enumerate(value):
                d['{}-{}'.format(name, i)] = value2
        else:
            d[name] = value
    return d

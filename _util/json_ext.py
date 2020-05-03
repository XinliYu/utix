import json
from json import JSONEncoder
from typing import Callable

import numpy as np

from utilx.general import import__, has_varkw
from utilx.ioex import write_all_lines, write_all_lines_to_stream, write_all_json_objs, read_all_json_objs

_register = {}


class JSerializationInfo(dict):
    def register(self, type_to_register: type, type_str: str, to_dict: Callable, from_dict: Callable = None):
        self[type_str] = (type_str, type_to_register, to_dict, from_dict)
        self[type_to_register] = self[type_str]

    def encode(self, o):
        t = type(o)
        if t in self:
            return self._encode(o, t)

    def _encode(self, o, t):
        info = self[t]
        d = {'__type__': info[0]}
        d.update(self[2](o))
        return d

    def decode(self, d: dict):
        c = d.get('__type__', None)
        if c and c in self:
            return self._decode(d, c)

    def _decode(self, d: dict, c: str):
        info = self[c]
        if info:
            c = info[3] or info[1]
            return c(**d) if has_varkw(c) else c(d)


def to_dict(o, js_info: JSerializationInfo = None) -> dict:
    t = type(o)
    if js_info and t in js_info:
        return js_info._encode(o, t)
    elif type(o) is np.ndarray:
        return {'__type__': 'np', 'data': o.tolist(), 'dtype': str(o.dtype)}
    elif hasattr(o, '__to_dict__'):
        return o.__to_dict__()


def from_dict(d: dict, import_func: Callable = import__, js_info: JSerializationInfo = None):
    c = d.get('__type__', None)
    if js_info and c in js_info:
        return js_info._decode(d, c)
    elif c == 'np':
        return np.array(d['data'], dtype=d['dtype'])
    else:
        c = import_func(c)
        if hasattr(c, '__from_dict__'):
            return c.__from_dict__(d)
        else:
            return c(**d) if has_varkw(c) else c(d)


def json_encode_ex(o):
    return to_dict(o) or JSONEncoder.default(None, o)


def json_decode_ex(o):
    return from_dict(o) if '__type__' in o else o


def dumps(obj, indent: int = None, **kwargs):
    return json.dumps(obj, indent=indent, default=json_encode_ex, **kwargs)


def dump(obj, file, indent: int = None, **kwargs):
    if isinstance(file, str):
        with open(file, 'w') as f:
            json.dump(obj, fp=f, indent=indent, default=json_encode_ex, **kwargs)
    else:
        json.dump(obj, fp=file, indent=indent, default=json_encode_ex, **kwargs)


def dumps_iter(kv_pair_iter, indent: int = None, **kwargs):
    for k, v in kv_pair_iter:
        yield json.dumps({k: v}, indent=indent, default=json_encode_ex, **kwargs)


def dump_iter(kv_pair_iter, file, indent: int = None, use_tqdm=False, tqdm_msg=None, append=False, **kwargs):
    it = dumps_iter(kv_pair_iter, indent=indent, **kwargs)
    if isinstance(file, str):
        write_all_lines(it, output_path=file, use_tqdm=use_tqdm, tqdm_msg=tqdm_msg, append=append)
    else:
        write_all_lines_to_stream(wf=file, iterable=it, use_tqdm=use_tqdm, tqdm_msg=tqdm_msg)


def loads(s: str, **kwargs):
    return json.loads(s, object_hook=json_decode_ex, **kwargs)


def load(file, **kwargs):
    if isinstance(file, str):
        with open(file, 'wr') as f:
            json.load(fp=f, object_hook=json_decode_ex, **kwargs)
    else:
        json.load(fp=file, object_hook=json_decode_ex, **kwargs)

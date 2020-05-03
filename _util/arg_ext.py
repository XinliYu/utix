import argparse
import ast
import inspect
import itertools
import json
import re
import sys as _sys
from argparse import _get_action_name, ArgumentError, SUPPRESS, Namespace, _UNRECOGNIZED_ARGS_ATTR
from collections import namedtuple
from copy import copy
from functools import partial
from gettext import gettext as _
from os import path
from pydoc import locate
from sys import argv
from typing import Tuple, Union, List, Callable, Dict, Any, Iterator

from numpy import iterable

import utilx.strex as strex
from utilx.dictex import tup2dict
from utilx.general import is_str, value_type, nonstr_iterable
from utilx.msgex import ensure_arg_not_empty, ensure_arg_not_none, ensure_key_exist, ensure_valid_python_name

"""
Utility functions related to argument parsing, object checking and object type conversion.
"""


# region json obj initialization

def get_obj_from_json_dict(jobj: dict, vars: dict = None, base_callable: callable = None, always_try_python_name_parse: bool = True, args=None):
    kwargs = {}

    for k, v in jobj.items():
        if isinstance(v, str) and v.isidentifier():
            v = getattr(args, v, vars.get(v, kwargs.get(v, v)) if vars else kwargs.get(v, v))

        k_splits = k.split(':')
        if len(k_splits) == 1:
            kwargs[k] = v
        elif len(k_splits) == 2:
            k, k_type = k_splits
            ensure_valid_python_name(k_type, extra_msg="the type of a field must be a valid Python identifier")
            ensure_arg_not_empty(arg_val=k_type, arg_name='k_type', extra_msg=f'the type name for field `{k}` is empty')
            callable_obj = vars.get(k_type, default=locate(k_type) if always_try_python_name_parse else None) if vars is not None \
                else (locate(k_type) if always_try_python_name_parse else None)
            ensure_arg_not_none(arg_val=k_type, arg_name='k_type', extra_msg=f"the type `{k_type}` for field `{k}` cannot be parsed as a Python callable")
            if isinstance(v, dict):
                kwargs[k] = get_obj_from_json_dict(jobj=v, vars=vars, base_callable=callable_obj)
            else:
                if not isinstance(v, str) and iterable(v) and any((x.kind == inspect.Parameter.VAR_POSITIONAL for x in inspect.signature(callable_obj).parameters.values())):
                    kwargs[k] = callable_obj(*v)
                else:
                    kwargs[k] = callable_obj(v)

    return base_callable(**kwargs) if base_callable is not None else kwargs


def get_obj_from_json_str(jstr: str, vars: dict = None, base_callable: callable = None, always_try_python_name_parse: bool = True, args=None):
    return get_obj_from_json_dict(jobj=json.loads(jstr), vars=vars, base_callable=base_callable, always_try_python_name_parse=always_try_python_name_parse, args=args)


def get_obj_from_json_file(file_path: str, base_callable: callable = None, always_try_python_name_parse: bool = True, multi_objs=False, jkey_vars='vars', jkey_args='args', jkey_modes='mode'):
    def _get_args():
        nonlocal args
        if args:
            args = [[k] + v for k, v in args.items()]
            args = get_parsed_args(*args, preset=vars)

    def _get_vars():
        nonlocal vars
        if vars:
            for k, v in vars.items():
                if isinstance(v, str):
                    try:
                        v = locate(v)
                    except:
                        pass
                vars[k] = v

    def _get_mode():
        nonlocal args, vars
        if hasattr(args, 'mode'):
            ensure_key_exist(key=args.mode, d=modes, dict_name=jkey_modes)
            mode = modes[args.mode]
            if jkey_vars in mode:
                vars.update(get_obj_from_json_dict(jobj=mode[jkey_vars], always_try_python_name_parse=always_try_python_name_parse))
                del mode[jkey_vars]
            if jkey_args in mode:
                args.update(get_obj_from_json_dict(jobj=mode[jkey_args], vars=vars, always_try_python_name_parse=always_try_python_name_parse))
                del mode[jkey_args]
            return mode

    with open(file_path) as fin:
        jobj: dict = json.loads(next(fin)) if multi_objs else json.load(fin)
        args: Union[dict, None] = jobj.get('args', None)
        vars: Union[dict, None] = jobj.get('vars', None)
        modes: Union[dict, None] = jobj.get('modes', None)
        _get_vars()
        _get_args()
        mode = _get_mode()

        if multi_objs:
            if mode:
                def _update(jobj):
                    jobj.upate(mode)
                    return jobj

                return [get_obj_from_json_dict(jobj=_update(json.loads(line)),
                                               vars=vars,
                                               base_callable=base_callable,
                                               always_try_python_name_parse=always_try_python_name_parse,
                                               args=args) for line in fin]
            else:
                return [get_obj_from_json_dict(jobj=json.loads(line),
                                               vars=vars,
                                               base_callable=base_callable,
                                               always_try_python_name_parse=always_try_python_name_parse,
                                               args=args) for line in fin]
        else:
            del jobj['args']
            del jobj['vars']
            if mode:
                jobj.update()
            return get_obj_from_json_dict(jobj=jobj,
                                          vars=vars,
                                          base_callable=base_callable,
                                          always_try_python_name_parse=always_try_python_name_parse,
                                          args=args)


# endregion

# region object checking


def get_args_from_callable(f: Callable, exclude_optional_args=False) -> Tuple[List, str, str]:
    args, star_arg_name, dstar_arg_name = [], None, None
    for x, p in inspect.signature(f).parameters.items():
        if p.kind == inspect.Parameter.VAR_POSITIONAL:
            star_arg_name = x
        elif p.kind == inspect.Parameter.VAR_KEYWORD:
            dstar_arg_name = x
        elif not exclude_optional_args or p.default == inspect.Parameter.empty:
            args.append(x)
    return args, star_arg_name, dstar_arg_name


def num_args_from_callable(f: Callable, exclude_optional_args=False) -> Tuple[int, str, str]:
    """
    Gets the number of arguments in the
    :param f:
    :param exclude_optional_args:
    :return:
    """
    num_args, star_arg_name, dstar_arg_name = 0, None, None
    for x, p in inspect.signature(f).parameters.items():
        if p.kind == inspect.Parameter.VAR_POSITIONAL:
            star_arg_name = x
        elif p.kind == inspect.Parameter.VAR_KEYWORD:
            dstar_arg_name = x
        elif not exclude_optional_args or p.default == inspect.Parameter.empty:
            num_args += 1
    return num_args, star_arg_name, dstar_arg_name


def is_arg_none(arg_val: str):
    return (not arg_val) or (arg_val.lower() == 'none')


# endregion

# region fast object init

def fast_init_obj(init_str: str, obj_dict: Dict[str, Callable]):
    # TODO

    obj_key = strex.first_startswith(init_str, obj_dict.keys())
    obj_key_len = len(obj_key)
    f = obj_dict[obj_key]
    if obj_key_len == len(init_str):
        return f

    arg_suffix = init_str[obj_key_len + 1:]
    if not arg_suffix:
        return f

    get_args_from_callable(f=f)


# endregion

# region common arg parsing


def solve_name_conflict(name: str, current_names: set, suffix_sep='', name_suffix_gen: Iterator = None):
    """
    Solves name conflict by automatically appending a suffix.

    >>> import utilx.argex as argx
    >>> print(argx.solve_name_conflict(name='para', current_names={'para', 'para1', 'para2', 'para3'}) == 'para4')
    >>> def suffix_gen():
    >>>     for x in range(1, 5):
    >>>         yield 'i' * x
    >>> print(argx.solve_name_conflict(name='para', current_names={'para', 'para_i', 'para_ii', 'para_iii'}, suffix_sep='_', name_suffix_gen=suffix_gen()) == 'para_iiii')

    :param name: the name to solve conflict.
    :param current_names: the set of all current names.
    :param suffix_sep: the separator to insert between the name and the suffix.
    :param name_suffix_gen: optionally provides generator of name suffixes; if this is not provided, the suffix will be numbers starting at 1.
    :return: the solved name with a possible suffix to avoid name conflict.
    """
    if name_suffix_gen is None:
        name_dd_idx = 1
        _name = name
        while name in current_names:
            name = _name + suffix_sep + str(name_dd_idx)
            name_dd_idx += 1
        current_names.add(name)
    else:
        _name = name
        name_suffix_gen = iter(name_suffix_gen)
        while name in current_names:
            name = _name + suffix_sep + str(next(name_suffix_gen))
        current_names.add(name)
    return name


def get_short_name(full_name: str, current_short_names: set = None, name_parts_sep: str = r'_|\-', short_name_suffix_sep='', short_name_suffix_gen: Iterator = None):
    if full_name:
        try:
            short_name = ''.join(part[0] for part in re.split(name_parts_sep, full_name))
        except:
            pass
        return solve_name_conflict(name=short_name, current_names=current_short_names, suffix_sep=short_name_suffix_sep, name_suffix_gen=short_name_suffix_gen)
    else:
        return ''


def first_not_none(*args):
    for arg in args:
        if arg is not None:
            return arg


def update_args(args, **kwargs):
    args = copy(args)
    if len(kwargs) == 0:
        return args
    for k, v in kwargs.items():
        if v is not None:
            setattr(args, k, v)
    return args


def get_args_or_default(*default_args):
    if len(argv) == 1:
        return default_args
    else:
        input_args = argv[1:]
        for i in range(len(input_args)):
            this_arg = input_args[i]
            arg_type = type(default_args[i])
            if arg_type in (float, int, bool):
                input_args[i] = arg_type(this_arg)
            elif arg_type in (tuple, list, set):
                if len(default_args[i]) == 0:
                    input_args[i] = [s.strip() for s in this_arg.split()]
                else:
                    arg_type2 = type(default_args[i][0])
                    input_args[i] = [arg_type2(s.strip()) for s in this_arg.split()]
        return tuple(input_args) + default_args[len(input_args):]


def iter_named_arg_combo(*args, **kwargs):
    combo_arg_vals = []
    for combo_arg_name in args:
        arg_val = kwargs.get(combo_arg_name)
        if type(arg_val) in (tuple, list, set):
            combo_arg_vals.append(arg_val)
        else:
            combo_arg_vals.append((arg_val,))

    for combo_vals in itertools.product(*combo_arg_vals):
        this_kwargs = kwargs.copy()
        for arg_idx, combo_arg_name in enumerate(args):
            this_kwargs[combo_arg_name] = combo_vals[arg_idx]
        yield this_kwargs


def apply_arg_combo(method: Callable, *args, unpack_for_single_result=False, return_first_only=False, **kwargs):
    output = []
    args = list(args)
    for arg_idx, arg in enumerate(args):
        if type(arg) not in (list, tuple):
            args[arg_idx] = (arg,)

    for arg_key, arg in kwargs.items():
        if type(arg) not in (list, tuple):
            kwargs[arg_key] = (arg,)

    for a1 in itertools.product(*args):
        for a2 in itertools.product(*kwargs.values()):
            this_arg = method(*a1, **dict(zip(kwargs, a2)))
            if return_first_only:
                return this_arg
            output.append(this_arg)

    if unpack_for_single_result and len(output) == 1:
        return output[0]
    return output


def get_obj_from_args(obj_args: Union[Callable, Tuple]):
    """
    Initialize an object from the provided `obj_args`.
    :param obj_args: provides a callable and its arguments, from which an object can be constructed.
            If `obj_args` is itself a callable, then `obj_args()` is returned.
            If `obj_args` is a tuple of length 1, then `obj_args[0]()` is returned.
            If `obj_args` is a tuple of length 2, then `obj_args[1]` must also be a tuple in which the items are treated as positional arguments, and `obj_args[0](*obj_args[1])` will be returned.
            If `obj_args` is a tuple of length 3, then `obj_args[1]` must also be a tuple in which the items are treated as positional arguments, `obj_args[2]` must be a dictionary which is treated as named arguments, and `obj_args[0](*obj_args[1], **obj_args[2])` will be returned.
            In any case of `obj_args` being a tuple, `obj_args[0]` must be a callable, e.g. a class or a function.
    :return: an object constructed from the provided `obj_args`.
    """
    if isinstance(obj_args, tuple):
        if not callable(obj_args[0]):
            raise ValueError("The first of the object argument tuple `obj_args[0]` must be a callable, e.g. a class or a function.")
        obj_args_len = len(obj_args)
        if obj_args_len == 2:
            return obj_args[0](**obj_args[1])
        elif obj_args_len == 3:
            return obj_args[0](*obj_args[1], **obj_args[2])
        elif obj_args_len == 1:
            return obj_args[0]()
        else:
            raise ValueError("The length of the object argument tuple `obj_args` should be 1, 2, or 3.")
    elif callable(obj_args):
        return obj_args()
    else:
        raise ValueError("The provided object argument `obj_args` is not supported.")


ArgInfo = namedtuple('ArgInfo', ('full_name', 'short_name', 'default_value', 'description', 'converter'), defaults=('', '', None, '', None))
ArgInfo.__doc__ = "Namedtuple for argument definition. Used for argument definition in `get_parsed_args`. "


def get_parsed_args(*arg_info_objs, preset_root: str = None, preset: [Union[Dict[str, Any], str]] = None, short_full_name_sep='/', return_seen_args=False, default_value_prefix: str = 'default_', **kwargs):
    """
    Parses terminal argument input. Suitable for both simple and complicated argument parsing. Also supports list and dictionary parsing.

    Simple argument parsing setup.
    ------------------------------
    >>> import utilx.argex as argx
    >>> # by simply specifying the default values, it tells the function there should be three terminal arguments `para1`, `para2` and `para3`,
    >>> # and it hints the function that `para1` is of type `int`, `para2` is of type `str`, and `para3` is of type `list`
    >>> args = argx.get_parsed_args(default_para1=1, default_para2='value', default_para3=[1, 2, 3, 4])
    >>> # 1) without any argument, this will print out "1 value [1, 2, 3, 4]";
    >>> # 2) set arguments `--para1 2 --para2 3 --para3 '[4,5,6,7]'` and this will print out "2 3 [4, 5, 6, 7]", where the '3' is of string type;
    >>> # 3) set arguments `--para1 2 --para2 3 --para3 5`  and this will print out "2 3 [5]", where the '3' is of string type, and the '5' is turned into a list;
    >>> # 4) the short names for these arguments are automatically generated, and they are 'p', 'p1', 'p2', try `-p 2 -p1 3 -p2 5`; we'll see more about short names later.
    >>> print(args.para1, args.para2, args.para3)
    >>> print(type(args.para1), type(args.para2), type(args.para3))

    Simple argument parsing setup without default values (not recommended).
    -----------------------------------------------------------------------
    >>> import utilx.argex as argx
    >>> # if no default values are needed, we could just specify the names;
    >>> # NOTE that without default values, there is no way to infer the type of each argument, unless it can be recognized as list, a tuple, a set or a dictionary;
    >>> # all other arguments will be of string type;
    >>> args = argx.get_parsed_args('para1', 'para2', 'para3')
    >>> # 1) without any argument, this will print out an empty line, and the types are '<class 'str'> <class 'str'> <class 'str'>';
    >>> # 2) try arguments `--para1 2 --para2 3 --para3 '[4,5,6,7]'` and this will print out "2 3 [4, 5, 6, 7]", with types '<class 'int'> <class 'str'> <class 'list'>'.
    >>> # 3) try arguments `--para1 2 --para2 3 --para3 5` and `-p 2 --p1 3 --p2 5`, and it will print out "2 3 5" with types '<class 'str'> <class 'str'> <class 'str'>'.
    >>> print(args.para1, args.para2, args.para3)
    >>> print(type(args.para1), type(args.para2), type(args.para3))

    Use 2-tuples to setup argument parsing.
    ---------------------------------------
    >>> import utilx.argex as argx
    >>> # we can provide argument info tuples;
    >>> # here every tuple is a 2-tuple, 1) the first being the name in the format of `fullname/shortname`, or just the `fullname`, and 2) the second being the default value;
    >>> # NOTE if the 'shortname' is not specified, the default is to use the first letter of the 'parts' of the full name as the short name.
    >>> # NOTE if the duplicate short name is found, an incremental number will be appended to the end to solve the name conflict.
    >>> args = argx.get_parsed_args(('para1_is_int/p', 1),('para2_is_str/p', 'value'),('para3_is_list', [1, 2, 3, 4]))  # short name not specified, and the default short name is `pil` by connecting the first letter of 'parts' of the full name

    >>> # 1) without any argument, this will print out "1 value [1, 2, 3, 4]";
    >>> # 2) set arguments by short names `-p 2 -p1 3 -pil '[4,5,6,7]'` and this will print out "2 3 [4, 5, 6, 7]", where the '3' is of string type;
    >>> # 3) set arguments `--para1_is_int 2 --para2_is_str 3 -para3_is_list '[4,5,6,7]'` and this will print out "2 3 [4, 5, 6, 7]", where the '3' is of string type,
    >>> print(args.para1_is_int, args.para2_is_str, args.para3_is_list)
    >>> print(type(args.para1_is_int), type(args.para2_is_str), type(args.para3_is_list))
    
    Use more explicit ArgInfo namedtuple to setup argument parsing.
    --------------------------------------------------------
    >>> import utilx.argex as argx
    >>> args = argx.get_parsed_args(argx.ArgInfo(full_name='para1_is_int', short_name='p', default_value=1), 
    >>>                             argx.ArgInfo(full_name='para2_is_str', short_name='p', default_value='value'),
    >>>                             argx.ArgInfo(full_name='para3_is_list', default_value=[1, 2, 3, 4]))
    >>> 
    >>> # try `-p 2 -p1 3 -pil '[4,5,6,7]'` and `--para1_is_int 2 --para2_is_str 3 --para3_is_list '[4,5,6,7]'` again, and it should print out '2 3 [4, 5, 6, 7]'
    >>> # try `-p 2 -p1 3 -pil 5`
    >>> print(args.para1_is_int, args.para2_is_str, args.para3_is_list)
    >>> print(type(args.para1_is_int), type(args.para2_is_str), type(args.para3_is_list))


    Use converters.
    ---------------
    >>> import utilx.argex as argx
    >>> args = argx.get_parsed_args(argx.ArgInfo(full_name='para1_is_int', short_name='p', default_value=1),
    >>>                             argx.ArgInfo(full_name='para2_is_str', short_name='p', default_value='value', converter=lambda x: '_' + x.upper()),
    >>>                             argx.ArgInfo(full_name='para3_is_list', default_value=[1, 2, 3, 4], converter=lambda x: x ** 2),
    >>>                             argx.ArgInfo(full_name='para4_is_dict', default_value={'a': 1, 'b': 2}, converter=lambda k, v: (k, k + str(v))))
    >>> # 1) without any argument, this will print out "1 _VALUE [1, 4, 9, 16] {'a': 'a1', 'b': 'b2'}";
    >>> # 2) try `-p 2 -p1 3 -pil '[4,5,6,7]' -pid "{'a':2, 'b':3}"` and `--para1_is_int 2 --para2_is_str 3 --para3_is_list '[4,5,6,7]' --para3_is_dict "{'a':2, 'b':3}"` again, 
    >>> #       and it should print out '2 _3 [16, 25, 36, 49] {'a': 'a2', 'b': 'b3'}'
    >>> print(args.para1_is_int, args.para2_is_str, args.para3_is_list, args.para4_is_dict)
    >>> print(type(args.para1_is_int), type(args.para2_is_str), type(args.para3_is_list), type(args.para4_is_dict))

    :param arg_info_objs: argument definition objects; define the argument name, default value, description, and value conversion. It can be
                            1) just the argument name;
                            you can specify both the full name and the short name in a single name string with the `short_full_name_sep` as the separator;
                            for example, by default `short_full_name_sep` is '/', then you can specify an argument name as 'para_1/p1' or 'learning_rate/lr';
                            2) a 2-tuple for the argument name (supports short name specification) and the default value;
                            3) a 3-tuple for the argument name (supports short name specification), the default value, and a description;
                            4) a 3-tuple for the argument name (supports short name specification), the default value, and a converter;
                            5) a 4-tuple for the argument name (supports short name specification), the default value, the description and the converter;
                            6) a 5-tuple for the full argument name, the short argument name, the default value, the description and the converter;
                            7) a :class:`ArgInfo` object, which is itself a :class:`namedtuple` of size 5.
                            NOTE the argument type is inferred from the default value.
                            1) If the default value is list, tuple, or set, the inferred argument type is the same container, and it will enforce the element type by the type of the first element in the default value;
                                for example, if the default value is `[1, "2", "3", "4"]`, the inferred type is a list of integers, regardless of the string values in the list;
                                if then the terminal input is `["1", "2", "3", "4"]`, it will still be recognized as a list of integers.
                            2) Otherwise, the inferred type is just the the type of the default value.
                            3) to change the above typing inference behavior, provide a `converter` for the argument in the `arg_info_objs`.
    :param preset_root: the path to the directory that stores presets of arguments.
    :param preset: the path/name of the preset relative to `preset_root`; a preset is a json file with predefined argument values; if `preset_root` is specified, then `preset` should be relative to the `preset_root`.
    :param short_full_name_sep: optional; the separator used to separate fullname and shortname in the `arg_info_objs`; the default is '/'.
    :param return_seen_args: optional; `True` to return the names of the arguments actually specified in the terminal; otherwise `False`.
    :param default_value_prefix: any named argument starting with this prefix will be treated as the default value for an argument of the same name without the prefix; the default is 'default_';
                                    for example, `default_learning_rate` indicates there is an argument named `learning_rate`, and the value of `default_learning_rate` is the default value for that argument;
                                    the argument need not be already specified in the `arg_info_objs`;
                                    for example, with 'default_' as the `default_value_prefix`, this function automatically adds the 'xxx' of any such parameter `default_xxx` found in `kwargs` to the recognized arguments.
    :param kwargs: specify the default values as named arguments.
    :return: just the parsed arguments if `return_seen_args` is `False`; otherwise, a tuple, the first being the parsed args, and the second being
    """
    arg_parser = ArgumentParser()
    arg_full_name_dd = set()
    arg_short_name_dd = set()
    converters = {}

    if type(preset) is str:
        if preset_root:
            preset = path.join(preset_root, preset)
        assert path.exists(preset), f"The argument preset file does not exits at {preset}."
        preset = json.load(open(preset))

    def _solve_arg_names(arg_name):
        arg_name, arg_short_name = strex.birsplit(arg_name, short_full_name_sep)
        if arg_short_name:
            arg_short_name = solve_name_conflict(name=arg_short_name, current_names=arg_short_name_dd)
            return (arg_name, arg_short_name) if arg_name not in arg_full_name_dd else (None, None)
        else:
            # automatically generates the short argument name
            return (arg_name, get_short_name(full_name=arg_name, current_short_names=arg_short_name_dd)) \
                if arg_name not in arg_full_name_dd \
                else (None, None)

    def _default_converter_multiple_values(x, ctype, vtype, converter):
        if converter is None:
            if nonstr_iterable(x):
                return ctype(vtype(xx) for xx in x)
            else:
                return ctype([vtype(x)])
        else:
            if nonstr_iterable(x):
                return ctype(converter(xx) for xx in x)
            else:
                return ctype([converter(x)])

    def _add_arg():
        if converter is not None and callable(converter):
            # if the converter is specified, then we leaves the argument string parsing to the converter
            arg_parser.add_argument('-' + arg_short_name, '--' + arg_name, help=description if description else '', default=default_value, type=str)
            if isinstance(default_value, (tuple, list, set)):
                converters[arg_name] = partial(_default_converter_multiple_values, ctype=type(default_value), vtype=value_type(default_value), converter=converter)
        else:
            # otherwise, we run the default argument parsing
            arg_value_type = type(default_value)
            if arg_value_type is bool and not default_value:
                arg_parser.add_argument('-' + arg_short_name, '--' + arg_name, help=description if description else '', required=False, action='store_true')
            else:
                if isinstance(default_value, (int, float, bool)):
                    converters[arg_name] = type(default_value)
                elif isinstance(default_value, (tuple, list, set)):
                    converters[arg_name] = partial(_default_converter_multiple_values, ctype=type(default_value), vtype=value_type(default_value), converter=None)
                arg_parser.add_argument('-' + arg_short_name, '--' + arg_name, help=description if description else '', default=default_value)
        arg_full_name_dd.add(arg_name)

    # region process argument definition tuples
    converter = None
    for arg_info_obj in arg_info_objs:
        if isinstance(arg_info_obj, str):
            arg_name: str = arg_info_obj
            default_value = description = ''
            converter = None
            arg_name, arg_short_name = _solve_arg_names(arg_name)
        elif isinstance(arg_info_obj, tuple):
            if len(arg_info_obj) == 5:  # the case when the _short name_ is separately specified.
                arg_name, arg_short_name, default_value, description, converter = arg_info_obj
                if not arg_short_name:
                    arg_name, arg_short_name = _solve_arg_names(arg_name)
                else:
                    arg_short_name = solve_name_conflict(name=arg_short_name, current_names=arg_short_name_dd)
                if converter is not None:
                    converters[arg_name] = converter
            else:
                if len(arg_info_obj) == 2:
                    # if the tuple has two elements, then it is parsed as the _argument name_ and the _default value_
                    arg_name, default_value = arg_info_obj
                    description = ''
                    converter = None
                elif len(arg_info_obj) == 3:
                    # if the tuple has three elements, then
                    #   1) if the last element is a string, then the tuple is parsed as the _argument name_, the _default value_ and the _argument description_,
                    #       and the argument value will be converted to a list derived from separating the argument value;
                    #   2) otherwise, the last element is parsed as parsed as the _argument name_, the _default value_ and the _argument converter_,
                    #       where the converter transforms the argument string to a desried argument value.
                    if isinstance(arg_info_obj[2], str):
                        arg_name, default_value, description = arg_info_obj
                        try:
                            converter = locate(description)
                            description = ''
                        except:
                            converter = None
                    else:
                        arg_name, default_value, converter = arg_info_obj
                        description = ''
                    if converter is not None:
                        converters[arg_name] = converter
                elif len(arg_info_obj) == 4:
                    # if the tuple has four elements, then the tuple is parsed as the _argument name_, the _default value_, the _argument description_, and the _argument converter_.
                    arg_name, default_value, description, converter = arg_info_obj
                    converters[arg_name] = locate(converter) if isinstance(converter, str) else converter
                else:
                    raise ValueError("Needs at least the argument name and the default value.")

                arg_name, arg_short_name = _solve_arg_names(arg_name)  # NOTE: re-definition of argument will be ignored
        else:
            raise TypeError(f'an argument info object should be a string or tuple; got {type(arg_info_obj)}')
        if arg_name:
            # default value overrides 1 - from the extra named arguments `kwargs`
            default_value_override = kwargs.get(arg_name, kwargs.get(default_value_prefix + arg_name, None))
            if default_value_override is not None:
                default_value = default_value_override

            # default value overrides 2 - from a preset dictionary; this has the highest priority
            if preset is not None:
                default_value_override = preset.get(arg_name, preset.get(default_value_prefix + arg_name, None))
                if default_value_override is not None:
                    default_value = default_value_override
            _add_arg()
    # endregion

    # region adds ad-hoc defined arguments
    description = ''
    if preset is not None:
        for arg_name, default_value in preset.items():
            if arg_name.startswith(default_value_prefix):
                arg_name = arg_name[len(default_value_prefix):]
            arg_name, arg_short_name = _solve_arg_names(arg_name)
            if arg_name:
                if default_value is None:
                    default_value = kwargs.get(arg_name, kwargs.get(default_value_prefix + arg_name, None))
                _add_arg()

    for arg_name, default_value in kwargs.items():
        if arg_name.startswith(default_value_prefix):
            arg_name = arg_name[len(default_value_prefix):]
            arg_name, arg_short_name = _solve_arg_names(arg_name)
            if arg_name:
                _add_arg()
    # endregion

    # region argument value conversion
    args, seen_actions = arg_parser.parse_args(argv[1:])
    for arg_name, arg_val in vars(args).items():
        if isinstance(arg_val, str):
            arg_val = arg_val.strip()
            if arg_val and arg_val[0] in ('\'', '"') and arg_val[-1] in ('\'', '"'):  # 'de-quote' the argument string
                arg_val = arg_val[1:-1]
            if len(arg_val) >= 2 and ((arg_val[0] == '[' and arg_val[-1] == ']') or (arg_val[0] == '(' and arg_val[-1] == ')')):
                arg_val = ast.literal_eval(arg_val)
                converter = converters.get(arg_name, None)
                if converter is not None:
                    arg_val = converter(arg_val)
            elif len(arg_val) >= 2 and (arg_val[0] == '{' and arg_val[-1] == '}'):
                arg_val = ast.literal_eval(arg_val)
                converter = converters.get(arg_name, None)
                if converter is not None:
                    arg_val = tup2dict(converter(k, v) for k, v in arg_val.items())
            elif arg_name in converters:
                arg_val = converters[arg_name](arg_val)
        elif arg_name in converters:
            converter = converters.get(arg_name, None)
            if isinstance(arg_val, (list, set, tuple)):
                arg_val = converter(arg_val)
            elif isinstance(arg_val, dict):
                arg_val = tup2dict(converter(k, v) for k, v in arg_val.items())
            else:
                arg_val = converter(arg_val)
        setattr(args, arg_name, arg_val)

    # endregion
    if return_seen_args:
        return args, tuple(x.dest for x in seen_actions)
    else:
        return args


def args2str(args, active_arg_name_info_tuples: Union[Tuple, List], default_short_name_map: dict = None, default_value_formatter: Callable = None, name_val_delimiter='_', name_parts_delimiter='-',
             prefix: str = None, suffix: str = None, extension_name=None):
    """
    Generates a string representation for the given arguments. This string representation can be used in file names or field names for quick identification of the argument setup.
    :param args: the arguments.
    :param active_arg_name_info_tuples: provides what arguments should be included in the string representation, and provides short-name and value-formatting function for these arguments. Can be:
                                        1) a three-tuple of the 'full name', the 'short name' and the 'value formatter';
                                            the 'short name' for an argument will appear in the string representation;
                                            the 'value format' should be a callable;
                                            for example, if `('learning_rate', 'lr', lambda x: str(x*100000))` is provided in this argument,
                                            and `args.learning_rate` is `1e-4`, and `name_val_delimiter` is '-', then string for this argument is `lr-10`;
                                        2) a two-tuple of the 'full name' and the 'short name', or a two-tuple of the 'full name' and the 'value formatter';
                                        3) just the 'full name'.
                                        In the second case or the third case when either the 'short name' or the 'value formatter' is missing, the default will apply.
    :param default_short_name_map: an optional convenient dictionary that provides mapping from an argument full name to a short name;
                            if this parameter is provided, then the short named need not be provided in `active_arg_name_info_tuples`;
                            if a short name is still provided in `active_arg_name_info_tuples`, then it overrides the short name in this dictionary.
    :param default_value_formatter: an optional function that serves as the default value formatter; if this parameter is not set, the internal default value formatter will be used as the default.
    :param name_val_delimiter: the delimiter between the short name and the value.
    :param name_parts_delimiter: the delimiter between the string of different arguments.
    :param prefix: adds this prefix string to the argument string.
    :param suffix: adds this suffix string to the argument string.
    :param extension_name: add this extension name to the end of the final argument string, after the `suffix`; this is convenient when using this method to return a file name.
    :return: the argument string.
    """
    short_name_dd = set()

    def _default_value_format(val):
        if val is None:
            return ''
        val_type = type(val)
        if val_type is float:
            return strex.clean_int_str(int(val * 1000000))
        elif val_type is int:
            return strex.clean_int_str(val)
        elif val_type is bool:
            return int(val)
        elif val_type in (list, tuple):
            return '_'.join((str(_default_value_format(x)) for x in val))
        else:
            return get_short_name(full_name=str(val))

    def _get_short_name():
        return get_short_name(full_name=arg_full_name, current_short_names=short_name_dd) if default_short_name_map is None or arg_full_name not in default_short_name_map else default_short_name_map[arg_full_name]

    name_parts = [prefix] if prefix else []
    for info_tup in active_arg_name_info_tuples:
        if len(info_tup) == 1:
            info_tup = info_tup[0]

        if is_str(info_tup):
            arg_full_name = info_tup
            short_name = _get_short_name()
            val_formatter = _default_value_format if default_value_formatter is None else default_value_formatter
        elif len(info_tup) == 2:
            arg_full_name, short_name_or_val_format = info_tup
            if is_str(short_name_or_val_format):
                short_name = short_name_or_val_format
                val_formatter = _default_value_format if default_value_formatter is None else default_value_formatter
            else:
                short_name = _get_short_name()
                val_formatter = short_name_or_val_format
        elif len(info_tup) == 3:
            arg_full_name, short_name, val_formatter = info_tup
        else:
            raise ValueError(f"The argument info tuple '{info_tup}' is not recognized.")

        val = val_formatter(getattr(args, arg_full_name))
        if val is not None:
            val = str(val)
            if short_name:
                name_parts.append(f'{short_name}{name_val_delimiter}{val}' if val else short_name)
            elif val:
                name_parts.append(val)
            else:
                raise ValueError(f"For argument {arg_full_name}, both its short name and value is empty.")

    if suffix:
        name_parts.append(suffix)

    main_name = name_parts_delimiter.join(name_parts)
    if extension_name:
        if extension_name[0] != '.':
            extension_name = '.' + extension_name
        return main_name + extension_name
    else:
        return main_name


def tuple_arg_parser(arg_or_args, target_class, positional_arg_types: Union[Tuple, List]):
    """
    Parses the provided argument(s) as a tuple of objects of the specified type.
    :param arg_or_args: the arg or args to parse; their types must be consistent with the types of the positional arguments of the `target_class`.
    :param target_class: the target argument type.
    :param positional_arg_types: the types of the positional arguments of the `target_class`.
    :return: a tuple of parsed arguments, all of type `target_class`.
    """

    def _inner_parse(field_arg):
        if isinstance(field_arg, positional_arg_types[0]):
            return target_class(field_arg)
        elif type(field_arg) in (tuple, list):
            arg_idx = 0
            for arg, arg_type in zip(arg_or_args, positional_arg_types):
                if arg_type is not None and not isinstance(arg, arg_type):
                    raise ValueError(f"Argument at position {arg_idx} of value {arg} has type {type(arg)}, while it should be of type {arg_type}.")
                arg_idx += 1
            return target_class(*arg_or_args)
        elif isinstance(field_arg, target_class):
            return field_arg

    if isinstance(arg_or_args, target_class):
        return target_class,
    elif type(arg_or_args) in (tuple, list):
        return tuple(_inner_parse(field_arg) for field_arg in arg_or_args)


# endregion

# region specialized arg parsing


def parse_score_name(score_name: str, must_be_non_empty=False):
    if not is_arg_none(score_name):
        score_name = score_name.strip()
        score_name_rank_reverse = True
        if score_name[0] == '-':
            score_name = score_name[1:]
            score_name_rank_reverse = False
        return score_name, score_name_rank_reverse
    else:
        if must_be_non_empty:
            raise ValueError("The score name is empty or 'none'.")
        return None, False


# endregion

# region ArgumentParser that exposes the actually specified parameters.
# TODO should update this script with updated python version

class ArgumentParser(argparse.ArgumentParser):
    def _parse_known_args(self, arg_strings, namespace):
        # replace arg strings that are file references
        if self.fromfile_prefix_chars is not None:
            arg_strings = self._read_args_from_files(arg_strings)

        # map all mutually exclusive arguments to the other arguments
        # they can't occur with
        action_conflicts = {}
        for mutex_group in self._mutually_exclusive_groups:
            group_actions = mutex_group._group_actions
            for i, mutex_action in enumerate(mutex_group._group_actions):
                conflicts = action_conflicts.setdefault(mutex_action, [])
                conflicts.extend(group_actions[:i])
                conflicts.extend(group_actions[i + 1:])

        # find all option indices, and determine the arg_string_pattern
        # which has an 'O' if there is an option at an index,
        # an 'A' if there is an argument, or a '-' if there is a '--'
        option_string_indices = {}
        arg_string_pattern_parts = []
        arg_strings_iter = iter(arg_strings)
        for i, arg_string in enumerate(arg_strings_iter):

            # all args after -- are non-options
            if arg_string == '--':
                arg_string_pattern_parts.append('-')
                for arg_string in arg_strings_iter:
                    arg_string_pattern_parts.append('A')

            # otherwise, add the arg to the arg strings
            # and note the index if it was an option
            else:
                option_tuple = self._parse_optional(arg_string)
                if option_tuple is None:
                    pattern = 'A'
                else:
                    option_string_indices[i] = option_tuple
                    pattern = 'O'
                arg_string_pattern_parts.append(pattern)

        # join the pieces together to form the pattern
        arg_strings_pattern = ''.join(arg_string_pattern_parts)

        # converts arg strings to the appropriate and then takes the action
        seen_actions = set()
        seen_non_default_actions = set()

        def take_action(action, argument_strings, option_string=None):
            seen_actions.add(action)
            argument_values = self._get_values(action, argument_strings)

            # error if this argument is not allowed with other previously
            # seen arguments, assuming that actions that use the default
            # value don't really count as "present"
            if argument_values is not action.default:
                seen_non_default_actions.add(action)
                for conflict_action in action_conflicts.get(action, []):
                    if conflict_action in seen_non_default_actions:
                        msg = _('not allowed with argument %s')
                        action_name = _get_action_name(conflict_action)
                        raise ArgumentError(action, msg % action_name)

            # take the action if we didn't receive a SUPPRESS value
            # (e.g. from a default)
            if argument_values is not SUPPRESS:
                action(self, namespace, argument_values, option_string)

        # function to convert arg_strings into an optional action
        def consume_optional(start_index):

            # get the optional identified at this index
            option_tuple = option_string_indices[start_index]
            action, option_string, explicit_arg = option_tuple

            # identify additional optionals in the same arg string
            # (e.g. -xyz is the same as -x -y -z if no args are required)
            match_argument = self._match_argument
            action_tuples = []
            while True:

                # if we found no optional action, skip it
                if action is None:
                    extras.append(arg_strings[start_index])
                    return start_index + 1

                # if there is an explicit argument, try to match the
                # optional's string arguments to only this
                if explicit_arg is not None:
                    arg_count = match_argument(action, 'A')

                    # if the action is a single-dash option and takes no
                    # arguments, try to parse more single-dash options out
                    # of the tail of the option string
                    chars = self.prefix_chars
                    if arg_count == 0 and option_string[1] not in chars:
                        action_tuples.append((action, [], option_string))
                        char = option_string[0]
                        option_string = char + explicit_arg[0]
                        new_explicit_arg = explicit_arg[1:] or None
                        optionals_map = self._option_string_actions
                        if option_string in optionals_map:
                            action = optionals_map[option_string]
                            explicit_arg = new_explicit_arg
                        else:
                            msg = _('ignored explicit argument %r')
                            raise ArgumentError(action, msg % explicit_arg)

                    # if the action expect exactly one argument, we've
                    # successfully matched the option; exit the loop
                    elif arg_count == 1:
                        stop = start_index + 1
                        args = [explicit_arg]
                        action_tuples.append((action, args, option_string))
                        break

                    # error if a double-dash option did not use the
                    # explicit argument
                    else:
                        msg = _('ignored explicit argument %r')
                        raise ArgumentError(action, msg % explicit_arg)

                # if there is no explicit argument, try to match the
                # optional's string arguments with the following strings
                # if successful, exit the loop
                else:
                    start = start_index + 1
                    selected_patterns = arg_strings_pattern[start:]
                    arg_count = match_argument(action, selected_patterns)
                    stop = start + arg_count
                    args = arg_strings[start:stop]
                    action_tuples.append((action, args, option_string))
                    break

            # add the Optional to the list and return the index at which
            # the Optional's string args stopped
            assert action_tuples
            for action, args, option_string in action_tuples:
                take_action(action, args, option_string)
            return stop

        # the list of Positionals left to be parsed; this is modified
        # by consume_positionals()
        positionals = self._get_positional_actions()

        # function to convert arg_strings into positional actions
        def consume_positionals(start_index):
            # match as many Positionals as possible
            match_partial = self._match_arguments_partial
            selected_pattern = arg_strings_pattern[start_index:]
            arg_counts = match_partial(positionals, selected_pattern)

            # slice off the appropriate arg strings for each Positional
            # and add the Positional and its args to the list
            for action, arg_count in zip(positionals, arg_counts):
                args = arg_strings[start_index: start_index + arg_count]
                start_index += arg_count
                take_action(action, args)

            # slice off the Positionals that we just parsed and return the
            # index at which the Positionals' string args stopped
            positionals[:] = positionals[len(arg_counts):]
            return start_index

        # consume Positionals and Optionals alternately, until we have
        # passed the last option string
        extras = []
        start_index = 0
        if option_string_indices:
            max_option_string_index = max(option_string_indices)
        else:
            max_option_string_index = -1
        while start_index <= max_option_string_index:

            # consume any Positionals preceding the next option
            next_option_string_index = min([
                index
                for index in option_string_indices
                if index >= start_index])
            if start_index != next_option_string_index:
                positionals_end_index = consume_positionals(start_index)

                # only try to parse the next optional if we didn't consume
                # the option string during the positionals parsing
                if positionals_end_index > start_index:
                    start_index = positionals_end_index
                    continue
                else:
                    start_index = positionals_end_index

            # if we consumed all the positionals we could and we're not
            # at the index of an option string, there were extra arguments
            if start_index not in option_string_indices:
                strings = arg_strings[start_index:next_option_string_index]
                extras.extend(strings)
                start_index = next_option_string_index

            # consume the next optional and any arguments for it
            start_index = consume_optional(start_index)

        # consume any positionals following the last Optional
        stop_index = consume_positionals(start_index)

        # if we didn't consume all the argument strings, there were extras
        extras.extend(arg_strings[stop_index:])

        # make sure all required actions were present and also convert
        # action defaults which were not given as arguments
        required_actions = []
        for action in self._actions:
            if action not in seen_actions:
                if action.required:
                    required_actions.append(_get_action_name(action))
                else:
                    # Convert action default now instead of doing it before
                    # parsing arguments to avoid calling convert functions
                    # twice (which may fail) if the argument was given, but
                    # only if it was defined already in the namespace
                    if (action.default is not None and
                            isinstance(action.default, str) and
                            hasattr(namespace, action.dest) and
                            action.default is getattr(namespace, action.dest)):
                        setattr(namespace, action.dest,
                                self._get_value(action, action.default))

        if required_actions:
            self.error(_('the following arguments are required: %s') %
                       ', '.join(required_actions))

        # make sure all required groups had one option present
        for group in self._mutually_exclusive_groups:
            if group.required:
                for action in group._group_actions:
                    if action in seen_non_default_actions:
                        break

                # if no actions were used, report the error
                else:
                    names = [_get_action_name(action)
                             for action in group._group_actions
                             if action.help is not SUPPRESS]
                    msg = _('one of the arguments %s is required')
                    self.error(msg % ' '.join(names))

        # return the updated namespace and the extra arguments
        return namespace, extras, seen_actions

    def parse_args(self, args=None, namespace=None):
        args, argv, seen_actions = self.parse_known_args(args, namespace)
        if argv:
            msg = _('unrecognized arguments: %s')
            self.error(msg % ' '.join(argv))
        return args, seen_actions

    def parse_known_args(self, args=None, namespace=None):
        if args is None:
            # args default to the system args
            args = _sys.argv[1:]
        else:
            # make sure that args are mutable
            args = list(args)

        # default Namespace built from parser defaults
        if namespace is None:
            namespace = Namespace()

        # add any action defaults that aren't present
        for action in self._actions:
            if action.dest is not SUPPRESS:
                if not hasattr(namespace, action.dest):
                    if action.default is not SUPPRESS:
                        setattr(namespace, action.dest, action.default)

        # add any parser defaults that aren't present
        for dest in self._defaults:
            if not hasattr(namespace, dest):
                setattr(namespace, dest, self._defaults[dest])

        # parse the arguments and exit if there are any errors
        try:
            namespace, args, seen_actions = self._parse_known_args(args, namespace)
            if hasattr(namespace, _UNRECOGNIZED_ARGS_ATTR):
                args.extend(getattr(namespace, _UNRECOGNIZED_ARGS_ATTR))
                delattr(namespace, _UNRECOGNIZED_ARGS_ATTR)
            return namespace, args, seen_actions
        except ArgumentError:
            err = _sys.exc_info()[1]
            self.error(str(err))

# endregion

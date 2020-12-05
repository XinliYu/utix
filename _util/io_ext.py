import _io
import gzip
import json
import os
import pickle
import random
import hashlib
import shutil
import sys
import urllib.request
import warnings
from itertools import chain
from os import path
from random import shuffle, randint
from time import time
from typing import Dict, Callable, Iterator, List, Iterable, Union, Any, Mapping
from zipfile import ZipFile
from utix.strex import add_prefix, strip__
import tqdm
import uuid
from utix.dictex import update_dict_by_addition, IndexDict, kvswap
from utix.general import hprint_pairs, hprint, hprint_message, get_hprint_str, tqdm_wrap, eprint_message, str2val, \
    apply_tqdm
from utix.iterex import chunk_iter, islice, next__, with_uuid, with_names
from utix.listex import iter_split_list, split_list_by_ratios
from utix.msgex import msg_arg_path_not_exist, msg_batch_file_writing_to_dir
from utix.timex import timestamp, tic, toc

import utix.pathex as paex

TYPE_FILENAME_OR_STREAM = Union[str, _io.TextIOWrapper]


def _get_input_file_stream(file, encoding, top, use_tqdm, display_msg, verbose):
    if isinstance(file, str):
        fin = open(file, encoding=encoding)
    else:
        fin = file
        if hasattr(file, 'name'):
            file = file.name
        else:
            file = 'an iterator'

    return tqdm_wrap((islice(fin, top) if top is not None else fin), use_tqdm=use_tqdm,
                     tqdm_msg=display_msg.format(file) if display_msg else None, verbose=verbose), fin


@apply_tqdm
def merge_files(files, output_path, skip_first_line=False):
    with open(output_path, 'w') as fout:
        if skip_first_line == 'keepone':
            file = next(iter(files))
            with open(file, 'r') as fin:
                for line in fin:
                    fout.writelines(line)
            skip_first_line = True

        for file in files:
            with open(file, 'r') as fin:
                if skip_first_line:
                    next(fin)
                for line in fin:
                    fout.writelines(line)


# region create files

def make_empty_file(file_path: str):
    """
    Creates an empty file a the specified path.
    :param file_path: the path to the empty file.
    """
    open(file_path, 'a').close()


def touch(file_path: str):
    """
    Modifies the timestamp of a file at the specified path. Equivalent to the linux `touch` command.
    :param file_path: provides the path to the file to modify its timestamp.
    """
    with open(file_path, 'a'):
        os.utime(file_path, None)


class open__:
    """
    Provides more options for opening a file, including automatically creating the parent directory, and tqdm wrap.
    """

    def __init__(self, file: str, mode: str = 'r', encoding=None, use_tqdm: bool = False, display_msg: str = None,
                 verbose=__debug__, create_dir=True, *args, **kwargs):
        self._file = file

        need_dir_exist = False
        if display_msg is None and (use_tqdm or verbose):
            binary = 'binary ' if 'b' in mode else ''
            if 'r' in mode:
                display_msg = f'read from {binary}file {file}'
            elif 'w' in mode:
                if path.exists(file):
                    display_msg = f'overwrite {binary}file {file}'
                else:
                    display_msg = f'write to {binary}file {file}'
                need_dir_exist = True
                use_tqdm = False
            elif 'a' in mode:
                display_msg = f'append to {binary}file {file}'
                need_dir_exist = True
                use_tqdm = False
            elif 'x' in mode:
                display_msg = f'write to {binary}file {file}'
                need_dir_exist = True
                use_tqdm = False
        else:
            need_dir_exist = 'w' in mode or 'a' in mode or 'x' in mode
            use_tqdm = not need_dir_exist

        if create_dir and need_dir_exist:
            os.makedirs(path.dirname(self._file), exist_ok=True)

        self._f = open(self._file, mode=mode, encoding=encoding, *args, **kwargs)
        self._tqdm_wrap = tqdm_wrap(it=self._f, use_tqdm=use_tqdm, tqdm_msg=display_msg, verbose=verbose)

    def __enter__(self):
        return self._tqdm_wrap

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._f.close()


# endregion

# region move/copy files


def _batch_copy(pid, src_paths, dst_dir, solve_conflict=True, use_tqdm=True, tqdm_msg=None, num_p=1):
    for src_path in tqdm_wrap(src_paths, use_tqdm, tqdm_msg):
        dst_path = path.join(dst_dir, path.basename(src_path))
        if solve_conflict:
            dst_path = paex.ensure_path_no_conflict(dst_path)
        if path.isdir(src_path):
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)


def batch_copy(src_paths, dst_dir, solve_conflict=True, use_tqdm=True, tqdm_msg=None, num_p=1):
    if num_p <= 1:
        _batch_copy(0, src_paths=src_paths, dst_dir=dst_dir, solve_conflict=solve_conflict, use_tqdm=use_tqdm,
                    tqdm_msg=tqdm_msg)
    else:
        from utix.mpex import parallel_process_by_pool
        parallel_process_by_pool(
            num_p=num_p,
            data_iter=src_paths,
            target=_batch_copy,
            args=(dst_dir, solve_conflict, use_tqdm, tqdm_msg, num_p),
            merge_output=False
        )


def batch_move(src_paths, dst_dir, solve_conflict=True, undo_move_on_failure=True, use_tqdm=True, tqdm_msg=None):
    if undo_move_on_failure:
        roll_back = []
        for src_path in tqdm_wrap(src_paths, use_tqdm, tqdm_msg):
            dst_path = path.join(dst_dir, path.basename(src_path))
            if solve_conflict:
                dst_path = paex.ensure_path_no_conflict(dst_path)
            try:
                shutil.move(src_path, dst_path)
            except Exception as e:
                for src_path, dst_path in roll_back:
                    shutil.move(dst_path, src_path)
                raise e
            roll_back.append((src_path, dst_path))
    else:
        for src_path in tqdm_wrap(src_paths, use_tqdm, tqdm_msg):
            dst_path = path.join(dst_dir, path.basename(src_path))
            if solve_conflict:
                dst_path = paex.ensure_path_no_conflict(dst_path)
            shutil.move(src_path, dst_path)


# endregion

# region remove files

def _print_file_removed(file_path: str):
    if path.exists(file_path):
        hprint_message('removed', file_path)
    else:
        eprint_message('removal failed', file_path)


def _print_file_existence(file_path: str, exists: bool):
    if exists:
        hprint_message("file exists", file_path)
    else:
        hprint_message("file not exist", file_path)


def remove_files_from_iter(file_path_iter, verbose=False):
    """
    Removes files returned by a file path iterator.
    :param file_path_iter: a file path iterator.
    :param verbose: `True` if file removal results are printed out on the terminal; otherwise `False`.
    """
    if verbose:
        for file_path in file_path_iter:
            if path.exists(file_path):
                os.remove(file_path)
                _print_file_removed(file_path)
            else:
                _print_file_existence(file_path, exists=False)
    else:
        for file_path in file_path_iter:
            if path.exists(file_path):
                os.remove(file_path)


def remove_if_exists(*file_or_dir_paths, verbose=False):
    def _remove(_path):
        if path.exists(_path):
            if path.isdir(_path):
                shutil.rmtree(_path)
                if verbose:
                    hprint_message("directory removed", _path)
            else:
                os.remove(_path)
                if verbose:
                    hprint_message("file removed", _path)
        elif verbose:
            hprint_message("no need to remove (path not exist)", _path)

    for _path in file_or_dir_paths:
        _remove(_path)


# endregion

# region read/write lines

def read_all_lines(file_path: str, encoding=None, use_tqdm: bool = False, disp_msg: str = None, lstrip=False,
                   rstrip=True, verbose=__debug__):
    """
    Works in the same way as `iter_all_lines` but returns everything all at once.
    """

    with open__(file_path, encoding=encoding, use_tqdm=use_tqdm, display_msg=disp_msg, verbose=verbose) as fin:
        return [strip__(line, lstrip=lstrip, rstrip=rstrip) for line in fin]


def iter_all_lines(file_path: str, use_tqdm: bool = False, tqdm_msg: str = None, lstrip=False, rstrip=True,
                   encoding=None, parse=None, verbose=__debug__):
    """
    Yields each line from the specified file, with all spaces at the end of each line stripped by default.
    :param file_path: the path of the file to read.
    :param use_tqdm: `True` if using tqdm to display progress; otherwise `False`.
    :param tqdm_msg: the message to display by the tqdm; the message can be a format pattern of single parameter `path`, e.g. 'read from file at {path}'.
    :param lstrip: `True` if the spaces at the beginning of each read line is stripped; otherwise `False`.
    :param rstrip: `True` if the spaces at the end of each read line is stripped; otherwise `False`.
    :param verbose: `True` if details of the execution should be printed on the terminal.
    :return: a list of lines read from the specified file, with all spaces at the end of each line stripped.
    """
    if parse:
        with open(file_path, 'r', encoding=encoding) as fin:
            fin = tqdm_wrap(fin, use_tqdm=use_tqdm,
                            tqdm_msg=tqdm_msg or get_hprint_str(f"`read` from file at {str(file_path)}"),
                            verbose=verbose)
            yield from (str2val(strip__(line, lstrip=lstrip, rstrip=rstrip)) for line in fin)
    else:
        with open(file_path, 'r', encoding=encoding) as fin:
            fin = tqdm_wrap(fin, use_tqdm=use_tqdm,
                            tqdm_msg=tqdm_msg or get_hprint_str(f"`read` from file at {str(file_path)}"),
                            verbose=verbose)
            yield from (strip__(line, lstrip=lstrip, rstrip=rstrip) for line in fin)


def iter_all_lines__(file_path: str, use_tqdm: bool = False, tqdm_msg: str = None, lstrip=False, rstrip=True,
                     line_filter: Union[str, Callable] = '#', verbose=__debug__):
    """
    The same as `iter_all_lines`, with the ability to skip lines.
    The `line_filter` can be 1) a string, and every line starting with this string will be skipped; 2) a callable that return `False` for a line that should be skipped.
    """
    if isinstance(line_filter, str):
        def _it():
            for line in fin:
                line = strip__(line, lstrip=lstrip, rstrip=rstrip)
                if not line.startswith(line_filter):
                    yield line
    elif callable(line_filter):
        def _it():
            for line in fin:
                line = strip__(line, lstrip=lstrip, rstrip=rstrip)
                if line_filter(line):
                    yield line
    else:
        def _it():
            return (strip__(line, lstrip=lstrip, rstrip=rstrip) for line in fin)

    with open(file_path, 'r') as fin:
        fin = tqdm_wrap(fin, use_tqdm=use_tqdm, tqdm_msg=tqdm_msg or get_hprint_str(f"`read` from file at {str(path)}"),
                        verbose=verbose)
        yield from _it()


def iter_multi_lines(file_path: str, index_file_patb: str, filter_file_path=None, filter=None, use_tqdm: bool = False,
                     tqdm_msg: str = None, lstrip=False, rstrip=True, verbose=__debug__):
    if filter_file_path is None or filter is None:
        with open(file_path, 'r') as fin, open(index_file_patb, 'r') as ifin:
            ifin = tqdm_wrap(ifin, use_tqdm=use_tqdm, tqdm_msg=tqdm_msg or get_hprint_str(
                f"`read` from file at {str(file_path)} with index file at {str(index_file_patb)}"), verbose=verbose)
            for n in ifin:
                yield tuple(strip__(line, lstrip=lstrip, rstrip=rstrip) for line in islice(fin, int(n)))
    else:
        with open(file_path, 'r') as fin, open(index_file_patb, 'r') as ifin, open(filter_file_path) as ffin:
            ifin = tqdm_wrap(zip(ifin, ffin), use_tqdm=use_tqdm, tqdm_msg=tqdm_msg or get_hprint_str(
                f"`read` from file at {str(file_path)} with index file at {str(index_file_patb)}"), verbose=verbose)
            for n, _filter in ifin:
                if _filter.strip() == filter:
                    yield tuple(strip__(line, lstrip=lstrip, rstrip=rstrip) for line in islice(fin, int(n)))
                else:
                    next__(fin, int(n))


def read_all_lines_or_none(file_path: str, use_tqdm: bool = False, tqdm_msg: str = None, lstrip=False, rstrip=True,
                           verbose=__debug__):
    if path.exists(file_path):
        return read_all_lines(file_path=file_path, use_tqdm=use_tqdm, disp_msg=tqdm_msg, lstrip=lstrip, rstrip=rstrip,
                              verbose=verbose)


def iter_all_lines_from_all_files(input_paths, sample_rate=1.0, lstrip=False, rstrip=True, use_tqdm=False,
                                  display_msg=None, verbose=__debug__, sort=False, sort_by_basename=False):
    """
    Iterates through all lines of a collection of input paths, with the options to sort input files, sub-sample lines, and strip the whitespaces at the start or the end of each line.
    """
    if isinstance(input_paths, str):
        input_paths = (input_paths,)
    else:
        input_paths = paex.sort_paths(input_paths, sort=sort, sort_by_basename=sort_by_basename)
    if sample_rate >= 1.0:
        for file in input_paths:
            with open__(file, use_tqdm=use_tqdm, display_msg=display_msg, verbose=verbose) as f:
                yield from (strip__(line, lstrip=lstrip, rstrip=rstrip) for line in f)
    else:
        for file in input_paths:
            with open__(file, use_tqdm=use_tqdm, display_msg=display_msg, verbose=verbose) as f:
                for line in f:
                    if random.uniform(0, 1) < sample_rate:
                        yield strip__(line, lstrip=lstrip, rstrip=rstrip)


def read_all_lines_from_all_files(input_path, *args, **kwargs):
    return list(iter_all_lines_from_all_files(input_path, *args, **kwargs))


def iter_all_lines_from_all_sub_dirs(input_path: str, pattern: str, sample_rate: float = 1.0, use_tqdm: bool = False,
                                     display_msg: str = None, verbose=__debug__) -> Iterator[str]:
    if path.isfile(input_path):
        all_files = [input_path]
    else:
        all_files = paex.get_sorted_files_from_all_sub_dirs(dir_path=input_path, pattern=pattern)

    return iter_all_lines_from_all_files(input_paths=all_files, sample_rate=sample_rate, lstrip=False, rstrip=True,
                                         use_tqdm=use_tqdm, display_msg=display_msg, verbose=verbose)


# endregion

# region write lines

def write_all_lines_to_stream(fout, iterable: Iterator[str], to_str: Callable[[Any], str] = None,
                              remove_blank_lines: bool = False, avoid_repeated_new_line: bool = True):
    def _write_text(text):
        if len(text) == 0:
            if not remove_blank_lines:
                fout.write('\n')
        else:
            fout.write(text)
            if not avoid_repeated_new_line or text[-1] != '\n':
                fout.write('\n')

    if to_str is None:
        to_str = str

    for item in iterable:
        _write_text(to_str(item))

    fout.flush()


def write_all_lines(iterable: Iterator, output_path: str, to_str: Callable = None, use_tqdm: bool = False,
                    display_msg: str = None,
                    append=False, encoding=None, verbose=__debug__, create_dir=True, remove_blank_lines: bool = False,
                    avoid_repeated_new_line: bool = True,
                    chunk_size=None, chunk_name_format='part_{:05}', chunked_file_ext_name='.txt'):
    iterable = tqdm_wrap(iterable, use_tqdm=use_tqdm, tqdm_msg=display_msg, verbose=verbose)
    if chunk_size is None:
        with open__(output_path, 'a+' if append else 'w+', encoding=encoding, create_dir=create_dir) as wf:
            write_all_lines_to_stream(fout=wf, iterable=iterable, to_str=to_str, remove_blank_lines=remove_blank_lines,
                                      avoid_repeated_new_line=avoid_repeated_new_line)
    else:
        chunked_file_ext_name = paex.make_ext_name(chunked_file_ext_name)
        for chunk_name, chunk in with_names(chunk_iter(iterable, chunk_size=chunk_size), name_format=chunk_name_format,
                                            name_suffix=chunked_file_ext_name):
            with open__(path.join(output_path, chunk_name), 'a+' if append else 'w+', encoding=encoding,
                        create_dir=create_dir) as wf:
                write_all_lines_to_stream(fout=wf, iterable=chunk, to_str=to_str, remove_blank_lines=remove_blank_lines,
                                          avoid_repeated_new_line=avoid_repeated_new_line)


def write_all_lines_to_dir(line_iter: Union[Iterator, Iterable], output_dir: str, output_file_size=-1,
                           num_output_files=1, file_name_pattern='part_{}.txt', make_output_dir_if_not_exists=True,
                           overwrite=False,
                           verbose=False, use_tqdm=True, tqdm_reading_msg=None, tqdm_writing_msg=None):
    if path.exists(output_dir):
        if not path.isdir(output_dir):
            raise ValueError(paex.msg_arg_not_a_dir(path_str=output_dir, arg_name='output_dir'))
        if overwrite:
            paex.ensure_dir_existence(output_dir, clear_dir=True, verbose=verbose)
    elif make_output_dir_if_not_exists:
        paex.ensure_dir_existence(output_dir, verbose=verbose)
    else:
        raise ValueError(msg_arg_path_not_exist(path_str=output_dir, arg_name='output_dir'))

    if not isinstance(line_iter, tqdm.std.tqdm):
        line_iter = tqdm.tqdm(line_iter)
    if tqdm_reading_msg:
        line_iter.set_description(tqdm_reading_msg)

    if output_file_size <= 0:
        lines = list(line_iter)
        num_lines = len(lines)
        if num_lines < num_output_files:
            warnings.warn(
                f"The total number of lines ({num_lines}) is less than the specified number of files ({num_output_files}).")
            num_output_files = num_lines

        if not file_name_pattern:
            file_name_pattern = 'part_{}.txt'
        for chunk_idx, chunk in enumerate(iter_split_list(list_to_split=lines, num_splits=num_output_files)):
            write_all_lines(iterable=chunk, output_path=path.join(output_dir, file_name_pattern.format(chunk_idx)),
                            use_tqdm=use_tqdm, display_msg=tqdm_writing_msg.format(chunk_idx))
    else:
        for chunk_idx, chunk in enumerate(chunk_iter(line_iter, output_file_size)):
            write_all_lines(iterable=chunk, output_path=path.join(output_dir, file_name_pattern.format(chunk_idx)),
                            use_tqdm=use_tqdm, display_msg=tqdm_writing_msg.format(chunk_idx))

    if verbose:
        hprint(msg_batch_file_writing_to_dir(path_str=output_dir, num_files=num_output_files))


# endregion

# region read/write json objs


def iter_json_objs(json_input, use_tqdm=True, disp_msg=None, verbose=__debug__, encoding=None, ignore_error=False,
                   top=None) -> Iterator[Dict]:
    """
    Iterates through all json objects in a file, or all json objects in all '.json' files in a directory, or a text line iterator.
    :param json_input: the path to a json file, or a text line iterator.
    :param disp_msg: the message to display for this reading.
    :param use_tqdm: `True` to use tqdm to display reading progress; otherwise, `False`.
    :param verbose: `True` to print out the `display_msg` regardless of `use_tqdm`.
    :return: a json object iterator.
    """

    def _iter_single_input(json_input):
        lines, fin = _get_input_file_stream(file=json_input, encoding=encoding, top=top, use_tqdm=use_tqdm,
                                            display_msg=disp_msg or 'read json object from {}', verbose=verbose)
        for line in lines:
            if line:
                try:
                    json_obj = json.loads(line)
                    yield json_obj
                except Exception as ex:
                    if ignore_error is True:
                        print(line)
                        print(ex)
                    elif ignore_error is 'silent':
                        continue
                    else:
                        print(line)
                        raise ex
        fin.close()

    if isinstance(json_input, str):
        if path.isfile(json_input):
            yield from _iter_single_input(json_input)
        else:
            for _json_input in paex.get_files_by_pattern(json_input, pattern='*.json', full_path=True, recursive=False,
                                                         sort=True):
                yield from _iter_single_input(_json_input)
    else:
        yield from _iter_single_input(json_input)


def read_all_json_objs(json_file: str, use_tqdm=False, display_msg=None, ignore_error=False, verbose=__debug__,
                       encoding=None, top=None):
    """
    The same as `iter_all_json_objs`, but reads all json objects all at once.
    """
    return list(iter_json_objs(json_input=json_file, use_tqdm=use_tqdm, disp_msg=display_msg, verbose=verbose,
                               ignore_error=ignore_error, encoding=encoding, top=top))


def iter_all_json_strs(json_obj_iter, process_func=None, indent=None, **kwargs):
    if process_func:
        for json_obj in json_obj_iter:
            try:
                yield json.dumps(process_func(json_obj), indent=indent, **kwargs)
            except Exception as ex:
                print(json_obj)
                raise ex
    else:
        for json_obj in json_obj_iter:
            try:
                yield json.dumps(json_obj, indent=indent, **kwargs)
            except Exception as ex:
                print(json_obj)
                raise ex


def iter_all_json_objs_from_all_sub_dirs(input_path: str, pattern: str = '*.json', use_tqdm: bool = False,
                                         display_msg: str = None, verbose=__debug__, encoding=None, ignore_error=False,
                                         top=None) -> Iterator[Dict]:
    if path.isfile(input_path):
        all_files = [input_path]
    else:
        all_files = paex.get_sorted_files_from_all_sub_dirs(dir_path=input_path, pattern=pattern)

    for json_file in all_files:
        yield from iter_json_objs(json_input=json_file, use_tqdm=use_tqdm, disp_msg=display_msg, verbose=verbose,
                                  encoding=encoding, ignore_error=ignore_error, top=top)


def write_all_json_objs(json_obj_iter, output_path, process_func=None, use_tqdm=False, disp_msg=None, append=False,
                        indent=None, verbose=__debug__, create_dir=True, **kwargs):
    write_all_lines(iterable=iter_all_json_strs(json_obj_iter, process_func, indent=indent, **kwargs),
                    output_path=output_path, use_tqdm=use_tqdm, display_msg=disp_msg, append=append, verbose=verbose,
                    create_dir=create_dir)


def write_json(obj, file_path: str, append=False, indent=0, create_dir=True, **kwargs):
    if create_dir:
        paex.ensure_dir_existence(path.dirname(file_path), verbose=False)
    if isinstance(obj, (list, tuple)):
        write_all_json_objs(json_obj_iter=obj, output_path=file_path, append=append, indent=indent, **kwargs)
    else:
        with open__(file_path, 'a' if append else 'w') as fout:
            fout.write(json.dumps(obj if isinstance(obj, Mapping) else vars(obj), indent=indent, **kwargs))


def iter_all_json_objs_with_line_text(json_file_path: str):
    with open(json_file_path) as f:
        for line in f:
            if line:
                json_obj = json.loads(line)
                yield json_obj, line


def _update_json_objs_internal_pool_wrap(args):
    return _update_json_objs_internal(*args)


def _update_json_objs_internal(pid, file_paths, stats, jobj_iter_creator, update_method, pre_loader, output_path,
                               one_file_per_process, verbose, args, kwargs):
    if pre_loader is None:
        def _get_jobj_iter(file_idx):
            return jobj_iter_creator(file_paths[file_idx])
    else:
        pre_load_jobjs = [list(jobj_iter_creator(data_file)) for data_file in file_paths]
        pre_load_results = pre_loader(pid, pre_load_jobjs, *args, **kwargs)

        def _get_jobj_iter(file_idx):
            return pre_load_jobjs[file_idx]

    for file_idx, file_path in enumerate(file_paths):  # updates each file in the source data
        this_total = this_update_count = 0

        def _iter():
            nonlocal this_total, this_update_count
            for jobj in _get_jobj_iter(file_idx):
                update_result = update_method(jobj, *args, **kwargs) if pre_loader is None else update_method(jobj,
                                                                                                              pre_load_results,
                                                                                                              *args,
                                                                                                              **kwargs)
                if update_result:
                    this_update_count += 1
                this_total += 1
                if update_result is None or isinstance(update_result, bool):
                    yield jobj
                else:
                    yield from update_result

        stats[0] += this_total
        stats[1] += this_update_count

        if output_path:
            if one_file_per_process:
                this_output_path = output_path.format(pid)
                write_all_json_objs(json_obj_iter=_iter(), output_path=this_output_path, use_tqdm=True,
                                    disp_msg=f'updating file {path.basename(this_output_path)}', append=True)
            else:
                file_base_name = path.basename(file_path)
                write_all_json_objs(json_obj_iter=_iter(), output_path=path.join(output_path, file_base_name),
                                    use_tqdm=True, disp_msg=f'updating file {file_base_name}')
        else:
            op_file_with_tmp(file_path,
                             lambda x: write_all_json_objs(json_obj_iter=_iter(), output_path=x, use_tqdm=True,
                                                           disp_msg=f'updating file {path.basename(x)}'))

        if verbose:
            hprint_pairs(("pid", pid), ("file", file_path), ("total", this_total), ("update", this_update_count))


def update_json_objs(json_files: List[str],
                     jobj_iter_creator: Callable, update_method: Callable, pre_loader: Callable = None,
                     num_p: int = 1, output_path: str = None, one_file_per_process=False, verbose=False,
                     *args, **kwargs):
    from multiprocessing import Manager
    from utix.mpex import parallel_process_by_pool

    if num_p > 1:
        manager = Manager()
        stats = manager.list([0, 0])

        parallel_process_by_pool(num_p=num_p,
                                 data_iter=json_files,
                                 target=_update_json_objs_internal,
                                 args=(
                                     stats, jobj_iter_creator, update_method, pre_loader, output_path,
                                     one_file_per_process,
                                     verbose, args, kwargs))
    else:
        stats = [0, 0]
        _update_json_objs_internal(pid=0, file_paths=json_files, stats=stats, jobj_iter_creator=jobj_iter_creator,
                                   update_method=update_method, pre_loader=pre_loader,
                                   output_path=output_path, one_file_per_process=one_file_per_process, verbose=verbose,
                                   args=args, kwargs=kwargs)

    if verbose:
        hprint_pairs(("overall total", stats[0]), ("overall updated", stats[1]))


# endregion


# region pack/unpack text files

def pack_text_file(file_path, output_path, sep=' ', top=None, use_tqdm=False, display_msg=None, verbose=__debug__):
    """
    Packs a text file a compressed pickle file.

    :param file_path: the path to the text file.
    :param output_path: the pickle file will be saved at this path.
    :param sep: the separator to split each line.
    :param use_tqdm: `True` to use tqdm to display packing progress; otherwise, `False`.
    :param display_msg: the message to print or display to indicate the data is being packed.
    :param verbose: `True` to print out as many internal messages as possible.
    """
    vocab = IndexDict()
    data = []
    if verbose:
        tic(f"Packing text file {file_path} by pickle.")
    with open(file_path, 'r') as f:
        for line in tqdm_wrap(f if top is None else islice(f, top), use_tqdm=use_tqdm, tqdm_msg=display_msg,
                              verbose=verbose):
            data.append(tuple(vocab.add(field) for field in line.strip('\n').split(sep)))

    if verbose:
        hprint_message('data size', len(data))
        hprint_message('vocab size', len(vocab))
        hprint_message('save to', output_path)
    pickle_save((sep, data, dict(vocab.to_dicts()[0])), output_path, compressed=True)

    if verbose:
        toc()


def unpack_text_file(data_path, output_path, use_tqdm=False, display_msg=None, verbose=__debug__):
    """
    Unpacks a compressed pickle file built by `pack_text_file`.

    :param data_path: the path to the pickle file.
    :param output_path: the output text file will be saved at this path.
    :param use_tqdm: `True` to use tqdm to display packing progress; otherwise, `False`.
    :param display_msg: the message to print or display to indicate the data is being unpacked.
    :param verbose: `True` to print out as much internal message as possible.
    """
    sep, data, vocab = pickle_load(data_path, compressed=True)
    vocab = kvswap(vocab)
    if verbose:
        hprint_message('data size', len(data))
        hprint_message('vocab size', len(vocab))

    def _line_iter():
        for fields in tqdm_wrap(data, use_tqdm=use_tqdm, tqdm_msg=display_msg, verbose=verbose):
            yield sep.join(vocab[field] for field in fields)

    if verbose:
        tic(f"Unpacking pickle file {data_path} to text.")
    write_all_lines(_line_iter(), output_path=output_path, use_tqdm=use_tqdm, display_msg=display_msg, verbose=verbose)


def pack_json_file(file_path, output_path, key_sep='|', top=None, use_tqdm=False, display_msg=None, verbose=__debug__,
                   chunk=None, chunk_suffix_digits=5):
    """
    Packs a json file a compressed pickle file.

    :param file_path: the path to the text file.
    :param output_path: the pickle file will be saved at this path.
    :param sep: the separator to split each line.
    :param use_tqdm: `True` to use tqdm to display packing progress; otherwise, `False`.
    :param display_msg: the message to print or display to indicate the data is being packed.
    :param verbose: `True` to print out as many internal messages as possible.
    """
    vocab = IndexDict()
    data = []
    if verbose:
        tic(f"Packing json file {file_path} by pickle.")

    def _vocab_key(k):
        _k = k.split(key_sep)
        return vocab.add(k) if len(_k) == 1 else tuple(vocab.add(x) for x in _k)

    def _replace_keys(d):
        return {_vocab_key(k): (_replace_keys(v) if isinstance(v, Mapping) else v) for k, v in d.items()}

    chunk_number = 1

    def _save_chunk():
        nonlocal chunk_number
        chunk_path = paex.append_to_main_name(output_path,
                                              ('{:0' + str(chunk_suffix_digits) + '}').format(chunk_number))
        if verbose:
            hprint_message('chunk number', chunk_number)
            hprint_message('chunk size', len(data))
            hprint_message('save to', chunk_path)
        pickle_save(data, chunk_path, compressed=True)
        chunk_number += 1
        data.clear()

    for jobj in iter_json_objs(json_input=file_path, use_tqdm=use_tqdm, disp_msg=display_msg, verbose=verbose,
                               top=top):
        data.append(_replace_keys(jobj))
        if chunk is not None and len(data) == chunk:
            _save_chunk()

    if chunk is not None:
        if len(data) != 0:
            _save_chunk()
        meta_path = paex.append_to_main_name(output_path, '_meta')
        if verbose:
            hprint_message('vocab size', len(vocab))
            hprint_message('save to', meta_path)
        pickle_save((key_sep, dict(vocab.to_dicts()[0])), meta_path, compressed=True)
    else:
        if verbose:
            hprint_message('data size', len(data))
            hprint_message('vocab size', len(vocab))
            hprint_message('save to', output_path)
        pickle_save((key_sep, data, dict(vocab.to_dicts()[0])), output_path, compressed=True)

    if verbose:
        toc()


def unpack_json_file(data_path, output_path, use_tqdm=False, display_msg=None, verbose=__debug__, chunk=False,
                     chunk_suffix_digits=5, separate_chunk_files=False):
    """
    Unpacks a compressed pickle file built by `pack_json_file`.

    :param data_path: the path to the pickle file.
    :param output_path: the output text file will be saved at this path.
    :param use_tqdm: `True` to use tqdm to display packing progress; otherwise, `False`.
    :param display_msg: the message to print or display to indicate the data is being unpacked.
    :param verbose: `True` to print out as much internal message as possible.
    """

    def _vocab_key(k):
        if isinstance(k, int):
            return vocab[k]
        else:
            return key_sep.join(vocab[x] for x in k)

    def _replace_keys(d):
        return {_vocab_key(k): _replace_keys(v) if isinstance(v, Mapping) else v for k, v in d.items()}

    def _jobj_iter():
        for jobj in tqdm_wrap(data, use_tqdm=use_tqdm, tqdm_msg=display_msg, verbose=verbose):
            yield _replace_keys(jobj)

    if verbose:
        tic(f"Unpacking pickle file {data_path} to text.")

    if chunk is False or chunk is None:
        key_sep, data, vocab = pickle_load(data_path, compressed=True)
        vocab = kvswap(vocab)
        if verbose:
            hprint_message('data size', len(data))
            hprint_message('vocab size', len(vocab))
        write_all_json_objs(_jobj_iter(), output_path=output_path, use_tqdm=use_tqdm, disp_msg=display_msg,
                            verbose=verbose)
    else:
        if isinstance(chunk, int):
            chunk = (chunk,)
        key_sep, vocab = pickle_load(paex.append_to_main_name(data_path, '_meta'), compressed=True)
        vocab = kvswap(vocab)
        if separate_chunk_files:
            for chunk_number in chunk:
                chunk_suffix = ('{:0' + str(chunk_suffix_digits) + '}').format(chunk_number)
                chunk_file = paex.append_to_main_name(data_path, chunk_suffix)
                data = pickle_load(chunk_file, compressed=True)
                chunk_output_file = paex.append_to_main_name(output_path, chunk_suffix)
                write_all_json_objs(_jobj_iter(), output_path=chunk_output_file, use_tqdm=use_tqdm,
                                    disp_msg=display_msg, verbose=verbose)
        else:
            def _jobj_iter2():
                nonlocal data
                for chunk_number in chunk:
                    chunk_suffix = ('{:0' + str(chunk_suffix_digits) + '}').format(chunk_number)
                    chunk_file = paex.append_to_main_name(data_path, chunk_suffix)
                    data = pickle_load(chunk_file, compressed=True)
                    yield from _jobj_iter()

            write_all_json_objs(_jobj_iter2(), output_path=output_path, use_tqdm=use_tqdm, disp_msg=display_msg,
                                verbose=verbose)


# endregion

def op_file_with_backup(file_path: str, op: Callable):
    bak_file = file_path + '.bak'
    shutil.copy2(file_path, bak_file)
    try:
        op(file_path)
        os.remove(bak_file)
    except Exception as ex:
        shutil.copy2(bak_file, file_path)
        os.remove(bak_file)
        raise ex


def op_file_with_tmp(file_path: str, op: Callable, move_tmp_to_target=True):
    if path.exists(file_path) and not path.isfile(file_path):
        raise ValueError(f'the target path {file_path} is not a file')
    tmp_file = file_path + '.tmp'
    op(tmp_file)
    if move_tmp_to_target:
        if path.exists(file_path):
            os.remove(file_path)
        os.rename(tmp_file, file_path)
    else:
        shutil.copy2(tmp_file, file_path)
        os.remove(tmp_file)


def iter_line_groups(input_file, line_group_separator='', strip_spaces_at_ends=True, ignore_empty_groups=True):
    """
    Iterates through line groups in a file.
    :param input_file: the path to the input file.
    :param line_group_separator: group delimiter; the default is empty line.
    :param strip_spaces_at_ends: `True` if spaces at both ends of each line are stripped.
    :param ignore_empty_groups: `True` if empty groups are not yielded.
    :return: an iterator to iterate through groups of lines in the specified file.
    """
    line_group = []
    with open(input_file) as f:
        for line in f:
            if strip_spaces_at_ends:
                line = line.strip()
            if line == line_group_separator:
                if line_group or (not ignore_empty_groups):
                    yield line_group
                    line_group = []
            else:
                line_group.append(line)
    if line_group or (not ignore_empty_groups):
        yield line_group


def iter_field_groups(input_file, line_group_separator='', field_separator='\t', strip_spaces_at_ends=True,
                      ignore_empty_groups=True, min_field_list_len=0, default_field_value=''):
    """
    Iterates through field groups in a file.
    :param input_file: the path to the input file.
    :param line_group_separator: group delimiter; the default is empty line.
    :param field_separator: field delimiter to separate a line into a field list; the default is tab '\t'.
    :param strip_spaces_at_ends: `True` if spaces at both ends of each line are stripped.
    :param ignore_empty_groups: `True` if empty groups are not yielded.
    :param min_field_list_len: minimum number of fields for each field list; there the actual number of fields is not sufficient, fields of value specified by `default_field_value` will be appended to the end.
    :param default_field_value: the default field value to append to the end of field list if it has less number of fields than specified by `min_num_fields_per_line`.
    :return: an iterator to iterate through groups of fields in the specified file.
    """
    field_group = []
    with open(input_file) as f:
        for line in f:
            if strip_spaces_at_ends:
                line = line.strip()
            if line == line_group_separator:
                if field_group or (not ignore_empty_groups):
                    yield field_group
                    field_group = []
            else:
                fields = line.split(field_separator)
                if len(fields) < min_field_list_len:
                    fields += [default_field_value] * (min_field_list_len - len(fields))
                field_group.append(fields)
    if field_group or (not ignore_empty_groups):
        yield field_group


def shuffle_lines(input_file, output_file, shuffle_skip_first_line=False, output_skip_first_line=False):
    with open(input_file) as fin:
        if shuffle_skip_first_line:
            first_line = next(fin)
            lines = list(fin)
            random.shuffle(lines)
            if output_skip_first_line:
                write_all_lines(iterable=lines, output_path=output_file, use_tqdm=True)
            else:
                write_all_lines(iterable=chain([first_line], lines), output_path=output_file, use_tqdm=True)
        else:
            lines = list(fin)
            random.shuffle(lines)
            write_all_lines(iterable=lines, output_path=output_file, use_tqdm=True)


def shuffle_and_split_lines(input_file, output_files, split_ratios, shuffle_skip_first_line=False,
                            output_skip_first_line=False, use_tqdm=True, display_msg=None, verbose=__debug__):
    with open(input_file) as fin:
        if shuffle_skip_first_line:
            first_line = next(fin)
            fin = tqdm_wrap(fin, use_tqdm, display_msg, verbose)
            lines = list(fin)
            random.shuffle(lines)
        else:
            fin = tqdm_wrap(fin, use_tqdm, display_msg, verbose)
            lines = list(fin)
            random.shuffle(lines)
        splits = split_list_by_ratios(list_to_split=lines, split_ratios=split_ratios)
        for output_path, split in zip(output_files, splits):
            if output_skip_first_line or not shuffle_skip_first_line:
                write_all_lines(iterable=split, output_path=output_path, use_tqdm=use_tqdm, display_msg=display_msg,
                                verbose=verbose, avoid_repeated_new_line=True)
            else:
                write_all_lines(iterable=chain([first_line], split), output_path=output_path, use_tqdm=use_tqdm,
                                display_msg=display_msg, verbose=verbose, avoid_repeated_new_line=True)


def sample_lines(input_file, output_file, sample_ratio, max_sample=None, use_tqdm=False):
    with open(input_file) as f, open(output_file, 'w+') as wf:
        if sample_ratio >= 1.0:
            write_all_lines(iterable=f, output_path=output_file, use_tqdm=use_tqdm)
        else:
            if use_tqdm:
                f = tqdm.tqdm(f)
            for line in f:
                if random.uniform(0, 1) < sample_ratio:
                    line = line.strip()
                    if line:
                        wf.write(line)
                        wf.write('\n')
                        if max_sample is not None:
                            max_sample -= 1
                            if max_sample <= 0:
                                break
        wf.flush()


def write_all_lines_split_files(output_path: str, iterable, num_split=0, use_tqdm=False):
    if num_split <= 1:
        write_all_lines(iterable=iterable, output_path=output_path, use_tqdm=use_tqdm)
    else:
        lines = list(iterable)
        num_lines = len(lines)
        output_dir_path = path.dirname(output_path)
        paex.ensure_dir_existence(output_dir_path)
        output_file_name = path.basename(output_path)
        output_file_main_name, output_file_ext_name = path.splitext(output_file_name)
        num_lines_per_split = num_lines // num_split + 1
        start = 0

        for i in tqdm.tqdm(range(0, num_split)) if use_tqdm else range(0, num_split):
            split_file_path = path.join(output_dir_path, f"{output_file_main_name}_{i}{output_file_ext_name}")
            with open(split_file_path, 'w+') as wf:
                for j in range(start, min(num_lines, start + num_lines_per_split)):
                    line: str = lines[j].strip()
                    wf.write(line)
                    if not line.endswith('\n'):
                        wf.write('\n')
                wf.flush()
            start += num_lines_per_split


def write_all_lines_for_list_iterable(output_path: str, iterable, item_concat='\n', end_of_list_sep='\n\n',
                                      use_tqdm=False):
    with open(output_path, 'w+') as wf:
        if use_tqdm:
            iterable = tqdm.tqdm(iterable)
        for l in iterable:
            wf.write(item_concat.join([str(item) for item in l]))
            wf.write(end_of_list_sep)
        wf.flush()


class DownloadProgressBar(tqdm.tqdm):

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download(url, dest_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=dest_path, reporthook=t.update_to)


def unzip(src_path, dest_dir, filter=None):
    with ZipFile(file=src_path) as zip_file:
        for file in tqdm.tqdm(iterable=zip_file.namelist(), total=len(zip_file.namelist())):
            # Extract each file to the `dest_dir` if it is in `filter`;
            # if you want to extract to current working directory, don't specify path.
            if filter is None or (isinstance(filter, list) and file in filter) or (
                    isinstance(filter, str) and file == filter):
                zip_file.extract(member=file, path=dest_dir)


def pickle_load(file_path: str, compressed: bool = False, encoding=None):
    with open(file_path, 'rb') if not compressed else gzip.open(file_path, 'rb') as f:
        if encoding is None or sys.version_info < (3, 0):
            return pickle.load(f)
        else:
            return pickle.load(f, encoding=encoding)


def pickle_save(data, file_path: str, compressed: bool = False):
    with open(file_path, 'wb+') if not compressed else gzip.open(file_path, 'wb+') as f:
        pickle.dump(data, f)


def pickle_save__(data, file_or_dir_path: str, extension_name=None, compressed: bool = False, auto_timestamp=True,
                  random_stamp=False, ensure_no_conflict=False, prefix=''):
    if path.isdir(file_or_dir_path):
        if auto_timestamp:
            file_or_dir_path = path.join(file_or_dir_path,
                                         ((prefix + '_') if prefix else '') + str(int(time() * 1000)) + (
                                             ('_' + str(randint(0, 1000))) if random_stamp else '') + (
                                             extension_name if extension_name else '.dat'))
        else:
            raise ValueError(
                "The specified `file_or_dir_path` is not a directory path, and `auto_timestamp` is not set `True`.")
    else:
        dir_name = path.dirname(file_or_dir_path)
        if not path.exists(dir_name):
            os.makedirs(dir_name)
        if auto_timestamp or random_stamp or prefix:
            file_name_split = path.splitext(path.basename(file_or_dir_path))
            file_name = ((prefix + '_') if prefix else '') + file_name_split[0] + (
                ('_' + str(int(time() * 1000))) if auto_timestamp else '') + (
                            ('_' + str(randint(0, 1000))) if random_stamp else '') + file_name_split[1]
            if ensure_no_conflict:
                file_name = paex.ensure_path_no_conflict(file_name)
            file_or_dir_path = path.join(dir_name, file_name)
        if extension_name:
            file_or_dir_path += extension_name
    with open(file_or_dir_path, 'wb+') if not compressed else gzip.open(file_or_dir_path, 'wb+') as f:
        pickle.dump(data, f)


def merge_counters_from_files(counter_files, computes_total=False):
    merged_dict = None
    mode = 0
    for dict_file in counter_files:
        if merged_dict is None:
            merged_dict = pickle_load(file_path=dict_file)
            if isinstance(merged_dict, dict):
                k, v = next(iter(merged_dict.items()))
                if isinstance(v, dict):
                    mode = 1
            elif isinstance(merged_dict, list):
                mode = 2
            else:
                raise NotImplementedError
        else:
            data = pickle_load(file_path=dict_file)
            if mode == 0:
                merged_dict += data
            elif mode == 1:
                for dict_key in data:
                    merged_dict[dict_key] += data[dict_key]
            elif mode == 2:
                for i, this_dict in enumerate(data):
                    merged_dict[i] += this_dict
    if computes_total:
        if mode == 0:
            merged_dict['__total__'] = sum(merged_dict.values())
        elif mode == 1:
            for dict_key in merged_dict:
                this_dict = merged_dict[dict_key]
                this_dict['__total__'] = sum(this_dict.values())
        elif mode == 2:
            for i, this_dict in enumerate(merged_dict):
                this_dict = merged_dict[i]
                this_dict['__total__'] = sum(this_dict.values())
    return merged_dict


def merge_dicts_from_files_by_addition(dict_files):
    """
    Merge dictionaries saved in files b
    :param dict_files:
    :return:
    """
    merged_dict = None
    mode = 0
    for dict_file in dict_files:
        if merged_dict is None:
            merged_dict = pickle_load(file_path=dict_file)
            if isinstance(merged_dict, dict):
                k, v = next(iter(merged_dict.items()))
                if isinstance(v, dict):
                    mode = 1
            elif isinstance(merged_dict, list):
                mode = 2
            else:
                raise NotImplementedError
        else:
            data = pickle_load(file_path=dict_file)
            if mode == 0:
                update_dict_by_addition(merged_dict, data)
            elif mode == 1:
                for dict_key in data:
                    update_dict_by_addition(merged_dict[dict_key], data[dict_key])
            elif mode == 2:
                for i, this_dict in enumerate(data):
                    update_dict_by_addition(merged_dict[i], this_dict)
    return merged_dict


def hash_file(file_path: str, hasher: Callable = hashlib.sha256, block_size=65536):
    hasher = hasher()
    with open(file_path, 'rb') as f:
        fb = f.read(block_size)
        while len(fb) > 0:
            hasher.update(fb)
            fb = f.read(block_size)

    return hasher.hexdigest()


def chunk_file(input_path: Union[str, List[str]], output_path, chunk_size, chunk_file_pattern='chunk_{}', use_uuid=True,
               use_tqdm=False, display_msg=None, verbose=__debug__):
    """
    Chunking one or more files.
    """
    if isinstance(input_path, str):
        with open(input_path) as f:
            f = tqdm_wrap(f, use_tqdm=use_tqdm,
                          tqdm_msg=display_msg or f'chunking the file at {input_path} with chunk size {chunk_size}',
                          verbose=verbose)
            ext_name = paex.get_ext_name(input_path)
            paex.ensure_dir_existence(output_path)
            it = chunk_iter(f, chunk_size=chunk_size)
            it = with_uuid(it) if use_uuid else enumerate(it)
            for chunk_idx, chunk in it:
                write_all_lines(chunk,
                                output_path=path.join(output_path, chunk_file_pattern.format(chunk_idx) + ext_name),
                                avoid_repeated_new_line=True)
    else:
        if verbose:
            hprint_message(f'chunking {len(input_path)} files, chunk size', chunk_size)
        ext_name = paex.get_ext_name(input_path[0])
        paex.ensure_dir_existence(output_path)
        it = chunk_iter(
            iter_all_lines_from_all_files(input_paths=input_path, use_tqdm=use_tqdm, display_msg=display_msg,
                                          verbose=verbose), chunk_size=chunk_size)
        it = with_uuid(it) if use_uuid else enumerate(it)
        for chunk_idx, chunk in it:
            write_all_lines(chunk, output_path=path.join(output_path, chunk_file_pattern.format(chunk_idx) + ext_name),
                            avoid_repeated_new_line=True)


# region cache

class Cache:

    # def __init__(self):
    #     self.available = False

    def __call__(self, obj, prefix=''):
        self.save(obj, prefix=prefix)

    # def __iter__(self):
    #     return self
    #
    # def __next__(self):
    #     raise NotImplementedError

    def load(self, file_name):
        raise NotImplementedError

    def save(self, obj, file_name, prefix, **kwargs):
        raise NotImplementedError

    def exists(self, file_name):
        raise NotImplementedError

    def iter_cache(self, prefix):
        raise NotImplementedError

    def has_cache(self, prefix):
        raise NotImplementedError

    def mark_complete(self, prefix: str):
        raise NotImplementedError

    def unmark_complete(self, prefix: str):
        raise NotImplementedError

    def is_complete(self, prefix: str):
        raise NotImplementedError

    def remove_cache(self, prefix: str, hard_remove=False):
        raise NotImplementedError

    def remove_cache_if_incomplete(self, prefix: str, hard_remove=False) -> bool:
        if self.has_cache(prefix) and not self.is_complete(prefix):
            self.remove_cache(prefix=prefix, hard_remove=hard_remove)
            return True
        return False

    def clear_removed(self):
        raise NotImplementedError


class SimpleFileCache(Cache):
    """
    Provides a
    """
    COMPLETION_MARK_FILE_EXTENSION = '.complete'
    REMOVAL_BACKUP_FOLDER = '.removed'

    def __init__(self, cache_dir, iter_file_ext_name='.bat', compressed=False, shuffle_file_order=False):
        super(SimpleFileCache, self).__init__()
        self.cache_dir = path.abspath(cache_dir)
        self._iter_file_ext_name = iter_file_ext_name if iter_file_ext_name[0] == '.' else '.' + iter_file_ext_name
        if self._iter_file_ext_name == SimpleFileCache.COMPLETION_MARK_FILE_EXTENSION:
            raise ValueError(
                f'the cache file extension name cannot be `{SimpleFileCache.COMPLETION_MARK_FILE_EXTENSION}`, which is used for cache completion mark files')

        self.shuffle_file_order = shuffle_file_order
        os.makedirs(self.cache_dir, exist_ok=True)
        # if self.cache_dir:
        #     if path.exists(self.cache_dir):
        #         self.cache_files = self._get_cache_files()
        #         if len(self.cache_files) == 0:
        #             self.cache_files = None
        #             self.available = False
        #         else:
        #             self._cache_file_iterator = iter(self.cache_files)
        #             self.available = True
        #     else:
        #         os.makedirs(self.cache_dir)
        #         self.available = False
        self.cache_compressed = compressed

    # def __next__(self):
    #     try:
    #         cache_file = next(self._cache_file_iterator)
    #     except StopIteration:
    #         if self.shuffle_file_order:
    #             shuffle(self.cache_files)
    #         raise StopIteration
    #     return pickle_load(cache_file, compressed=self.cache_compressed)

    def _get_file_iter(self, pattern):
        return paex.iter_files_by_pattern(self.cache_dir, pattern=pattern, full_path=True, recursive=False)

    def _get_cache_files(self, prefix=''):
        pattern = add_prefix(prefix, f'*{self._iter_file_ext_name}')
        cache_files = list(self._get_file_iter(pattern))
        if not cache_files:
            warnings.warn(f"no cache file is found at path `{self.cache_dir}` with pattern `{pattern}`")
        elif self.shuffle_file_order:
            shuffle(cache_files)
        return cache_files

    def iter_cache(self, prefix: str):
        for cache_file in self._get_cache_files(prefix):
            yield pickle_load(cache_file, compressed=self.cache_compressed)

    def remove_cache(self, prefix: str, hard_remove=False):
        pattern = add_prefix(prefix, f'*{self._iter_file_ext_name}')

        if hard_remove:
            for cache_file in self._get_file_iter(pattern):
                os.remove(cache_file)
        else:
            backup_dir = path.join(self.cache_dir, SimpleFileCache.REMOVAL_BACKUP_FOLDER, timestamp())
            os.makedirs(backup_dir, exist_ok=True)
            for cache_file in self._get_file_iter(pattern):
                os.rename(cache_file, paex.replace_dir(cache_file, backup_dir))

    def clear_removed(self):
        shutil.rmtree(path.join(self.cache_dir, SimpleFileCache.REMOVAL_BACKUP_FOLDER))

    def has_cache(self, prefix: str) -> bool:
        pattern = add_prefix(prefix, f'*{self._iter_file_ext_name}')
        try:
            next(self._get_file_iter(pattern))
            return True
        except:
            return False

    def _complete_mark_file(self, prefix: str) -> str:
        return path.join(self.cache_dir, prefix + SimpleFileCache.COMPLETION_MARK_FILE_EXTENSION)

    def mark_complete(self, prefix: str):
        make_empty_file(self._complete_mark_file(prefix))

    def unmark_complete(self, prefix: str):
        remove_if_exists(self._complete_mark_file(prefix))

    def is_complete(self, prefix: str) -> bool:
        return path.exists(self._complete_mark_file(prefix))

    def load(self, file_name):
        file_name = path.join(self.cache_dir, file_name)
        if path.exists(file_name):
            if path.isfile(file_name):
                return pickle_load(file_name, compressed=self.cache_compressed)
        else:
            dir_path, base_name = path.dirname(file_name), path.basename(file_name)
            main_name, ext_name = path.splitext(base_name)
            pattern = f'{main_name}_*{ext_name}'
            file_names = paex.get_files_by_pattern(dir_or_dirs=dir_path, pattern=pattern, recursive=False)
            output = []
            for file_name in file_names:
                output.append(pickle_load(file_name, compressed=self.cache_compressed))
            return output

    def exists(self, file_name):
        return path.exists(path.join(self.cache_dir, file_name))

    def save(self, obj, file_name=None, prefix='', auto_timestamp=False, random_stamp=False,
             ensure_no_conflict=False):  # TODO clumsy function parameters
        if file_name:
            pickle_save__(data=obj, file_or_dir_path=path.join(self.cache_dir, file_name), extension_name='',
                          compressed=self.cache_compressed, auto_timestamp=auto_timestamp, random_stamp=random_stamp,
                          ensure_no_conflict=ensure_no_conflict, prefix='')
        else:
            pickle_save__(data=obj, file_or_dir_path=self.cache_dir, extension_name=self._iter_file_ext_name,
                          compressed=self.cache_compressed, auto_timestamp=True, random_stamp=True,
                          ensure_no_conflict=True, prefix=prefix)


# endregion

# region dict/text IO

def write_dict_as_text(d, output_path, sep='\t', use_tqdm=False, display_msg=None, verbose=__debug__):
    with open(output_path, 'w') as fout:
        for k, v in tqdm_wrap(d.items(), use_tqdm=use_tqdm,
                              tqdm_msg=display_msg or f'writing dict to text file at {output_path}', verbose=verbose):
            fout.write(str(k) + sep + str(v) + '\n')


def read_dict_from_text(file_path, keytype=None, valtype=None, sep='\t', strip_key=True, strip_value=True, format=None,
                        use_tqdm=False, display_msg=None, verbose=__debug__):
    if format is None or format == 'default':
        def _get_kv():
            k, v = line.split(sep, maxsplit=1)
            k = k.strip() if strip_key else k
            v = v.strip() if strip_value else v
            return k, v

        with open(file_path, 'r') as fin:
            fin = tqdm_wrap((line.rstrip('\n') for line in fin), use_tqdm=use_tqdm,
                            tqdm_msg=display_msg or f'reading dict from text file at {file_path}', verbose=verbose)
            if keytype is None:
                if valtype is None:
                    return dict(line.split(sep, maxsplit=1) for line in fin)
                else:
                    d = {}
                    for line in fin:
                        k, v = _get_kv()
                        d[k] = valtype(v)
                    return d
            elif valtype is None:
                d = {}
                for line in fin:
                    k, v = _get_kv()
                    d[keytype(k)] = v
                return d
            else:
                d = {}
                for line in fin:
                    k, v = _get_kv()
                    d[keytype(k)] = valtype(v)
                return d
    elif format == 'dictstr':
        with open(file_path, 'r') as fin:
            fin = tqdm_wrap((line.rstrip('\n') for line in fin), use_tqdm=use_tqdm,
                            tqdm_msg=display_msg or f'reading dict from text file at {file_path}', verbose=verbose)
            dictstr = []
            for line in fin:
                k, v = line.strip().split(':')
                dictstr.append(':'.join((k.replace('\'', '"'), v)))
            dictstr = ''.join(dictstr)
            d = json.loads(dictstr)
            if keytype is None and valtype is None:
                return d
            elif keytype is None:
                return {k: valtype(v) for k, v in d.items()}
            elif valtype is None:
                return {keytype(k): v for k, v in d.items()}
            else:
                return {keytype(k): valtype(v) for k, v in d.items()}
# endregion

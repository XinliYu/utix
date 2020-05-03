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
from os import path
from random import shuffle, randint
from time import time
from typing import Dict, Callable, Iterator, List, Iterable, Union, Any
from zipfile import ZipFile
from utilx.strex import add_prefix, strip__
import tqdm

from utilx.dictex import update_dict_by_addition
from utilx.general import hprint_pairs, hprint, hprint_message, get_hprint_str, tqdm_wrap, eprint_message, str2val
from utilx.iterex import chunk_iter, islice, next__
from utilx.listex import iter_split_list
from utilx.mpex import parallel_process_by_pool
from utilx.msgex import msg_arg_path_not_exist, msg_batch_file_writing_to_dir
from utilx.pathex import iter_files_by_pattern, get_sorted_files_from_all_sub_dirs, ensure_path_no_conflict, get_files_by_pattern, ensure_dir_existence, msg_arg_not_a_dir, replace_dir
from utilx.timex import timestamp

TYPE_FILENAME_OR_STREAM = Union[str, _io.TextIOWrapper]


# region create files

def make_empty_file(file_path: str):
    open(file_path, 'a').close()


def touch(file_path: str):
    with open(file_path, 'a'):
        os.utime(file_path, None)


class open__:
    def __init__(self, file: str, mode: str, *args, **kwargs):
        self._file = file
        self._mode = mode
        self._args = args
        self._kwargs = kwargs

    def __enter__(self):
        if 'w' in self._mode or 'a' in self._mode:
            os.makedirs(path.dirname(self._file), exist_ok=True)
        self._f = open(self._file, self._mode, *self._args, **self._kwargs)
        return self._f

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._f.close()


# endregion

# region move/copy files


def _batch_copy(pid, src_paths, dst_dir, solve_conflict=True, use_tqdm=True, tqdm_msg=None, num_p=1):
    for src_path in tqdm_wrap(src_paths, use_tqdm, tqdm_msg):
        dst_path = path.join(dst_dir, path.basename(src_path))
        if solve_conflict:
            dst_path = ensure_path_no_conflict(dst_path)
        if path.isdir(src_path):
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)


def batch_copy(src_paths, dst_dir, solve_conflict=True, use_tqdm=True, tqdm_msg=None, num_p=1):
    if num_p <= 1:
        _batch_copy(0, src_paths=src_paths, dst_dir=dst_dir, solve_conflict=solve_conflict, use_tqdm=use_tqdm, tqdm_msg=tqdm_msg)
    else:
        parallel_process_by_pool(
            num_p=num_p,
            data_iter=src_paths,
            target=_batch_copy,
            args=(dst_dir, solve_conflict, use_tqdm, tqdm_msg, num_p),
            cross_merge_output=False
        )


def batch_move(src_paths, dst_dir, solve_conflict=True, undo_move_on_failure=True, use_tqdm=True, tqdm_msg=None):
    if undo_move_on_failure:
        roll_back = []
        for src_path in tqdm_wrap(src_paths, use_tqdm, tqdm_msg):
            dst_path = path.join(dst_dir, path.basename(src_path))
            if solve_conflict:
                dst_path = ensure_path_no_conflict(dst_path)
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
                dst_path = ensure_path_no_conflict(dst_path)
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


def remove_if_exists(file_or_dir_path, verbose=False):
    def _remove():
        if path.exists(file_or_dir_path):
            if path.isdir(file_or_dir_path):
                shutil.rmtree(file_or_dir_path)
                if verbose:
                    hprint_message("directory removed", file_or_dir_path)
            else:
                os.remove(file_or_dir_path)
                if verbose:
                    hprint_message("file removed", file_or_dir_path)
        elif verbose:
            hprint_message("no need to remove (path not exist)", file_or_dir_path)

    if isinstance(file_or_dir_path, str):
        _remove()
    else:
        file_or_dir_paths = file_or_dir_path
        for file_or_dir_path in file_or_dir_paths:
            _remove()


# endregion

# region read/write lines

def read_all_lines(file_path: str, use_tqdm: bool = False, tqdm_msg: str = None, lstrip=False, rstrip=True, verbose=__debug__):
    """
    Works in the same way as `iter_all_lines` but returns everything all at once.
    """

    with open(file_path, 'r') as fin:
        fin = tqdm_wrap(fin, use_tqdm=use_tqdm, tqdm_msg=tqdm_msg or get_hprint_str(f"`read` from file at {str(file_path)}"), verbose=verbose)
        return [strip__(line, lstrip=lstrip, rstrip=rstrip) for line in fin]


def iter_all_lines(file_path: str, use_tqdm: bool = False, tqdm_msg: str = None, lstrip=False, rstrip=True, encoding=None, parse=None, verbose=__debug__):
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
            fin = tqdm_wrap(fin, use_tqdm=use_tqdm, tqdm_msg=tqdm_msg or get_hprint_str(f"`read` from file at {str(file_path)}"), verbose=verbose)
            yield from (str2val(strip__(line, lstrip=lstrip, rstrip=rstrip)) for line in fin)
    else:
        with open(file_path, 'r', encoding=encoding) as fin:
            fin = tqdm_wrap(fin, use_tqdm=use_tqdm, tqdm_msg=tqdm_msg or get_hprint_str(f"`read` from file at {str(file_path)}"), verbose=verbose)
            yield from (strip__(line, lstrip=lstrip, rstrip=rstrip) for line in fin)


def iter_all_lines__(file_path: str, use_tqdm: bool = False, tqdm_msg: str = None, lstrip=False, rstrip=True, line_filter: Union[str, Callable] = '#', verbose=__debug__):
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
        fin = tqdm_wrap(fin, use_tqdm=use_tqdm, tqdm_msg=tqdm_msg or get_hprint_str(f"`read` from file at {str(path)}"), verbose=verbose)
        yield from _it()


def iter_multi_lines(file_path: str, index_file_patb: str, filter_file_path=None, filter=None, use_tqdm: bool = False, tqdm_msg: str = None, lstrip=False, rstrip=True, verbose=__debug__):
    if filter_file_path is None or filter is None:
        with open(file_path, 'r') as fin, open(index_file_patb, 'r') as ifin:
            ifin = tqdm_wrap(ifin, use_tqdm=use_tqdm, tqdm_msg=tqdm_msg or get_hprint_str(f"`read` from file at {str(file_path)} with index file at {str(index_file_patb)}"), verbose=verbose)
            for n in ifin:
                yield tuple(strip__(line, lstrip=lstrip, rstrip=rstrip) for line in islice(fin, int(n)))
    else:
        with open(file_path, 'r') as fin, open(index_file_patb, 'r') as ifin, open(filter_file_path) as ffin:
            ifin = tqdm_wrap(zip(ifin, ffin), use_tqdm=use_tqdm, tqdm_msg=tqdm_msg or get_hprint_str(f"`read` from file at {str(file_path)} with index file at {str(index_file_patb)}"), verbose=verbose)
            for n, _filter in ifin:
                if _filter.strip() == filter:
                    yield tuple(strip__(line, lstrip=lstrip, rstrip=rstrip) for line in islice(fin, int(n)))
                else:
                    next__(fin, int(n))


def read_all_lines_or_none(file_path: str, use_tqdm: bool = False, tqdm_msg: str = None, lstrip=False, rstrip=True, verbose=__debug__):
    if path.exists(file_path):
        return read_all_lines(file_path=file_path, use_tqdm=use_tqdm, tqdm_msg=tqdm_msg, lstrip=lstrip, rstrip=rstrip, verbose=verbose)


def iter_all_lines_from_all_files(file_paths, sample_rate=1.0, use_tqdm=False, tqdm_msg=None):
    if sample_rate >= 1.0:
        for file in file_paths:
            with open(file) as f:
                if use_tqdm:
                    f = tqdm.tqdm(f)
                    if tqdm_msg:
                        f.set_description(tqdm_msg)
                    else:
                        f.set_description(f"reading lines from {path.basename(file)}")
                yield from f
    else:
        for file in file_paths:
            with open(file) as f:
                if use_tqdm:
                    f = tqdm.tqdm(f)
                    if tqdm_msg:
                        f.set_description(tqdm_msg)
                    else:
                        f.set_description(f"reading lines from {path.basename(file)}")
                for line in f:
                    if random.uniform(0, 1) < sample_rate:
                        yield line


def iter_all_lines_from_all_sub_dirs(dir_or_file_path: str, pattern: str, sample_rate: float = 1.0, use_tqdm: bool = False, tqdm_msg: str = None) -> Iterator[str]:
    if path.isfile(dir_or_file_path):
        all_files = [dir_or_file_path]
    else:
        all_files = get_sorted_files_from_all_sub_dirs(dir_path=dir_or_file_path, pattern=pattern)

    return iter_all_lines_from_all_files(file_paths=all_files, sample_rate=sample_rate, use_tqdm=use_tqdm, tqdm_msg=tqdm_msg)


# endregion

# region write lines

def write_all_lines_to_stream(wf, iterable: Iterator[str], to_str: Callable[[Any], str] = None, use_tqdm: bool = False, tqdm_msg: str = None, remove_blank_lines: bool = True, avoid_repeated_new_line: bool = True):
    def _write_text(text):
        if len(text) == 0:
            if not remove_blank_lines:
                wf.write('\n')
        else:
            wf.write(text)
            if not avoid_repeated_new_line or text[-1] != '\n':
                wf.write('\n')

    if use_tqdm:
        iterable = tqdm.tqdm(iterable)
        if tqdm_msg is not None:
            iterable.set_description(tqdm_msg)
    if to_str is None:
        to_str = str

    for item in iterable:
        _write_text(to_str(item))

    wf.flush()


def write_all_lines(iterable: Iterator, output_path: str, to_str: Callable = None, use_tqdm: bool = False, tqdm_msg: str = None, append=False, encoding=None, verbose=__debug__):
    if verbose:
        mode = "append" if append else ('overwrite' if path.exists(output_path) else 'write')
        if not tqdm_msg:
            tqdm_msg = get_hprint_str("`{mode}` to file at {path}")
        tqdm_msg.format(mode=mode, path=output_path)
        if not use_tqdm:
            print(tqdm_msg)
    with open(output_path, 'a+' if append else 'w+', encoding=encoding) as wf:
        write_all_lines_to_stream(wf=wf, iterable=iterable, to_str=to_str, use_tqdm=use_tqdm, tqdm_msg=tqdm_msg)


def write_all_lines_to_dir(line_iter: Union[Iterator, Iterable], output_dir: str, output_file_size=-1, num_output_files=1, file_name_pattern='part_{}.txt', make_output_dir_if_not_exists=True, overwrite=False,
                           verbose=False, use_tqdm=True, tqdm_reading_msg=None, tqdm_writing_msg=None):
    if path.exists(output_dir):
        if not path.isdir(output_dir):
            raise ValueError(msg_arg_not_a_dir(path_str=output_dir, arg_name='output_dir'))
        if overwrite:
            ensure_dir_existence(output_dir, clear_dir=True, verbose=verbose)
    elif make_output_dir_if_not_exists:
        ensure_dir_existence(output_dir, verbose=verbose)
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
            warnings.warn(f"The total number of lines ({num_lines}) is less than the specified number of files ({num_output_files}).")
            num_output_files = num_lines

        if not file_name_pattern:
            file_name_pattern = 'part_{}.txt'
        for chunk_idx, chunk in enumerate(iter_split_list(list_to_split=lines, num_splits=num_output_files)):
            write_all_lines(iterable=chunk, output_path=path.join(output_dir, file_name_pattern.format(chunk_idx)), use_tqdm=use_tqdm, tqdm_msg=tqdm_writing_msg.format(chunk_idx))
    else:
        for chunk_idx, chunk in enumerate(chunk_iter(line_iter, output_file_size)):
            write_all_lines(iterable=chunk, output_path=path.join(output_dir, file_name_pattern.format(chunk_idx)), use_tqdm=use_tqdm, tqdm_msg=tqdm_writing_msg.format(chunk_idx))

    if verbose:
        hprint(msg_batch_file_writing_to_dir(path_str=output_dir, num_files=num_output_files))


# endregion

# region read/write json objs


def iter_all_json_objs(json_file, display_msg=None, use_tqdm=False, verbose=__debug__, encoding=None) -> Iterator[Dict]:
    """
    Iterates through all json objects in a file or a text line iterator.
    :param json_file: the path to a json file, or a text line iterator.
    :param display_msg: the message to display for this reading.
    :param use_tqdm: `True` to use tqdm to display reading progress; otherwise, `False`.
    :param verbose: `True` to print out the `display_msg` regardless of `use_tqdm`.
    :return: a json object iterator.
    """
    if isinstance(json_file, str):
        with open(json_file, encoding=encoding) as fin:
            fin = tqdm_wrap(fin, use_tqdm=use_tqdm, tqdm_msg=display_msg or f"read from json file at {str(json_file)}", verbose=verbose)
            for line in fin:
                if line:
                    try:
                        json_obj = json.loads(line)
                    except Exception as ex:
                        print(line)
                        raise ex
                    yield json_obj
    else:
        json_file = tqdm_wrap(json_file, use_tqdm=use_tqdm, tqdm_msg=display_msg or f"read a json iterator {str(json_file)}", verbose=verbose)
        for line in json_file:
            if line:
                try:
                    json_obj = json.loads(line)
                except Exception as ex:
                    print(line)
                    raise ex
                yield json_obj


def read_all_json_objs(json_file: str, display_msg=None, use_tqdm=False, verbose=__debug__):
    """
    The same as `iter_all_json_objs`, but reads all json objects all at once.
    """
    return list(iter_all_json_objs(json_file=json_file, display_msg=display_msg, use_tqdm=use_tqdm, verbose=verbose))


def iter_all_json_strs(json_obj_iter, process_func=None):
    if process_func:
        for json_obj in json_obj_iter:
            try:
                yield json.dumps(process_func(json_obj))
            except Exception as ex:
                print(json_obj)
                raise ex
    else:
        for json_obj in json_obj_iter:
            try:
                yield json.dumps(json_obj)
            except Exception as ex:
                print(json_obj)
                raise ex


def write_all_json_objs(json_obj_iter, output_path, process_func=None, use_tqdm=False, tqdm_msg=None, append=False):
    write_all_lines(iterable=iter_all_json_strs(json_obj_iter, process_func), output_path=output_path, use_tqdm=use_tqdm, tqdm_msg=tqdm_msg, append=append)


def write_json(d: dict, file_path: str, append=False, indent=None):
    with open__(file_path, 'a' if append else 'w') as fout:
        fout.write(json.dumps(d, indent=indent))


def iter_all_json_objs_with_line_text(json_file_path: str):
    with open(json_file_path) as f:
        for line in f:
            if line:
                json_obj = json.loads(line)
                yield json_obj, line


def _update_json_objs_internal_pool_wrap(args):
    return _update_json_objs_internal(*args)


def _update_json_objs_internal(pid, file_paths, stats, jobj_iter_creator, update_method, pre_loader, output_path, one_file_per_process, verbose, args, kwargs):
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
                update_result = update_method(jobj, *args, **kwargs) if pre_loader is None else update_method(jobj, pre_load_results, *args, **kwargs)
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
                write_all_json_objs(json_obj_iter=_iter(), output_path=this_output_path, use_tqdm=True, tqdm_msg=f'updating file {path.basename(this_output_path)}', append=True)
            else:
                file_base_name = path.basename(file_path)
                write_all_json_objs(json_obj_iter=_iter(), output_path=path.join(output_path, file_base_name), use_tqdm=True, tqdm_msg=f'updating file {file_base_name}')
        else:
            op_file_with_tmp(file_path, lambda x: write_all_json_objs(json_obj_iter=_iter(), output_path=x, use_tqdm=True, tqdm_msg=f'updating file {path.basename(x)}'))

        if verbose:
            hprint_pairs(("pid", pid), ("file", file_path), ("total", this_total), ("update", this_update_count))


def update_json_objs(json_files: List[str],
                     jobj_iter_creator: Callable, update_method: Callable, pre_loader: Callable = None,
                     num_p: int = 1, output_path: str = None, one_file_per_process=False, verbose=False,
                     *args, **kwargs):
    from multiprocessing import Manager
    from utilx.mpex import parallel_process_files_by_pool

    if num_p > 1:
        manager = Manager()
        stats = manager.list([0, 0])
        parallel_process_files_by_pool(num_p=num_p,
                                       file_paths=json_files,
                                       target=_update_json_objs_internal_pool_wrap,
                                       args=(stats, jobj_iter_creator, update_method, pre_loader, output_path, one_file_per_process, verbose, args, kwargs))
    else:
        stats = [0, 0]
        _update_json_objs_internal(pid=0, file_paths=json_files, stats=stats, jobj_iter_creator=jobj_iter_creator, update_method=update_method, pre_loader=pre_loader,
                                   output_path=output_path, one_file_per_process=one_file_per_process, verbose=verbose, args=args, kwargs=kwargs)

    if verbose:
        hprint_pairs(("overall total", stats[0]), ("overall updated", stats[1]))


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


def op_file_with_tmp(file_path: str, op: Callable):
    tmp_file = file_path + '.tmp'
    op(tmp_file)
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


def iter_field_groups(input_file, line_group_separator='', field_separator='\t', strip_spaces_at_ends=True, ignore_empty_groups=True, min_field_list_len=0, default_field_value=''):
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


def shuffle_lines(input_file, output_file):
    lines = list(open(input_file))
    random.shuffle(lines)
    write_all_lines(iterable=lines, output_path=output_file, use_tqdm=True)


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
        ensure_dir_existence(output_dir_path)
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


def write_all_lines_for_list_iterable(output_path: str, iterable, item_concat='\n', end_of_list_sep='\n\n', use_tqdm=False):
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
            if filter is None or (isinstance(filter, list) and file in filter) or (isinstance(filter, str) and file == filter):
                zip_file.extract(member=file, path=dest_dir)


def pickle_load(file_path: str, compressed: bool = False, encoding=None):
    with open(file_path, 'rb') if not compressed else gzip.open(file_path, 'rb') as f:
        if encoding is None or sys.version_info < (3, 0):
            return pickle.load(f)
        else:
            return pickle.load(f, encoding=encoding)


def pickle_save(file_path: str, data, compressed: bool = False):
    with open(file_path, 'wb+') if not compressed else gzip.open(file_path, 'wb+') as f:
        pickle.dump(data, f)


def pickle_save__(file_or_dir_path: str, data, extension_name=None, compressed: bool = False, auto_timestamp=True, random_stamp=False, ensure_no_conflict=False, prefix=''):
    if path.isdir(file_or_dir_path):
        if auto_timestamp:
            file_or_dir_path = path.join(file_or_dir_path,
                                         ((prefix + '_') if prefix else '') + str(int(time() * 1000)) + (('_' + str(randint(0, 1000))) if random_stamp else '') + (extension_name if extension_name else '.dat'))
        else:
            raise ValueError("The specified `file_or_dir_path` is not a directory path, and `auto_timestamp` is not set `True`.")
    else:
        dir_name = path.dirname(file_or_dir_path)
        if not path.exists(dir_name):
            os.makedirs(dir_name)
        if auto_timestamp or random_stamp or prefix:
            file_name_split = path.splitext(path.basename(file_or_dir_path))
            file_name = ((prefix + '_') if prefix else '') + file_name_split[0] + (('_' + str(int(time() * 1000))) if auto_timestamp else '') + (('_' + str(randint(0, 1000))) if random_stamp else '') + file_name_split[1]
            if ensure_no_conflict:
                file_name = ensure_path_no_conflict(file_name)
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
    COMPLETION_MARK_FILE_EXTENSION = '.complete'
    REMOVAL_BACKUP_FOLDER = '.removed'

    def __init__(self, cache_dir, iter_file_ext_name='.bat', compressed=False, shuffle_file_order=False):
        super(SimpleFileCache, self).__init__()
        self.cache_dir = path.abspath(cache_dir)
        self._iter_file_ext_name = iter_file_ext_name if iter_file_ext_name[0] == '.' else '.' + iter_file_ext_name
        if self._iter_file_ext_name == SimpleFileCache.COMPLETION_MARK_FILE_EXTENSION:
            raise ValueError(f'the cache file extension name cannot be `{SimpleFileCache.COMPLETION_MARK_FILE_EXTENSION}`, which is used for cache completion mark files')

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
        return iter_files_by_pattern(self.cache_dir, pattern=pattern, full_path=True, recursive=False)

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
                os.rename(cache_file, replace_dir(cache_file, backup_dir))

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
            file_names = get_files_by_pattern(dir_or_dirs=dir_path, pattern=pattern, recursive=False)
            output = []
            for file_name in file_names:
                output.append(pickle_load(file_name, compressed=self.cache_compressed))
            return output

    def exists(self, file_name):
        return path.exists(path.join(self.cache_dir, file_name))

    def save(self, obj, file_name=None, prefix='', auto_timestamp=False, random_stamp=False, ensure_no_conflict=False):  # TODO clumsy function parameters
        if file_name:
            pickle_save__(path.join(self.cache_dir, file_name),
                          data=obj,
                          extension_name='',
                          compressed=self.cache_compressed,
                          auto_timestamp=auto_timestamp,
                          random_stamp=random_stamp,
                          ensure_no_conflict=ensure_no_conflict,
                          prefix='')
        else:
            pickle_save__(self.cache_dir,
                          data=obj,
                          extension_name=self._iter_file_ext_name,
                          compressed=self.cache_compressed,
                          auto_timestamp=True,
                          random_stamp=True,
                          ensure_no_conflict=True,
                          prefix=prefix)
# endregion

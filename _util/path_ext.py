import itertools
import shutil
import os
import re
import shutil
import warnings
from enum import IntEnum
from os import path, listdir, sep
from pathlib import Path
from sys import argv
from typing import List, Union, Iterator, Tuple

from utix.general import hprint_pairs, hprint_message, eprint_message, hprint, is_str
from utix.msgex import msg_create_dir, msg_arg_not_a_dir, msg_clear_dir, msg_arg_multi_path_not_exist
from utix.timex import timestamp
import platform
import glob
import re

REG_PATTERN_URL_OR_WINDOWS_ABS_PATH_PREFIX = r'^[a-zA-Z]+\:((\/\/)|\\)'
DEFAULT_MULTI_PATH_DELIMITER = ':'


# region basic path operations

def join__(root_dir, *parts, verbose=False):
    paths = tuple(path.join(root_dir, part) if isinstance(part, str) else path.join(root_dir, *part) for part in parts)
    if verbose:
        print_basic_path_info(paths)
    return paths


def sort_paths(paths, sort, sort_by_basename=False):
    if sort_by_basename:
        if sort is True or sort == 'alphabetic':
            return sorted(paths, key=lambda x: path.basename(x))
        elif sort == 'index':
            return sorted(paths, key=lambda x: int(re.search(r'[0-9]+', path.basename(x)).group()))
        return paths
    else:
        if sort is True or sort == 'alphabetic':
            return sorted(paths)
        elif sort == 'index':
            return sorted(paths, key=lambda x: int(re.search(r'[0-9]+', x).group()))
        return paths


def make_ext_name(ext_name: str):
    """
    Ensures the extension name `ext_name` starts with a dot.
    If `ext_name` is None or empty, then `None` is returned.
    """
    if ext_name:
        return '.' + ext_name if ext_name[0] != '.' else ext_name


def make_file_name(s: str, special_chr_rep='', space_rep='_'):
    """
    Makes a valid file name out of any given string.

    >>> s = "what's on my costo list"
    >>> make_file_name(s)
    'whats_on_my_costo_list'

    >>> make_file_name(s, special_chr_rep='_')
    'what_s_on_my_costo_list'

    >>> make_file_name(s, special_chr_rep='', space_rep='')
    'whatsonmycostolist'

    """
    if special_chr_rep == space_rep:
        return re.sub(r'[^a-zA-Z0-9_\-\.]+', special_chr_rep, s)
    else:
        return re.sub(r'\s+', space_rep, re.sub(r'[^a-zA-Z0-9_\-\.\s]+', special_chr_rep, s))


def abspath__(pathstr):
    return path.expanduser(pathstr) if pathstr[0] == '~' else path.abspath(pathstr)


def get_main_name(path_str: str) -> str:
    """
    Gets the main file name of a path. For example, the main name of path `a/b/c.d` is `c`.
    :param path_str: gets the main name of this path.
    :return: the main name of the specified path.
    """
    return path.splitext(path.basename(path_str))[0]


def get_ext_name(path_str: str) -> str:
    return path.splitext(path.basename(path_str))[1]


def get_main_name_ext_name(path_str: str) -> Tuple[str, str]:
    return path.splitext(path.basename(path_str))


def abs_path_wrt_file(file_path: str, rel_path: str):
    """
    Gets the absolute path of the relative path with respect to a file path.
    This method finds the absolute path of `rel_path` by considering it as being relative to the directory path of the `file_path`.
    If `rel_path` is already absolute, then this method simply returns `rel_path`.
    :param file_path: the `rel_path` is considered as being relative to the directory path this `file_path`.
    :param rel_path: the relative path.
    :return: the absolute path of `rel_path` with respect to the `file_path`.
    """
    if rel_path[0] == '/' or (rel_path[0] == '~' and rel_path[1] == '/') or re.match(pattern=REG_PATTERN_URL_OR_WINDOWS_ABS_PATH_PREFIX, string=file_path):
        return rel_path
    else:
        return path.abspath(path.join(path.dirname(file_path), rel_path))


def path_or_name_with_timestamp(path_or_name: str, extname: str = None, timestamp_scale=100, timestampe_sep='_', extname_sep='.'):
    """
    Appends a timestamp to the end of a path or name.

    >>> import utix.pathex as paex
    >>> paex.path_or_name_with_timestamp('a/b/c/d') # get something like 'a/b/c/d_158886811034'
    >>> paex.path_or_name_with_timestamp('a/b/c/d/') # get something like 'a/b/c/d/158886811034'

    :param path_or_name: the path or name to attach the timestamp;
                            if this ends with either `path.sep` or `path.altsep` (i.e. usually '/' or '\'), then it will be treated as you want to get a name under a directory, and an underscore `timestampe_sep` will be placed in the front of the timestamp.
    :param extname: the extension name, if necessary.
    :param timestamp_scale: a timestamp is `time() * timestamp_scale`.
    :param timestampe_sep: the separator between the name and the timestamp.
    :param extname_sep: the separator for the extension name.
    :return: the name or path with timestamp attached.
    """

    if extname:
        if extname[0] == extname_sep:
            extname = extname[1:]
        if path_or_name[-1] in (path.sep, path.altsep):
            return f'{path_or_name}{timestamp(scale=timestamp_scale)}{extname_sep}{extname}'
        else:
            return f'{path_or_name}{timestampe_sep}{timestamp(scale=timestamp_scale)}{extname_sep}{extname}'
    else:
        if path_or_name[-1] in (path.sep, path.altsep):
            return f'{path_or_name}{timestamp(scale=timestamp_scale)}'
        else:
            return f'{path_or_name}{timestampe_sep}{timestamp(scale=timestamp_scale)}'


def append_timestamp(path_str: str, timestamp_scale=100):
    return append_to_main_name(path_str, '_' + timestamp(scale=timestamp_scale))


def append_to_main_name(path_str: str, main_name_suffix: str):
    """
    Appends a suffix to the main name of a path.
    For example, `append_to_main_name('../test.csv', '_fixed')` renames the path as `../test_fixed.csv` (appending suffix `_fixed` to the main name `test` of this path).
    :param path_str: appends the suffix to the main name of this path.
    :param main_name_suffix: the suffix to append to the main name of the provided path.
    :return: a new path string with the suffix appended to the main name.
    """
    path_splits = path.splitext(path_str)
    return path_splits[0] + str(main_name_suffix) + path_splits[1]


def add_to_main_name(path_str: str, prefix: str = '', suffix: str = ''):
    dir_name = path.dirname(path_str)
    base_name = path.basename(path_str)
    path_splits = path.splitext(base_name)
    return path.join(dir_name, str(prefix) + path_splits[0] + str(suffix) + path_splits[1])


def replace_ext(file_path: str, new_ext: str, main_name_suffix=''):
    if new_ext[0] != '.':
        new_ext = '.' + new_ext
    return path.splitext(file_path)[0] + main_name_suffix + new_ext


def replace_dir(file_path: str, new_dir_path: str):
    return path.join(new_dir_path, path.basename(file_path))


def check_path_existence(*path_or_paths):
    has_error = False
    for item in path_or_paths:
        if not path.exists(item):
            eprint_message("path not found", item)
            has_error = True
    if not has_error:
        hprint_message("Path existence check passed!")


def ensure_parent_dir_existence(*dir_path_or_paths, clear_dir=False, verbose=__debug__):
    ensure_dir_existence(*(path.dirname(p) for p in dir_path_or_paths), clear_dir=clear_dir, verbose=verbose)
    return dir_path_or_paths[0] if len(dir_path_or_paths) == 1 else dir_path_or_paths


def ensure_dir_existence(*dir_path_or_paths, clear_dir=False, verbose=__debug__):
    """
    Creates a directory if the path does not exist. Optionally, set `clear_dir` to `True` to clear an existing directory.

    >>> import utix.pathex as pathx
    >>> import os
    >>> path1, path2 = 'test/_dir1', 'test/_dir2'
    >>> pathx.print_basic_path_info(path1)
    >>> pathx.print_basic_path_info(path2)

    Pass in a single path.
    ----------------------
    >>> pathx.ensure_dir_existence(path1)
    >>> os.remove(path1)

    Pass in multiple paths.
    -----------------------
    >>> pathx.ensure_dir_existence(path1, path2)
    >>> os.remove(path1)
    >>> os.remove(path2)

    Pass in multiple paths as a tuple.
    ----------------------------------
    >>> # this is useful when this method is composed with another function that returns multiple paths.
    >>> def foo():
    >>>     return path1, path2
    >>> pathx.ensure_dir_existence(foo())

    :param dir_path_or_paths: one or more paths to check.
    :param clear_dir: clear the directory if they exist.
    :return: the input directory paths; this function has guaranteed their existence.
    """
    if len(dir_path_or_paths) == 1 and not isinstance(dir_path_or_paths[0], str):
        dir_path_or_paths = dir_path_or_paths[0]

    for dir_path in dir_path_or_paths:
        if not path.exists(dir_path):
            if verbose:
                hprint(msg_create_dir(dir_path))
            os.makedirs(dir_path)
        elif not path.isdir(dir_path):
            raise ValueError(msg_arg_not_a_dir(path_str=dir_path, arg_name='dir_path_or_paths'))
        elif clear_dir is True:
            if verbose:
                hprint(msg_clear_dir(dir_path))
            shutil.rmtree(dir_path)
            os.makedirs(dir_path)
        elif is_str(clear_dir) and bool(clear_dir):
            for file in iter_files_by_pattern(dir_or_dirs=dir_path, pattern=clear_dir, recursive=False):
                os.remove(file)

        if verbose:
            print_basic_path_info(dir_path)

    return dir_path_or_paths[0] if len(dir_path_or_paths) == 1 else dir_path_or_paths


def ensure_path_no_conflict(path_to_solve: str, name_idx_delimiter='_'):
    if not path.exists(path_to_solve):
        return path_to_solve

    idx = 1
    main_name, ext_name = path.splitext(path_to_solve)
    path_to_solve = path.join(f"{main_name}{name_idx_delimiter}{idx}{ext_name}")

    while path.exists(path_to_solve):
        idx += 1
        path_to_solve = path.join(f"{main_name}{name_idx_delimiter}{idx}{ext_name}")

    return path_to_solve


# endregion

def print_basic_path_info(*path_or_paths):
    for item in path_or_paths:
        if isinstance(item, str):
            hprint_pairs(("path", item), ("is file", path.isfile(item)), ("exists", path.exists(item)))
        else:
            hprint_pairs((item[0], item[1]), ("is file", path.isfile(item[1])), ("exists", path.exists(item[1])))


def iter_all_sub_dirs(dir_path: str):
    return map(lambda x: x[0], os.walk(dir_path))


def get_all_sub_dirs(dir_path: str) -> List[str]:
    return [x[0] for x in os.walk(dir_path)]


def iter_all_immediate_sub_dirs(dir_path: str, full_path=False):
    for name in os.listdir(dir_path):
        _full_path = os.path.join(dir_path, name)
        if os.path.isdir(_full_path):
            yield _full_path if full_path else name


def iter_paired_files(dir_path, main_file_reg_pattern, paired_file_format_pattern, sort=True):
    for file_name in iter_files(dir_path=dir_path, full_path=False, sort=sort):
        match = re.search(main_file_reg_pattern, file_name)
        if match is not None and match.group(0) == file_name:
            yield path.join(dir_path, file_name), path.join(dir_path, paired_file_format_pattern.format(match.group(1)))


class FullPathMode(IntEnum):
    RelativePath = 0
    FullPath = 1
    BaseName = 2
    FullPathRelativePathTuple = 3


def iter_files(dir_path: str, full_path: bool = True, sort=True):
    """
    Iterate through the paths or file names of all files in a folder (NOT including its sub-folders) at path `dir_path`.
    :param dir_path: the path to the folder.
    :param full_path: `True` if the full path to each file should be returned; otherwise, only the file name will be returned.
    :return: an iterator that returns each file in a folder at path `dir_path`, NOT including files in the sub-folders.
    """
    if sort == 'glob':
        files = glob.glob(dir_path + '/*')
        if full_path is True or full_path == FullPathMode.FullPath:
            yield from (x for x in files if path.isfile(x))
        else:
            yield from (path.basename(x) for x in files if path.isfile(x))
    else:
        files = sort_paths(listdir(dir_path), sort=sort, sort_by_basename=False)  # NOTE `listdir` already returns basenames; therefore the `use_basename` is set `False` here.
        if full_path is True or full_path == FullPathMode.FullPath:
            for f in files:
                file = path.join(dir_path, f)
                if path.isfile(file):
                    yield file
        else:
            for f in files:
                if path.isfile(path.join(dir_path, f)):
                    yield f


def get_latest_file(dir_path: str, pattern: str = '*'):
    files = glob.glob(path.join(dir_path, pattern))
    return max(filter(path.isfile, files), key=path.getctime)


def iter_files_by_pattern(dir_or_dirs: str, pattern: str = '*', full_path: Union[FullPathMode, bool] = True, recursive=True):
    """
    Iterate through the paths or file names of all files in a folder at path `dir_path` of a specified `pattern`.
    :param dir_or_dirs: the path to the folder.
    :param pattern: only iterate through files of this pattern, e.g. '*.json'; a pattern starts with `**/` indicates to recursively search all sub folders.
    :param full_path: `True` if the full path to each file should be returned; otherwise, only the file name will be returned.
    :param recursive: `True` if recursively searching all sub folders, equivalent to adding prefix `**/` in front of `pattern`; `False` has no actual effect.
    :return: an iterator that returns each file of the specified pattern in a folder at path `dir_path`.
    """

    def _iter_files(dir_path):
        nonlocal pattern
        if path.isdir(dir_path):
            if recursive and not pattern.startswith('**/'):
                pattern = '**/' + pattern
            if full_path is True or full_path == FullPathMode.FullPath:
                p = Path(path.abspath(dir_path))
                for f in p.glob(pattern):
                    if path.isfile(f):
                        yield str(f)
            elif full_path is False or full_path == FullPathMode.BaseName:
                p = Path(path.abspath(dir_path))
                for f in p.glob(pattern):
                    if path.isfile(f):
                        yield path.basename(f)
            elif full_path == FullPathMode.RelativePath:
                dir_path = path.abspath(dir_path)
                len_dir_path = len(dir_path)
                p = Path(path.abspath(dir_path))
                for f in p.glob(pattern):
                    if path.isfile(f):
                        f = str(f)
                        yield f[len_dir_path + 1:] if f[len_dir_path] == os.sep else f[len_dir_path:]
            elif full_path == FullPathMode.FullPathRelativePathTuple:
                dir_path = path.abspath(dir_path)
                len_dir_path = len(dir_path)
                p = Path(path.abspath(dir_path))
                for f in p.glob(pattern):
                    if path.isfile(f):
                        f = str(f)
                        yield (f, f[len_dir_path + 1:]) if f[len_dir_path] == os.sep else (f, f[len_dir_path:])

    if isinstance(dir_or_dirs, str):
        yield from _iter_files(dir_or_dirs)
    else:
        for dir_path in dir_or_dirs:
            yield from _iter_files(dir_path)


def get_files_by_pattern(dir_or_dirs: Union[str, Iterator], pattern: str = '*', full_path: Union[FullPathMode, bool] = True, recursive=True, sort=False, sort_use_basename=False):
    """
    Gets the paths or file names of all files in a folder at path(s) specified by `dir_or_dirs` of a specified `pattern`.
    :param dir_or_dirs: the path to one or more folders where this method is going to search for files of the provided pattern.
    :param pattern: search for files of this pattern, e.g. '*.json'; a pattern starts with `**/` indicates to recursively search all sub folders, or equivalently set the parameter `recursive` as `True`.
    :param full_path: `True` if the full path to each file should be returned; otherwise, only the file name will be returned.
    :param recursive: `True` if recursively searching all sub folders, equivalent to adding prefix `**/` in front of `pattern`; otherwise, `False`.
    :param sort: `True` if to sort the returned file paths (by their strings); otherwise, `False`.
    :return: a list of file paths of the specified file name pattern in the folder or folders specified in `dir_or_dirs`.
    """

    def _proc1(f, len_dir_path):
        f = str(f)
        return f[len_dir_path + 1:] if f[len_dir_path] == os.sep else f[len_dir_path:]

    def _proc2(f, len_dir_path):
        f = str(f)
        return (f, f[len_dir_path + 1:]) if f[len_dir_path] == os.sep else (f, f[len_dir_path:])

    def _get_files(dir_path):
        nonlocal pattern
        if path.isdir(dir_path):
            if recursive and not pattern.startswith('**/'):
                pattern = '**/' + pattern

            if full_path is True or full_path == FullPathMode.FullPath:
                p = Path(path.abspath(dir_path))
                results = [str(f) for f in p.glob(pattern) if path.isfile(f)]
            elif full_path is False or full_path == FullPathMode.BaseName:
                p = Path(dir_path)
                results = [path.basename(f) for f in p.glob(pattern) if path.isfile(f)]
            elif full_path == FullPathMode.RelativePath:
                dir_path = path.abspath(dir_path)
                p = Path(dir_path)
                len_dir_path = len(dir_path)
                results = [_proc1(f, len_dir_path) for f in p.glob(pattern) if path.isfile(f)]
            elif full_path == FullPathMode.FullPathRelativePathTuple:
                dir_path = path.abspath(dir_path)
                p = Path(dir_path)
                len_dir_path = len(dir_path)
                results = [_proc2(f, len_dir_path) for f in p.glob(pattern) if path.isfile(f)]
            return sort_paths(results, sort=sort, sort_by_basename=sort_use_basename)

    return _get_files(dir_or_dirs) if isinstance(dir_or_dirs, str) else sum([_get_files(dir_path) for dir_path in dir_or_dirs], [])


def exists_files_by_pattern(dir_path: str, pattern: str, recursive=True):
    if path.isdir(dir_path):
        if recursive and not pattern.startswith('**/'):
            pattern = '**/' + pattern
        p = Path(dir_path)
        try:
            f = str(next(p.glob(pattern)))
            return bool(f)
        except:
            return False


def get_sorted_files_from_all_sub_dirs(dir_path: str, pattern: str, full_path: bool = True):
    files = []
    sub_dirs = get_all_sub_dirs(dir_path)
    sub_dirs.sort()
    for sub_dir in sub_dirs:
        sub_dir_files = get_files_by_pattern(dir_or_dirs=sub_dir, pattern=pattern, full_path=full_path, recursive=False)
        sub_dir_files.sort()
        files.extend(sub_dir_files)
    return files


def get_sorted_files_from_all_sub_dirs__(target_path: str, full_path: bool = True):
    if path.isfile(target_path):
        embeds_file_paths = [target_path]
    elif path.isdir(target_path):
        embeds_file_paths = get_sorted_files_from_all_sub_dirs(dir_path=target_path, pattern='*', full_path=full_path)
    else:
        embeds_file_paths = get_sorted_files_from_all_sub_dirs(dir_path=path.dirname(target_path), pattern=path.basename(target_path), full_path=full_path)
    return embeds_file_paths


def get_files(dir_path: str, full_path: bool = True):
    return list(iter_files(dir_path=dir_path, full_path=full_path))


def replace_path_part(pathstr: str, replacement: str, part_index, sep=path.sep):
    path_parts = pathstr.split(sep)
    path_parts[part_index] = replacement
    return sep.join(path_parts)


def replace_path_tail(pathstr: str, replacement: str, seg_sep=sep):
    """
    Replace the final segments in a path.
    For example, if `seg_sep` is '/', `path_str` is `a/b/c/d`, and `replacement` is `e/f` (`/e/f` is also fine), then the returned path is `a/b/e/f`.
    :param pathstr: the path string.
    :param replacement: replacement for the final segments.
    :param seg_sep: the path separator.
    :return: a new path string with the final segments replaced.
    """
    if replacement[0] == seg_sep:
        replacement = replacement[1:]
    rep_parts = replacement.split(seg_sep)
    path_parts = pathstr.split(seg_sep)
    return seg_sep.join(path_parts[:-len(rep_parts)] + rep_parts)


def get_path_tail(pathstr: str, num_parts: int, seg_sep=sep):
    return seg_sep.join(pathstr.split(seg_sep)[-num_parts:])


def get_path_head(pathstr: str, num_parts: int, seg_sep=sep):
    return seg_sep.join(pathstr.split(seg_sep)[:num_parts])


def add_subfolder(file_path: str, subfolder: str):
    data_dir, basename = path.split(file_path)
    return path.join(data_dir, subfolder, basename)


def add_subfolder_and_replace_ext(file_path: str, subfolder: str, new_ext: str):
    data_dir, basename = path.split(file_path)
    return path.join(data_dir, subfolder, path.splitext(basename)[0] + new_ext)


def path_fix(path_to_fix: str):
    if path.exists(path_to_fix):
        return path_to_fix
    fixed_path = path.join('.', path_to_fix)
    return fixed_path if path.exists(fixed_path) else path_to_fix


def script_folder():
    return path.dirname(path.abspath(argv[0]))


def insert_dir(path_str: str, dir_name: str, index: int, path_sep=sep):
    path_components = path_str.split(path_sep)
    path_components.insert(index, dir_name)
    return path_sep.join(path_components)


def file_count(dir_path: str):
    return len([name for name in listdir(dir_path) if path.isfile(path.join(dir_path, name))]) if path.exists(dir_path) \
        else 0


def clean_file_name(file_name: str, rep='_'):
    return re.sub(r'[^\w\-_\. ]', rep, file_name)


def solve_abs_path_with_root_path(root_path, rel_path):
    if rel_path[0] == '.':
        if len(rel_path) == 1:
            return root_path
        elif rel_path[1] == os.sep:
            return path.abspath(path.join(root_path, rel_path[2:]))
        elif rel_path[1] == '.':
            return path.abspath(path.join(root_path, rel_path))
    elif rel_path[0] != os.sep:
        return path.abspath(path.join(root_path, rel_path))
    else:
        return rel_path


# region batch file operations

def batch_join(*paths, remove_non_exists=False):
    if len(paths) == 1:
        return paths[0]

    joins = [path.join(*ps) for ps in itertools.product(([p] if isinstance(p, str) else p) for p in paths)]
    return [p for p in joins if path.exists(p)] if remove_non_exists else joins


# endregion

# region multi-path

def _solve_multi_path(multi_path_str, file_pattern=None, multi_path_delimiter=DEFAULT_MULTI_PATH_DELIMITER, sort=True, verbose=__debug__):
    if verbose:
        hprint_message('solving multi-file paths from input', multi_path_str)

    # region STEP1: get all paths

    # split the path by the subdir delimiter; special treatment for Windows system.
    input_paths = [file_or_dir_path for file_or_dir_path in multi_path_str.split(multi_path_delimiter) if file_or_dir_path]
    if platform.system() == 'Windows' and multi_path_delimiter == ':' and len(input_paths[0]) == 1 and input_paths[0].isalpha() and input_paths[1][0] == '\\':
        input_paths = [f'{input_paths[0]}:{input_paths[1]}'] + input_paths[2:]

    # replace the final segments of the first path to generate all actual subdir/file paths.
    for i in range(1, len(input_paths)):
        input_paths[i] = replace_path_tail(input_paths[0], input_paths[i])

    # region STEP2: check the path existence.
    path_exists = [False] * len(input_paths)
    has_available_path = False
    for path_idx, possible_path in enumerate(input_paths):
        path_exists[path_idx] = path.exists(possible_path)
        if path_exists[path_idx]:
            has_available_path = True
        if verbose:
            hprint_pairs(('path', possible_path), ('exists', path_exists[path_idx]))
    # endregion

    # region STEP3: if the `file_pattern` is specified, then expand each existing dir path as files that match the provided pattern.
    if file_pattern:
        expanded_input_paths = []
        expanded_path_exists = []
        for input_path, path_exist in zip(input_paths, path_exists):
            if path_exist and path.isdir(input_path):
                files = get_files_by_pattern(input_path, file_pattern)
                if files:
                    expanded_input_paths.extend(files)
                    expanded_path_exists.extend([True] * len(files))
                    has_available_path = True
                    if verbose:
                        hprint_pairs(('extending path', input_path), ('pattern', file_pattern), ('num found files', len(files)))
            else:  # ! keeps the original path if 1) it does not exist; 2) it is a file.
                expanded_input_paths.append(input_path)
                expanded_path_exists.append(path_exist)

        if len(expanded_input_paths) == 0:
            warnings.warn(f"File pattern '{file_pattern}' specified, but no file of this pattern is found.")

        input_paths = expanded_input_paths
        path_exists = expanded_path_exists

    # endregion

    # returns the solved paths, their existence flags, and a single boolean value indicating if any of the path exists.
    if sort:
        input_paths, path_exists = zip(*sorted(zip(input_paths, path_exists)))
    return input_paths, path_exists, has_available_path


def solve_multi_path(multi_path: Union[str, Iterator[str]], file_pattern=None, multi_path_delimiter=DEFAULT_MULTI_PATH_DELIMITER, sort=True, verbose=__debug__, raise_error_if_none_of_the_paths_exist=False, remove_non_existent_paths=False):
    """
    Solves a multi-path of format like `root_dir/sub_path1:sub_path2:sub_path3:...`. The sub paths can point to either directories or files, and can be mixed like `root_dir/sub_dir1:sub_dir2:sub_file1:sub_file2:sub_dir3:...`.
    If `file_pattern` is not specified, then the returned paths will be `root_dir/sub_dir1`, `root_dir/sub_dir2`, `root_dir/sub_file1`, `root_dir/sub_file2`, `root_dir/dir3`, ...
    Otherwise, this method will search for files of the specified `file_pattern` in all directory paths that exist in the file system, and replace these directory paths by paths to the found files.

    >>> import os
    >>> from os import path
    >>> import shutil
    >>> import utix.pathex as pathex
    >>> test_root = path.join('.', 'tmp_root')
    >>> if os.path.exists(test_root):
    >>>     shutil.rmtree(test_root)
    >>> tmp_dir1 = path.join(test_root, 'tmp1')
    >>> tmp_dir2 = path.join(test_root, 'tmp2')
    >>> tmp_dir3 = path.join(test_root, 'tmp3')
    >>> os.makedirs(tmp_dir1)
    >>> os.makedirs(tmp_dir2)
    >>> open(path.join(tmp_dir2, '1.json'), 'w').close()
    >>> open(path.join(tmp_dir2, '2.json'), 'w').close()
    >>> multi_path = f'{tmp_dir1}:tmp2:tmp3'
    >>> pathex.solve_multi_path(multi_path=multi_path)
    solving multi-file paths from input: .\tmp_root\tmp1:tmp2:tmp3
    path: .\tmp_root\tmp1 exists: True
    path: .\tmp_root\tmp2 exists: True
    path: .\tmp_root\tmp3 exists: False
    >>> pathex.solve_multi_path(multi_path=multi_path, file_pattern='*.json')
    solving multi-file paths from input: .\tmp_root\tmp1:tmp2:tmp3
    path: .\tmp_root\tmp1 exists: True
    path: .\tmp_root\tmp2 exists: True
    path: .\tmp_root\tmp3 exists: False
    extending path: .\tmp_root\tmp2 pattern: *.json num found files: 2
    >>> pathex.solve_multi_path(multi_path=multi_path, file_pattern='*.json', remove_non_existent_paths=True)
    >>> shutil.rmtree(test_root)

    :param multi_path: the string(s) for the multi-path.
    :param file_pattern: the pattern of the files to search in each directory paths that exist in the file system.
    :param multi_path_delimiter: the character that separates sub paths.
    :param sort: `True` if to sort the returned paths; otherwise `False`.
    :param verbose: `True` if to print out internal message; otherwise `False`.
    :param raise_error_if_none_of_the_paths_exist: `True` to raise a `Value` error if none of the paths specified by `multi_path_str` exists.
    :param remove_non_existent_paths: `True` to remove non-existent paths from the return; setting this to `True` also makes this method return only the file paths. 
    :return: If `remove_non_exist_paths` is `False`, then it returns a three-tuple: 
                1) a list of path strings solved from the provided `multi_path_str`; 
                2) a list of Booleans of the same size as the returned paths, indicating if each path exists; 
                3) a Boolean value indicating if any path is available.
             If `remove_non_exist_paths` is `True`, then just returns the existent paths; or `None` if no path exsits.
    
    """
    # A frequent scenario during data pre-processing is that we need different files or different sub directories.
    # This method solves paths like `.../subdir1:subdir2:subdir3` or `.../file1:file2:file3`, so that you could input multiple subdir/file paths at the same time.

    if isinstance(multi_path, str):
        resolved_paths, path_exists, has_available_path = _solve_multi_path(multi_path, file_pattern=file_pattern, multi_path_delimiter=multi_path_delimiter, sort=sort, verbose=verbose)
    else:
        resolved_paths, path_exists, has_available_path = [], [], []

        for multi_path_str_ in multi_path:
            _resolved_paths, _path_exists, _has_available_path = _solve_multi_path(multi_path_str_, file_pattern=file_pattern, multi_path_delimiter=multi_path_delimiter, sort=sort, verbose=verbose)
            resolved_paths.append(_resolved_paths)
            path_exists.append(_path_exists)

        resolved_paths, path_exists, has_available_path = sum(resolved_paths, []), sum(path_exists, []), any(has_available_path)

    if raise_error_if_none_of_the_paths_exist and not has_available_path:
        raise ValueError(msg_arg_multi_path_not_exist(multi_path, 'multi_path'))

    if remove_non_existent_paths:
        return [resolved_path for resolved_path, path_exist in zip(resolved_paths, path_exists) if path_exist] if has_available_path else None
    else:
        return resolved_paths, path_exists, has_available_path

# endregion

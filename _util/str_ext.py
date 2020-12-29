import hashlib
import re
import string
import math
import warnings
from collections import defaultdict
import ast
# region misc
from typing import Tuple, Callable, Dict

from utix.general import is_str, try____, str2bool, str2bool__, str2num, str2val__
from utix.general import iter__
from utix.external.parse import parse


def find_nth(_s: str, sub: str, n: int, start: int = None, end: int = None):
    start = _s.find(sub, start, end)
    while start >= 0 and n > 1:
        start = _s.find(sub, start + len(sub), end)
        n -= 1
    return start


def find_nth_overlap(_s: str, sub: str, n: int, start: int = None, end: int = None):
    start = _s.find(sub, start, end)
    while start >= 0 and n > 1:
        start = _s.find(sub, start + 1, end)
        n -= 1
    return start


def find_whitespace_nth(_s: str, n: int, start: int = 0, end: int = None, ignore_consecutive_whitespaces=False):
    if end is None:
        end = len(_s)
    if not ignore_consecutive_whitespaces:
        while start < end:
            if _s[start].isspace():
                n -= 1
                if n == 0:
                    return start
                start += 1
    else:
        prev_space_flag = False
        while start < end:
            if _s[start].isspace():
                if not prev_space_flag:
                    n -= 1
                    if n == 0:
                        return start
                prev_space_flag = True
            else:
                prev_space_flag = False
            start += 1
    return -1


def add_prefix(prefix: str, s: str, sep='_'):
    return (prefix + sep + s) if prefix else s


def add_suffix(suffix: str, s: str, sep='_'):
    return (s + sep + suffix) if suffix else s


def positional_extract(s: str, sep: str, n: int, filter: Callable = None, sentinel: Callable = None):
    parts = tuple(iter__(s.split(sep), filter=filter, sentinel=sentinel))
    num_parts = len(parts)
    if num_parts > n:
        return parts[:n]
    elif num_parts == n:
        return parts
    else:
        return parts + ((None,) * (n - num_parts))


def cut(s: str, cut_before=None, cut_after=None):
    if cut_before is not None:
        try:
            s = s[(s.index(cut_before) + 1):]
        except ValueError:
            pass
    if cut_after is not None:
        try:
            s = s[:s.rindex(cut_after)]
        except ValueError:
            pass
    return s


# region split

def split__(s: str, sep: str = None, maxsplit: int = -1, n: int = None, pad=None, remove_empty_split: bool = False, parse: bool = False, lstrip: bool = False, rstrip: bool = False, cut_before=None, cut_after=None):
    """
    String split with rich options.

    >>> from utix.strex import split__

    Use the `n` parameter to enforce the return has the specified size.
    -------------------------------------------------------------------
    >>> r1, r2 = split__('a, b, c', sep=',', n=2, lstrip=True, rstrip=True)
    >>> print(r1 == 'a', r2 == 'b')
    >>> r1, r2 = split__('a', sep=',', n=2, lstrip=True, rstrip=True)
    >>> print(r1 == 'a', r2 is None)

    Remove empty splits.
    --------------------
    >>> print(split__('a, b, ,,, c, d, ,, e', sep=',', remove_empty_split=True, lstrip=True, rstrip=True))

    Value parsing.
    --------------
    >>> print(split__('1,2,3,4', sep=',', parse=True) == [1, 2, 3, 4])
    >>> print(split__('1\t2\t[3,4,5,6]\t{"a":7, "b":8}', sep='\t', parse=True) == [1, 2, [3, 4, 5, 6], {'a': 7, 'b': 8}])

    :param s: the string to split.
    :param sep: the delimiter according which to split the string; `None` (the default value) means split according to any whitespace, and discard empty strings from the result.
    :param maxsplit: maximum number of splits to do; -1 (the default value) means no limit.
    :param n: the size of the returned split tuple; if the actual number of splits is less, `pad` will be appended to the end.
    :param pad: padding the results if `n` is larger than the number of actual splits.
    :param remove_empty_split: `True` to remove empty splits; otherwise `False`.
    :param parse: `True` to parse every split as its likely value.
    :param lstrip: `True` to left-strip each split; otherwise `False`.
    :param rstrip: `True` to right-strip each split; otherwise `False`.
    :return:
    """
    s = cut(s, cut_before, cut_after)

    splits = s.split(sep, maxsplit=maxsplit)

    def _process(x: str):
        x = strip__(x, lstrip=lstrip, rstrip=rstrip)
        if x:
            if parse:
                val, succ = str2val__(x, success_label=True)
                return val if succ else x
            else:
                return x
        else:
            return x

    if parse:
        splits = [_process(x) for x in splits]
    elif lstrip or rstrip:
        splits = [strip__(x, lstrip=lstrip, rstrip=rstrip) for x in splits]
    if remove_empty_split:
        splits = [x for x in splits if x]

    if n:
        if len(splits) < n:
            splits += ((pad,) * (n - len(splits)))
        elif len(splits) > n:
            splits = splits[:n]
    return splits


def iter_split_tups(s: str, item_sep='_', kv_sep='@', lstrip_key=True, rstrip_key=True, lstrip_val=True, rstrip_val=True, conversion: Dict[str, Callable] = None, none_strs=('none', 'null')):
    def _map(item):
        k, v = bisplit(item, kv_sep)
        k = strip__(k, lstrip=lstrip_key, rstrip=rstrip_key)
        if v is not None:
            v = strip__(v, lstrip=lstrip_val, rstrip=rstrip_val)
        if v in none_strs:
            v = None
        elif conversion is not None and k in conversion:
            v = conversion[k](v)
        return k, v

    return map(_map, s.split(item_sep))

    # endregion


def strip__(s: str, lstrip: bool, rstrip: bool, chars=None) -> str:
    if lstrip and rstrip:
        return s.strip(chars)
    elif lstrip:
        return s.lstrip(chars)
    else:
        return s.rstrip(chars)


def literal_eval(s: str, verbose=__debug__):
    try:
        return ast.literal_eval(s)
    except ValueError as err:
        if verbose and any((x in s for x in ('"', '(', ')', '[', ']', '{', '}'))):
            warnings.warn(f"evaluation of `{s}` failed with message `{str(err)}`; the literal string is returned; make sure this is desired")
        return s


# endregion

# region conversion

def to_int(s, value_for_nan: int = 0) -> int:
    return value_for_nan if s == 'nan' or (isinstance(s, float) and math.isnan(s)) else int(s)


def to_bool(s, value_for_nan: bool = False) -> bool:
    return value_for_nan if s == 'nan' or (isinstance(s, float) and math.isnan(s)) else bool(s)


# endregion

# region split

def bisplit(s: str, sep: str):
    splits = s.split(sep, maxsplit=1)
    return (splits[0], None) if len(splits) == 1 else splits


def birsplit(s: str, sep: str):
    splits = s.rsplit(sep, maxsplit=1)
    return (splits[0], None) if len(splits) == 1 else splits


def split_with_trim(s: str, sep: str, strip_begin=False, strip_end=False, maxsplit: int = -1):
    if strip_begin and strip_end:
        return [ss.strip() for ss in s.split(sep=sep, maxsplit=maxsplit)]
    elif strip_begin:
        return [ss.lstrip() for ss in s.split(sep=sep, maxsplit=maxsplit)]
    elif strip_end:
        return [ss.rstrip() for ss in s.split(sep=sep, maxsplit=maxsplit)]
    else:
        return s.split(sep=sep, maxsplit=maxsplit)


def split_by_chars_with_trim(s: str, sep_chars: str, trim_start=False, trim_end=False, maxsplit: int = -1):
    """
    Splits the string `s` by one or more chars. Allows specifying options to strip the begging or/and the end of each split.
    :param s: the string to split.
    :param sep_chars: the chars used to separate the provided string.
    :param trim_start: `True` if the beginning of each split must be stripped; otherwise `False`.
    :param trim_end: `True` if the end of each split must be stripped; otherwise `False`.
    :param maxsplit: maximum number of splits to do. -1 (the default value) means no limit.
    :return: a list of the substrings in the string `s`, using `sep_chars` as the separation chars.
    """

    splits = s.split(sep=sep_chars, maxsplit=maxsplit) if len(sep_chars) == 1 or (sep_chars[0] == '\\' and len(sep_chars) == 2) \
        else re.split(make_re(sep_chars), string=s, maxsplit=maxsplit)

    if trim_start and trim_end:
        return [s.strip() for s in splits]
    elif trim_start:
        return [s.lstrip() for s in splits]
    elif trim_end:
        return [s.rstrip() for s in splits]
    else:
        return splits


# endregion

# region starts/ends with

def first_startswith(src: str, targets) -> str:
    return next(x for x in targets if src.startswith(x))


def first_endswith(src: str, targets) -> Tuple[int, str]:
    return next(x for x in targets if src.endswith(x))


# endregion


# region regular expressions
REGEX_SPECIAL_CHRS = set('.*?+\|-()[]^$:={}<>#')
REGEX_MATCH_ALL_PUNCTUATION_EXCEPT_FOR_HYPHEN = r'[^\P{P}-]+'


def make_re(s: str) -> str:
    """
    Converts the provided string `s` to a valid regular expression. Every regular expression special character in `s` will be escaped with a prefix `\\`, except for `\\` itself.
    For example, '-_' will be converted to '\\-_' because '-' can be used as a special character in the regular expression.
    For example, '\\t' will stay as '\\t' because the backslash `\\` will not be escaped.
    :param s: the string to be made a valid regular expression.
    :return: a string equivalent to the provided `s` when used as a regular expression.
    """
    output = []
    escaped = False
    for c in s:
        if escaped:
            output.append(c)
            escaped = False
        elif c == '\\':
            output.append('\\')
            escaped = True
        elif c in REGEX_SPECIAL_CHRS:
            output.append('\\')
            output.append(c)
        else:
            output.append(c)
    return ''.join(output)


def join_re(*regexps: str, compiled=False, compile_flags: re.RegexFlag = 0, whole_string=False):
    regex = '|'.join((f'({x})' for x in regexps))
    if whole_string:
        regex = fr'({regex})\Z'
    return re.compile(regex, flags=compile_flags) if compiled else regex


# endregion

def differ_bag_of_splits(text1: str, text2: str, sep=None, de_duplicate=True):
    if not bool(sep):
        split1 = text1.split()
        split2 = text2.split()
    else:
        split1 = text1.split(sep)
        split2 = text2.split(sep)

    if de_duplicate:
        return set(split2).symmetric_difference(set(split1))
    else:
        split_dict = defaultdict(int)
        for s in split1:
            split_dict[s] += 1
        for s in split2:
            split_dict[s] -= 1
        output = []
        for k, v in split_dict.items():
            if v != 0:
                output += [k] * abs(v)
        return output


def get_1st_int_from_str(s: str) -> str:
    """
    Gets the first integer in the string. For example, the first integer in string 'I have 15 apples and 20 pears.' is '15'. The string of the integer is returned.
    :param s: the string to search for an integer.
    :return: a substring of the input string `s` which represents the first integer found in `s`.
    """
    return re.findall(r'\d+', s)[0]


def hash_str(s: str):
    return int(hashlib.sha1(s.encode('utf-8')).hexdigest(), 16) % (10 ** 16)


def split_by_cap_letters(str_to_split: str, skip_consecutive_cap_letters=False):
    """
    Splits the string by capital letters in it. For example, "PlayMusic" will be split as "Play" and "Music".
    :param str_to_split: the string to split.
    :return:
    """
    if str_to_split.isupper():
        return [str_to_split]
    results = re.findall(r'[A-Z]+[^A-Z]+' if skip_consecutive_cap_letters else r'[A-Z][^A-Z]*', str_to_split)
    return results if results else [str_to_split]


def remove_punctuation_except_for_hyphen(text: str) -> str:
    text = re.sub(pattern=REGEX_MATCH_ALL_PUNCTUATION_EXCEPT_FOR_HYPHEN, repl=' ', string=text)
    return re.sub(pattern=r'\s+', repl=' ', string=text).strip()


def remove_punctuation_except_for_hyphen2(text: str):
    translate_table = {ord(char): None for char in string.punctuation if char != '-'}
    text = text.translate(translate_table)
    return ' '.join(text.split())


def re_clip_left(string: str, pattern: str, include_match: True):
    """
    Clips a substring from the given `string`, starting from the first character up to the position (or end position if `include_match` is `True`) of the first pattern match.
    :param string: the string to clip.
    :param pattern: the pattern to match.
    :param include_match: `True` if the matched text is included at the end of the returned substring.
    :return: a substring from the beginning of the given `string` up to the first pattern match.
    """
    re_match = re.search(pattern=pattern, string=string)
    return (string[:re_match.start()] + re_match.group(0)) if include_match else string[:re_match.start()]


def clean_int_str(num):
    num = int(num)
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


# region string filtering


class StringFilter:
    """
    Represents a collection of string patterns.
    You can test if a string matches any of these patterns by the `in` operation.
    If the `regexp_only` flag is turned on for the `__init__` function, then a string pattern can be of five types, indicated by the first character,
    1) contains: indicated by `*`; for example, pattern `*abc` means the string must contains a substring 'abc';
    2) startswith: indicated by `^`; for example, pattern `^abc` means the string must start with the substring 'abc';
    3) endswith: indicated by `$`; for example, pattern `$abc` means the string must end with the substring 'abc';
    4) regular expression: indicated by `@`; for example, pattern `@a.*c` means the string must be a whole match to the regular expression 'a.*c'.
        NOTE must put `@` as the first character to indicate this is a regular expression; `a.*c` will be treated as a literal as in 5), unless you set `regexp_only` flag as `True` for the `__init__` function.
    5) literal: for any pattern not starting with `*`, `^`, `$`, `@`, then the pattern will be treated as a literal; for example, `abc` means the string must be exactly 'abc';
                    a special case is a string pattern starting with a backslash '\', where the backslash will be discarded and the remaining will be a literal;
                    for example, `\\@a.*c` will exactly match a whole string 'a.*c' rather than being parsed a a regular expression.

    If the `regexp_only` flag is turned on for the `__init__` function, then only regular expressions are accepted, and no `@` is necessary.
    For example, `a.*c` will now be treated as a regular expression, and a string like `aaac` will be 'in' this filter.

    .. code-block:: python
    sfilter = StringFilter('@a.*C', 'def', '$gh')
    print('abc' in sfilter) # True
    print('ac' in sfilter) # True
    print('def' in sfilter) # True
    print('ghi' in sfilter) # False
    print('abcd' in sfilter) # False; NOTE only matches the whole string, even the substring 'abc' is in this filter

    sfilter = StringFilter(r'@[2-9]\d{2}-\d{3}-\d{4}', r'@\d{5}')
    print(19121 in sfilter) # True; you can pass any object here; as long as its string representation matches any pattern, `True` will be returned.
    print('212-334-2134' in sfilter) # True
    """

    def __init__(self, *filters: str, regexp_only=False):
        """
        :param filters: a sequence of strings as filter patterns.
        :param regexp_only: `True` if the filter patterns are regular expressions;
                            `False` if to also accept convenient patterns starting with `*` (contains), `^` (startswith), `$` (endswith), where now regular expressions must start with `@`, and all others are treated as literals.
        """
        if regexp_only:
            self._re = join_re(*filters, compiled=True, whole_string=True)
        else:
            regexps = []
            for filter_str in filters:
                if filter_str[0] == '*':
                    regexps.append(f'.*{make_re(filter_str[1:])}.*')
                elif filter_str[0] == '^':
                    regexps.append(f'{make_re(filter_str[1:])}.*')
                elif filter_str[0] == '$':
                    regexps.append(f'.*{make_re(filter_str[1:])}')
                elif filter_str[0] == '@':
                    regexps.append(filter_str[1:])
                elif filter_str[0] == '\\':
                    regexps.append(make_re(filter_str[1:]))
                else:
                    regexps.append(make_re(filter_str))
            self._re = join_re(*regexps, compiled=True, whole_string=True)

    def __contains__(self, item):
        return bool(self._re.match(str(item)))


class NamedFieldExtractor:
    """
    Represents a named field extractor from a string.
    Uses regular expression and the `parse` package (https://pypi.org/project/parse/)
    """

    __slots__ = ('_parser', '_type')

    def __init__(self, pattern):
        if is_str(pattern):
            self._parser, self._type = try____(pattern,
                                               func=re.compile,
                                               afunc=parse.compile,
                                               post_error_raise_check=lambda x: '(?P<' in x,
                                               extra_msg=f"tried to parse the provided format object `{pattern}` for `{pattern}` as a regular expression but failed")
        elif type(pattern) is NamedFieldExtractor:
            self._parser = pattern._parser
            self._type = pattern._type
        else:
            self._parser = pattern
            self._type = type(pattern) is parse.Parser

    def parse(self, s: str) -> Dict[str, str]:
        return self._parser.parse(s).named if self._type else self._parser.match(s).groupdict()

    # endregion

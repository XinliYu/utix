import logging
import warnings
from os import path
from typing import Union

from IPython.utils.path import ensure_dir_exists
from numpy import iterable

from _util.general_ext import hprint, eprint, get_pairs_str_for_hprint_and_regular_print, hprint_message, eprint_message


# region misc
def _ends_with_period(msg: str):
    return msg and msg[-1] == '.' and (len(msg) == 1 or msg[-2] != '.')


def extra_msg_wrap(func):
    def _wrap(*args, **kwargs):
        if 'extra_msg' in kwargs:
            extra_msg = kwargs.get('extra_msg')
            del kwargs['extra_msg']
            base_msg = func(*args, **kwargs)
            if _ends_with_period(base_msg):
                base_msg = base_msg[:-1]
            return (base_msg + (f'; {extra_msg}' if extra_msg else '.')) if base_msg else extra_msg
        else:
            return func(*args, **kwargs)

    return _wrap


# endregion

# region IO/Path messages

@extra_msg_wrap
def msg_not_a_dir(path_str):
    return f"the specified path `{path_str}` is not a directory"


@extra_msg_wrap
def msg_arg_not_a_dir(path_str, arg_name):
    return f"the specified path `{path_str}` in argument `{arg_name}` is not a directory"


@extra_msg_wrap
def msg_arg_multi_path_not_exist(path_str, arg_name):
    return f"none of the path(s) specified in `{path_str}` in argument `{arg_name}` exist"


@extra_msg_wrap
def msg_arg_path_not_exist(path_str, arg_name):
    return f"the specified path `{path_str}` by argument/variable `{arg_name}` does not exist"


@extra_msg_wrap
def msg_batch_file_writing_to_dir(path_str, num_files):
    return f"total `{num_files}` files written to directory `{path_str}`"


@extra_msg_wrap
def msg_create_dir(path_str):
    return f"directory `{path_str}` does not exist. Now create the directory"


@extra_msg_wrap
def msg_clear_dir(path_str):
    return f"directory `{path_str}` exists. Clearing any contents in it"


# endregion

# region argument checking messages
@extra_msg_wrap
def msg_positive_value_expected(arg_val, arg_name):
    return f"the argument/variable `{arg_name}` is expected to be a positive number; got `{arg_val}` instead"


@extra_msg_wrap
def msg_invalid_arg_value(arg_val, arg_name):
    return f"the value `{arg_val}` for argument/variable `{arg_name}` is invalid"


@extra_msg_wrap
def msg_values_sum_to_one_expected(sum_val, arg_name):
    return f"the values in argument/variable `{arg_name}` is expected to sum to one; got `{sum_val}` instead"


@extra_msg_wrap
def msg_arg_none_or_empty(arg_name):
    return f"argument/variable `{arg_name}` is none or empty"


@extra_msg_wrap
def msg_arg_none(arg_name):
    return f"argument/variable `{arg_name}` is none"


@extra_msg_wrap
def msg_arg_empty(arg_val, arg_name):
    return f"argument/variable `{arg_name}` of type `{type(arg_val)}` is empty"


@extra_msg_wrap
def msg_at_least_one_arg_should_avail(arg_names):
    return f"all arguments/variables in {','.join([f'`{arg_name}`' for arg_name in arg_names])} are none or empty"


@extra_msg_wrap
def msg_arg_not_callable(arg_name):
    return f"argument/variable `{arg_name}` is not callable"


@extra_msg_wrap
def msg_arg_not_iterable(arg_name):
    return f"argument/variable `{arg_name}` is not iterable"


@extra_msg_wrap
def msg_arg_not_of_type(arg_name, desired_type):
    return f"argument/variable `{arg_name}` is not of desired type `{desired_type}`"


@extra_msg_wrap
def msg_arg_not_of_types(arg_name, desired_types):
    return f"argument/variable `{arg_name}` must be one of the following types: {', '.join(str(x) for x in desired_types)}"


@extra_msg_wrap
def msg_arg_not_tuple_or_list(arg_name):
    return f"argument/variable `{arg_name}` must be a tuple or a list"


@extra_msg_wrap
def msg_arg_not_of_recognized_type(arg_val, arg_name):
    return f"the type `{type(arg_val)}` of argument/variable `{arg_name}` is not of recognized type/format"


# endregion

# region object checking messages
@extra_msg_wrap
def msg_name_already_defined(obj, member_name: str):
    return f"name `{member_name}` is already defined in object {obj}"


@extra_msg_wrap
def msg_name_not_defined(obj, member_name: str):
    return f"name `{member_name}` is not defined in object {obj}"


@extra_msg_wrap
def msg_key_not_exist(key, dict_name):
    return f"key `{key}` does not exist in dictionary {dict_name}"


@extra_msg_wrap
def msg_not_valid_python_name(name):
    return f"name `{name}` is not a valid Python name"


# endregion


# region argument check

def _warning_or_error(error_type, msg, warning_method, warning_category=None, *args, **kwargs):
    if warning_method is True:
        warnings.warn(msg, category=warning_category)
    elif callable(warning_method):
        warning_method(msg, warning_category=warning_category)
    else:
        raise error_type(msg, *args, **kwargs)


def ensure_arg_type(arg_val, arg_name, desired_type, extra_msg: str = None, warning=False, warning_category=None, *args, **kwargs):
    if type(arg_val) is not desired_type:
        _warning_or_error(error_type=ValueError, msg=msg_arg_not_of_type(arg_name=arg_name, desired_type=desired_type, extra_msg=extra_msg), warning_method=warning, warning_category=warning_category, *args, **kwargs)


def ensure_arg_types(arg_val, arg_name, desired_types, extra_msg: str = None, warning=False, warning_category=None, *args, **kwargs):
    if type(arg_val) not in desired_types:
        _warning_or_error(error_type=ValueError, msg=msg_arg_not_of_types(arg_name=arg_name, desired_type=desired_types, extra_msg=extra_msg), warning_method=warning, warning_category=warning_category, *args, **kwargs)


def ensure_arg_tuple_or_list(arg_val, arg_name, extra_msg: str = None, warning=False, warning_category=None, *args, **kwargs):
    if not isinstance(arg_val, tuple) and not isinstance(arg_val, list):
        _warning_or_error(error_type=ValueError, msg=msg_arg_not_tuple_or_list(arg_name=arg_name, extra_msg=extra_msg), warning_method=warning, warning_category=warning_category, *args, **kwargs)


def ensure_arg_iterable(arg_val, arg_name, extra_msg: str = None):
    if not iterable(arg_val):
        raise ValueError(msg_arg_not_iterable(arg_name=arg_name, extra_msg=extra_msg))


def ensure_arg_not_none(arg_val, arg_name, extra_msg: str = None):
    if arg_val is None:
        raise ValueError(msg_arg_none(arg_name=arg_name, extra_msg=extra_msg))


def ensure_arg_callable(arg_val, arg_name, extra_msg: str = None):
    if not callable(arg_val):
        raise ValueError(msg_arg_not_callable(arg_name=arg_name, extra_msg=extra_msg))
    return arg_val


def ensure_arg_not_none_or_empty(arg_val, arg_name, extra_msg: str = None):
    if arg_val is None:
        raise ValueError(msg_arg_none_or_empty(arg_name=arg_name, extra_msg=extra_msg))


def ensure_arg_not_empty(arg_val, arg_name, extra_msg: str = None):
    if len(arg_val) == 0:
        raise ValueError(msg_arg_empty(arg_val=arg_val, arg_name=arg_name, extra_msg=extra_msg))


def ensure_positive_arg(arg_val, arg_name, extra_msg: str = None):
    if arg_val is not None and arg_val <= 0:
        raise ValueError(msg_positive_value_expected(arg_val=arg_val, arg_name=arg_name, extra_msg=extra_msg))


def ensure_positive_arg_or_none(arg_val, arg_name, extra_msg: str = None):
    if arg_val is not None and arg_val <= 0:
        raise ValueError(msg_positive_value_expected(arg_val=arg_val, arg_name=arg_name, extra_msg=extra_msg))


def ensure_sum_to_one_arg(arg_val, arg_name, extra_msg: str = None, warning=False):
    sum_val = sum(arg_val)
    if sum_val != 1.0:
        msg = msg_values_sum_to_one_expected(sum_val=sum_val, arg_name=arg_name, extra_msg=extra_msg)
        if warning:
            warnings.warn(msg)
        else:
            raise ValueError(msg)


# endregion

# region IO/path check

def assert_path_exist(path_str, arg_name, extra_msg: str = None):
    if not path.exists(path_str):
        raise ValueError(msg_arg_path_not_exist(path_str=path_str, arg_name=arg_name, extra_msg=extra_msg))


# endregion

# region object check

def assert_name_not_defined(obj, member_name, extra_msg: str = None):
    if hasattr(obj, member_name):
        raise ValueError(msg_name_already_defined(obj=obj, member_name=member_name, extra_msg=extra_msg))


def ensure_name_defined(obj, member_name, extra_msg: str = None):
    if not hasattr(obj, member_name):
        raise ValueError(msg_name_not_defined(obj=obj, member_name=member_name, extra_msg=extra_msg))


def ensure_key_exist(key, d, dict_name, extra_msg: str = None):
    if key not in d:
        raise ValueError(msg_key_not_exist(key=key, dict_name=dict_name, extra_msg=extra_msg))


def ensure_valid_python_name(name: str, extra_msg: str = None):
    if not name.isidentifier():
        raise ValueError(msg_not_valid_python_name(name=name, extra_msg=extra_msg))


# endregion

# region logging

def get_logger(name: str, log_dir_path='.', logging_level=logging.DEBUG, log_format="%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s", append=False, file_ext='log'):
    """
    Convenient method to get a file-based logger.
    :param name: the name of the logger; also used as the main name for the log file.
    :param log_dir_path: the path to the directory containing the log file; if the log file does not exist, a new file will be created; if the log file already exists, then it is overwritten if `append` is `False`, or new log lines are appended to the existing file if `append` is set `True`.
    :param logging_level: provides the logging level; the default is the lowest level `logging.DEBUG`.
    :param log_format: the format for each logging message; by default it includes logging time, process id, logger name, logging level and the message; check https://docs.python.org/3/library/logging.html#logrecord-attributes for more about logging format directives.
    :param append: `True` if appending new log lines to the log file; `False` if the existing log file should be overwritten.
    :param file_ext: the extension name for the log file.
    :return: a file-based logger.
    """
    logger = logging.getLogger(name)
    ensure_dir_exists(log_dir_path)
    handler = logging.FileHandler(path.join(log_dir_path, f'{name}.{file_ext if file_ext else "log"}'), 'a+' if append else 'w+')
    handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(handler)
    logger.setLevel(logging_level)
    return logger


TYPE_LOGGER = Union[logging.Logger, str]


class LoggableBase:
    """
    A base class that wraps a logger and provides convenient common logging methods.
    """

    def __init__(self, logger: TYPE_LOGGER = None, logger_flush_interval=20, print_out=__debug__, color_print=True, *args, **kwargs):
        """
        :param logger: provides a logger instance or the path to a log file. If a log file path is provided, then a file-based logger will be automatically created.
        :param logger_flush_interval: specifies the number of log lines between two flush operations.
        :param print_out: `True` if printing out the message on the terminal if the log line contains a message; otherwise, `False`.
        :param color_print: `True` if to enable colorful print out on the terminal, and the color directives in the message will be recognized; otherwise, `False`.
        """
        if isinstance(logger, str):
            main_name, ext_name = path.splitext(path.basename(logger))
            self.logger = get_logger(name=main_name, log_dir_path=path.dirname(logger), file_ext=ext_name)
        else:
            self.logger = logger
        self._has_logger = logger is not None
        self._print_out = print_out
        self._color_print = color_print
        if color_print:
            self._debug_print = self._info_print = hprint
            self._error_print = eprint
        else:
            self._debug_print = self._info_print = self._error_print = print
        self._warn_print = warnings.warn  # warn print is always `warnings.warn`

        if self._has_logger:
            self._log_lines_count = 0
            self._logger_flush_interval = logger_flush_interval

    def log_flush(self):
        if self._has_logger and self._log_lines_count >= self._logger_flush_interval:
            for handler in self.logger.handlers:
                if hasattr(handler, 'flush'):
                    handler.flush()

    def _log_with_print(self, log_method, print_method, msg: str, print_out=False):
        log_method(msg)
        if (self._print_out or print_out) and msg:
            print_method(msg)
        self._log_lines_count += 1
        self.log_flush()

    def _log_msg_with_title(self, log_method, print_method, title: str, content: str, print_out=False):
        if self._has_logger:
            log_method(f"{title}: {content}")
            self._log_lines_count += 1
            self.log_flush()

        if self._print_out or print_out:
            print_method(title, content)

    def _log_pairs(self, *pairs, log_method, pair_msg_gen_method, print_out=False):
        colored_msg, uncolored_msg = pair_msg_gen_method(*pairs)
        if self._has_logger:
            log_method(uncolored_msg)
            self._log_lines_count += 1
            self.log_flush()

        if self._print_out or print_out:
            print(colored_msg if self._color_print else uncolored_msg)

    def _log_multiple_with_print(self, log_method, print_method, *msgs, print_out=False):
        msg_count = len(msgs)
        if self._has_logger:
            for i in range(msg_count):
                msg = msgs[i]
                log_method(msg)
                if self._print_out and msg:
                    print_method(msg)
            self._log_lines_count += msg_count
            self.log_flush()
        elif self._print_out or print_out:
            for i in range(msg_count):
                msg = msgs[i]
                if msg:
                    print_method(msg)

    def error(self, msg: str, print_out=True):
        """
        Logs one logging.ERROR level message.
        :param msg: the message to log.
        """
        if self._has_logger:
            self._log_with_print(self.logger.error, self._error_print, msg, print_out=print_out)

    def error_message(self, title, msg, print_out=True, print_method=eprint_message):
        self._log_msg_with_title(log_method=self.logger.info, print_method=print_method, title=title, content=msg, print_out=print_out)

    def error_multiple(self, *msgs, print_out=True):
        """
        Logs multiple logging.ERROR level messages.
        :param msgs: the messages to log.
        """
        if self._has_logger:
            self._log_multiple_with_print(self.logger.error, self._error_print, *msgs, print_out=print_out)

    def debug(self, msg: str, print_out=False):
        """
        Logs one logging.DEBUG level message.
        :param msg: the message to log.
        """
        if self._has_logger:
            self._log_with_print(self.logger.debug, self._debug_print, msg, print_out=print_out)

    def debug_multiple(self, *msgs, print_out=False):
        """
        Logs multiple logging.DEBUG level messages.
        :param msgs: the messages to log.
        """
        if self._has_logger:
            self._log_multiple_with_print(self.logger.debug if self._has_logger else None, self._debug_print, *msgs, print_out=print_out)

    def debug_pairs(self, *pairs, print_out=False):
        self._log_pairs(*pairs, log_method=self.logger.debug if self._has_logger else None, pair_msg_gen_method=get_pairs_str_for_hprint_and_regular_print, print_out=print_out)

    def debug_message(self, title, msg, print_out=False, print_method=hprint_message):
        if self._has_logger:
            self._log_msg_with_title(log_method=self.logger.debug if self._has_logger else None, print_method=print_method, title=title, content=msg, print_out=print_out)

    def warning(self, msg: str, print_out=True, print_method=None):
        """
        Logs one logging.WARNING level message.
        :param msg: the message to log.
        """
        if self._has_logger:
            self._log_with_print(self.logger.warning, print_method if print_method else self._warn_print, msg, print_out=print_out)

    def warning_multiple(self, *msgs, print_out=True, print_method=None):
        """
        Logs multiple logging.WARNING level messages.
        :param msgs: the messages to log.
        """
        if self._has_logger:
            self._log_multiple_with_print(self.logger.warning if self._has_logger else None, print_method if print_method else self._warn_print, *msgs, print_out=print_out)

    def info(self, msg: str, print_out=True, print_method=None):
        """
        Logs one logging.INFO level message.
        :param msg: the message to log.
        """
        if self._has_logger:
            self._log_with_print(self.logger.info if self._has_logger else None, print_method if print_method else self._info_print, msg, print_out=print_out)

    def info_multiple(self, *msgs, print_out=True, print_method=None):
        """
        Logs multiple logging.INFO level messages.
        :param msgs: the messages to log.
        """
        if self._has_logger:
            self._log_multiple_with_print(self.logger.info, print_method if print_method else self._info_print, *msgs, print_out=print_out)

    def info_message(self, title, msg, print_out=True, print_method=hprint_message):
        self._log_msg_with_title(log_method=self.logger.info if self._has_logger else None, print_method=print_method, title=title, content=msg, print_out=print_out)

    def info_pairs(self, *pairs, print_out=True, pairs_str_gen_method=get_pairs_str_for_hprint_and_regular_print):
        self._log_pairs(*pairs, log_method=self.logger.info if self._has_logger else None, pair_msg_gen_method=pairs_str_gen_method, print_out=print_out)

# endregion

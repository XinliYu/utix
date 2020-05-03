from collections import defaultdict
from time import time
from typing import Callable, List
import datetime

DEFAULT_TOC_MSG = 'Done!'

_tic_toc_time_stack = []
_tic_toc_dict = defaultdict(list)

tic_toc_always_enabled = False


class TicToc:
    def __init__(self, update_interval: int = 1):
        self.start_time = self.last_time = time()
        self.index = 0
        self.recent_runtime = self.avg_runtime = 0.
        self.update_interval = update_interval

    def tic(self):
        self.last_time = time()

    def toc(self):
        self.index += 1
        if self.index != 1 and self.index % self.update_interval == 0:
            last_time = self.last_time
            self.last_time = time()
            self.recent_runtime = (self.last_time - last_time) / self.update_interval
            self.avg_runtime = (self.last_time - self.start_time) / self.index
            return True
        else:
            return False


def tic(msg: str = None, key=None, newline=False, verbose=True):
    if __debug__ or tic_toc_always_enabled:
        cur_time = time()
        time_stack = _tic_toc_time_stack if key is None else _tic_toc_dict[key]
        if time_stack:
            if not time_stack[-1][1]:
                print()
            time_stack[-1][3] = True  # sets nested flag
        time_stack.append([cur_time, newline, msg, False])

        if msg and verbose:
            print("{} ({}).".format((msg[:-1] if msg[-1] == '.' and (len(msg) == 1 or msg[-2] != '.') else msg), datetime.datetime.now().strftime("%I:%M %p on %B %d, %Y")), end='\n' if newline else ' ')


def toc(msg: str = DEFAULT_TOC_MSG, key=None, print_out=True):
    if __debug__ or tic_toc_always_enabled:
        curr_time = time()

        if key is None:
            time_stack = _tic_toc_time_stack
            if len(time_stack) == 0:
                return
        else:
            if key in _tic_toc_dict:
                time_stack = _tic_toc_dict[key]
                if len(time_stack) == 0:
                    del _tic_toc_dict[key]
                    return
            else:
                return

        last_time, tic_new_line, tic_msg, nested = time_stack.pop()
        if time_stack:
            time_stack[-1][1] = True  # sets newline flag

        time_diff = curr_time - last_time

        if print_out:
            if nested and tic_msg:
                print("{} ({}, {:.5f} secs elapsed).".format(msg, tic_msg, time_diff))
            elif msg:
                print("{} ({:.5f} secs elapsed).".format(msg, time_diff))
            else:
                print("{:.5f} secs elapsed.".format(time_diff))
        return time_diff


def time_fun(funcs_to_compare: List[Callable], epochs=3, iterations=100000):
    keyed_func_list = [[i, func, 0] for i, func in enumerate(funcs_to_compare)]
    from random import shuffle
    for i in range(epochs):
        shuffle(keyed_func_list)
        tic()
        for rcd in keyed_func_list:
            for j in range(iterations):
                rcd[1]()
            rcd[2] += toc(print_out=False)
    keyed_func_list.sort()
    for rcd in keyed_func_list:
        print('func:{}, secs:{}'.format(rcd[0], rcd[2] / iterations))


def timestamp(scale=100) -> str:
    return str(int(time() * scale))

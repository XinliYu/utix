# This script shows how to use hprint to highlight part of the string.
# hprint costs 200% more time for short print, and 600% more time for long print.
# with cython, it reduces to 70% more time for short print, and 150% more time for long print.

import timeit

from utix._utilc.general_ext import hprint_message, hprint
# from utix._utilc.test import hprint

def target():
    hprint('test for `short` highlight')


hprint_time_short = timeit.timeit(target, number=10000)


def target():
    print('test for `short` highlight')


print_time_short = timeit.timeit(target, number=10000)


def target():
    hprint('hprint can be used for simple print with `highlight`;\nthe part to `highlight` must be enclosed by a pair of `backticks` (``); \nuse two backticks ```` to escape the backtick.')


hprint_time_long = timeit.timeit(target, number=10000)


def target():
    print('hprint can be used for simple print with `highlight`;\nthe part to `highlight` must be enclosed by a pair of `backticks` (``); \nuse two backticks ```` to escape the backtick.')


print_time_long = timeit.timeit(target, number=10000)

hprint_message(title='short hprint', content=hprint_time_short)
hprint_message(title='short print', content=print_time_short)
hprint_message(title='long hprint', content=hprint_time_long)
hprint_message(title='long print', content=print_time_long)
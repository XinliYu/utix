import timeit

from utix._util.general_ext import xsum, hprint_message
from utix._util.dict_ext import xfdict

data = [xfdict({'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5})] * 100


def target():
    return sum(data, xfdict({}))


sum_time = timeit.timeit(target, number=1000)


def target():
    return xsum(data)


accu_sum_time = timeit.timeit(target, number=1000)

data = [[1, 2, 3, 4, 5]] * 100


def target():
    return sum(data, [])


list_sum_time = timeit.timeit(target, number=1000)


def target():
    return xsum(data)


list_accu_sum_time = timeit.timeit(target, number=1000)

data = [2.0] * 10000


def target():
    return sum(data)


num_sum_time = timeit.timeit(target, number=1000)


def target():
    return xsum(data)


num_accu_sum_time = timeit.timeit(target, number=1000)

hprint_message('sum time', sum_time)
hprint_message('xsum time', accu_sum_time)
hprint_message('list sum time', list_sum_time)
hprint_message('list xsum time', list_accu_sum_time)
hprint_message('num sum time', num_sum_time)
hprint_message('num xsum time', num_accu_sum_time)
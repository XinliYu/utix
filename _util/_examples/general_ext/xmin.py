import timeit

from _util.general_ext import xsum, hprint_message, xmin, xmax
from _util.dict_ext import xfdict

obj1 = xfdict({'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5})
obj2 = xfdict({'a': 2, 'b': 0, 'c': 4, 'd': 5, 'e': 1, 'f': 3})
obj3 = xfdict({'a': 0, 'b': 4, 'c': -1, 'f': 2})

print(xmin((obj1, obj2, obj3)))
print(xmin((obj3, obj1, obj2)))
print(xmin((obj2, obj3, obj1)))

print(xmax((obj1, obj2, obj3)))
print(xmax((obj3, obj1, obj2)))
print(xmax((obj2, obj3, obj1)))
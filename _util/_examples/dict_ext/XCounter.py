from utix._util.dict_ext import XCounter

a = XCounter()
a += {'int': 1, 'float': 0.55, 'list': [1, 2, 3], 'tuple': (1, 2, 3)}
a += {'int': 1, 'float': 0.55, 'list': [1, 2, 3], 'tuple': (1, 2, 3)}

print(a)

a = XCounter()
a += {'int': 1, 'float': 0.55, 'set': {1, 2, 3}}
a += {'int': 1, 'float': 0.55, 'set': {1, 2, 3}}

print(a)
print(a - a)
print((a / 3))

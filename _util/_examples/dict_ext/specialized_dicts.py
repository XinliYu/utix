from utilc.dict_ext import listdict, setdict, tupledict

a = listdict()
a += {'a': 1, 'b': 2}
a += {'a': 3, 'b': 4}
a += {'a': 5, 'b': 6}
print(a)


a = setdict()
a += {'a': 1, 'b': 2}
a += {'a': 3, 'b': 4}
a += {'a': 5, 'b': 6}
print(a)


a = tupledict()
a += {'a': 1, 'b': 2}
a += {'a': 3, 'b': 4}
a += {'a': 5, 'b': 6}
print(a)

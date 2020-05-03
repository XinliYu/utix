from utilx.dict_ext import hdict

d = hdict(a=1, b=2, c=3)
print(d['a'])
print(d['b'])
print(d['c'])

del d['a']
print(d)

d['d'] = 4
print(d)

del d['d']
print(d)

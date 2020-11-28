from utix.dict_ext import mk_accu

a = mk_accu(a=1, b=2)
b = mk_accu(a=1, b=2, c=3)
c = a + b
print(c)

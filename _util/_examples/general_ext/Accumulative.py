from _util.general_ext import Accumulative, FieldsRepr


class Test(Accumulative, FieldsRepr):
    __slots__ = ('field1', 'field2')

    def __init__(self, field1, field2):
        Accumulative.__init__(self)
        FieldsRepr.__init__(self)
        self.field1 = field1
        self.field2 = field2


a = Test(1, 2)
b = Test(3, 4)
print(a + b)
print(a | b)

a += b
print(a)

a /= 2
print(a)

a *= 2
print(a)

a //= 2
print(a)


import random
from collections import Callable
from functools import partial
from itertools import chain
from typing import Union, Tuple, Callable, List

import numpy as np
import utilx.npex as npx
import pyro

from _util.general_ext import shape_after_broadcast, shape_after_broadcast__, is_list_or_tuple, is_num
from _util.np_ext import make_symmetric_matrix


def shuffle__(_iterable):
    """
    A variant of the `random.shuffle` with simple support for an iterable (e.g. a tuple).
    NOTE this function has a return, while `random.shuffle` has no return.
    """
    if not isinstance(_iterable, list):
        _iterable = list(_iterable)
    random.shuffle(_iterable)
    return _iterable


def rnd_partial(func, *args, **kwargs):
    if type(func) is RndFunc or not any(isinstance(arg, RndTensor) for arg in chain(args, kwargs.values())):
        return partial(func, *args, **kwargs)
    else:
        return partial(RndFunc(func), *args, **kwargs)


def rnd_partial__(func, *args, **kwargs):
    gen_type = type(func)
    if gen_type is partial:
        return func
    elif gen_type is RndFunc or not any(isinstance(arg, RndTensor) for arg in chain(args, kwargs.values())):
        return partial(func, *args, **kwargs)
    else:
        return partial(RndFunc(func), *args, **kwargs)


class RndGen:
    __slots__ = ('generator',)

    def __init__(self, generator, *args, **kwargs):
        self.generator = rnd_partial__(generator, *args, **kwargs)

    def __call__(self, shape=None):
        return self.generator(shape)

    # region add

    @staticmethod
    def _add1(gen1, gen2, shape):
        return gen1(shape) + gen2(shape)

    @staticmethod
    def _add2(gen, other, shape):
        return gen(shape) + other

    @staticmethod
    def _radd2(gen, other, shape):
        return other + gen(shape)

    def __iadd__(self, other):
        if hasattr(other, 'generator'):
            self.generator = partial(RndGen._add1, self.generator, other.generator)
        else:
            self.generator = partial(RndGen._add2, self.generator, other)
        return self

    def __add__(self, other):
        if hasattr(other, 'generator'):
            return RndGen(generator=partial(RndGen._add1, self.generator, other.generator))
        else:
            return RndGen(generator=partial(RndGen._add2, self.generator, other))

    def __radd__(self, other):
        if hasattr(other, 'generator'):
            return RndGen(generator=partial(RndGen._add1, other.generator, self.generator))
        else:
            return RndGen(generator=partial(RndGen._radd2, self.generator, other))

    # endregion 

    # region mul

    @staticmethod
    def _mul1(gen1, gen2, shape):
        return gen1(shape) + gen2(shape)

    @staticmethod
    def _mul2(gen, other, shape):
        return gen(shape) + other

    @staticmethod
    def _rmul2(gen, other, shape):
        return other + gen(shape)

    def __imul__(self, other):
        if hasattr(other, 'generator'):
            self.generator = partial(RndGen._mul1, self.generator, other.generator)
        else:
            self.generator = partial(RndGen._mul2, self.generator, other)
        return self

    def __mul__(self, other):
        if hasattr(other, 'generator'):
            return RndGen(generator=partial(RndGen._mul1, self.generator, other.generator))
        else:
            return RndGen(generator=partial(RndGen._mul2, self.generator, other))

    def __rmul__(self, other):
        if hasattr(other, 'generator'):
            return RndGen(generator=partial(RndGen._mul1, other.generator, self.generator))
        else:
            return RndGen(generator=partial(RndGen._rmul2, self.generator, other))

    # endregion


class RndTensor:
    __slots__ = ('generator', 'shape')

    def __init__(self, generator: Callable, *args, shape=None, atom_dim=None, **kwargs):
        self.shape = None if shape is None else ((shape,) if type(shape) is int else tuple(shape))
        self.generator = rnd_partial__(generator, *(*args, (self.shape[:-atom_dim] if atom_dim else self.shape)), **kwargs)

    def __call__(self):
        return self.generator()

    # region add

    @staticmethod
    def _add1(gen1, gen2):
        return gen1() + gen2()

    @staticmethod
    def _add2(gen, other):
        return gen() + other

    @staticmethod
    def _radd2(gen, other):
        return other + gen()

    def __iadd__(self, other):
        if hasattr(other, 'generator'):
            self.generator = partial(RndTensor._add1, self.generator, other.generator)
        else:
            self.generator = partial(RndTensor._add2, self.generator, other)
        return self

    def __add__(self, other):
        if hasattr(other, 'generator'):
            return RndTensor(generator=partial(RndTensor._add1, self.generator, other.generator), shape=shape_after_broadcast__(self, other))
        else:
            return RndTensor(generator=partial(RndTensor._add2, self.generator, other), shape=shape_after_broadcast__(self, other))

    def __radd__(self, other):
        if hasattr(other, 'generator'):
            return RndTensor(generator=partial(RndTensor._add1, other.generator, self.generator), shape=shape_after_broadcast__(other, self))
        else:
            return RndTensor(generator=partial(RndTensor._radd2, self.generator, other), shape=shape_after_broadcast__(other, self))

    # endregion 

    # region mul

    @staticmethod
    def _mul1(gen1, gen2):
        return gen1() + gen2()

    @staticmethod
    def _mul2(gen, other):
        return gen() + other

    @staticmethod
    def _rmul2(gen, other):
        return other + gen()

    def __imul__(self, other):
        if hasattr(other, 'generator'):
            self.generator = partial(RndTensor._mul1, self.generator, other.generator)
        else:
            self.generator = partial(RndTensor._mul2, self.generator, other)
        return self

    def __mul__(self, other):
        if hasattr(other, 'generator'):
            return RndTensor(generator=partial(RndTensor._mul1, self.generator, other.generator), shape=shape_after_broadcast__(self, other))
        else:
            return RndTensor(generator=partial(RndTensor._mul2, self.generator, other), shape=shape_after_broadcast__(self, other))

    def __rmul__(self, other):
        if hasattr(other, 'generator'):
            return RndTensor(generator=partial(RndTensor._mul1, other.generator, self.generator), shape=shape_after_broadcast__(other, self))
        else:
            return RndTensor(generator=partial(RndTensor._rmul2, self.generator, other), shape=shape_after_broadcast__(other, self))
    # endregion


class RndFunc:
    __slots__ = ('func',)

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*((arg() if isinstance(arg, RndTensor) else arg) for arg in args), **{k: (v() if isinstance(v, RndTensor) else v) for k, v in kwargs.items()})


class UniformGen(RndGen):
    """
    An uniform random number generator that wraps around the `numpy.random.uniform`. Its `low` and `high` can be composed with a random scalar.

    >>> import utilx.rnd_ext as rndx
    >>> unigen1 = rndx.UniformGen(low=0, high=1)
    >>> print(unigen1((2, 3)))
    >>> unigen2 = rndx.UniformGen(low=rndx.UniformScalar(0, 5), high=rndx.UniformScalar(5, 10))
    >>> print(unigen2((2, 3)))
    """

    def __init__(self, low, high):
        super(UniformGen, self).__init__(np.random.uniform, low, high)


def make_rnd_gen(_t):
    """
    A convenient function to convert a 2-tuple to a random uniform generator, or convert a single int/float number to a uniform distribution from 0 to that number. The parameter is directly returned if it is not a tuple.
    """
    return UniformGen(*_t) if is_list_or_tuple(_t) and len(_t) == 2 else (UniformGen(0, _t) if is_num(_t) else _t)


class CategoricalGen(RndGen):
    """
    An random category generator that wraps around the `numpy.random.choice`.

    >>> import utilx.rnd_ext as rndx
    >>> catgen = rndx.CategoricalGen(range(10))
    >>> print(catgen((2, 3)))
    """

    def __init__(self, categories, replace=True, p=None):
        super(CategoricalGen, self).__init__(np.random.choice, categories, replace=replace, p=p)


class UniformArray(RndTensor):
    def __init__(self, low, high, shape=None):
        super(UniformArray, self).__init__(np.random.uniform, low, high, shape=shape)


class UniformScalar(UniformArray):
    def __init__(self, low, high):
        super(UniformScalar, self).__init__(low, high)


class NormalGen(RndGen):
    def __init__(self, mean, var):
        """
        An normal random number generator. Its `mean` and `var` can be composed with a random scalar.

        >>> import utilx.rnd_ext as rndx
        >>> ngen = rndx.NormalGen(mean=0, var=1)
        >>> print(ngen((2, 3)))
        >>> ngen = rndx.NormalGen(mean=rndx.UniformScalar(1, 3), var=rndx.UniformScalar(0, 5))
        >>> print(ngen((2, 3)))
        """
        super(NormalGen, self).__init__(np.random.normal, mean, var)


class NormalArray(RndTensor):
    def __init__(self, mean, var, shape=None):
        super(NormalArray, self).__init__(np.random.normal, mean, var, shape=shape)


class NormalScalar(NormalArray):
    def __init__(self, mean, var):
        super(NormalScalar, self).__init__(mean, var)


class MultiVariateNormalArray(RndTensor):
    def __init__(self, mean, cov, shape=None):
        super(MultiVariateNormalArray, self).__init__(np.random.multivariate_normal, mean, cov, shape=shape, atom_dim=1)


hstack = RndFunc(np.hstack)
hstack__ = RndFunc(npx.hstack__)
choice = RndFunc(np.random.choice)


def random_fill_diagonal(mat: np.ndarray, dist: Union[Tuple, RndGen], first_k: int = None, wrap=False):
    """
    Fills the diagonal of the matrix with random numbers.

    >>> import utilx.rnd_ext as rndx
    >>> import numpy as np
    >>> a = np.zeros((5, 5))
    >>> rndx.random_fill_diagonal(a, dist=(0, 3), first_k=3)
    >>> print(a)

    :param mat: the matrix.
    :param dist: either a tuple representing the low and high of a uniform random number generator, or a random number generator.
    :param first_k: fill the first k diagonal elements specified by this parameter.
    :param wrap: see the `wrap` parameter of `numpy.fill_diagonal` function.
    """
    size = min(mat.shape)
    if first_k is None or first_k >= size:
        np.fill_diagonal(mat, make_rnd_gen(dist)(size), wrap=wrap)
    else:
        mat[np.diag_indices(first_k)] = make_rnd_gen(dist)(first_k)


def random_square_matrix(dist: Union[Tuple, Callable], n) -> np.ndarray:
    return make_rnd_gen(dist)((n, n))


def random_symmetric_matrix(dist: Union[Tuple, RndGen], n) -> np.ndarray:
    """
    Creates a random symmetric matrix.
    :param dist: either a tuple representing the low and high of a uniform random number generator, or a random number generator.
    :param n: the size of the random symmetric matrix to generate.
    :return: a random symmetric matrix, whose random numbers are generated according to `dist`.
    """
    m = make_rnd_gen(dist)((n, n))
    make_symmetric_matrix(m)
    return m


def random_matrix_replace(x: np.ndarray, dist: Union[Tuple, RndGen], num_replaces: Union[int, List, Tuple, range, Callable], dim0: bool = True):
    if num_replaces and dist:
        dist = make_rnd_gen(dist)
        type_num_replaces = type(num_replaces)

        if dim0:
            k = x.shape[0]

            def _replace(num_exceptions):
                if num_exceptions > 0:
                    exception_indices = np.random.choice(idxes, num_exceptions, replace=False)
                    x[i, exception_indices] = dist(num_exceptions)
                if i != k - 1:
                    idxes[i] = i

        else:
            k = x.shape[1]

            def _replace(num_exceptions):
                if num_exceptions > 0:
                    exception_indices = np.random.choice(idxes, num_exceptions, replace=False)
                    x[exception_indices, i] = dist(num_exceptions)
                if i != k - 1:
                    idxes[i] = i

        idxes = list(range(1, k))  # this works with `if i != k - 1: idxes[i] = i` to rule out the 'current' row or column
        if type_num_replaces is int:
            for i in range(k):
                _replace(num_replaces)
        elif type_num_replaces in (tuple, list, range):
            for i in range(k):
                _replace(random.choice(num_replaces))
        elif callable(num_replaces):
            _replace(int(num_replaces()))


# a = MultiVariateNormalArray(mean=NormalArray(0, 1, 3), cov=np.eye(3), shape=(3, 5, 3))
# b = MultiVariateNormalArray(mean=NormalArray(0, 1, 3), cov=np.eye(3), shape=(3, 1, 3))
# print(a.shape)
# print(b.shape)
# print((a + b).shape)
# print(((a + b)()))
#
#
def rnd_corr_data(*corr_data, feature_size=50, shuffle=False, non_corr_data=None):
    if not corr_data:
        corr_data = (MultiVariateNormalArray(mean=NormalArray(0, 1, feature_size), cov=np.eye(feature_size), shape=(32, feature_size)) * UniformScalar(),
                     MultiVariateNormalArray(mean=NormalArray(0, 1, feature_size), cov=np.eye(feature_size), shape=(50, feature_size)))
    if not non_corr_data:
        non_corr_data = UniformArray(low=0, )
    return npx.shuffle_columns(hstack((*corr_data, non_corr_data)))

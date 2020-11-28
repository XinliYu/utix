import types
import warnings
from itertools import product
from typing import Callable, Union

import numpy as np
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utix.general import bool2obj, hasattr_or_exec
from utix.listex import nested_lists_regular_shape, nested_lists_get

"""
Numpy-related utilities.
Requires numpy, scipy and scikit-learn.
"""


def empty__(shape, ref=None):
    """
    Returns a non-initialized array, with the specified `shape`, and with the same data type and order as the `ref` array if `ref` is not `None`.
    :param ref: the returned array will have the same data type and the same order as this `ref`, if this parameter is assigned.
    :param shape: the shape for the returned array.
    :return: a non-initialized array, with the specified `shape`, and with the same data type and order as the `ref` array.
    """
    return np.empty(shape) if ref is None else np.empty(shape, dtype=ref.dtype, order='C' if ref.flags.c_contiguous else 'F')


def zeros__(shape: tuple, ref):
    """
    Returns an array filled with the scalar value 0, with the specified `shape`, and with the same data type and order as the `ref` array if `ref` is not `None`.
    :param ref: the returned array will have the same data type and the same order as this `ref`, if this parameter is assigned.
    :param shape: the shape for the returned array.
    :return: an array filled with the scalar value 0, with the specified `shape`, and with the same data type and order as the `ref` array.
    """
    return np.zeros(shape) if ref is None else np.zeros(shape, dtype=ref.dtype, order='C' if ref.flags.c_contiguous else 'F')


def ones__(shape: tuple, ref):
    """
    Returns an array filled with the scalar value 1, with the specified `shape`, and with the same data type and order as the `ref` array if `ref` is not `None`.
    :param ref: the returned array will have the same data type and the same order as this `ref`, if this parameter is assigned.
    :param shape: the shape for the returned array.
    :return: an array filled with the scalar value 1, with the specified `shape`, and with the same data type and order as the `ref` array.
    """
    return np.ones(shape) if ref is None else np.ones(shape, dtype=ref.dtype, order='C' if ref.flags.c_contiguous else 'F')


def column(_x):
    return np.expand_dims(np.array(_x), 1)


def row(_x):
    return np.expand_dims(np.array(_x), 0)


def sequential(shape, start=0, step=1, order='C'):
    """
    Creates a numpy array of the specified shape filled with sequential integers.
    
    >>> import utix.np_ext as npx
    >>> print(sequential((2, 3))
    >>>       == np.array([[0, 1, 2],
    >>>                    [3, 4, 5]]))
    >>> print(sequential((2, 3), order='F')
    >>>       == np.array([[0, 2, 4],
    >>>                    [1, 3, 5]]))

    :param shape: the shape of the numpy array to create.
    :param start: the first integer of the sequence.
    :param step: the increase between two adjacent values in the sequence.
    :param order: see `numpy.reshape`.
    :return: a numpy array of the specified shape; the first value in this returned array is `start`, and increase by `step` for each fill.
    """
    _len = np.prod(shape) * step
    return np.arange(start, start + _len, step).reshape(shape, order=order)


def shuffle_rows__(x):
    idx = np.arange(len(x))
    idx_shuf = np.random.shuffle(idx)
    return x[idx_shuf], idx_shuf


def shuffle_transpose(x) -> None:
    """
    Shuffles the transpose (columns if `x` is a 2D array) of a numpy array in-place.

    >>> import utix.np_ext as npx
    >>> a = npx.sequential((2, 3))
    >>> npx.shuffle_transpose(a)
    >>> print(a)

    """
    np.random.shuffle(np.transpose(x))


shuffle_columns = shuffle_transpose
shuffle_columns.__doc__ = "Alias for `shuffle_transpose`."


def vstack__(*x):
    """
    A convenient version for `vstack` where you could pass in arrays to stack as variable positional parameters.
    >>> import utix.np_ext as npx
    >>> import numpy as np
    >>> a = npx.vstack__(npx.sequential((1,2)), npx.sequential((2,2)))
    >>> print(a == np.array([[0, 1],
    >>>                      [0, 1],
    >>>                      [2, 3]]))

    """
    return np.vstack(x)


def hstack__(*x):
    """
    A convenient version for `hstack` where you could pass in arrays to stack as variable positional parameters.
    >>> import utix.np_ext as npx
    >>> import numpy as np
    >>> a = npx.vstack__(npx.sequential((2,1)), npx.sequential((2,2)))
    >>> print(a == np.array([[0, 0, 1],
    >>>                      [1, 2, 3]]))

    """
    return np.hstack(x)


def make_symmetric_matrix(x: np.ndarray, upper2lower=True):
    n = x.shape[0]
    i = np.tril_indices(n, -1) if upper2lower else np.triu_indices(n, 1)
    x[i] = x.T[i]


def row_wise_where(arr: np.ndarray, targets: np.ndarray):
    if isinstance(targets, list):
        targets = np.array([targets]).T
    else:
        targets = targets[:, np.newaxis]
    return np.argmax((np.hstack((arr, targets)) - targets) == 0, axis=-1)


def np_fullprint(*args, **kwargs):
    opt = np.get_printoptions()
    np.set_printoptions(threshold='nan')
    print(*args, **kwargs)
    np.set_printoptions(**opt)


def nums_to_prob(*nums):
    """
    Simply converts a list of non-negative numbers to a probability distribution.
    :param nums: a list of non-negative numbers. You must ensure these numbers are non-negative; the method does not check whether they are.
    :return: a probability distribution derived by normalizing the numbers.
    """
    a = np.array(nums)
    return (a / np.sum(a)).tolist()


def array__(x, padding=0, dtype=None, **kwargs):
    """
    A variant of `numpy.array` that accepts ragged nested lists, padding it so that to have a regular-shaped array.

    For example,
    >>> import utix.npex as npx
    >>> npx.array__([[1,2], [3]])
    array([[1, 2],
           [3, 0]])
    >>> npx.array__([[[1],[2,3],[4,5,6]], [[7], [8, 9, 10]], [[11],[12, 13]]])
    array([[[ 1,  0,  0],
            [ 2,  3,  0],
            [ 4,  5,  6]],
    <BLANKLINE>
           [[ 7,  0,  0],
            [ 8,  9, 10],
            [ 0,  0,  0]],
    <BLANKLINE>
           [[11,  0,  0],
            [12, 13,  0],
            [ 0,  0,  0]]])

    :param x: the list, or a nested list to convert to a numpy array.
    :param padding: the padding value.
    :param dtype: the numpy data type for the array.
    :param kwargs: other arguments for `numpy.array` function.
    :return: a regular-shaped array with all the values from `x`, and pad it with the `pad_value` to make it regular-shaped.
    """
    shape = nested_lists_regular_shape(x)
    result = np.full(shape, fill_value=padding, dtype=dtype, **kwargs)
    for index in product(*(range(_) for _ in shape[:-1])):
        row = nested_lists_get(x, index)
        if row is not None:
            result[index][:len(row)] = row
    return result


class numpy_local_seed:
    def __init__(self, seed):
        self._seed = seed
        self._prev_state = None

    def __enter__(self):
        if self._seed >= 0:
            self._prev_state = np.random.get_state()
            np.random.seed(self._seed)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self._prev_state is not None:
            np.random.set_state(self._prev_state)


def sort_by_diagonal(x, *sort_together, reverse=False):
    idx = np.argsort(np.diag(x))
    if reverse:
        idx = idx[::-1]
    if sort_together:
        return (x[idx[:, None], idx], *((xx[idx] if isinstance(xx, np.ndarray) else [xx[i] for i in idx]) for xx in sort_together))
    else:
        return x[idx[:, None], idx]


def axis_to_front(x, source):
    if type(source) is int:
        return np.moveaxis(x, source=source, destination=0)
    else:
        return np.moveaxis(x, source=source, destination=np.arange(len(source)))


def axis_to_rear(x, source):
    if type(source) is int:
        return np.moveaxis(x, source=source, destination=-1)
    else:
        return np.moveaxis(x, source=source, destination=np.arange(-len(source), 0))


def iter_slices(x, dim):
    """
    Iterates through slices at the specified dimensions of the numpy array.

    >>> import utix.np_ext as npx
    >>> a = npx.sequential((2, 2, 3)) # array([[[ 0,  1,  2],
    >>>                               #         [ 3,  4,  5]],
    >>>                               #        [[ 6,  7,  8],
    >>>                               #         [ 9, 10, 11]]]
    >>> print(a)
    >>> print(list(npx.iter_slices(a, -1))) # [array([0, 1, 2]), array([3, 4, 5]), array([6, 7, 8]), array([ 9, 10, 11])], which are results of `a[0,0,:]`, `a[0,1,:]`, `a[1,0,:]` and `a[1,1,:]`
    >>> print(list(npx.iter_slices(a, 1))) # [array([0, 3]), array([1, 4]), array([2, 5]), array([6, 9]), array([ 7, 10]), array([ 8, 11])], which are results of `a[0,:,0]`, `a[0,:,1]`, `a[1,:,0]` and `a[1,:,1]`
    >>> print(list(npx.iter_slices(a, (1, 2)))) # [array([[0, 1, 2], [3, 4, 5]]), array([[ 6,  7,  8], [ 9, 10, 11]])], which are results of `a[0,:,:]` and `a[1,:,:]`
    >>> print(list(npx.iter_slices(a, (0, 2)))) # [array([[0, 1, 2], [6, 7, 8]]), array([[ 3,  4,  5], [ 9, 10, 11]])], which are results of `a[:,0,:]` and `a[:,1,:]`

    :param x: the numpy array.
    :param dim: the slices at these specified dimension(s).
    :return: an iterator through the slices at the specified dimensions.
    """
    if type(dim) is int:
        x = np.moveaxis(x, source=dim, destination=-1)
        for i in np.ndindex(x.shape[:-1]):
            yield x[i]
    else:
        x = np.moveaxis(x, source=dim, destination=np.arange(-len(dim), 0))
        for i in np.ndindex(x.shape[:-len(dim)]):
            yield x[i]


def apply_to_slices(x, func, dim, func_return_shape=None, in_place=True, dtype=None):
    """
    Applies a function `func` to slices at the specified dimensions of the numpy array, which either update values in-place, or creates a new array with the return values of `func`.

    >>> import utix.np_ext as npx
    >>> x = npx.sequential((2, 2, 3)) # array([[[ 0,  1,  2],
    >>>                               #         [ 3,  4,  5]],
    >>>                               #        [[ 6,  7,  8],
    >>>                               #         [ 9, 10, 11]]]
    >>> x2 = npx.apply_to_slices(x, lambda x: sum(x), dim=-1, func_return_shape=1, in_place=False)
    >>> print(x2) # [[[ 3] [12]] [[21] [30]]]
    >>> print(x2.shape==(2, 2, 1))

    :param x: the numpy array
    :param func: the function to apply on the slices of the array `x`.
    :param dim: the slices at these specified dimension(s).
    :param func_return_shape: the return shape of the `func`; leave this as `None` if the return shape is the same as the slice shape; otherwise must specify the correct shape here.
    :param in_place: `True` to update the values in-place, which works only if the return shape of `func` is the same as the slice shape; otherwise, specify this as `False`, and a new array will be created.
    :param dtype: only effective if `in_place` is `False`; the date type of the new array.
    :return: the original numpy array `x` or a new numpy array with the return values of `func`.
    """

    if type(dim) is int:
        dim = (dim,)
    len_dim: int = len(dim)
    if type(func_return_shape) is int:
        func_return_shape = (func_return_shape,)

    _dst = np.arange(-len_dim, 0)
    x = np.moveaxis(x, source=dim, destination=_dst)
    slice_shape = x.shape[-len_dim:]

    if in_place:
        if slice_shape == func_return_shape:
            func_return_shape = None
        else:
            raise ValueError(f'the `func_return_shape` does not match the slice shape {slice_shape}')
    elif func_return_shape is None:
        func_return_shape = slice_shape

    if func_return_shape is None:
        _x = x
    else:
        if len(func_return_shape) != len_dim:
            raise ValueError(f'the `func_return_shape` must have the same size as `dim`, which is `{dim}`; '
                             f'in this case `func_return_shape` should be a list/tuple of length {len_dim}; got {func_return_shape}')
        else:
            new_shape = x.shape[:-len_dim] + func_return_shape
        _x = empty__(new_shape, ref=x) if dtype is None else np.empty(new_shape, dtype=dtype)

    for i in np.ndindex(x.shape[:-len_dim]):
        _x[i] = func(x[i])

    return _x if func_return_shape is None else np.moveaxis(_x, source=_dst, destination=dim)


def reduce_by_pca(x, n_components: int, sample_dim: int = -2, feature_dim: int = -1, standardizer: Union[bool, Callable] = True, pca_module=PCA, **kwargs):
    """
    Dimension reduction on `x` at the specified sample dimension `sample_dim` and the feature dimension `feature_dim`.
    After PCA, the size of the feature dimension will be reduced to `n_components`.
    `x` may contain more than two dimensions.

    The default.
    ------------
    >>> import utix.np_ext as npx
    >>> import numpy as np
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.preprocessing import StandardScaler
    >>> x = np.array([
    >>>     [0.387, 4878, 5.42],
    >>>     [0.723, 12104, 5.25],
    >>>     [1, 12756, 5.52],
    >>>     [1.524, 6787, 3.94],
    >>> ])
    >>> # the last second dimension is the sample dimension, and the last dimension is the feature dimension
    >>> np.allclose(npx.reduce_by_pca(x, 3), np.array([[-0.94063993, 1.62373172, -0.06406894],
    >>>                                                [-0.7509653, -0.63365168, 0.35357757],
    >>>                                                [-0.6710958, -1.11766206, -0.29312477],
    >>>                                                [2.36270102, 0.12758202, 0.00361615]]))

    Choose the sample dimension and the feature dimension.
    ------------------------------------------------------
    >>> x = np.asarray(tuple(npx.sequential((5, 5), start=i) for i in range(10)))
    >>> print(x.shape == (10, 5, 5))
    >>> print(npx.reduce_by_pca(x, sample_dim=1, feature_dim=2, n_components=3).shape == (10, 5, 3)) # the last dimension is the feature dimension; reduced the feature dimension from 5 to size 3
    >>> print(npx.reduce_by_pca(x, sample_dim=1, feature_dim=0, n_components=3).shape == (3, 5, 5)) # the first dimension is the feature dimension; reduced the feature dimension from 10 to size 3
    >>> pca = PCA(n_components=3)
    >>> np.allclose(npx.reduce_by_pca(x, sample_dim=1, feature_dim=2, n_components=3),
    >>>             np.asarray(tuple(pca.fit_transform(StandardScaler().fit_transform(_x)) for _x in x)))
    >>> np.allclose(npx.reduce_by_pca(x, sample_dim=1, feature_dim=0, n_components=3),
    >>>             np.moveaxis(np.asarray(tuple(pca.fit_transform(StandardScaler().fit_transform(x[...,i].T)) for i in range(x.shape[-1]))), (1, 2), (1, 0)))

    :param x: the numpy array with at least two dimensions for dimension reduction.
    :param sample_dim: the sample dimension.
    :param feature_dim: the feature dimension.
    :param n_components: reduce the size of the feature dimension to this specified number.
    :param kwargs: the other arguments to pass into `~sklearn.decomposition.PCA`.
    :param standardizer: a scikit-learn
    :return: the `x` with the reduced feature dimension.
    """

    len_x_shape = len(x.shape)
    if len_x_shape <= 1:
        raise ValueError(f'the numpy array must have at least two dimension to perform dimension reduction; got `{len_x_shape}`')
    if sample_dim < 0:
        sample_dim += len_x_shape
    if feature_dim < 0:
        feature_dim += len_x_shape

    if sample_dim == feature_dim:
        raise ValueError(f'the `sample_dim` and `feature_dim` cannot be the same; got both {sample_dim}')

    if issparse(x):
        warnings.warn('PCA does not support sparse matrix by its mathematical nature; the sparse matrix is converted to a full matrix')
        x = x.todense()

    pca = pca_module(n_components=n_components, **kwargs)
    standardizer = hasattr_or_exec(bool2obj(obj=standardizer, obj_for_true=StandardScaler), 'fit_transform', types.MethodType)

    if len_x_shape == 2:
        if sample_dim == 0 and feature_dim == 1:
            return pca.fit_transform(standardizer.fit_transform(x)) if standardizer is not None else pca.fit_transform(x)
        else:
            return pca.fit_transform(standardizer.fit_transform(x.T)) if standardizer is not None else pca.fit_transform(x.T)
    else:
        return apply_to_slices(x, lambda _x: (pca.fit_transform(standardizer.fit_transform(_x)) if standardizer is not None else pca.fit_transform(_x)), dim=(sample_dim, feature_dim), func_return_shape=(x.shape[sample_dim], n_components), in_place=False, dtype=np.float)


def rotate2d(degree):
    theta = np.radians(degree)
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))

# scikit-learn essentials


# endregion

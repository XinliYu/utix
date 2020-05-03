import inspect
import warnings
from itertools import product, chain
from typing import Union, List, Callable, Dict, Mapping

import torch
from numpy import iterable
from torch import optim
from torch.nn import parallel
import torch.nn.parallel as nnP

from _util.general_ext import iterable__
from _util.list_ext import nested_lists_regular_shape, nested_lists_get
from allennlp.data.iterators.data_iterator import TensorDict
from allennlp.models import Model
from framework.constants import LOSS_KEY
from pytorch_pretrained_bert import BertAdam


def log_avoid_nan(tensor: torch.Tensor, fill_for_zero: float = 1e-13) -> torch.Tensor:
    """
    Takes logarithm of every number in a tensor, but avoids zeros by replacing zeros by a very small positive number.
    :param tensor: take logarithm of this tensor.
    :param fill_for_zero: replace zeros by this tiny non-negative number.
    :return: a tensor of the same size of the input tensor, with the values being the logarithm of the original numbers.
    """
    return tensor.masked_fill(tensor == 0, fill_for_zero).log()


def log_avoid_nan_(tensor: torch.Tensor, fill_for_zero: float = 1e-13) -> torch.Tensor:
    """
    The in-place version for `log_avoid_nan`.
    """
    return tensor.masked_fill_(tensor == 0, fill_for_zero).log_()


def zeros__(shape: tuple, ref: torch.Tensor = None, **kwargs) -> torch.Tensor:
    """
    Returns a tensor filled with the scalar value 0, with the specified `shape`, and with the same data type and on the same device as the `ref` tensor if `ref` is not `None`.
    :param ref: the returned tensor will have the same data type and be on the same device as this `ref`, if this parameter is assigned.
    :param shape: the shape for the returned tensor.
    :return: a tensor filled with the scalar value 0, with the specified `shape`, and with the same data type and on the same device as the `ref` tensor.
    """
    return torch.zeros(shape, **kwargs) if ref is None else torch.zeros(shape, dtype=ref.dtype, device=ref.device, **kwargs)


def ones__(shape: tuple, ref: torch.Tensor = None, **kwargs) -> torch.Tensor:
    """
    Returns a tensor filled with the scalar value 1, with the specified `shape`, and with the same data type and on the same device as the `ref` tensor if `ref` is not `None`.
    :param ref: the returned tensor will have the same data type and be on the same device as this `ref`, if this parameter is assigned.
    :param shape: the shape for the returned tensor.
    :return: a tensor filled with the scalar value 1, with the specified `shape`, and with the same data type and on the same device as the `ref` tensor.
    """
    return torch.ones(shape, **kwargs) if ref is None else torch.ones(shape, dtype=ref.dtype, device=ref.device, **kwargs)


def batch_tril_ones(batch_size, mat_size: int, diagonal=0, ref: torch.Tensor = None):
    """
    Creates a batch of identical lower-triangular (square) matrices filled with `1`s.
    :param batch_size: the batch size, i.e. the number of identical lower-triangular matrices to create.
    :param mat_size: the matrix size.
    :param diagonal: see the meaning of `diagonal` parameter for `torch.tril`.
    :param ref: the created tensors will have the same data type and on the same device as this reference tensor.
    :return: a batch of identical lower-triangular (square) matrices filled with `1`s.
    """

    return ones__((mat_size, mat_size), ref=ref).tril_(diagonal=diagonal).unsqueeze(0).repeat(batch_size, 1, 1)


def tril_ones(mat_size: int, diagonal=0, ref: torch.Tensor = None):
    return ones__((mat_size, mat_size), ref=ref).tril_(diagonal=diagonal)


def batch_triu_ones(batch_size, mat_size: int, diagonal=0, ref: torch.Tensor = None):
    return ones__((mat_size, mat_size), ref=ref).triu_(diagonal=diagonal).unsqueeze(0).repeat(batch_size, 1, 1)


def triu_ones(mat_size: int, diagonal=0, ref: torch.Tensor = None):
    return ones__((mat_size, mat_size), ref=ref).triu_(diagonal=diagonal)


def has_varkw(func: Callable) -> bool:
    return (inspect.getfullargspec(func.forward).varkw is not None) if isinstance(func, torch.nn.Module) else (inspect.getfullargspec(func).varkw is not None)


# region tensor detaching

def _detach_single_tensor_recursive(x):
    return x.detach().cpu() if isinstance(x, torch.Tensor) else (detach_tensors_to_cpu(recursive=True, **x)[1] if isinstance(x, dict) else (detach_tensors_to_cpu(*x, recursive=True)[0] if not isinstance(x, str) and iterable(x) else x))


def _detach_single_tensor_recursive_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else (detach_tensors_to_numpy(recursive=True, **x)[1] if isinstance(x, dict) else (detach_tensors_to_numpy(*x, recursive=True)[0] if not isinstance(x, str) and iterable(x) else x))


def detach_tensors_to_cpu(*args, recursive=False, **kwargs):
    """
    Detaches tensors in the `args` and the values of `kwargs` from the computational graph and moves them to the CPU (if necessary).
    """
    if not recursive:
        return tuple(x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in args), {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in kwargs}
    else:
        return tuple(_detach_single_tensor_recursive(x) for x in args), {k: _detach_single_tensor_recursive(v) for k, v in kwargs}


def detach_tensors_to_numpy(*args, recursive=False, **kwargs):
    """
    Detaches tensors in the `args` and the values of `kwargs` from the computational graph and converts them to numpy arrays.
    """
    if not recursive:
        return tuple(x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in args), {k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v) for k, v in kwargs}
    else:
        return tuple(_detach_single_tensor_recursive_numpy(x) for x in args), {k: _detach_single_tensor_recursive_numpy(v) for k, v in kwargs}


# endregion

def batch_weighted_sum_of_features(batch_features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Computes weighted sum of features vectors for a batch.
    :param batch_features: the data batch tensor of shape `batch_size * seq_len * feature_dim`.
    :param weights: the weight matrix of shape `batch_size * seq_len`.
    :return: a `batch_size * feature_dim` tensor, where the `i`th row is a weighted sum by applying weights at `weights[i]` to feature vectors in `x[i]`.
    """

    # the `unsqueeze` method unsqueezes the `similarities` to a `batch_size * 1 * seq_len` tensor,
    # so that it can then be properly applied on `x`, a `batch_size * seq_len * feature_dim` tensor to get the weighted sum.
    return (weights.unsqueeze(-2) @ batch_features).squeeze()


def config_torch_cuda(model, cuda_device_index: Union[int, List[int]], parallel_init):
    if (isinstance(cuda_device_index, int) and cuda_device_index >= 0) or (isinstance(cuda_device_index, list) and len(cuda_device_index) != 0):
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            if __debug__:
                print("CUDA is not available; fall back to CPU.")
        else:
            device, device_index = first_torch_cuda_device(cuda_device_index)
            if 0 <= device_index < gpu_count:
                model.cuda(device)
            else:
                if __debug__:
                    print("cuda:{} is not available; fall back to CPU.".format(device_index))
            if parallel_init and isinstance(cuda_device_index, list) and len(cuda_device_index) > 1:
                return parallel_init(model, [cuda_index for cuda_index in cuda_device_index if cuda_index < gpu_count])
    return model


def first_torch_cuda_device(cuda_device_index: Union[int, List[int]]):
    """
    Gets the cuda device object of the specified index, or the first index in a list of specified cuda device indices.
    :param cuda_device_index: a non-negative integer or a list of non-negative integers as cuda devices indices.
    :return: a tuple, the first being the torch cuda device object, the second being the device index; if `cuda_device_index` is `None`, negative or empty, then `None, -1` is returned.
    """
    if isinstance(cuda_device_index, int):
        if cuda_device_index >= 0:
            return torch.device('cuda:{}'.format(cuda_device_index)), cuda_device_index
    elif isinstance(cuda_device_index, list) and len(cuda_device_index) != 0:
        return torch.device('cuda:{}'.format(cuda_device_index[0])), cuda_device_index[0]
    return None, -1


def get_adam_optimizer(model, lr=0.001, **kwargs):
    """
    A convenient function to get the Adam optimizer.
    :param model: the model to apply this optimizer on.
    :param lr: the learning rate; the default is 1e-3 for small-sized an medium-sized models; for large-sized models, use a smaller learning rate.
    :return: an Adam optimizer.
    """
    return optim.Adam(model.parameters(), lr=lr, **kwargs)


def get_bert_optimizer(model, lr=2e-5, **kwargs):
    """
    A convenient function to get the BERT-Adam optimizer.
    :param model: the model to apply this optimizer on.
    :param lr: the learning rate; ! for a complicated model like bert, a tiny learning rate like 2e-5 is usually desired.
    :return: a BERT-Adam optimizer.
    """
    if lr > 2e-4:
        warnings.warn(f"for a complicated model like bert, a tiny learning rate like 2e-5 is usually desired; got `{lr}`")
    return BertAdam(model.parameters(), lr=lr, **kwargs)


def tensor__(x, padding=0, dtype=None, **kwargs):
    shape = nested_lists_regular_shape(x)
    result = torch.full(shape, fill_value=padding, dtype=dtype, **kwargs)
    for index in product(*(range(x) for x in shape[:-1])):
        row = nested_lists_get(x, index)
        if row is not None:
            result[index][:len(row)] = torch.tensor(row, dtype=dtype)
    return result


def has_tensor(obj) -> bool:
    """
    Given a possibly complex data structure,
    check if it has any torch.Tensors in it.
    """
    if isinstance(obj, torch.Tensor):
        return True
    elif isinstance(obj, dict):
        return any(has_tensor(value) for value in obj.values())
    elif isinstance(obj, (list, tuple)):
        return any(has_tensor(item) for item in obj)
    else:
        return False


def move_tensor_dict_to_device(tensor_dict: Mapping, device_id: int, non_blocking=True):
    # ! returns a new dictionary
    # so that the original tensor dict is not modified;
    #   the original tensor dict might be cached;
    #   if we replace CPU tensors by GPU tensors in-place, it can cause memory overflow
    if device_id < 0:
        return {k: v.cpu() for k, v in tensor_dict.items() if isinstance(v, torch.Tensor)}
    else:
        return {k: v.cuda(device_id, non_blocking=non_blocking) for k, v in tensor_dict.items() if isinstance(v, torch.Tensor)}


def parallel_tensor_dict(tensor_dicts: List[Mapping],
                         model: Model,
                         device_ids: List,
                         loss_key='loss',
                         atom_types=(str,)) -> Dict[str, torch.Tensor]:
    """
    Performs a forward pass using multiple GPUs.  This is a simplification
    of torch.nn.parallel.data_parallel to support the allennlp model
    interface.
    """
    if len(tensor_dicts) > len(device_ids):
        raise ValueError("the number of tensor dicts must be the same as the number of device ids")

    # region 1 - copy data and model to multiple GPUS

    # NOTE, there can be fewer tensor dicts,
    # and in this case the number of used device ids might be less than the number of provided device ids
    moved = [move_tensor_dict_to_device(tensor_dict, device_id) for tensor_dict, device_id in zip(tensor_dicts, device_ids)]
    used_device_ids = device_ids[:len(moved)]

    # must replicate the model to the GPUs every time, because its parameters have been updated
    replicas = nnP.replicate(model, used_device_ids)

    # endregion

    # region 2 - get the outputs

    # the outputs must be a dictionary of results returned by each GPU
    outputs = nnP.parallel_apply(replicas,
                                 [()] * len(tensor_dicts),  # no positional argument
                                 moved,  # the tensor dict as named arguments
                                 used_device_ids)

    # endregion

    # region 3 - gather the results on the first GPU

    result = {}
    for k, v in outputs[0].items():
        if k == loss_key:  # special treatment for the loss key
            result[k] = nnP.gather([output[k].unsqueeze(0) for output in outputs],
                                   target_device=used_device_ids[0],
                                   dim=0).mean()
        else:
            if isinstance(v, torch.Tensor):
                result[k] = [nnP.gather([output[k]], target_device=used_device_ids[0], dim=0) for output in outputs]
            elif iterable__(v, atom_types=atom_types):
                result[k] = tuple(chain([output[k] for output in outputs]))
            else:
                result[k] = tuple(output[k] for output in outputs)

    # endregion

    return result


def move_to_device(obj, cuda_device: int):
    """

    Given a structure (possibly) containing Tensors on the CPU,
    move all the Tensors to the specified GPU (or do nothing, if they should be on the CPU).
    """
    # pylint: disable=too-many-return-statements
    if cuda_device < 0 or not has_tensor(obj):
        return obj
    elif isinstance(obj, torch.Tensor):
        return obj.cuda(cuda_device, non_blocking=True)
    elif isinstance(obj, dict):
        return {key: move_to_device(value, cuda_device) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, cuda_device) for item in obj]
    elif isinstance(obj, tuple) and hasattr(obj, '_fields'):
        # This is the best way to detect a NamedTuple, it turns out.
        return obj.__class__(*[move_to_device(item, cuda_device) for item in obj])
    elif isinstance(obj, tuple):
        return tuple([move_to_device(item, cuda_device) for item in obj])
    else:
        return obj


def allen_data_parallel(batch_group: List[TensorDict],
                        model: Model,
                        cuda_devices: List) -> Dict[str, torch.Tensor]:
    """
    Performs a forward pass using multiple GPUs.  This is a simplification
    of torch.nn.parallel.data_parallel to support the allennlp model
    interface.
    """
    assert len(batch_group) <= len(cuda_devices)

    moved = [move_to_device(batch, device) for batch, device in zip(batch_group, cuda_devices)]

    used_device_ids = cuda_devices[:len(moved)]
    # Counterintuitively, it appears replicate expects the source device id to be the first element
    # in the device id list. See torch.cuda.comm.broadcast_coalesced, which is called indirectly.
    replicas = nnP.replicate(model, used_device_ids)

    # We pass all our arguments as kwargs. Create a list of empty tuples of the
    # correct shape to serve as (non-existent) positional arguments.
    inputs = [()] * len(batch_group)
    outputs = nnP.parallel_apply(replicas, inputs, moved, used_device_ids)

    # Only the 'loss' is needed.
    # a (num_gpu, ) tensor with loss on each GPU
    if LOSS_KEY in outputs[0]:
        result = {LOSS_KEY: nnP.gather([output[LOSS_KEY].unsqueeze(0) for output in outputs],
                                       target_device=used_device_ids[0],
                                       dim=0).mean()}
    else:
        result = {}

    for key in outputs[0]:
        if key == 'tags':
            result[key] = list(chain([output[key] for output in outputs]))
        elif key != LOSS_KEY:
            result[key] = [nnP.gather([output[key]], target_device=used_device_ids[0], dim=0) for output in outputs]
    return result

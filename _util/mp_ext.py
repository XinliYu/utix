from collections import defaultdict, Counter
from enum import IntEnum
from multiprocessing import get_context, Pool, Manager
from multiprocessing.queues import SimpleQueue, Queue
from multiprocessing.context import BaseContext
from time import sleep
from typing import Tuple, Callable, List, Union, Iterator, Iterable

from tqdm import tqdm

import _util.dict_ext as dex
from _util.general_ext import hprint_pairs, hprint_message
from _util.iter_ext import split_iter, slice_iter, slices__
from _util.time_ext import tic, toc


class MPTarget:
    __slots__ = ('use_queue',)

    def __init__(self, use_queue=False):
        self.use_queue = use_queue

    def target(self, pid, data, *args):
        raise NotImplementedError

    def __call__(self, pid, data, *args):
        if self.use_queue:
            iq: SimpleQueue = data
            oq: SimpleQueue = args[0]
            while not iq.empty():
                oq.put(self.target(pid, iq.get(), *args[1:]))
        else:
            return self.target(pid, data, *args)


def dispatch_data(num_p: int, data_iter: Union[Iterator, Iterable, List], args: Tuple, print_out=__debug__):
    if num_p <= 0:
        raise ValueError(f"The number of processes specified in `nump_p` must be positive, but it is {num_p}.")

    tic("Splitting task", verbose=print_out)
    splits = split_iter(it=data_iter, num_splits=num_p, use_tqdm=print_out)
    toc(print_out=print_out)

    num_p = len(splits)
    if num_p == 0:
        raise ValueError(f"The number of data splits is zero. Possibly no data was read from the provided iterator.")
    else:
        job_args = [None] * num_p
        for pidx in range(num_p):
            if print_out:
                hprint_pairs(('pid', pidx), ('workload', len(splits[pidx])))
            job_args[pidx] = (pidx, splits[pidx]) + args
        return job_args


def start_and_wait_jobs(jobs: [Union[List, Tuple]], interval: float = 0.01):
    for p in jobs:
        p.start()
        if interval != 0:
            sleep(interval)
    for p in jobs:
        p.join()


def parallel_process(num_p, data_iter: Union[Iterator, Iterable, List], target: Callable, args: Tuple, ctx: BaseContext = None, print_out=__debug__):
    if isinstance(target, MPTarget):
        target.use_queue = False
    if ctx is None:
        ctx = get_context('spawn')
    job_args = dispatch_data(num_p=num_p, data_iter=data_iter, args=args, print_out=print_out)
    jobs = [None] * num_p
    for i in range(num_p):
        jobs[i] = ctx.Process(target=target, args=job_args[i])
    start_and_wait_jobs(jobs)


def parallel_process_by_pool(num_p, data_iter: Union[Iterator, Iterable, List], target: Callable, args: Tuple = (), print_out=__debug__, cross_merge_output: bool = False, mergers: Union[List, Tuple] = None):
    if isinstance(target, MPTarget):
        target.use_queue = False
    job_args = dispatch_data(num_p=num_p, data_iter=data_iter, args=args, print_out=print_out)
    pool = Pool(processes=num_p)
    rst = pool.starmap(target, job_args)
    return merge_results(result_collection=list(zip(*rst)), mergers=mergers) if cross_merge_output else rst


def merge_results(result_collection, mergers: Union[List, Tuple] = None):
    def _default_merger_1(results, merge_method: str):
        if merge_method == 'list':
            return sum(results, [])
        elif merge_method == 'list_dict':
            return dex.merge_list_dicts(results, in_place=True)
        elif merge_method == 'set_dict':
            return dex.merge_set_dicts(results, in_place=True)
        elif merge_method == 'counter_dict':
            return dex.merge_counter_dicts(results, in_place=True)
        elif merge_method == 'sum':
            return sum(results)
        raise ValueError(f"The provided results does not support the default merge method {merge_method}.")

    def _default_merger_2(results):
        size_results = len(results)
        if size_results == 0:
            return results

        rst_type = type(results[0])
        if rst_type in (int, float, bool):
            return sum(results)
        elif rst_type is list:
            return sum(results, [])
        elif rst_type is tuple:
            return sum(results, ())
        elif rst_type in (dict, defaultdict):
            val_type = None
            for i in range(size_results):
                if len(results[i]) != 0:
                    val_type = type(next(iter(results[i].values())))
                    break
            if val_type is None:
                return {}
            elif val_type is list:
                return dex.merge_list_dicts(results[i:], in_place=True)
            elif val_type is set:
                return dex.merge_set_dicts(results[i:], in_place=True)
            elif val_type is Counter:
                return dex.merge_counter_dicts(results[i:], in_place=True)
        raise ValueError("The provided results does not support the default merge.")

    return tuple((merger(results) if callable(merger) else _default_merger_1(results, merger)) for results, merger in zip(result_collection, mergers)) if mergers \
        else tuple(_default_merger_2(results) for results in result_collection)


def parallel_process_by_queue(num_p, data_iter, target, args, ctx: BaseContext = None, task_unit_size=5000, print_out=__debug__):
    if isinstance(target, MPTarget):
        target.use_queue = True
    if ctx is None:
        ctx = get_context('spawn')
    iq = Queue(ctx=ctx)
    oq: Manager = ctx.Manager().Queue()

    tic(f"Creating input queue with task unit size {task_unit_size}", verbose=print_out)
    cnt_task_unit = 0
    for item in tqdm(slices__(data_iter, task_unit_size)):
        iq.put(item)
        cnt_task_unit += 1
    jobs = [None] * num_p
    for i in range(num_p):
        jobs[i] = ctx.Process(target=target, args=(i, iq, oq) + args)
    toc()

    tic(f"Working on {cnt_task_unit} task units with {num_p} processes", verbose=print_out)
    start_and_wait_jobs(jobs)

    out = []
    while not oq.empty():
        out.append(oq.get_nowait())
    toc()
    return out


# region obsolete methods

def dispatch_files(num_p, file_paths: List[str], args: Tuple):
    num_files = len(file_paths)
    if __debug__:
        hprint_message(f"Dispatching {num_p} processes for {num_files} files ...")
    num_files_per_process = int(num_files / num_p)
    num_files_overflow = num_files - num_files_per_process * num_p
    file_idx_start = 0
    job_args = [None] * num_p
    for pidx in range(num_p):
        file_idx_end = file_idx_start + num_files_per_process + (pidx < num_files_overflow)
        if num_p == 1:
            curr_file_paths = file_paths
        elif pidx == num_p - 1:
            curr_file_paths = file_paths[file_idx_start:]
        else:
            curr_file_paths = file_paths[file_idx_start:file_idx_end]
            file_idx_start = file_idx_end
        if __debug__:
            hprint_pairs(('pid', pidx), ('num of files', len(curr_file_paths)))
        job_args[pidx] = (pidx, curr_file_paths) + args
    return job_args


def parallel_process_files(num_p, file_paths: List[str], target: Callable, args: Tuple, ctx: BaseContext = None):
    if ctx is None:
        ctx = get_context('spawn')
    job_args = dispatch_files(num_p=num_p, file_paths=file_paths, args=args)
    jobs = [None] * num_p
    for i in range(num_p):
        jobs[i] = ctx.Process(target=target, args=job_args[i])
    start_and_wait_jobs(jobs)


def parallel_process_files_by_pool(num_p, file_paths: List[str], target: Callable, args: Tuple):
    job_args = dispatch_files(num_p=num_p, file_paths=file_paths, args=args)
    pool = Pool(processes=num_p)
    return pool.map(target, job_args)

# endregion

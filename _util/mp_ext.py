import warnings
from collections import defaultdict, Counter
from functools import partial
from itertools import islice, chain
from multiprocessing import get_context, Pool, Manager, freeze_support
from multiprocessing.context import BaseContext
from multiprocessing.queues import SimpleQueue, Queue
from time import sleep
from typing import Tuple, Callable, List, Union, Iterator, Iterable

from tqdm import tqdm
from os import path
import utix.dictex as dex
import utix.pathex as paex
import utix.ioex as ioex
from utix.general import hprint_pairs, hprint_message, cpu_count, wprint_message
from utix.iterex import split_iter, slices__, chunk_iter
from utix.timex import tic, toc
import uuid


# region basics

def get_num_p(num_p=None):
    """
    Gets the number of workers given the suggested number specified by `num_p`.
    If `num_p` is `None` or 0 or a negative number, the number of CPUs minus 1 or 2 will be returned.
    Otherwise, the smaller number between `num_p` and the number of CPUs will be returned.

    :param num_p: the suggested number of workers.
    """
    if num_p is None or num_p <= 0:
        num_workers = cpu_count()
        if num_workers <= 8:
            return num_workers - 1
        else:
            return num_workers - 2
    else:
        return min(num_p, cpu_count())


class MPResultTuple(tuple):
    pass


class MPTarget:
    """
    Wraps a multi-processing target callable and provides rich options for convenient multi-processing operations.
    """
    __slots__ = ('use_queue',
                 '_target',
                 '_pass_each_data_item',
                 '_pass_pid',
                 '_wait_time',
                 '_unpack_singleton_result',
                 '_data_from_files',
                 '_remove_none',
                 '_result_dump_path',
                 '_result_dump_method',
                 '_result_dump_file_pattern',
                 '_is_target_iter',
                 '_always_return_results',
                 'name')

    def __init__(self, use_queue=False, target=None, pass_each=False, pass_pid=True, wait_time=0.5, name='target', unpack_singleton_result=False, common_func=False, data_from_files=False, remove_none=False,
                 result_dump_dir=None, result_dump_name=None, result_dump_method=ioex.pickle_save, always_return_results=False, is_target_iter=False):
        """`

        :param use_queue: `True` to use queue for passing multi-processing data.
        :param target: the multi-processing target callable.
        :param pass_each: will pass each item from the input (e.g. one file, or one single data item) to the `target` function, rather than passing a list of them to the `target`; in multi-processing, each `target` actually receives a list of assigned items, but very often the `target` is a normal function intented for processing just one single file or one single data item, and in this case we must set this `pass_each` attribute to `True`.
        :param pass_pid:
        :param wait_time:
        :param name:
        :param unpack_singleton_result:
        :param common_func: a convenience parameter; `True` to indicate `pass_pid` is `False`, `pass_each_data_item` is `True` and `unpack_singleton_result` is `True`.
        :param data_from_files: `True` to indicate the `target` is to process data from files; in this case, if `pass_each_data_item` is also `True`, then each line of the file will be passed to the `target`, or otherwise all lines read from the files will be passed to the `target`.
        :param remove_none: effective only if `pass_each_data_item` is set `True`; set this to `True` if to ignore `None` result produced by the `target`.
        `"""
        if common_func:
            pass_pid = False
            pass_each = True
            unpack_singleton_result = True
        self.use_queue = use_queue
        self._target = target
        self._pass_each_data_item = pass_each
        self._pass_pid = pass_pid
        self._wait_time = wait_time
        self._unpack_singleton_result = unpack_singleton_result
        self._data_from_files = data_from_files
        self._remove_none = remove_none
        self._result_dump_path = result_dump_dir
        self._result_dump_method = result_dump_method or ioex.pickle_save
        self._result_dump_file_pattern = result_dump_name
        self._is_target_iter = is_target_iter
        self._always_return_results = always_return_results
        self.name = name

    def target(self, pid, data, *args):
        if self._target is not None:
            if self._pass_pid:
                rst = self._target(pid, data, *args)
            else:
                rst = self._target(data, *args)
            return list(rst) if self._is_target_iter else rst
        else:
            raise NotImplementedError

    def __call__(self, pid, data, *args):
        hprint_message('initialized', f'{self.name}{pid}')
        no_job_cnt = 0
        if self._pass_each_data_item:
            if not self._result_dump_path and self.use_queue:
                # TODO file based queue
                iq: Queue = data
                oq: Queue = args[0]
                flags = args[1]
                while True:
                    while not iq.empty():
                        data = iq.get()
                        if self._data_from_files:
                            data = ioex.iter_all_lines_from_all_files(input_paths=data, use_tqdm=True)
                            _data = (self.target(pid, dataitem, *args[2:]) for dataitem in data)
                            oq.put(MPResultTuple((x for x in _data if x is not None) if self._remove_none else _data))
                        else:
                            if self._unpack_singleton_result and len(data) == 1:
                                oq.put(self.target(pid, data[0], *args[2:]))
                            else:
                                oq.put(MPResultTuple(self.target(pid, dataitem, *args[2:]) for dataitem in data))
                    if not flags or flags[0]:
                        return
                    no_job_cnt += 1
                    if no_job_cnt % 10 == 0:
                        hprint_pairs(('no jobs for', f'{self.name}{pid}'), ('wait for', self._wait_time))
                    sleep(self._wait_time)
            else:
                if self._data_from_files:
                    data = ioex.iter_all_lines_from_all_files(input_paths=data, use_tqdm=True)
                    _data = (self.target(pid, dataitem, *args) for dataitem in data)
                    output = MPResultTuple((x for x in _data if x is not None) if self._remove_none else _data)
                elif self._unpack_singleton_result and len(data) == 1:
                    output = self.target(pid, data[0], *args)
                else:
                    data = tqdm(data, desc=f'pid: {pid}')
                    _data = (self.target(pid, dataitem, *args) for dataitem in data)
                    # use a fake data type `MPResultTuple` (actually just a tuple) to inform the outside multi-processing method that the output comes from each data item
                    output = MPResultTuple((x for x in _data if x is not None) if self._remove_none else _data)
        elif not self._result_dump_path and self.use_queue:
            iq: Queue = data
            oq: Queue = args[0]
            flags = args[1]
            while True:
                while not iq.empty():
                    data = iq.get()
                    if self._data_from_files:
                        data = ioex.iter_all_lines_from_all_files(input_paths=data, use_tqdm=True)
                    result = self.target(pid, data, *args[2:])
                    oq.put(result[0] if self._unpack_singleton_result and hasattr(result, '__len__') and hasattr(result, '__getitem__') and len(result) == 1 else result)
                if not flags or flags[0]:
                    return
                no_job_cnt += 1
                if no_job_cnt % 10 == 0:
                    hprint_pairs(('no jobs for', f'{self.name}{pid}'), ('wait for', self._wait_time))
                sleep(self._wait_time)
        else:
            if self._data_from_files:
                data = ioex.iter_all_lines_from_all_files(input_paths=data, use_tqdm=True)
            output = self.target(pid, data, *args)
            if self._unpack_singleton_result and hasattr(output, '__len__') and hasattr(output, '__getitem__') and len(output) == 1:
                output = output[0]
        if self._result_dump_path:
            dump_path = path.join(self._result_dump_path, (ioex.pathex.append_timestamp(str(uuid.uuid4())) + '.mpb' if self._result_dump_file_pattern is None else self._result_dump_file_pattern.format(pid)))
            self._result_dump_method(output, dump_path)
            return dump_path if not self._always_return_results else output
        else:
            return output


# class MPWriteOutput:
#     def __init__(self, target, output_path):
#         self._target = target
#         self._output_path = output_path
#
#     def __call__(self, *args):
#         output = self._target(*args)


def _default_result_merge(results):
    if isinstance(results[0], list):
        if all((isinstance(result, list) for result in results[1:])):
            results = tqdm(results)
            results.set_description('merging lists')
            return sum(results, [])
    elif isinstance(results[0], tuple):
        if all((isinstance(result, tuple) for result in results[1:])):
            results = tqdm(results)
            results.set_description('merging tuples')
            return sum(results, ())
    elif isinstance(results[0], dict):
        if all((isinstance(result, dict) for result in results[1:])):
            output = results[0]
            results = tqdm(results[1:])
            results.set_description('merging dicts')
            for d in results:
                output.update(d)
            return output
    return results


# endregion

def mp_chunk_file(input_path, output_path, chunk_size, num_p, chunk_file_pattern='chunk_{}', use_tqdm=False, display_msg=None, verbose=__debug__, global_chunking=False):
    if global_chunking or isinstance(input_path, str) or len(input_path) == 1:
        ioex.chunk_file(input_path, output_path=output_path, chunk_size=chunk_size, chunk_file_pattern=chunk_file_pattern, use_uuid=True, use_tqdm=use_tqdm, display_msg=display_msg, verbose=verbose)
    else:
        parallel_process_by_pool(num_p=num_p, data_iter=input_path,
                                 target=MPTarget(target=ioex.chunk_file, pass_each=True, pass_pid=False),
                                 args=(output_path, chunk_size, chunk_file_pattern, True, use_tqdm, display_msg, verbose))


class MPProvider:
    def __init__(self, create_iterator, chunk_size, wait_time=1, name='provider', pass_each_data_item=False):
        self.create_iterator = create_iterator
        self.chunk_size = chunk_size
        self._wait_time = wait_time
        self.name = name
        self.pass_each_data_item = pass_each_data_item

    def __call__(self, pid, iq: Queue, data, *args):
        if self.pass_each_data_item:
            it = chain(*(chunk_iter(self.create_iterator(dataitem, *args), chunk_size=self.chunk_size, as_list=True) for dataitem in data))
        else:
            it = chunk_iter(self.create_iterator(data, *args), chunk_size=self.chunk_size, as_list=True)
        hprint_message('initialized', f'{self.name}{pid}')
        while True:
            while not iq.full():
                try:
                    obj = next(it)
                except StopIteration:
                    return
                iq.put(obj)
            hprint_pairs(('full queue for', f'{self.name}{pid}'), ('wait for', self._wait_time))
            sleep(self._wait_time)


def mp_read(data_iter, provider, producer, provider_args=(), producer_args=(), num_providers=1, num_producers=4, ctx=None, checking_interval=0.5, print_out=True):
    provider_jobs = [None] * num_providers
    producer_jobs = [None] * num_producers

    if ctx is None:
        ctx = get_context()
    manager = ctx.Manager()
    iq = manager.Queue()
    oq: Queue = manager.Queue()
    flags = manager.list([False])
    if isinstance(producer, MPTarget):
        producer.use_queue = True

    provider_args = dispatch_data(num_p=num_providers, data_iter=data_iter, args=provider_args, print_out=print_out)

    for i in range(num_providers):
        provider_jobs[i] = ctx.Process(target=provider, args=(i, iq) + provider_args[i][1:])
    for i in range(num_producers):
        producer_jobs[i] = ctx.Process(target=producer, args=(i, iq, oq, flags) + producer_args)

    start_jobs(provider_jobs)
    start_jobs(producer_jobs)

    while True:
        while not oq.empty():
            objs = oq.get()
            yield from objs
        sleep(checking_interval)
        any_active_provider = any((job.is_alive() for job in provider_jobs))
        any_active_producer = any((job.is_alive() for job in producer_jobs))
        if not any_active_provider:
            if not flags[0]:
                flags[0] = True
                hprint_message('all providers done!')
            if not any_active_producer:
                hprint_message('all jobs done!')
                break


def _default_mp_read_lines(files):
    return ioex.read_all_lines_from_all_files(input_path=files, use_tqdm=True)


def get_mp_cache_files(num_p, file_paths, sort=True, verbose=__debug__, cache_dir_path=None, chunk_size=100000, sort_use_basename=False, rebuild_on_change=True):
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    else:
        file_paths = paex.sort_paths(file_paths, sort=sort, sort_by_basename=sort_use_basename)

    num_file_paths = len(file_paths)
    if verbose:
        hprint_pairs(('number of files', num_file_paths), ('num_p', num_p))
    if num_file_paths < num_p:
        if cache_dir_path is None:
            if len(file_paths) == 1:
                cache_dir_path = paex.add_to_main_name(file_paths[0], prefix='.mp.')
            else:
                cache_dir_path = path.join(path.dirname(file_paths[0]), '.mp')
        cache_file_ext_name = paex.get_ext_name(file_paths[0])

        tic('Constructs multi-processing cache files at path ' + path.join(cache_dir_path, '*' + cache_file_ext_name))

        mp_cache_file_paths = None
        files_id_path = cache_dir_path + '.id'
        if path.exists(cache_dir_path):
            if path.exists(files_id_path):
                old_files_id = ioex.read_all_text(files_id_path).strip()
                new_files_id = ioex.get_files_id(file_paths)  # the file paths are already sorted above, so the files_id would be the same for the same files if they are not changed
                if new_files_id != old_files_id:
                    hprint_message(f'Files are changed; rebuilding cache at', cache_dir_path)
                    import shutil, os
                    shutil.rmtree(cache_dir_path)  # removes file cache
                    os.remove(files_id_path)  # removes the id file
                else:
                    mp_cache_file_paths = paex.get_files_by_pattern(dir_or_dirs=cache_dir_path, pattern='*' + cache_file_ext_name, full_path=True, recursive=False, sort=sort, sort_use_basename=sort_use_basename)
                    if not mp_cache_file_paths:
                        wprint_message('Cache directory exists, but nothing there', cache_dir_path)
            else:
                hprint_message(f'Files id does not exist; rebuilding cache at', cache_dir_path)
                import shutil
                shutil.rmtree(cache_dir_path)  # removes file cache
        if not mp_cache_file_paths:
            ioex.write_all_text(ioex.get_files_id(file_paths), files_id_path)
            ioex.write_all_lines(iterable=ioex.iter_all_lines_from_all_files(file_paths), output_path=cache_dir_path, create_dir=True, chunk_size=chunk_size, chunked_file_ext_name=cache_file_ext_name)
            mp_cache_file_paths = paex.get_files_by_pattern(dir_or_dirs=cache_dir_path, pattern='*' + cache_file_ext_name, full_path=True, recursive=False, sort=sort, sort_use_basename=sort_use_basename)

        if mp_cache_file_paths:
            hprint_message(
                title='number of multi-processing cache files',
                content=len(mp_cache_file_paths)
            )
        else:
            raise IOError('multi-processing cache files are not found')
        file_paths = mp_cache_file_paths
        num_p = min(num_p, len(file_paths))
        toc('Done!')
    return num_p, file_paths


def mp_read_from_files(num_p, input_path, target=None, args=(), sort=True, verbose=__debug__, cache_dir_path=None, chunk_size=100000, sort_use_basename=False, result_merge='default'):
    """
    Read files with multiple processes. Suitable for faster reading of files that are not frequently changed.
    """
    num_p, input_path = get_mp_cache_files(
        num_p=num_p,
        file_paths=input_path,
        sort=sort,
        verbose=verbose,
        cache_dir_path=cache_dir_path,
        chunk_size=chunk_size,
        sort_use_basename=sort_use_basename
    )

    if target is None:
        target = MPTarget(target=partial(ioex.read_all_lines_from_all_files, use_tqdm=True), pass_pid=False)
    output = parallel_process_by_pool(num_p=num_p, data_iter=input_path, target=target, args=args, verbose=verbose)
    if isinstance(output[0], MPResultTuple):
        output = sum(output, ())
    if result_merge == 'default':
        return _default_result_merge(output)
    elif result_merge == 'chain':
        return chain(*output)
    else:
        return output


def mp_read_write_files(num_p, input_file_paths, output_path, target, args=(), output_multiple_files=True, input_cache_dir_path=None, sort=True, verbose=__debug__, chunk_size=100000, sort_use_basename=False):
    num_p, file_paths = get_mp_cache_files(
        num_p=num_p,
        file_paths=input_file_paths,
        sort=sort,
        verbose=verbose,
        cache_dir_path=input_cache_dir_path,
        chunk_size=chunk_size,
        sort_use_basename=sort_use_basename
    )

    parallel_process_by_pool(num_p=num_p, data_iter=file_paths, target=target, args=args, verbose=verbose)
    if target is None:
        target = MPTarget(target=partial(ioex.read_all_lines_from_all_files, use_tqdm=True), pass_pid=False)


# def mp_read(input_path, num_p, base_iter, args, cache_size=100000, chunk_size=1000, no_chunking=False):
#     if no_chunking:
#         pass
#     else:
#         input_chunk_path = paex.append_to_main_name(input_path, '_chunks')
#
#         if not path.exists(input_chunk_path):
#
#             pass
#         else:
#             if isinstance(target, MPTarget):
#                 target.use_queue = True
#             if ctx is None:
#                 ctx = get_context('spawn')
#             iq = Queue(ctx=ctx)
#             oq: Manager = ctx.Manager().Queue()
#
#             tic(f"Creating input queue with task unit size {task_unit_size}", verbose=print_out)
#             cnt_task_unit = 0
#             for item in tqdm(slices__(data_iter, task_unit_size)):
#                 iq.put(item)
#                 cnt_task_unit += 1
#             jobs = [None] * num_p
#             for i in range(num_p):
#                 jobs[i] = ctx.Process(target=target, args=(i, iq, oq) + args)
#             toc()
#
#             tic(f"Working on {cnt_task_unit} task units with {num_p} processes", verbose=print_out)
#             start_and_wait_jobs(jobs)
#
#             out = []
#             while not oq.empty():
#                 out.append(oq.get_nowait())
#             toc()
#             return out


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


def start_jobs(jobs: [Union[List, Tuple]], interval: float = 0.01):
    for p in jobs:
        p.start()
        if interval != 0:
            sleep(interval)


def start_and_wait_jobs(jobs: [Union[List, Tuple]], interval: float = 0.01):
    start_jobs(jobs, interval=interval)
    for p in jobs:
        p.join()


def parallel_process(num_p, data_iter: Union[Iterator, Iterable, List], target: Callable, args: Tuple, ctx: BaseContext = None, verbose=__debug__):
    if isinstance(target, MPTarget):
        target.use_queue = False
    if ctx is None:
        ctx = get_context('spawn')
    job_args = dispatch_data(num_p=num_p, data_iter=data_iter, args=args, print_out=verbose)
    jobs = [None] * num_p
    for i in range(num_p):
        jobs[i] = ctx.Process(target=target, args=job_args[i])
    start_and_wait_jobs(jobs)


def parallel_process_by_pool(num_p,
                             data_iter: Union[Iterator, Iterable, List],
                             target: Callable,
                             args: Tuple = (),
                             verbose=__debug__,
                             merge_output: bool = False,
                             mergers: Union[List, Tuple] = None,
                             debug=False,
                             return_job_splits=False,
                             load_dumped_results=False,
                             result_dump_load_method=ioex.pickle_load):
    """
    Parallel process data with multiprocessing pool. In comparison to spark, this method is more flexible and efficient to process medium sized data on the local machine.
    :param num_p: the number of processors to use.
    :param data_iter: the data iterator.
    :param target:
    :param args:
    :param verbose:
    :param merge_output:
    :param mergers:
    :param debug:
    :return:
    """
    if num_p == 1:
        if isinstance(target, MPTarget):
            return target.target(0, data_iter, *args)
        else:
            return target(0, data_iter, *args)

    if num_p is None or num_p <= 0:
        num_p = cpu_count()
    if isinstance(target, MPTarget):
        target.use_queue = False

    if debug > 1:
        num_p = debug
        data_iter = islice(data_iter, 500)

    job_args = dispatch_data(num_p=num_p, data_iter=data_iter, args=args, print_out=verbose)
    if debug is True or debug == 1:
        rst = target(*job_args[0])
    else:
        pool = Pool(processes=num_p)
        try:
            rst = pool.starmap(target, job_args)
        except Exception as err:
            pool.close()
            pool.join()
            raise err

        pool.close()
        pool.join()

    if load_dumped_results:
        if isinstance(rst[0], str) and path.isfile(rst[0]):
            rst = [result_dump_load_method(file_path) for file_path in rst]
        else:
            warnings.warn(f'Expected to load results from dumped files; in this case the returned result from each process must be a file path; got {type(rst[0])}')

    if debug == 1:
        raise ValueError('debug is set True or 1, in this case the result merge will not work; change debug to an integer higher than 2')

    if merge_output:
        rst = merge_results(result_collection=list(zip(*rst)), mergers=mergers)

    return (rst, (job_arg[1] for job_arg in job_args)) if return_job_splits else rst


def merge_results(result_collection, mergers: Union[List, Tuple] = None):
    def _default_merger_1(results, merge_method: str):
        if merge_method == 'list':
            return sum(results, [])
        elif merge_method == 'dict':
            return dex.merge_dicts(results, in_place=True)
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
        err = None
        rst_type = type(results[0])
        if rst_type in (int, float, bool):
            return sum(results)
        elif rst_type is list:
            return sum(results, [])
        elif rst_type is tuple:
            return sum(results, ())
        elif rst_type in (dict, defaultdict):
            peek_value = None
            for i in range(size_results):
                if len(results[i]) != 0:
                    peek_value = next(iter(results[i].values()))
                    break
            if isinstance(peek_value, list):
                return dex.merge_list_dicts(results[i:], in_place=True)
            elif isinstance(peek_value, set):
                return dex.merge_set_dicts(results[i:], in_place=True)
            elif isinstance(peek_value, Counter):
                return dex.merge_counter_dicts(results[i:], in_place=True)
            else:
                try:
                    return dex.sum_dicts(results[i:], in_place=True)
                except Exception as err:
                    pass
        raise ValueError(f"The provided results does not support the default merge. Error: {err or f'type {rst_type} not supported'}.")

    return tuple((merger(results) if callable(merger) else _default_merger_1(results, merger)) for results, merger in zip(result_collection, mergers)) if mergers \
        else tuple(_default_merger_2(results) for results in tqdm(result_collection))


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

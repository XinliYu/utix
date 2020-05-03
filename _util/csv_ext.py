import csv

from tqdm import tqdm


# region CSV reading
from utilx.general import str2val, tqdm_wrap


class CsvReader:
    pass


# class RowToVec:
#     def __init__(self, conversions: dict, columns_to_index=None):
#         self._convs = conversions
#         self._cols2idx = {col: {} for col in columns_to_index}
#
#     def __call__(self, row):
#         for k, v in self._convs.items():


# def row2vec_iter(row_iter: Union[Iterable, Iterator], separator:str='\t', header:bool=True, to_vec_funcs:dict=None, use_pandas=False, pandas_chunk_size=512) -> List[List]:
#     if use_pandas:
#         for chunk in pd.read_csv(filename, chunksize=chunksize):


# endregion


# region CSV writing

# endregion

def write_dicts_to_csv(row_dicts, output_path, append=False):
    with open(output_path, 'a+' if append else 'w+') as csv_fout:
        writer = csv.DictWriter(csv_fout, fieldnames=row_dicts[0].keys())
        if not append:
            writer.writeheader()
        writer.writerows(row_dicts)


def csv_write_first_line(csv_f, first_line: str):
    csv_f.write(first_line)
    if first_line[-1] != '\n':
        csv_f.write('\n')


def csv_write_line(csv_f, csv_tups: tuple, separator='\t'):
    csv_f.write(separator.join(csv_tups))
    csv_f.write('\n')


def extract_csv_columns_by_index_to_file(csv_file_path, output_file_path, col_idxes, separator='\t', skip_first_line=False):
    with open(csv_file_path, 'r') as f, open(output_file_path, 'w+') as w:
        if skip_first_line:
            next(f)
        for line in f:
            splits = line.split(separator)
            col_count = len(col_idxes)
            for i, col_idx in enumerate(col_idxes):
                split = splits[col_idx]
                w.write(split.strip())
                if i != col_count - 1:
                    w.write(separator)
            w.write('\n')
        w.flush()


def extract_csv_columns_by_index(csv_file_path, col_idxes, separator='\t', skip_first_line=False):
    outputs = []
    with open(csv_file_path, 'r') as f:
        if skip_first_line:
            next(f)
        if len(col_idxes) == 1:
            for line in f:
                splits = line.strip().split(separator)
                col_idx = col_idxes[0]
                outputs.append(splits[col_idx].strip())
        else:
            for line in f:
                splits = line.strip().split(separator)
                outputs.append(tuple(splits[col_idx].strip() for col_idx in col_idxes))
    return outputs


def iter_csv(csv_file_path, col_idxes=None, separator='\t', skip_first_line=False, parse=False, use_tqdm=False, display_msg=None, verbose=__debug__):
    """
    Iterates through each line of a csv file.
    :param csv_file_path: the path to the csv file.
    :param col_idxes: only reads specific columns of the csv file.
    :param separator: the separator for the csv file.
    :param skip_first_line: `True` to skip the first line of the csv file.
    :param parse: `True` to parse the strings in each field as their likely Python values.
    :param use_tqdm: `True` to use tqdm to track reading progress.
    :param verbose: `True` to print out `display_msg` on the terminal; effective if `use_tqdm` is `False`.
    :param display_msg: the message to display for this reading; if `use_tqdm` is set `True`, the message will be passed to tqdm; otherwise it will be printed out if `verbose` is `True`.
    :return: an iterator that yields a tuple at a time, corresponding to one line of the csv file.
    """
    with tqdm_wrap(open(csv_file_path, 'r'), use_tqdm=use_tqdm, tqdm_msg=display_msg, verbose=verbose) as f:
        if parse:
            if skip_first_line:
                next(f)
            if col_idxes is None:
                for line in f:
                    yield tuple(str2val(s.strip()) for s in line.split(separator))
            else:
                for line in f:
                    splits = line.split(separator)
                    if isinstance(col_idxes, int):
                        yield str2val(splits[col_idxes].strip()),
                    else:
                        yield tuple(str2val(splits[col_idx].strip()) for col_idx in col_idxes)
        else:
            if skip_first_line:
                next(f)
            if col_idxes is None:
                for line in f:
                    yield tuple(s.strip() for s in line.split(separator))
            else:
                for line in f:
                    splits = line.split(separator)
                    if isinstance(col_idxes, int):
                        yield splits[col_idxes].strip(),
                    else:
                        yield tuple(splits[col_idx].strip() for col_idx in col_idxes)


def iter_csv_dict(csv_file_path: str, delimiter='\t'):
    with open(csv_file_path, newline='') as csvfile:
        yield from csv.DictReader(csvfile, delimiter=delimiter)


def filter_csv(csv_file_path, filter_func, col_idxes=None, separator='\t', skip_first_line=False, output_first_line=None):
    for tups in iter_csv(csv_file_path=csv_file_path, col_idxes=col_idxes, separator=separator, skip_first_line=skip_first_line):
        filtered_tups = filter_func(tups)
        if filtered_tups is not None:
            yield filtered_tups


def filter_csv_to_file(csv_file_path, filter_func, output_file_path=None, col_idxes=None, separator='\t', skip_first_line=False, output_first_line=None, use_tqdm=False):
    with open(output_file_path, 'w') as w:
        if output_first_line:
            csv_write_first_line(w, output_first_line)
        csv_iter = iter_csv(csv_file_path=csv_file_path, col_idxes=col_idxes, separator=separator, skip_first_line=skip_first_line)
        if use_tqdm:
            csv_iter = tqdm(csv_iter)
        for tups in csv_iter:
            filtered_tups = filter_func(tups)
            if filtered_tups is not None:
                csv_write_line(csv_f=w, csv_tups=filtered_tups, separator=separator)

import csv
from itertools import islice, chain
from os import path
from tqdm import tqdm
import utix.pathex as paex

from utix.dictex import IndexDict, kvswap
from utix.general import str2val, tqdm_wrap, hprint_message, exclude_none
from utix.ioex import pickle_load, pickle_save, read_all_lines


# region CSV reading

def iter_feature_data(csv_file_path, num_meta_data_fields=0, num_label_fields=1, parse=True, use_tqdm=True, disp_msg=None, verbose=__debug__, fields_as_list=True):
    _labels_end = num_meta_data_fields + num_label_fields
    for fields in iter_csv(csv_file_path, header=False, parse=parse, use_tqdm=use_tqdm, disp_msg=disp_msg, verbose=verbose, fields_as_list=fields_as_list):
        meta_data = fields[:num_meta_data_fields] if num_meta_data_fields else None
        if num_label_fields == 1:
            labels = fields[num_meta_data_fields]
        elif num_label_fields:
            labels = fields[num_meta_data_fields:_labels_end]
        else:
            labels = None

        feats = fields[_labels_end:] if _labels_end < len(fields) else None
        yield tuple(exclude_none([meta_data, labels, feats]))


def iter_feature_group_sizes(csv_file_path, keyed=True, use_tqdm=True, disp_msg=None, verbose=__debug__):
    if keyed:
        for key, group_size in iter_csv(csv_file_path, header=False, parse=False, use_tqdm=use_tqdm, disp_msg=disp_msg, verbose=verbose, fields_as_list=False):
            yield key, int(group_size)
    else:
        return map(int, read_all_lines(csv_file_path, use_tqdm=use_tqdm, disp_msg=disp_msg, verbose=verbose))


# endregion


# region CSV writing

def write_keyed_lists(keyed_lists, output_path, sep='\t', append=False, use_tqdm=True, disp_msg=None, verbose=__debug__):
    with open(output_path, 'a+' if append else 'w+') as csv_fout:
        for k, l in tqdm_wrap(keyed_lists, use_tqdm=use_tqdm, tqdm_msg=disp_msg, verbose=verbose):
            csv_fout.write(str(k) + sep + sep.join(map(str, l)))
            csv_fout.write('\n')


# endregion

def write_dicts_to_csv(row_dicts, output_path, append=False):
    if not path.exists(output_path):
        append = False
    with open(output_path, 'a+' if append else 'w+') as csv_fout:
        writer = csv.DictWriter(csv_fout, fieldnames=row_dicts[0].keys())
        if not append:
            writer.writeheader()
        writer.writerows(row_dicts)


def csv_write_first_line(csv_f, first_line: str):
    csv_f.write(first_line)
    if first_line[-1] != '\n':
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


def pack_csv(csv_path, output_path, sep='\t', data_seps=' ', header=True, top=None, use_tqdm=False, display_msg=None, verbose=__debug__):
    """
    Packs a csv file a compressed pickle file.

    :param csv_path: the input csv file path.
    :param output_path: the pickle file will be saved at this path.
    :param sep: the csv field separator.
    :param data_seps: the separator to further split the field data; can use a dictionary to specify a data separator for each field.
    :param header: `True` if the csv file has a header; otherwise, `False`.
    :param use_tqdm: `True` to use tqdm to display packing progress; otherwise, `False`.
    :param display_msg: the message to print or display to indicate the data is being packed.
    :param verbose: `True` to print out as much internal message as possible.
    """
    vocab = IndexDict()
    data = []
    with open(csv_path, 'r') as f:
        if header:
            header = next(f).strip('\n').split(sep)
            if isinstance(data_seps, dict):
                for i, h in enumerate(header):
                    if h in data_seps:
                        data_seps[i] = data_seps[h]
        else:
            header = None

        for line in tqdm_wrap(f if top is None else islice(f, top), use_tqdm=use_tqdm, tqdm_msg=display_msg, verbose=verbose):
            fields = line.strip('\n').split(sep)
            if data_seps is None:
                fields = tuple(vocab.add(field) for field in fields)
            elif isinstance(data_seps, str):
                fields = tuple(tuple(vocab.add(x) for x in field.split(data_seps)) for field in fields)
            else:
                fields = tuple(tuple(vocab.add(x) for x in field.split(data_seps.get(i, ' '))) for i, field in enumerate(fields))
            data.append(fields)
    if verbose:
        hprint_message('data size', len(data))
        hprint_message('vocab size', len(vocab))
    pickle_save((sep, data_seps, header, data, dict(vocab.to_dicts()[0])), output_path, compressed=True)


def unpack_csv(data_path, output_csv_path, use_tqdm=False, display_msg=None, verbose=__debug__):
    """
    Unpacks a compressed pickle file built by `pack_csv` to a csv file.

    :param data_path: the path to the pickle file.
    :param output_csv_path: the output csv file will be saved at this path.
    :param use_tqdm: `True` to use tqdm to display packing progress; otherwise, `False`.
    :param display_msg: the message to print or display to indicate the data is being unpacked.
    :param verbose: `True` to print out as much internal message as possible.
    """
    sep, data_seps, header, data, vocab = pickle_load(data_path, compressed=True)
    vocab = kvswap(vocab)
    if verbose:
        hprint_message('data size', len(data))
        hprint_message('vocab size', len(vocab))

    def _tup_iter():
        for fields in tqdm_wrap(data, use_tqdm=use_tqdm, tqdm_msg=display_msg, verbose=verbose):
            if data_seps is None:
                yield tuple(vocab.get(field) for field in fields)
            elif isinstance(data_seps, str):
                yield tuple(data_seps.join([vocab.get(x) for x in field]) for field in fields)
            else:
                yield tuple(data_seps.get(i, ' ').join([vocab.get(x) for x in field]) for i, field in enumerate(fields))

    write_csv(_tup_iter(), output_csv_path=output_csv_path, sep=sep, header=header)


def write_csv(tup_iter, output_csv_path, sep='\t', header=None, append=False, encoding='utf-8', create_dir=True):
    """
    Writes tuples/lists to a csv file.

    :param tup_iter: an iterator of tuples or lists.
    :param output_csv_path: the output csv file will be saved at this path.
    :param sep: the csv field separator.
    :param header: provide the header for the csv file; the header will be written as the first line of the csv file if `append` is set `False`.
    :param append: `True` to use append mode; otherwise, `False`.
    :param encoding: specifies the csv file encoding; the default is 'utf-8'.
    """
    if create_dir:
        paex.ensure_dir_existence(path.dirname(output_csv_path), verbose=False)

    with open(output_csv_path, 'a' if append else 'w', encoding=encoding) as csv_f:
        if header is not None and append is False:
            csv_f.write(sep.join(header))
            csv_f.write('\n')
        for tup in tup_iter:
            csv_f.write(sep.join(str(x) for x in tup))
            csv_f.write('\n')


def iter_csv(csv_input, col_idxes=None, sep='\t', encoding='utf-8', header=True, parse=False, use_tqdm=True, disp_msg=None, allow_missing_cols=False, verbose=__debug__, fields_as_list=False):
    """
    Iterates through each line of a csv file.

    :param csv_file_path: the path to the csv file.
    :param col_idxes: only reads specific columns of the csv file.
    :param sep: the separator for the csv file.
    :param encoding: the encoding for the csv file.
    :param header: `True` to indicate the csv file has a header line; otherwise `False`; speically, can specify 'skip' to skip th first line, no matter whether it is a header.
    :param parse: `True` to parse the strings in each field as their likely Python values.
    :param use_tqdm: `True` to use tqdm to track reading progress.
    :param disp_msg: the message to display for this reading; if `use_tqdm` is set `True`, the message will be passed to tqdm; otherwise it will be printed out if `verbose` is `True`.
    :param verbose: `True` to print out as much internal message as possible.
    :return: an iterator that yields a tuple at a time, corresponding to one line of the csv file.
    """
    if isinstance(csv_input, str):
        if path.isdir(csv_input):
            files = paex.get_files_by_pattern(csv_input, pattern='*.csv', recursive=False, full_path=True)
            files = [open(file, 'r', encoding=encoding) for file in files]
            if header is True or header == 'skip':
                for file in files[1:]:
                    next(file)
            f = chain(*files)
        else:
            f = open(csv_input, 'r', encoding=encoding)
        is_file = True
    else:
        f = csv_input
        is_file = False

    result_type = list if fields_as_list else tuple
    # region special process for the first line
    is_col_idxes_int = isinstance(col_idxes, int)
    if header == 'skip':
        next(f)
    elif header is not None and header is not False:
        if header is True:
            if col_idxes is None or is_col_idxes_int:
                next(f)
            else:
                header = next(f).split(sep)

        if isinstance(col_idxes, str):
            for i, col_name in enumerate(header):
                if col_idxes == col_name:
                    col_idxes = i
                    is_col_idxes_int = True
                    break
            if not is_col_idxes_int:
                raise ValueError(f"the name of the only column `{col_idxes}` does not appear in the csv header")
        else:
            header = {k: i for i, k in enumerate(header)}
            if col_idxes is not None:
                if allow_missing_cols:
                    # if a column does not exist in the header, then replace the column index by `None`
                    col_idxes = tuple((header.get(x, None) if isinstance(x, str) else x) for x in col_idxes)
                else:
                    try:
                        col_idxes = tuple((header[x] if isinstance(x, str) else x) for x in col_idxes)
                    except KeyError as keyerr:
                        raise ValueError(f"name '{keyerr.args[0]}' does not appear in the csv header")
    # endregion
    f = tqdm_wrap(f, use_tqdm=use_tqdm, tqdm_msg=disp_msg, verbose=verbose)
    if parse:
        if col_idxes is None:
            for line in f:
                yield result_type(str2val(s.strip()) for s in line.split(sep))
        elif allow_missing_cols:
            for line in f:
                splits = line.split(sep)
                if isinstance(col_idxes, int):
                    yield str2val(splits[col_idxes].strip()) if col_idxes < len(splits) else None,
                else:
                    yield result_type((str2val(splits[col_idx].strip()) if (col_idx is not None and col_idx < len(splits)) else None) for col_idx in col_idxes)
        else:
            for line in f:
                splits = line.split(sep)
                if isinstance(col_idxes, int):
                    yield str2val(splits[col_idxes].strip()),
                else:
                    yield result_type(str2val(splits[col_idx].strip()) for col_idx in col_idxes)
    else:
        if col_idxes is None:
            for line in f:
                yield result_type(s.strip() for s in line.split(sep))
        elif allow_missing_cols:
            for line in f:
                splits = line.split(sep)
                if isinstance(col_idxes, int):
                    yield splits[col_idxes].strip() if col_idxes < len(splits) else None,
                else:
                    yield result_type((splits[col_idx].strip() if (col_idx is not None and col_idx < len(splits)) else None) for col_idx in col_idxes)
        else:
            for line in f:
                splits = line.split(sep)
                if isinstance(col_idxes, int):
                    yield splits[col_idxes].strip(),
                else:
                    try:
                        if len(splits) < 11:
                            continue
                        yield result_type(splits[col_idx].strip() for col_idx in col_idxes)
                    except Exception as err:
                        print(splits)
                        raise err

    if is_file:
        f.close()


def iter_csv_dict(csv_file_path: str, sep='\t'):
    with open(csv_file_path, newline='') as csvfile:
        yield from csv.DictReader(csvfile, delimiter=sep)


def iter_csv_tuple(csv_file_path: str, sep='\t'):
    with open(csv_file_path, newline='') as csvfile:
        for line in csvfile:
            line = line.rstrip('\r\n')
            yield line.split(sep, 1)


def filter_csv(csv_file_path, filter_func, col_idxes=None, separator='\t', skip_first_line=False, output_first_line=None):
    for tups in iter_csv(csv_file_path=csv_file_path, col_idxes=col_idxes, sep=separator, header=skip_first_line):
        filtered_tups = filter_func(tups)
        if filtered_tups is not None:
            yield filtered_tups


def filter_csv_to_file(csv_file_path, filter_func, output_file_path=None, col_idxes=None, separator='\t', skip_first_line=False, output_first_line=None, use_tqdm=False):
    with open(output_file_path, 'w') as w:
        if output_first_line:
            csv_write_first_line(w, output_first_line)
        csv_iter = iter_csv(csv_file_path=csv_file_path, col_idxes=col_idxes, sep=separator, header=skip_first_line)
        if use_tqdm:
            csv_iter = tqdm(csv_iter)
        for tups in csv_iter:
            filtered_tups = filter_func(tups)
            if filtered_tups is not None:
                write_csv_line(csv_f=w, csv_tups=filtered_tups, sep=separator)

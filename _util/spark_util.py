import datetime
import itertools
import re
from calendar import monthrange
from functools import partial
from functools import reduce
from os import path
from time import sleep
from typing import Mapping
from typing import Tuple, Union

import pyspark.sql.functions as F
from pyspark.sql import SparkSession, Window, DataFrame, Column
from tqdm import tqdm

from utix.general import tqdm_wrap


# region misc utilities

def spark():
    """
    Gets the default cluster spark session.
    """
    return SparkSession.builder \
        .appName("data") \
        .master("spark://0.0.0.0:7077") \
        .getOrCreate()


def local_spark(app_name='data', num_cpus='*') -> SparkSession:
    """
    Gets the default local spark session.
    """
    return SparkSession.builder \
        .appName(app_name) \
        .master("local[{}]".format(num_cpus)) \
        .getOrCreate()


def num_worker_workers(spark):
    """
    Gets the number of available spark workers.
    """
    return spark._jsc.sc().getExecutorMemoryStatus().size()


def wait_for_workers(spark, target=50, check_interval=10, print_out=True):
    """
    Blocks the process if the specified `target` number of workers are not currently registered for spark.
    """
    import time
    cur_num_workders = num_worker_workers(spark)
    while cur_num_workders < target:
        if print_out:
            print('Target Number of Workers: {}, Current Number of Workers: {}'.format(target, cur_num_workders))
        time.sleep(check_interval)
    if print_out:
        'Target number of workers reached.'


def get_columns_by_pattern(df, reg_pattern):
    """
    A convenient function to retrieve the column names of the dataframe by the specified regular expression.
    """
    reg_pattern = '^' + reg_pattern + '$'
    return [_col for _col in df.columns if re.match(reg_pattern, _col)]


def column_as_set(df: DataFrame, col):
    """
    Collects the values in one column as a set.
    """
    return set(row[col] for row in tqdm(df.select(col).distinct().collect()))


def columns_as_set(df: DataFrame, *cols):
    """
    Collects the values of several columns as a set of tuples.
    """
    return set(tuple(row[col] for col in cols) for row in tqdm(df.select(*cols).distinct().collect()))


def trim_columns(df, *cols) -> DataFrame:
    """
    Trims texts in the specified columns.
    """
    for col in cols:
        df = df.withColumn(col, F.trim(F.col(col)))
    return df


def compare(df_base, df_compare, success_criteria, result_save_dir, comparison_name):
    if isinstance(success_criteria, str):
        success_criteria = F.col(success_criteria)

    df_loss = join_on_columns(
        df1=df_base.where(success_criteria),
        df2=df_compare.where(~success_criteria),
        join_key_col_names1=['query'],
        col_name_suffix=True
    ).cache()
    show_count(df_loss, 'df_loss')
    write_df_as_json(df_loss, path.join(result_save_dir, f'{comparison_name}_loss' if comparison_name else 'loss'), num_files=1, repartition=True)

    df_gain = join_on_columns(
        df1=df_base.where(~success_criteria),
        df2=df_compare.where(success_criteria),
        join_key_col_names1=['query'],
        col_name_suffix=True
    ).cache()
    show_count(df_gain, 'df_gain')
    write_df_as_json(df_gain, path.join(result_save_dir, f'{comparison_name}_gain' if comparison_name else 'gain'), num_files=1, repartition=True)

    df_both_success = join_on_columns(
        df1=df_base.where(success_criteria),
        df2=df_compare.where(success_criteria),
        join_key_col_names1=['query'],
        col_name_suffix=True
    ).cache()
    show_count(df_both_success, 'df_both_trig')
    write_df_as_json(df_loss, path.join(result_save_dir, f'{comparison_name}_both_success' if comparison_name else 'both_success'), num_files=1, repartition=True)

    df_both_fail = join_on_columns(
        df1=df_base.where(~success_criteria),
        df2=df_compare.where(~success_criteria),
        join_key_col_names1=['query'],
        col_name_suffix=True
    ).cache()
    show_count(df_both_fail, 'df_both_not_trig')
    write_df_as_json(df_loss, path.join(result_save_dir, f'{comparison_name}_both_fail') if comparison_name else 'both_fail', num_files=1, repartition=True)
    return df_loss, df_gain, df_both_success, df_both_fail


# endregion

# region convenient grouping & counting

def group_with_cnt_column(df, *cols, cnt_col_name=None) -> Tuple[DataFrame, str]:
    """
    A convenient function to group the dataframe by the specified columns, and adds a counting column to count the number of items in each group.
    """

    if cnt_col_name is None:
        cnt_col_name = '__'.join(cols) + '___cnt'
    return df.groupBy(*cols).agg(F.count("*").alias(cnt_col_name)), cnt_col_name


def group_with_max_column(df, mx_trg_col, *group_cols, mx_col_name=None) -> Tuple[DataFrame, str]:
    """
    A convenient function to group the dataframe by the specified columns, and adds a column to save the maximum value of the `mx_trg_col` in each group.
    """
    if mx_col_name is None:
        mx_col_name = '__'.join(group_cols) + f'__{mx_trg_col}___mx'
    return df.groupBy(*group_cols).agg(F.max(mx_trg_col).alias(mx_col_name)), mx_col_name


def group_with_rank_column(df, group_cols, order_cols, ascending=True, rank_col_name=None):
    """
    A convenient function to group the dataframe by the specified columns, order the rows in each group by columns specified in `order_cols`, and adds a column to save the ranking of each group.
    """
    if rank_col_name is None:
        rank_col_name = '__'.join(group_cols) + '_' + '__'.join(order_cols) + f'___rank'
    if not ascending:
        order_cols = [F.col(colname).desc() if isinstance(colname, str) else colname for colname in order_cols]
    w = Window().partitionBy(*group_cols).orderBy(*order_cols)
    return df.withColumn(rank_col_name, F.row_number().over(w)), rank_col_name


# endregion

# region convenient join

def join_on_columns(df1, df2, join_key_col_names1, join_key_col_names2=None, col_name_suffix=None, *args, **kwargs):
    """
    A convenient function to join two dataframes based on identify of columns of specific names. Supports adding suffixes to column names to avoid column name conflict.
    """
    if isinstance(join_key_col_names1, str):
        join_key_col_names1 = [join_key_col_names1]
    else:
        join_key_col_names1 = list(join_key_col_names1)

    # if the join keys for `df2` is different from the join keys of `df1`, then
    if join_key_col_names2 is not None:
        df2 = df2.rename(df2, {name2: name1 for name1, name2 in zip(join_key_col_names1, join_key_col_names2)})

    if col_name_suffix not in (None, False):
        # adding suffixes to the non-join-key column names
        if col_name_suffix is True:
            suffix1, suffix2 = '_1', '_2'
        elif isinstance(col_name_suffix, (tuple, list)):
            suffix1, suffix2 = col_name_suffix
        else:
            raise ValueError('`col_name_suffix` should be a list, or tuple of length 2, or one of `None`, `True`, `False`')

        df1 = rename_by_adding_suffix(df1, suffix=suffix1, excluded_col_names=join_key_col_names1)
        df2 = rename_by_adding_suffix(df2, suffix=suffix2, excluded_col_names=join_key_col_names1)
    return df1.join(df2, join_key_col_names1, *args, **kwargs)


def join_multiple_on_columns(dfs, join_key_col_names, col_name_suffix=None, *args, **kwargs):
    """
    A convenient function to join multiple dataframes based on identify of columns of specific names. Supports adding suffixes to column names to avoid column name conflict.
    """

    def _add_col_suffix(df, df_idx):
        if col_name_suffix not in (None, False):
            return df
        elif col_name_suffix is True:
            return rename_by_adding_suffix(df, suffix=f'_{df_idx}', excluded_col_names=join_key_col_names)
        elif isinstance(col_name_suffix, (tuple, list)):
            return rename_by_adding_suffix(df, suffix=col_name_suffix[df_idx], excluded_col_names=join_key_col_names)
        else:
            raise ValueError('`col_name_suffix` should be a list, or tuple, or one of `None`, `True`, `False`')

    if isinstance(join_key_col_names, str):
        join_key_col_names = [join_key_col_names]

    df = _add_col_suffix(dfs[0], 0)

    if isinstance(join_key_col_names[0], str):
        for df_idx, df2 in enumerate(dfs[1:]):
            df = df.join(_add_col_suffix(df2, df_idx + 1), join_key_col_names, *args, **kwargs)
    else:
        col_names1 = join_key_col_names[0]
        for df_idx, (df2, col_names2) in enumerate(zip(dfs[1:], join_key_col_names[1:])):
            df = join_on_columns(df, _add_col_suffix(df2, df_idx + 1), col_names1, col_names2, *args, **kwargs)
    return df


# endregion


def deduplicate(df, *keycols, top=1) -> DataFrame:
    dff, rankcol = group_with_rank_column(df, group_cols=keycols, order_cols=keycols)
    return dff.where(F.col(rankcol) <= top).drop(rankcol)


def deduplicate2(df, *keycols):
    dd = set()
    unique_rows = []
    for row in tqdm(df.collect()):
        k = tuple(row[col] for col in keycols)
        if k not in dd:
            dd.add(k)
            unique_rows.append(k)
    return spark.createDataFrame(unique_rows)


def get_most_frequent_map(df, src_col, trg_col, no_tie=True, to_dict=True, use_tqdm=True, display_msg='construction map from {} to {}', verbose=__debug__) -> Union[DataFrame, dict]:
    """
    Gets a map from the values in the `src_col` to the most frequently associated values in the `trg_col`.
    :param df: the spark dataframe.
    :param src_col: the values of this column are keys of the map.
    :param trg_col: the values of this column are values of the map.
    :param no_tie: `True` to break ties; otherwise `False`.
    :param to_dict: `True` to return a python dictionary; `False` to only return a dataframe with the two columsn `src_col` and `trg_col`.
    :param use_tqdm: use tqdm to report progress when constructing the map.
    :param display_msg: the progress message pattern.
    :param verbose: `True` to print out as many internal messages as possible.
    :return:
    """
    df1, cnt_col_name = group_with_cnt_column(df, src_col, trg_col)
    if no_tie:
        df2, rank_col_name = group_with_rank_column(df1, group_cols=(src_col,), order_cols=(cnt_col_name,), ascending=False)
        df = df2.where(F.col(rank_col_name) == 1).select(src_col, trg_col)
    else:
        df2, mx_col_name = group_with_max_column(df1, cnt_col_name, src_col)
        df1 = df1.alias('df1')
        df2 = df2.alias('df2')

        df = df1.join(df2, (F.col(cnt_col_name) == F.col(mx_col_name)) & (F.col(f"df1.{src_col}") == F.col(f"df2.{src_col}"))).select(F.col(f"df1.{src_col}"), F.col(f"df1.{trg_col}"))

    if to_dict:
        if no_tie:
            it = tqdm_wrap(df.select(src_col, trg_col).collect(), use_tqdm=use_tqdm, tqdm_msg=display_msg.format(src_col, trg_col), verbose=verbose)
            return dict(it)
        else:
            d = {}
            it = tqdm_wrap(df.collect(), use_tqdm=use_tqdm, tqdm_msg=display_msg.format(src_col, trg_col), verbose=verbose)
            for row in it:
                d[row[src_col]] = row[trg_col]
            return d
    else:
        return df


def get_concat_paras(col_names, name_val_delimiter=':', col_delimiter='|'):
    concat_paras = []
    for col_idx, col_name in enumerate(col_names):
        concat_paras.extend([F.lit(col_name), F.lit(name_val_delimiter), F.col(col_name)])
        if col_idx != len(col_names) - 1:
            concat_paras.append(F.lit(col_delimiter))
    return concat_paras


def agg_by_categories(df: DataFrame, category_columns, target_columns, agg_funs=((F.count, '*', 'count'),), category_label_column_name='category', output_path=None, includes_overall=True):
    num_categories = len(category_columns)
    target_columns = tuple(target_columns)
    agg_col_names = tuple(agg_fun[2] for agg_fun in agg_funs)
    output_df = (
        df.groupBy(*target_columns).agg(*[agg_fun[0](agg_fun[1]).alias(agg_fun[2]) for agg_fun in agg_funs])
            .withColumn(category_label_column_name, F.lit("overall"))
            .drop(*category_columns)
            .select(*((category_label_column_name,) + target_columns + agg_col_names))
    ) if includes_overall else None
    for cat_size in range(1, num_categories + 1):
        for cat_combo in itertools.combinations(category_columns, cat_size):
            cat_cols = get_concat_paras(cat_combo)
            gdf = (
                df.groupBy(*(cat_combo + target_columns)).agg(*[agg_fun[0](agg_fun[1]).alias(agg_fun[2]) for agg_fun in agg_funs])
                    .withColumn(category_label_column_name, F.concat(*cat_cols))
                    .drop(*category_columns)
                    .select(*((category_label_column_name,) + target_columns + agg_col_names))
            )
            if output_df is None:
                output_df = gdf
            else:
                output_df = output_df.union(gdf)
    if output_path:
        output_df.coalesce(1).write.csv(output_path, mode='overwrite', header='true')
    return output_df


def merge_data(df, merge_columns, cnt_column_and_avg_columns, list_columns=None, max_columns=None, min_columns=None):
    _cnt_column_and_avg_columns = []
    all_cnt_avg_cols = []
    for cnt_col, avg_cols in cnt_column_and_avg_columns:
        if isinstance(avg_cols, str):
            avg_cols = get_columns_by_pattern(df, avg_cols)
        _cnt_column_and_avg_columns.append((cnt_col, avg_cols))
        all_cnt_avg_cols.append(cnt_col)
        all_cnt_avg_cols.extend(avg_cols)
    cnt_column_and_avg_columns = _cnt_column_and_avg_columns

    selected_columns = list(merge_columns)
    for cnt_col, avg_cols in cnt_column_and_avg_columns:
        selected_columns.append(cnt_col)
        selected_columns.extend((F.col(cnt_col) * F.col(avg_col)).alias(avg_col) for avg_col in avg_cols)

    for cols in (list_columns, max_columns, min_columns):
        if cols:
            selected_columns += cols

    df = df.select(*selected_columns)

    agg_columns = (
            [F.sum(F.col(cnt_total_col)).alias(cnt_total_col) for cnt_total_col in all_cnt_avg_cols] +
            ([(F.flatten(F.collect_list(F.col(list_col)))).alias(list_col) for list_col in list_columns] if list_columns else []) +
            ([F.max(F.col(max_col)).alias(max_col) for max_col in max_columns] if max_columns else []) +
            ([F.min(F.col(min_col)).alias(min_col) for min_col in min_columns] if min_columns else [])
    )
    df = df.groupBy(*merge_columns).agg(*agg_columns)

    selected_columns = list(merge_columns)
    for cnt_col, avg_cols in cnt_column_and_avg_columns:
        selected_columns.append(cnt_col)
        selected_columns.extend((F.col(avg_col) / F.col(cnt_col)).alias(avg_col) for avg_col in avg_cols)
    for cols in (list_columns, max_columns, min_columns):
        if cols:
            selected_columns += cols

    return df.select(*selected_columns)


# region input

def read_json(spark, input_path, mode='', schema=None, read_interval=10):
    """
    Reads from json sources.
    """
    if not mode or isinstance(input_path, str):
        return spark.read.json(input_path, schema=schema)
    if isinstance(mode, str):
        mode = set(mode.split(','))
    else:
        mode = set()

    def _read():
        if 'late_union' in mode:
            dfs = []
            for _path in input_path:
                print('reading from {}'.format(_path))
                dfs.append(spark.read.json(_path, schema=schema))
                if read_interval != 0:
                    sleep(read_interval)

            print('union {} dataframes'.format(len(dfs)))
            return reduce(DataFrame.union, dfs)
        else:
            df = None
            for _path in input_path:
                print('reading from {}'.format(_path))
                if df is None:
                    df = spark.read.json(_path, schema=schema)
                else:
                    df = df.union(spark.read.json(_path, schema=schema))
                if read_interval != 0:
                    sleep(read_interval)
            return df

    if 'on_error' in mode:
        try:
            return spark.read.json(input_path, schema=schema)
        except:
            sleep(read_interval)
            return _read()
    else:
        return _read()


def solve_input(spark, _input, format='json', schema=None, wait_time_on_error=10):
    """
    Pre-processes spark input of different format and returns the desired dataframe.
    This is useful for processing a spark function input argument, allowing the input be either a path or a dataframe.
    """
    if isinstance(_input, DataFrame):
        return _input

    if spark is None:
        raise ValueError('spark instance must be provided to load a {} file at `{}`'.format(format, _input))

    if format == 'json':
        return read_json(spark, _input, mode='on_error', schema=schema, read_interval=wait_time_on_error)
    else:
        raise ValueError('format `{}` is not supported yet'.format(format))


# endregion

def _proc_output_path(output_path, sub_dir, disp_msg):
    if sub_dir is not None:
        output_path = path.join(output_path, sub_dir)
    if disp_msg is not None:
        print(disp_msg.format(output_path))
    return output_path


def overwrite_single_csv(df, output_path, sub_dir=None, disp_msg='csv file saved at {}', **kwargs):
    output_path = _proc_output_path(output_path, sub_dir, disp_msg)
    df.coalesce(1).write.mode('overwrite').csv(output_path, **kwargs)


def overwrite_single_json(df, output_path, sub_dir=None, disp_msg='json file saved at {}', **kwargs):
    output_path = _proc_output_path(output_path, sub_dir, disp_msg)
    df.coalesce(1).write.mode('overwrite').json(output_path, **kwargs)


def overwrite_json(df, output_path, sub_dir=None, disp_msg='json files saved at {}', coalesce=None, **kwargs):
    output_path = _proc_output_path(output_path, sub_dir, disp_msg)
    if coalesce is not None:
        df = df.coalesce(coalesce)
    df.write.mode('overwrite').json(output_path, **kwargs)


def single_value(df):
    return df.collect()[0][0]


def parse_metric_cols(metric_cols):
    reverses, parsed = [], []
    for metric_col in metric_cols:
        reverse = metric_col[0] == '-'
        reverses.append(reverse)
        parsed.append(metric_col[1:] if reverse else metric_col)
    return reverses, metric_cols


def iter_cols_distinct(df, *cols):
    for row in df.select(*cols).distinct().collect():
        yield [row[col] for col in cols]


def replace_col(df, col, cond, new_val, in_place=True):
    if in_place:
        if isinstance(cond, Column):
            return df.withColumn(col, F.when(cond, new_val).otherwise(df[col]))
        elif not callable(cond):
            return df.withColumn(col, F.when(F.col(col) == cond, new_val).otherwise(df[col]))
        else:
            return df.withColumn(col, F.when(cond(F.col(col)), new_val).otherwise(df[col]))
    else:
        tmp_col = "__tmp__" + col
        if not callable(cond):
            df = df.withColumn(tmp_col, F.when(F.col(col) == cond, new_val).otherwise(df[col]))
        else:
            df = df.withColumn(tmp_col, F.when(cond(F.col(col)), new_val).otherwise(df[col]))
        return df.drop(col).withColumnRenamed(tmp_col, col)


def replace_col__(df, col, cond, new_val, other_val, in_place=True):
    if in_place:
        if not callable(cond):
            return df.withColumn(col, F.when(F.col(col) == cond, new_val).otherwise(other_val))
        else:
            return df.withColumn(col, F.when(cond(F.col(col)), new_val).otherwise(other_val))
    else:
        tmp_col = "__tmp__" + col
        if not callable(cond):
            df = df.withColumn(tmp_col, F.when(F.col(col) == cond, new_val).otherwise(other_val))
        else:
            df = df.withColumn(tmp_col, F.when(cond(F.col(col)), new_val).otherwise(other_val))
        return df.drop(col).withColumnRenamed(tmp_col, col)


def extract_eq(df, cols, vals):
    for col, val in zip(cols, vals):
        df = df.where(F.col(col) == val)
    return df


def single_row(df, cols, vals):
    rows = extract_eq(df, cols, vals).collect()
    if len(rows) != 1:
        raise ValueError('expect to have single returned row')
    return rows[0]


def agg__(grouped, cols, with_count=True, round=4, agg_name='avg', reverses=None):
    if reverses is None:
        agg_cols = tuple(F.round(getattr(F, agg_name)(col), round).alias(agg_name + '_' + col) for col in cols)
    elif agg_name == 'max':
        agg_cols = tuple(F.round((F.min if reverse else F.max)(col), round).alias(('min_' if reverse else 'max_') + col) for col, reverse in zip(cols, reverses))
    elif agg_name == 'min':
        agg_cols = tuple(F.round((F.max if reverse else F.min)(col), round).alias(('max_' if reverse else 'min_') + col) for col, reverse in zip(cols, reverses))
    else:
        raise ValueError('the aggregation {} does no support reverse'.format(agg_name))

    if with_count:
        agg_cols = (F.count('*').alias('count'),) + agg_cols
    return grouped.agg(*agg_cols)


def dump_extreme_sample(output_path, df, metric_cols, group_cols, group_iter=None, extreme_quantile=0.1, min_sample=None, limit=100, coalesce=None, summary_csv_cols=None, summary_csv_sep='\t', summary_csv_header=True):
    if group_iter is None:
        group_iter = iter_cols_distinct(df, group_cols)
    reverses, metric_cols = parse_metric_cols(metric_cols)
    # worse_metrics = agg__(df.groupBy(*group_cols), metric_cols, with_count=False, agg_name='max', reverses=reverses).cache()
    for group in tqdm(group_iter):
        # group_worst_metrics = single_row(worse_metrics, group_cols, group)
        for reverse, metric_col in zip(reverses, metric_cols):
            extreme_sample = extract_eq(df, group_cols, group)
            if reverse:
                extreme_sample = extreme_sample.orderBy(F.col(metric_col))  # .where(F.col(metric_col) < group_worst_metrics['min_' + metric_col] * (1 + extreme_quantile))
            else:
                extreme_sample = extreme_sample.orderBy(F.col(metric_col).desc())  # .where(F.col(metric_col) > group_worst_metrics['max_' + metric_col] * (1 - extreme_quantile))

            extreme_sample = extreme_sample.limit(limit)
            if summary_csv_cols:
                extreme_sample = extreme_sample.cache()
                extreme_sample_summary = extreme_sample.groupBy(*summary_csv_cols).agg(F.count('*').alias('count'), F.avg(metric_col).alias(metric_col))
                extreme_sample_summary = (extreme_sample_summary.orderBy(F.col(metric_col))) if reverse else (extreme_sample_summary.orderBy(F.col(metric_col).desc())).limit(limit)
                # extreme_sample = extreme_sample.join(extreme_sample_summary.select(*summary_csv_cols), list(summary_csv_cols), how='inner')
            # else:
            #     extreme_sample = extreme_sample.limit(limit)

            sub_dir = '__'.join(tuple(str(x) for x in group) + (metric_col,))
            overwrite_json(
                df=extreme_sample,
                output_path=output_path,
                sub_dir=sub_dir,
                coalesce=coalesce
            )
            if summary_csv_cols:
                overwrite_single_csv(
                    df=extreme_sample_summary,
                    output_path=output_path,
                    sub_dir=sub_dir + '__summary_csv',
                    sep=summary_csv_sep,
                    header=summary_csv_header)
                del extreme_sample
                del extreme_sample_summary


def write_df_as_json(df,
                     output_path,
                     num_files=10,
                     compress=False,
                     show_counts=True,
                     chunk_field='date',
                     chunk_ranges=None,
                     cache_before_writing=False,
                     repartition=False,
                     gaptime=30,
                     check_edge_chunks=True,
                     left_edge_chunk_end_offset=None,
                     right_edge_chunk_start_offset=None):
    def _write(df, output_path):
        print('write to{} files{}'.format((' ' + str(num_files) if num_files else ''), (' with compression' if compress else '')))
        if num_files:
            df = df.repartition(num_files) if repartition else df.coalesce(num_files)
        if cache_before_writing or show_counts:
            if cache_before_writing:
                df = df.cache()
            print('writing {} rows to: {}'.format(df.count(), output_path))
        else:
            print('writing to ' + output_path)
        if compress:
            df.write.mode('overwrite').json(output_path, compression='gzip')
        else:
            df.write.mode('overwrite').json(output_path)

    if chunk_ranges is None:
        _write(df=df, output_path=output_path)
    else:
        if chunk_field == 'rowid':
            chunk_field = '__rowid'
            # df = df.withColumn(chunk_field, F.row_number().over(Window.orderBy(F.monotonically_increasing_id())) - 1).drop(tmp_field).cache()
            df = df.withColumn(chunk_field, F.monotonically_increasing_id()).cache()
            df_min_max_row_id = df.select(F.min(chunk_field), F.max(chunk_field)).collect()[0]
            min_row_id, max_row_id = df_min_max_row_id[0], df_min_max_row_id[1]
            print('min row id {}, max row id {}'.format(min_row_id, max_row_id))
            if isinstance(chunk_ranges, (int, float)):
                chunk_size = int((max_row_id - min_row_id) / float(chunk_ranges))
                chunk_sep_ids = list(range(min_row_id, max_row_id + 1, chunk_size)) + [max_row_id]
                for i in range(len(chunk_sep_ids) - 1):
                    chunk_start, chunk_end = chunk_sep_ids[i], chunk_sep_ids[i + 1]
                    print('writing data chunk from {} to {}'.format(chunk_start, chunk_end))
                    _write(df=df.where((F.col(chunk_field) >= chunk_start) & (F.col(chunk_field) < chunk_end)).drop(chunk_field), output_path=path.join(output_path, '{:04}'.format(i)))
                    if gaptime > 0:
                        sleep(gaptime)
            else:
                for chunk_start, chunk_end in chunk_ranges:
                    print('writing data chunk from {} to {}'.format(chunk_start, chunk_end))
                    _write(df=df.where(((F.col(chunk_field) >= chunk_start) & F.col(chunk_field) < chunk_end)).drop(chunk_field), output_path=path.join(output_path, '{}_{}'.format(chunk_start, chunk_end)))
                    if gaptime > 0:
                        sleep(gaptime)
            return df
        else:
            df = df.cache()
            print('cached {} rows'.format(df.count()))
            if check_edge_chunks:
                chunk_start_no_offset, chunk_end_no_offset = chunk_start, chunk_end = chunk_ranges[0][0], chunk_ranges[-1][1]
                if left_edge_chunk_end_offset is not None:
                    chunk_start -= left_edge_chunk_end_offset
                if right_edge_chunk_start_offset is not None:
                    chunk_end += right_edge_chunk_start_offset
                _output_path = path.join(output_path, 'before_{}'.format(chunk_start_no_offset))
                _write(df=df.where(F.col(chunk_field) < chunk_start), output_path=_output_path)
                _output_path = path.join(output_path, 'after_{}'.format(chunk_end_no_offset))
                _write(df=df.where(F.col(chunk_field) >= chunk_end), output_path=_output_path)
            for chunk_start, chunk_end in chunk_ranges:
                print('writing data chunk from {} to {}'.format(chunk_start, chunk_end))
                _output_path = path.join(output_path, '{}_{}'.format(chunk_start, chunk_end))
                if chunk_start == chunk_end:
                    _write(df=df.where(F.col(chunk_field) == chunk_start), output_path=_output_path)
                else:
                    _write(df=df.where((chunk_start <= F.col(chunk_field)) & (F.col(chunk_field) < chunk_end)), output_path=_output_path)
                if gaptime > 0:
                    sleep(gaptime)
            return df


def merge_splits_counts_and_averages(df, group_cols, cnt_col_and_avg_cols, split_idx_col='split_index'):
    # `_cnt_column_and_avg_columns`, saves cnt_col/avg_cols tuples
    # `all_cnt_avg_cols`, saves all cnt_col, avg_cols in a sequence
    _cnt_column_and_avg_columns, all_cnt_avg_cols = [], []
    for cnt_col, avg_cols in cnt_col_and_avg_cols:
        if isinstance(avg_cols, str):
            avg_cols = get_columns_by_pattern(df, avg_cols)
        _cnt_column_and_avg_columns.append((cnt_col, avg_cols))
        all_cnt_avg_cols.append(cnt_col)
        all_cnt_avg_cols.extend(avg_cols)
    cnt_col_and_avg_cols = _cnt_column_and_avg_columns

    # then only chooses one row in each group;
    # we assume in each group the stat values in columns 'avg_cols' are the same;
    # we must consider `split_idx_col` as one grouping field, since data were grouped in each split
    selected_columns = list(group_cols) + [split_idx_col]
    window = Window.partitionBy(*selected_columns).orderBy(F.lit(0))
    df = df.withColumn('__rank', F.row_number().over(window)).where(F.col('__rank') == 1).drop('__rank')

    # reverts stat averages in each group
    selected_columns = list(group_cols)
    for cnt_col, avg_cols in cnt_col_and_avg_cols:
        selected_columns.append(cnt_col)
        selected_columns.extend((F.col(cnt_col) * F.col(avg_col)).alias(avg_col) for avg_col in avg_cols)
    df = df.select(*selected_columns)

    # summing over all counts and the stats reverted from the averages accross the splits
    agg_columns = [F.sum(F.col(col)).alias(col) for col in all_cnt_avg_cols]
    df = df.groupBy(*group_cols).agg(*agg_columns)

    # recomputes the averages
    selected_columns = list(group_cols)
    for cnt_col, avg_cols in cnt_col_and_avg_cols:
        selected_columns.append(cnt_col)
        selected_columns.extend((F.col(avg_col) / F.col(cnt_col)).alias(avg_col) for avg_col in avg_cols)

    return df.select(*selected_columns)


def merge_splits(df, key_cols, group_cnt_avg_cols, list_cols=None, max_cols=None, min_cols=None, split_idx_col='split_index'):
    agg_columns = (
            ([(F.flatten(F.collect_list(F.col(list_col)))).alias(list_col) for list_col in list_cols] if list_cols else []) +
            ([F.max(F.col(max_col)).alias(max_col) for max_col in max_cols] if max_cols else []) +
            ([F.min(F.col(min_col)).alias(min_col) for min_col in min_cols] if min_cols else [])
    )
    if agg_columns:
        out_df = df.groupBy(*key_cols).agg(*agg_columns)
    else:
        out_df = df.select(*key_cols).distinct()

    for group_cols, cnt_col_and_avg_cols in group_cnt_avg_cols:
        avg_stat_df = merge_splits_counts_and_averages(df, group_cols=group_cols, cnt_col_and_avg_cols=cnt_col_and_avg_cols, split_idx_col=split_idx_col)
        out_df = out_df.join(avg_stat_df, list(group_cols), how='left')

    return out_df


def _sort_by_key(arr, sorting_key, reverse):
    if isinstance(sorting_key, str):
        return sorted(arr, key=lambda x: x[sorting_key], reverse=reverse)
    elif isinstance(sorting_key, (list, tuple)):
        return sorted(arr, key=lambda x: tuple(x[_key] for _key in sorting_key), reverse=reverse)
    else:
        return sorted(arr, key=sorting_key, reverse=reverse)


def sort_nested_array(df, array_field, sorting_key, reverse=False):
    return df.withColumn(array_field, F.udf(partial(_sort_by_key, sorting_key=sorting_key, reverse=reverse), df.schema[array_field].dataType)(F.col(array_field)))


def one_from_each_group(df, group_cols, order_cols=None):
    if not isinstance(group_cols, (tuple, list)):
        group_cols = (group_cols,)
    if order_cols is None:
        order_cols = group_cols
    elif not isinstance(order_cols, (tuple, list)):
        order_cols = (order_cols,)
    rank_col_name = '--'.join(group_cols).replace('.', '_') + '_' + '--'.join(group_cols).replace('.', '_') + '---rank'
    w = Window().partitionBy(*group_cols).orderBy(*order_cols)
    return df.withColumn(rank_col_name, F.row_number().over(w)).where(F.col(rank_col_name) == 1).drop(rank_col_name)


# region paths organized by dates

def _format_days_for_spark_path(days):
    return '{' + ','.join(days) + '}'


def _parse_days(days_expr, year, month, days_chunk_size=0):
    days = []
    splits = days_expr.split(',')
    for split in splits:
        subsplits = split.split('-')
        if len(subsplits) == 1:
            day = subsplits[0]
            if day == '*':
                days = list(range(1, monthrange(year, month)[1] + 1))
                break
            else:
                days.append(int(day))
        else:
            days.extend(list(range(int(subsplits[0]), int(subsplits[1]) + 1)))
    days = ['{:02}'.format(day) for day in sorted(set(days))]
    if days_chunk_size != 0:
        days = [days[i:i + days_chunk_size] for i in range(0, len(days), days_chunk_size)]
        return [_format_days_for_spark_path(days_split) for days_split in days], days
    else:
        return [_format_days_for_spark_path(days)], [days]


def _parse_hours(hours_expr):
    splits = hours_expr.split(',')
    hours = []
    for split in splits:
        subsplits = split.split('-')
        if len(subsplits) == 1:
            hours.append(int(subsplits[0]))
        else:
            hours.extend(list(range(int(subsplits[0]), int(subsplits[1]) + 1)))
    return _format_days_for_spark_path(['{:02}'.format(hour) for hour in hours])


def get_day_chunks(year, months_days_expr, days_chunk_size=0):
    start_end_dates = []
    for month, days_expr in map(lambda x: x.split('/'), months_days_expr.split(';')):
        month = int(month)
        parsed_month = '{:02}'.format(month)
        parsed_days_path_parts, days_splits = _parse_days(year=year, month=month, days_expr=days_expr, days_chunk_size=days_chunk_size)
        for days_path_part, days_split in zip(parsed_days_path_parts, days_splits):
            start_end_dates.append(('{}-{}-{}'.format(year, parsed_month, days_split[0]), '{}-{}-{}'.format(year, parsed_month, days_split[-1])))
    return start_end_dates


# def create_date_ranges(year, date_range_expr, day_delta):
#     for month, days_expr in map(lambda x: x.split('/'), date_range_expr.split(';')):
#         month = int(month)

def get_day_chunks_and_paths(path_pattern, year, months_days_expr, days_chunk_size=0, hours_expr=None):
    """
    Generates one or more input paths organized by days, e.g. 'xxx/2020/07/{01,02,03,04}/*'.

    :param path_pattern: the path pattern with three fields 'year', 'month', 'days', and optionally 'hours', e.g. 'xxx/{year}/{month}/{days}/{hours}'.
    :param year: will replace the 'year' field in the input path pattern.

    """

    out_input_paths, start_end_dates = [], []
    input_pattern_has_hours = '{hours}' in path_pattern
    if input_pattern_has_hours:
        if hours_expr is None:
            hours_expr = '*'
        else:
            hours_expr = _parse_hours(hours_expr)
    for month, days_expr in map(lambda x: x.split('/'), months_days_expr.split(';')):
        month = int(month)
        parsed_month = '{:02}'.format(month)
        parsed_days_path_parts, days_splits = _parse_days(year=year, month=month, days_expr=days_expr, days_chunk_size=days_chunk_size)
        for days_path_part, days_split in zip(parsed_days_path_parts, days_splits):
            out_input_paths.append(path_pattern.format(year=year, month=parsed_month, days=days_path_part, hours=hours_expr) if input_pattern_has_hours else path_pattern.format(year=year, month=parsed_month, days=days_path_part))
            end_day = datetime.datetime(year=year, month=month, day=int(days_split[-1])) + datetime.timedelta(days=1)
            start_end_dates.append(('{}-{}-{}'.format(year, parsed_month, days_split[0]), '{}-{}-{}'.format(year, '{:02}'.format(end_day.month), '{:02}'.format(end_day.day))))
    return out_input_paths, start_end_dates


# endregion


# region quick join

def exclude(df1, df2, col_map):
    for col1, col2 in col_map.items():
        df1 = df1.withColumnRenamed(col1, col2)
    df1 = df1.join(df2, list(col_map.values()), how='leftanti')
    for col1, col2 in col_map.items():
        df1 = df1.withColumnRenamed(col2, col1)
    return df1


# end

# region nested structs

def fold(df, group_cols, fold_col_name, cols_to_fold=None, agg_cols=None):
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    fold_col = F.collect_list(F.struct([col_name for col_name in df.columns if (col_name not in group_cols) and (cols_to_fold is None or col_name in cols_to_fold)])).alias(fold_col_name)
    if agg_cols is None:
        agg_cols = [fold_col]
    else:
        agg_cols = list(agg_cols) + [fold_col]
    return df.groupBy(*group_cols).agg(*agg_cols)


def explode_as_flat_columns(df, col_to_explode, explode_prefix=None, sub_col_names=None, add_id=False, new_cols=None):
    if add_id is True:
        add_id = '_id'
    if add_id:
        df = df.withColumn(add_id, F.monotonically_increasing_id())
    if sub_col_names is None:
        sub_col_names = df.schema[col_to_explode].dataType.elementType.names

    has_explode_prefix = bool(explode_prefix)
    if not explode_prefix:
        explode_prefix = '__explode'

    df = df.withColumn(explode_prefix, F.explode(col_to_explode))

    if has_explode_prefix:
        for sub_col_name in sub_col_names:
            if new_cols is None or sub_col_name not in new_cols:
                df = df.withColumn('{}_{}'.format(explode_prefix, sub_col_name), F.col('{}.{}'.format(explode_prefix, sub_col_name)))
    else:
        for sub_col_name in sub_col_names:
            if new_cols is None or sub_col_name not in new_cols:
                df = df.withColumn(sub_col_name, F.col('{}.{}'.format(explode_prefix, sub_col_name)))

    if new_cols is not None:
        for key, col in new_cols.items():
            if isinstance(col, str):
                df = df.withColumn(key, F.col(col))
            else:
                df = df.withColumn(key, col)

    df = df.drop(explode_prefix).drop(col_to_explode)
    return df


# endregion

# region quick counts & ratio

def get_counts(dfs, group_cols, count_col_name='count', min_count=-1, ratio_col_name='ratio', min_ratio=-1, ratio_digits=4, col_name_suffix=True, total=None):
    def _get_cnts(df, total):
        if ratio_col_name is not None and total is None:
            total = df.count()
        if isinstance(group_cols, str):
            df = df.groupBy(group_cols)
        else:
            df = df.groupBy(*group_cols)
        df = df.agg(F.count('*').alias(count_col_name)).orderBy(F.col(count_col_name).desc())
        if min_count >= 0:
            df = df.where(F.col(count_col_name) > min_count)
        if ratio_col_name is not None:
            if ratio_digits >= 0:
                df = df.withColumn(ratio_col_name, F.round(F.col(count_col_name) / total, ratio_digits))
            else:
                df = df.withColumn(ratio_col_name, F.col(count_col_name) / total)
            if min_ratio >= 0:
                df = df.where(F.col(ratio_col_name) > min_count)
        return df

    if not isinstance(dfs, (list, tuple)):
        return _get_cnts(dfs, total)
    else:
        if isinstance(total, (tuple, list)):
            return join_multiple_on_columns([_get_cnts(df, _total) for df, _total in zip(dfs, total)], join_key_col_names=group_cols, col_name_suffix=col_name_suffix, how='outer')
        else:
            return join_multiple_on_columns([_get_cnts(df, total) for df in dfs], join_key_col_names=group_cols, col_name_suffix=col_name_suffix, how='outer')


def show_counts(df, group_cols, count_col_name='count', min_count=0, ratio_col_name='ratio', min_ratio=-1, ratio_digits=4, show_limit=100, truncate=False):
    get_counts(df, group_cols=group_cols, count_col_name=count_col_name, min_count=min_count, ratio_col_name=ratio_col_name, min_ratio=min_ratio, ratio_digits=ratio_digits).show(show_limit, truncate=truncate)
    return df


def show_count(df, title):
    cnt = df.count()
    print('{}: {}'.format(title, cnt))
    return cnt


def show_ratio(df1, df2, title1, title2):
    cnt1, cnt2 = df1.count(), df2.count()
    print('{}: {}'.format(title1, cnt1))
    print('{}: {}'.format(title2, cnt2))
    print('ratio: {}'.format(cnt1 / float(cnt2)))
    return cnt1, cnt2


def cache__(df, name=None, unpersist=None):
    """
    Caches the specified dataframe, immediately materialize it, and un-caches the dataframes specified in `unpersist` after the materialization of `df` cache.
    This is a very useful utility function to make fine-grained cache optimization, while avoiding code clutter.
    """

    def _cache(df, name):
        df = df.cache()
        if name is None:
            df.count()
        else:
            show_count(df, name)
        return df

    if isinstance(df, Mapping):
        out = tuple(_cache(_df, name) for name, _df in df.items())

    elif isinstance(df, (list, tuple)):
        if name is None:
            out = tuple(_cache(_df, None) for _df in df)
        else:
            out = tuple(_cache(_df, name) for name, _df in zip(name, df))
    else:
        out = _cache(df, name)

    if unpersist is not None:
        if isinstance(unpersist, DataFrame):
            unpersist.unpersist()
        else:
            for _df in unpersist:
                _df.unpersist()
    return out


# endregion

# region shortcut methods


def rename(df, name_map):
    if isinstance(name_map, Mapping):
        name_map = name_map.items()
    for src, trg in name_map:
        if src != trg:
            df = df.withColumnRenamed(src, trg)
    return df


def rename_by_adding_suffix(df, suffix, col_names_to_change=None, excluded_col_names=None):
    """
    Appending a `suffix` to every field name specified in `col_names_to_change` except for those in `excluded_col_names`.
    If `col_names_to_change` is not specified, then every field in the dataframe will get the suffix.
    """
    if col_names_to_change is None:
        col_names_to_change = df.columns

    return rename(df, ((col_name, col_name + str(suffix)) for col_name in col_names_to_change if (excluded_col_names is None or col_name not in excluded_col_names)))


def rename_by_replace(df, src, trg):
    """
    Renames every column of the dataframe if the column name contains `src`, replacing `src` by `trg`.
    """
    for col_name in df.columns:
        if src in col_name:
            df = df.withColumnRenamed(col_name, col_name.replace(src, trg))
    return df


def filter_on_columns(df, df_filter, df_col_names, filter_col_names=None):
    if isinstance(df_col_names, str):
        df_col_names = [df_col_names]
    if isinstance(filter_col_names, str):
        filter_col_names = [filter_col_names]
    if filter_col_names:
        if set(filter_col_names) != set(df_filter.columns):
            df_filter = df_filter.select(*filter_col_names)
    else:
        if set(df_col_names) != set(df_filter.columns):
            df_filter = df_filter.select(*df_col_names)
    df_filter = df_filter.distinct()
    if filter_col_names:
        df_filter = rename(df_filter, zip(filter_col_names, df_col_names))
    return df.join(df_filter, df_col_names, how='inner')


def fold_as_struct_by_prefix(df, prefixes, prefix_connector='_', excluded_col_names=None):
    """
    Folding each set of fields with each of the specified `prefixes` into a struct.
    For example, if the dataframe has three fields 'customer_field1', 'customer_field2', 'customer_field3', 'global_field1', 'global_field2', 'global_field3',
    then calling `fold_as_struct_by_prefix(df, [customer, global], '_', None)` will create two fields 'customer' and 'global' like `{ "customer": { "field1": ..., "field2": ..., "field3": ... }, "global": { "field1": ..., "field2": ..., "field3": ... }}`
    """
    if isinstance(excluded_col_names, str):
        excluded_col_names = [excluded_col_names]
    for prefix in prefixes:
        prefix_ = prefix
        if not prefix.endswith(prefix_connector):
            prefix_ = prefix + prefix_connector
        cols = [col_name for col_name in df.columns if col_name.startswith(prefix_) if col_name not in excluded_col_names]
        df = df.withColumn(prefix, F.struct(*(F.col(col_name).alias(col_name[len(prefix_):]) for col_name in cols))).drop(*cols)
    return df


def fold_as_struct_by_suffix(df, suffixes, suffix_connector='_', excluded_col_names=None):
    """
    The same as `fold_as_struct_by_prefix` by we fold fields based on the suffixes.
    """
    if isinstance(excluded_col_names, str):
        excluded_col_names = [excluded_col_names]
    for suffix in suffixes:
        suffix_ = suffix
        if not suffix.startswith(suffix_connector):
            suffix_ = suffix_connector + suffix
        cols = [col_name for col_name in df.columns if col_name.endswith(suffix_) if col_name not in excluded_col_names]
        df = df.withColumn(suffix, F.struct(*(F.col(col_name).alias(col_name[:-len(suffix_)]) for col_name in cols))).drop(*cols)
    return df


# endregion


def withDefault(col_name, default=0):
    return F.when(F.col(col_name).isNull(), F.lit(default)).otherwise(F.col(col_name))


def has_col(df, col_name):
    try:
        df.select(F.col(col_name))
        return True
    except:
        return False


def select(df, *args):
    cols = []
    for arg in args:
        if isinstance(arg, Mapping):
            cols += [F.col(k).alias(v) for k, v in arg.items()]
        else:
            cols.append(arg)
    return df.select(*cols)


def _solve_col(col_or_colname):
    if isinstance(col_or_colname, str):
        return F.col(col_or_colname)
    else:
        return col_or_colname


def aggregate(df, group_cols, agg_cols, max_cols=None, min_cols=None, collect_list_cols=None,
              concat_list_cols=None,
              count_col_name='count',
              collect_list_col_name='occurrences',
              max_cols_prefix='max_',
              min_cols_prefix='min_',
              compute_avg=False,
              avg_by_count_col=False,
              agg_col_prefix=''):
    """
    Summing values of the `agg_cols` grouped by the `group_cols`. A count column will be automatically added.
    """
    if agg_cols is None:
        if count_col_name:
            agg_cols = (F.count('*').alias(agg_col_prefix + count_col_name),)
        else:
            agg_cols = ()
    else:
        if compute_avg:
            if avg_by_count_col:
                _agg_cols = agg_cols
                if count_col_name:
                    agg_cols = (F.sum(count_col_name).alias(agg_col_prefix + count_col_name),)
                else:
                    agg_cols = ()
                agg_cols += tuple((F.sum(col_name) / F.sum(count_col_name)).alias(agg_col_prefix + col_name) for col_name in _agg_cols)
            else:
                _agg_cols = agg_cols
                if count_col_name:
                    agg_cols = (F.count('*').alias(agg_col_prefix + count_col_name),)
                else:
                    agg_cols = ()
                agg_cols += tuple(F.avg(col_name).alias(agg_col_prefix + col_name) for col_name in _agg_cols)
        else:
            _agg_cols = agg_cols
            if count_col_name:
                agg_cols = (F.count('*').alias(agg_col_prefix + count_col_name),)
            else:
                agg_cols = ()
            agg_cols += tuple(F.sum(col_name).alias(agg_col_prefix + col_name) for col_name in _agg_cols)
    if max_cols:
        agg_cols += tuple(F.max(col_name).alias(max_cols_prefix + col_name) for col_name in max_cols)
    if min_cols:
        agg_cols += tuple(F.min(col_name).alias(min_cols_prefix + col_name) for col_name in min_cols)
    if concat_list_cols:
        agg_cols += tuple(F.flatten(F.collect_list(col_name)).alias(col_name) for col_name in concat_list_cols)

    if collect_list_cols:
        agg_cols += (F.collect_list(
            F.struct(*collect_list_cols)
        ).alias(collect_list_col_name),)
    return df.groupBy(*group_cols).agg(*agg_cols)


def prev_and_next(df, group_cols, order_cols, prev_next_col_names=None, shared_col_names=None, suffix_prev='_first', suffix_next='_second', null_next_indicator_col_name=None):
    if not isinstance(group_cols, (list, tuple)):
        group_cols = (group_cols,)
    if not isinstance(order_cols, (list, tuple)):
        order_cols = (order_cols,)
    if prev_next_col_names is None:
        prev_next_col_names = [col_name for col_name in df.columns if col_name not in group_cols and col_name not in order_cols and (shared_col_names is None or col_name not in shared_col_names)]
    for col_name in prev_next_col_names:
        df = df.withColumn(col_name + suffix_next, F.lead(F.col(col_name)).over(Window.partitionBy(*group_cols).orderBy(*order_cols)))
        df = df.withColumnRenamed(col_name, col_name + suffix_prev)
    if null_next_indicator_col_name:
        df = df.where(F.col(null_next_indicator_col_name + suffix_next).isNotNull())
    return df

# # region paths organized by dates
#
# def _format_days_for_spark_path(days):
#     return '{' + ','.join(days) + '}'
#
#
# def _parse_days(days_expr, year, month, days_chunk_size=0):
#     days = []
#     splits = days_expr.split(',')
#     for split in splits:
#         subsplits = split.split('-')
#         if len(subsplits) == 1:
#             day = subsplits[0]
#             if day == '*':
#                 days = list(range(1, monthrange(year, month)[1] + 1))
#                 break
#             else:
#                 days.append(int(day))
#         else:
#             days.extend(list(range(int(subsplits[0]), int(subsplits[1]) + 1)))
#     days = ['{:02}'.format(day) for day in sorted(set(days))]
#     if days_chunk_size != 0:
#         days = [days[i:i + days_chunk_size] for i in range(0, len(days), days_chunk_size)]
#         return [_format_days_for_spark_path(days_split) for days_split in days], days
#     else:
#         return [_format_days_for_spark_path(days)], [days]
#
#
# def get_input_paths_organized_by_dates(input_path_pattern, year, months_days_expr, days_chunk_size=0):
#     out_input_paths, start_end_dates = [], []
#     for month, days_expr in map(lambda x: x.split('/'), months_days_expr.split(';')):
#         month = int(month)
#         parsed_month = '{:02}'.format(month)
#         parsed_days_path_parts, days_splits = _parse_days(year=year, month=month, days_expr=days_expr, days_chunk_size=days_chunk_size)
#         for days_path_part, days_split in zip(parsed_days_path_parts, days_splits):
#             out_input_paths.append(input_path_pattern.format(year=year, month=parsed_month, days=days_path_part))
#             start_end_dates.append(('{}-{}-{}'.format(year, parsed_month, days_split[0]), '{}-{}-{}'.format(year, parsed_month, days_split[-1])))
#     return out_input_paths, start_end_dates
#
# # endregion

import itertools
from functools import reduce
from typing import Tuple, Union

from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType
from pyspark.sql.window import Window
from tqdm import tqdm
import re
from utix.general import tqdm_wrap
from copy import copy


def local_spark(app_name='data', parallel='*') -> SparkSession:
    if 'spark' in globals():
        return globals()['spark']
    else:
        return SparkSession.builder \
            .appName(app_name) \
            .master("local[{}]".format(parallel)) \
            .getOrCreate()


def def_local_spark(app_name='data', parallel='*'):
    if 'spark' not in globals():
        global spark
        spark = local_spark(app_name=app_name, parallel=parallel)


def group_with_cnt_column(df, *cols, cnt_col_name=None) -> Tuple[DataFrame, str]:
    if cnt_col_name is None:
        cnt_col_name = '__'.join(cols) + '___cnt'
    return df.groupBy(*cols).agg(F.count("*").alias(cnt_col_name)), cnt_col_name


def group_with_max_column(df, mx_trg_col, *group_cols, mx_col_name=None) -> Tuple[DataFrame, str]:
    if mx_col_name is None:
        mx_col_name = '__'.join(group_cols) + f'__{mx_trg_col}___mx'
    return df.groupBy(*group_cols).agg(F.max(mx_trg_col).alias(mx_col_name)), mx_col_name


def group_with_rank_column(df, group_cols, orderby_cols, ascending=True, rank_col_name=None):
    if rank_col_name is None:
        rank_col_name = '__'.join(group_cols) + '_' + '__'.join(orderby_cols) + f'___rank'
    if not ascending:
        orderby_cols = [F.col(colname).desc() if isinstance(colname, str) else colname for colname in orderby_cols]
    w = Window().partitionBy(*group_cols).orderBy(*orderby_cols)
    return df.withColumn(rank_col_name, F.row_number().over(w)), rank_col_name


def deduplicate(df, *keycols, top=1) -> DataFrame:
    dff, rankcol = group_with_rank_column(df, group_cols=keycols, orderby_cols=keycols)
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


def column_as_set(df: DataFrame, col):
    return set(row[col] for row in tqdm(df.select(col).distinct().collect()))


def columns_as_set(df: DataFrame, *cols):
    return set(tuple(row[col] for col in cols) for row in tqdm(df.select(*cols).distinct().collect()))


def trim_columns(df, *cols) -> DataFrame:
    """
    Trims texts in the specified columns.
    """
    for col in cols:
        df = df.withColumn(col, F.trim(F.col(col)))
    return df


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
        df2, rank_col_name = group_with_rank_column(df1, group_cols=(src_col,), orderby_cols=(cnt_col_name,), ascending=False)
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


def get_columns_by_pattern(df, column_name_pattern):
    column_name_pattern = '^' + column_name_pattern + '$'
    return [_col for _col in df.columns if re.match(column_name_pattern, _col)]


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


def get_input_paths_organized_by_dates(input_path_pattern, year, months_days_expr, days_chunk_size=0):
    out_input_paths, start_end_dates = [], []
    for month, days_expr in map(lambda x: x.split('/'), months_days_expr.split(';')):
        month = int(month)
        parsed_month = '{:02}'.format(month)
        parsed_days_path_parts, days_splits = _parse_days(year=year, month=month, days_expr=days_expr, days_chunk_size=days_chunk_size)
        for days_path_part, days_split in zip(parsed_days_path_parts, days_splits):
            out_input_paths.append(input_path_pattern.format(year=year, month=parsed_month, days=days_path_part))
            start_end_dates.append(('{}-{}-{}'.format(year, parsed_month, days_split[0]), '{}-{}-{}'.format(year, parsed_month, days_split[-1])))
    return out_input_paths, start_end_dates

# endregion

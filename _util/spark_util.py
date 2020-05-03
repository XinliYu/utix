import itertools

from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F


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

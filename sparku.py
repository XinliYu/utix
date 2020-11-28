USE_CYTHON = False
if USE_CYTHON:
    try:
        from utix._utilc.spark_ext import *
    except:
        from utix._util.spark_util import *
else:
    from utix._util.spark_util import *

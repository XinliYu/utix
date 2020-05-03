USE_CYTHON = False
if USE_CYTHON:
    try:
        from _utilc.spark_ext import *
    except:
        from _util.spark_util import *
else:
    from _util.spark_util import *

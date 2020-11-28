USE_CYTHON = False
if USE_CYTHON:
    try:
        from utix._utilc.stat_util import *
    except:
        from utix._util.stat_util import *
else:
    from utix._util.stat_util import *

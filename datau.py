USE_CYTHON = False
if USE_CYTHON:
    try:
        from utix._utilc.data_util import *
    except:
        from utix._util.data_util import *
else:
    from utix._util.data_util import *

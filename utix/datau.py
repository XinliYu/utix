USE_CYTHON = False
if USE_CYTHON:
    try:
        from _utilc.data_util import *
    except:
        from _util.data_util import *
else:
    from _util.data_util import *

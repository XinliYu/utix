USE_CYTHON = False
if USE_CYTHON:
    try:
        from _utilc.stat_util import *
    except:
        from _util.stat_util import *
else:
    from _util.stat_util import *

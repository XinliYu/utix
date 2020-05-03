USE_CYTHON = False
if USE_CYTHON:
    try:
        from _utilc.time_ext import *
    except:
        from _util.time_ext import *
else:
    from _util.time_ext import *

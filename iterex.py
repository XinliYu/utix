USE_CYTHON = False
if USE_CYTHON:
    try:
        from _utilc.iter_ext import *
    except:
        from _util.iter_ext import *
else:
    from _util.iter_ext import *

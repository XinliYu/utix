USE_CYTHON = False
if USE_CYTHON:
    try:
        from _utilc.path_ext import *
    except:
        from _util.path_ext import *
else:
    from _util.path_ext import *

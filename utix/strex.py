USE_CYTHON = False
if USE_CYTHON:
    try:
        from _utilc.str_ext import *
    except:
        from _util.str_ext import *
else:
    from _util.str_ext import *

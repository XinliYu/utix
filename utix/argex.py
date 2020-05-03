USE_CYTHON = False
if USE_CYTHON:
    try:
        from _utilc.arg_ext import *
    except:
        from _util.arg_ext import *
else:
    from _util.arg_ext import *

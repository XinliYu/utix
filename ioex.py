USE_CYTHON = False
if USE_CYTHON:
    try:
        from _utilc.io_ext import *
    except:
        from _util.io_ext import *
else:
    from _util.io_ext import *

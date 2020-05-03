USE_CYTHON = False
if USE_CYTHON:
    try:
        from _utilc.np_ext import *
    except:
        from _util.np_ext import *
else:
    from _util.np_ext import *

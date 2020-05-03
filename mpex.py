USE_CYTHON = False
if USE_CYTHON:
    try:
        from _utilc.mp_ext import *
    except:
        from _util.mp_ext import *
else:
    from _util.mp_ext import *

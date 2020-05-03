USE_CYTHON = False
if USE_CYTHON:
    try:
        from _utilc.general_ext import *
    except:
        from _util.general_ext import *
else:
    from _util.general_ext import *
del USE_CYTHON

USE_CYTHON = False
if USE_CYTHON:
    try:
        from _utilc.dict_ext import *
    except:
        from _util.dict_ext import *
else:
    from _util.dict_ext import *
del USE_CYTHON

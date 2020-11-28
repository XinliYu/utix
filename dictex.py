USE_CYTHON = False
if USE_CYTHON:
    try:
        from utix._utilc.dict_ext import *
    except:
        from utix._util.dict_ext import *
else:
    from utix._util.dict_ext import *
del USE_CYTHON

USE_CYTHON = False
if USE_CYTHON:
    try:
        from utix._utilc.general_ext import *
    except:
        from utix._util.general_ext import *
else:
    from utix._util.general_ext import *
del USE_CYTHON

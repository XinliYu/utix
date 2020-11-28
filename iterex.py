USE_CYTHON = False
if USE_CYTHON:
    try:
        from utix._utilc.iter_ext import *
    except:
        from utix._util.iter_ext import *
else:
    from utix._util.iter_ext import *

USE_CYTHON = False
if USE_CYTHON:
    try:
        from utix._utilc.str_ext import *
    except:
        from utix._util.str_ext import *
else:
    from utix._util.str_ext import *

USE_CYTHON = False
if USE_CYTHON:
    try:
        from utix._utilc.arg_ext import *
    except:
        from utix._util.arg_ext import *
else:
    from utix._util.arg_ext import *

USE_CYTHON = False
if USE_CYTHON:
    try:
        from utix._utilc.path_ext import *
    except:
        from utix._util.path_ext import *
else:
    from utix._util.path_ext import *

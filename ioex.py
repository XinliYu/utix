USE_CYTHON = False
if USE_CYTHON:
    try:
        from utix._utilc.io_ext import *
    except:
        from utix._util.io_ext import *
else:
    from utix._util.io_ext import *

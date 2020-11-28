USE_CYTHON = False
if USE_CYTHON:
    try:
        from utix._utilc.time_ext import *
    except:
        from utix._util.time_ext import *
else:
    from utix._util.time_ext import *

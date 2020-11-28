USE_CYTHON = False
if USE_CYTHON:
    try:
        from utix._utilc.np_ext import *
    except:
        from utix._util.np_ext import *
else:
    from utix._util.np_ext import *

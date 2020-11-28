USE_CYTHON = False
if USE_CYTHON:
    try:
        from utix._utilc.list_ext import *
    except:
        from utix._util.list_ext import *
else:
    from utix._util.list_ext import *

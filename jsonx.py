USE_CYTHON = False
if USE_CYTHON:
    try:
        from utix._utilc.json_ext import *
    except:
        from utix._util.json_ext import *
else:
    from utix._util.json_ext import *
del USE_CYTHON
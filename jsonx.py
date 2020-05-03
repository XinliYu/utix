USE_CYTHON = False
if USE_CYTHON:
    try:
        from _utilc.json_ext import *
    except:
        from _util.json_ext import *
else:
    from _util.json_ext import *
del USE_CYTHON
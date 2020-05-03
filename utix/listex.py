USE_CYTHON = False
if USE_CYTHON:
    try:
        from _utilc.list_ext import *
    except:
        from _util.list_ext import *
else:
    from _util.list_ext import *

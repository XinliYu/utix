USE_CYTHON = False
if USE_CYTHON:
    try:
        from utix._utilc.mp_ext import *
    except:
        from utix._util.mp_ext import *
else:
    from utix._util.mp_ext import *

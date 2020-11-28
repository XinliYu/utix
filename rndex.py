USE_CYTHON = False
if USE_CYTHON:
    try:
        from utix._utilc.rnd_ext import *
    except:
        from utix._util.rnd_ext import *
else:
    from utix._util.rnd_ext import *

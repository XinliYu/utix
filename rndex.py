USE_CYTHON = False
if USE_CYTHON:
    try:
        from _utilc.rnd_ext import *
    except:
        from _util.rnd_ext import *
else:
    from _util.rnd_ext import *

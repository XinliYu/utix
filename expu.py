USE_CYTHON = False
if USE_CYTHON:
    try:
        from _utilc.exp_util import *
    except:
        from _util.exp_util import *
else:
    from _util.exp_util import *

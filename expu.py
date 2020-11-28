USE_CYTHON = False
if USE_CYTHON:
    try:
        from utix._utilc.exp_util import *
    except:
        from utix._util.exp_util import *
else:
    from utix._util.exp_util import *

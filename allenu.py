USE_CYTHON = False
if USE_CYTHON:
    try:
        from _utilc.allen_util import *
    except:
        from _util.allen_util import *
else:
    from _util.allen_util import *

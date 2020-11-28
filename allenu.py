USE_CYTHON = False
if USE_CYTHON:
    try:
        from utix._utilc.allen_util import *
    except:
        from utix._util.allen_util import *
else:
    from utix._util.allen_util import *

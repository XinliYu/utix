USE_CYTHON = False
if USE_CYTHON:
    try:
        from utix._utilc.torch_util import *
    except:
        from utix._util.torch_util import *
else:
    from utix._util.torch_util import *

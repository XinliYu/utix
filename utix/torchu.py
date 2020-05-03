USE_CYTHON = False
if USE_CYTHON:
    try:
        from _utilc.torch_util import *
    except:
        from _util.torch_util import *
else:
    from _util.torch_util import *

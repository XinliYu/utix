USE_CYTHON = False
if USE_CYTHON:
    try:
        from _utilc.graph_util import *
    except:
        from _util.graph_util import *
else:
    from _util.graph_util import *
del USE_CYTHON

USE_CYTHON = False
if USE_CYTHON:
    try:
        from utix._utilc.graph_util import *
    except:
        from utix._util.graph_util import *
else:
    from utix._util.graph_util import *
del USE_CYTHON

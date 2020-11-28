USE_CYTHON = False
if USE_CYTHON:
    try:
        from utix._utilc.faiss_util import *
    except:
        from utix._util.faiss_util import *
else:
    from utix._util.faiss_util import *
del USE_CYTHON

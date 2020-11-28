USE_CYTHON = False
if USE_CYTHON:
    try:
        from utix._utilc.data_loading import *
    except:
        from utix._util.data_loading import *
else:
    from utix._util.data_loading import *

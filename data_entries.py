USE_CYTHON = False
if USE_CYTHON:
    try:
        from utix._utilc.data_entries import *
    except:
        from utix._util.data_entries import *
else:
    from utix._util.data_entries import *

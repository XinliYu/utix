USE_CYTHON = False
if USE_CYTHON:
    try:
        from utix._utilc.csv_ext import *
    except:
        from utix._util.csv_ext import *
else:
    from utix._util.csv_ext import *

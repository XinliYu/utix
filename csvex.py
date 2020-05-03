USE_CYTHON = False
if USE_CYTHON:
    try:
        from _utilc.csv_ext import *
    except:
        from _util.csv_ext import *
else:
    from _util.csv_ext import *

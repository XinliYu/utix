USE_CYTHON = False
if USE_CYTHON:
    try:
        from utix._utilc.plot_ext import *
    except:
        from utix._util.plot_ext import *
else:
    from utix._util.plot_ext import *

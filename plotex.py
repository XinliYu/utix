USE_CYTHON = False
if USE_CYTHON:
    try:
        from _utilc.plot_ext import *
    except:
        from _util.plot_ext import *
else:
    from _util.plot_ext import *

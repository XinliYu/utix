USE_CYTHON = False
if USE_CYTHON:
    try:
        from _utilc.msg_ext import *
    except:
        from _util.msg_ext import *
else:
    from _util.msg_ext import *

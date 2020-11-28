USE_CYTHON = False
if USE_CYTHON:
    try:
        from utix._utilc.msg_ext import *
    except:
        from utix._util.msg_ext import *
else:
    from utix._util.msg_ext import *

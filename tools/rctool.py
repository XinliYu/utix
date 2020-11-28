USE_CYTHON = False
if USE_CYTHON:
    try:
        from utix._utilc.tools.rctrool import *
    except:
        from utix._util.tools.rctool import *
else:
    from utix._util.tools.rctool import *

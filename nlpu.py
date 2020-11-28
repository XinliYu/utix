USE_CYTHON = False
if USE_CYTHON:
    try:
        from utix._utilc.nlp_util import *
    except:
        from utix._util.nlp_util import *
else:
    from utix._util.nlp_util import *

USE_CYTHON = False
if USE_CYTHON:
    try:
        from _utilc.nlp_util import *
    except:
        from _util.nlp_util import *
else:
    from _util.nlp_util import *

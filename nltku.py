USE_CYTHON = False
if USE_CYTHON:
    try:
        from _utilc.nltk_util import *
    except:
        from _util.nltk_util import *
else:
    from _util.nltk_util import *

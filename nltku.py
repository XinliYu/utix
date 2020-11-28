USE_CYTHON = False
if USE_CYTHON:
    try:
        from utix._utilc.nltk_util import *
    except:
        from utix._util.nltk_util import *
else:
    from utix._util.nltk_util import *

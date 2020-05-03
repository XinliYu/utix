USE_CYTHON = True
if USE_CYTHON:
    try:
        import _utilc._external.scp as scp
    except:
        import _util._external.scp as scp
else:
    import _util._external.scp as scp
del USE_CYTHON

USE_CYTHON = True
if USE_CYTHON:
    try:
        import _utilc._external.parse as parse
    except:
        import _util._external.parse as parse
else:
    import _util._external.parse as parse
del USE_CYTHON

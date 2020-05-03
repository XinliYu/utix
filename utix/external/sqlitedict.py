USE_CYTHON = True
if USE_CYTHON:
    try:
        import _utilc._external.sqlitedict as sqlitedict
    except:
        import _util._external.sqlitedict as sqlitedict
else:
    import _util._external.sqlitedict as sqlitedict
del USE_CYTHON

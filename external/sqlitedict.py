USE_CYTHON = True
if USE_CYTHON:
    try:
        import utix._utilc._external.sqlitedict as sqlitedict
    except:
        import utix._util._external.sqlitedict as sqlitedict
else:
    import utix._util._external.sqlitedict as sqlitedict
del USE_CYTHON

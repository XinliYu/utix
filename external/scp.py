USE_CYTHON = True
if USE_CYTHON:
    try:
        import utix._utilc._external.scp as scp
    except:
        import utix._util._external.scp as scp
else:
    import utix._util._external.scp as scp
del USE_CYTHON

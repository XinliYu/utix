USE_CYTHON = True
if USE_CYTHON:
    try:
        import utix._utilc._external.parse as parse
    except:
        import utix._util._external.parse as parse
else:
    import utix._util._external.parse as parse
del USE_CYTHON

from distutils.core import setup
from Cython.Build import cythonize

_root = '../../'
setup(
    ext_modules=cythonize([f"{_root}/_util/_external/parse.py",
                           f"{_root}/_util/_external/sqlitedict.py",
                           f"{_root}/_util/_external/sqlitedict.py"])
)

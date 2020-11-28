from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(["../_util/general.py", "../_util/dictex.py", "../_util/graphu.py"])
)

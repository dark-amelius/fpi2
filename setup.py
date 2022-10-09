from distutils.core import setup
from Cython.Build import cythonize
import numpy

# Compiles Cython files
setup(ext_modules=cythonize('helpers.pyx'),
      include_dirs=[numpy.get_include()]
)
print("done")
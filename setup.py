from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(name="cython_sampler", ext_modules=cythonize('cython_sampler.pyx'),include_dirs=[np.get_include()])
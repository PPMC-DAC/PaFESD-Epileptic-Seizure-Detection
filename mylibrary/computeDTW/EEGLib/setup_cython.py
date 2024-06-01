from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

from distutils import sysconfig

sysconfig.get_config_vars()['CC'] = 'gcc-12'
sysconfig.get_config_vars()['CXX'] = 'g++-12'

# First create an Extension object with the appropriate name and sources
ext = Extension(name="EEGLIB",
                sources=['cython_wrapper.pyx', 'EEG_lib.cpp'],
                language="c++",
                extra_compile_args=['-std=c++17', '-O3', '-fopenmp', '-Wno-sign-compare'],
                extra_link_args=['-std=c++17','-fopenmp']) #'-Wl,-rpath,/usr/local/gcc-12/lib64/'

# Use cythonize on the extension object.
setup(name="EEGLIB",
      ext_modules=cythonize(ext),
      include_dirs=[numpy.get_include()])
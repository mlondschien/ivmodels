from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "integrand",
        sources=["ivmodels/tests/integrand.pyx"],
        libraries=["gsl", "gslcblas"],
        library_dirs=["/opt/homebrew/lib"],
        include_dirs=["/opt/homebrew/include", np.get_include()],
        extra_compile_args=["-fPIC"]
    )
]

setup(
    name="integrand",
    ext_modules=cythonize(ext_modules),
)
# Available at setup time due to pyproject.toml
import eigenpip
from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

__version__ = "0.2.1"

ext_modules = [
    Pybind11Extension("planar",
        ["planar/bindings.cpp"],
        depends=["planar/bindings.hpp"],
        include_dirs=[eigenpip.get_include()],
        cxx_std=17,
        ),
]

setup(
    name="planar",
    version=__version__,
    author="James A. Preiss",
    author_email="jamesalanpreiss@gmail.com",
    #url="https://github.com/jpreiss/mlpfile",
    #description="Multilayer perceptron file format and evaluation.",
    #long_description=open("README.md", "r").read(),
    #long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    packages=["planar"],
    package_data={
        "planar": ["planar/bindings.cpp", "planar/planar.hpp"]
    },
    #requires=["numpy"],
    zip_safe=False,  # TODO: Understand.
    python_requires=">=3.7",
)

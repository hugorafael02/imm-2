from setuptools import setup, Extension
import pybind11
import sys
import os

# Detecção de Sistema Operacional
if sys.platform == 'win32':
    # Configurações para Visual Studio (Windows)
    cpp_args = ['/O2', '/openmp', '/EHsc', '/std:c++14']
    link_args = []
else:
    # Configurações para GCC (Linux/AWS)
    cpp_args = ['-O3', '-fopenmp', '-std=c++11']
    link_args = ['-fopenmp']

ext_modules = [
    Extension(
        'imm_module',
        # Apenas bindings.cpp é necessário, pois ele inclui o imm.cpp via #include
        ['bindings.cpp'], 
        
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=cpp_args,
        extra_link_args=link_args,
    ),
]

setup(
    name='imm_module',
    version='1.0',
    ext_modules=ext_modules,
)
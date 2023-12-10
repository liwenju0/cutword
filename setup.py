#! -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(
    name='bytepiece',
    version='0.6.3',
    python_requires='>=3',
    description='Smarter Byte-based Tokenizer',
    long_description=open('README_en.md', encoding="utf-8").read(),
    long_description_content_type='text/markdown',
    license='Apache License 2.0',
    url='https://github.com/bojone/bytepiece',
    author='bojone',
    author_email='bojone@spaces.ac.cn',
    install_requires=['numpy', 'tqdm'],
    packages=find_packages(),
    ext_modules=cythonize('bytepiece/*.pyx'),
    package_data={'bytepiece': ['*.pyx']},
    include_package_data=True
)

#! -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(
    name='cutword',
    version='0.0.1',
    python_requires='>=3',
    description='Just Cut Word Faster',
    long_description=open('README.md', encoding="utf-8").read(),
    long_description_content_type='text/markdown',
    license='Apache License 2.0',
    url='https://github.com/liwenju0/cutword',
    author='liwenju',
    author_email='liwenjudetiankong@126.com',
    install_requires=['numpy', 'tqdm'],
    packages=find_packages(),
    ext_modules=cythonize('cutword/*.pyx'),
    package_data={'cutword': ['*.pyx']},
    include_package_data=True
)

# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='sample',
    version='0.1.0',
    description='NFP_PROJECT',
    long_description=readme,
    author='Mike Kritzell',
    author_email='mkrite1@kent.edu',
    url='https://github.com/kritzell33/NFP_Project',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)


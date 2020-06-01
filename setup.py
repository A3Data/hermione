from setuptools import setup, find_packages
import re
import os

exec(open('hermione/_version.py').read())

setup(
    name='hermione-ml',
    version=__version__,
    author='A3Data',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Click',
        'conda'
    ],
    entry_points='''
        [console_scripts]
        hermione=hermione.cli:cli
    ''',
)

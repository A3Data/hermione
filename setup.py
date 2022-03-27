from setuptools import setup, find_packages
import re
import os

from hermione._version import __version__

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()
    
def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

extra_files = package_files('hermione/data')

setup(
    name='hermione-ml',
    version=__version__,
    author='A3Data',
    author_email='hermione@a3data.com.br',
    url='https://github.com/A3Data/hermione',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development'
      ],
      keywords='machine learning mlops devops artificial intelligence',
      license='Apache License 2.0',
    install_requires=[
        'Click',
        'Jinja2'
    ],
    entry_points='''
        [console_scripts]
        hermione=hermione.cli.main:cli
    ''',
    python_requires='>=3.8.0',
    package_data={'': extra_files},
)

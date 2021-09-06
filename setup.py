from setuptools import setup, find_packages

from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext


setup(
    name='classicalgsg',
    version='0.0.1a.dev3',

    author="Nazanin Donyapour",
    author_email="nazanin@msu.edu",
    description="ClassicalGSG",
    license="MIT",
    url="https://github.com/ADicksonLab/ClassicalGSG",
    classifiers=[
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
        'Programming Language :: Python :: 3'
    ],

    # package
    packages=find_packages(where='src'),

    package_dir={'' : 'src'},

    # if this is true then the package_data won't be included in the
    # dist. Use MANIFEST.in for this
    include_package_data=True,

    # py_modules=[
    #     'src/LogpPredictor',
    # ],

    # SNIPPET: this is a general way to do this
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],

    entry_points = {},

    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'skorch',
        'tabulate',
        'ParmEd',
    ],
)

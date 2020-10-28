from setuptools import setup, find_packages

setup(
    name='classicalgsg',
    version='0.0.1',
    py_modules=['classicalgsg'],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
    ],
)

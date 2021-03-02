import PhyPraKit  # from this directory
import sys

from setuptools import setup
from setuptools.command.test import test as TestCommand

# class for running unit tests
# from: https://pytest.org/latest/goodpractices.html
class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def run_tests(self):
        import pytest
        errcode = pytest.main(self.pytest_args)
        sys.exit(errcode)

setup(
    name='PhyPraKit',
    version=PhyPraKit.__version__,
    author='Guenter Quast',
    author_email='Guenter.Quast@online.de',
    packages=['PhyPraKit'],
    scripts=[],
    classifiers=[
    'Development Status :: 5 - stable',
    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3.5',
    ],
    url='http://www.ekp.kit.edu/~quast/',
    license='GNU Public Licence',
    description='Tools for data visualisation and analysis in Physics Lab Courses',
    #long_description='todo: add long description',  # open('README.txt').read()
    long_description=open('README.rst').read(),
    setup_requires=[\
        "NumPy >= 1.19.1",
        "SciPy >= 1.5.1",
        "matplotlib >= 3.3.0",
        "iminuit < 2", ]
)

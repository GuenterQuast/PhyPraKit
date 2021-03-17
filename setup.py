import PhyPraKit  # from this directory
import sys

from setuptools import setup

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
    'Programming Language :: Python :: 3.6',
    ],
    url='http://www.etp.kit.edu/~quast/',
    license='GNU Public Licence',
    description='Tools for data visualisation and analysis in Physics Lab Courses',
    long_description=open('README.rst').read(),
    setup_requires=[\
        "NumPy >= 1.19",
        "SciPy >= 1.5",
        "matplotlib >= 3",
        "iminuit < 2",
        "kafe2 >=2.3.0-pre2",]
)

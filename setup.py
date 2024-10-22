import pathlib
from setuptools import setup
import sys

pkg_name = "PhyPraKit"

# The directory containing this file
HERE = pathlib.Path(__file__).parent
# import _version_info from package
sys.path[0] = pkg_name
import _version_info

_version = _version_info._get_version_string()

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name=pkg_name,
    packages=[pkg_name],
    version=_version,
    author="Guenter Quast",
    author_email="Guenter.Quast@online.de",
    url="http://www.etp.kit.edu/~quast/",
    license="GNU Public Licence",
    description="Tools for data visualisation and analysis in Physics Lab Courses",
    long_description="README.md",
    long_description_content_type="text/markdown",
    scripts=[
        "tools/run_phyFit.py",
        "tools/plotData.py",
        "tools/csv2yml.py",
        "tools/plotCSV.py",
        "tools/smoothCSV.py",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        #    'Development Status :: 4 - Beta',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    install_requires=[
        "NumPy >= 1.17",
        "SciPy >= 1.3",
        "matplotlib >= 3",
        "iminuit >=2",
        "kafe2 >=2.8.0",
    ],
)

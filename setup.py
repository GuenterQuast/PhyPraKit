import PhyPraKit  # from this directory
import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()


setup(
    name='PhyPraKit',
    packages=['PhyPraKit'],
    version=PhyPraKit.__version__,
#
    author='Guenter Quast',
    author_email='Guenter.Quast@online.de',
    url='http://www.etp.kit.edu/~quast/',
    license='GNU Public Licence',
    description='Tools for data visualisation and analysis in Physics Lab Courses',
    long_description=README,
    scripts=['tools/run_phyFit.py', 'tools/plotData.py',
             'tools/csv2yml.py', 'tools/plotCSV.py' ],
    classifiers=[
    'Development Status :: 5 - Production/Stable',
    #    'Development Status :: 4 - Beta',
    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3.6',
    ],
    setup_requires=[\
        "NumPy >= 1.19",
        "SciPy >= 1.5",
        "matplotlib >= 3",
        "iminuit >1.99",
        "kafe2 >=2.6.0",]
)

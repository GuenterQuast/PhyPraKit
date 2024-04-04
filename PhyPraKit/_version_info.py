"""
.. module:: _version_info
   :platform: >=3.6
   :synopsis: Version 1.2.7 of PhyPraKit, rel. Apr. 2024

.. moduleauthor:: Guenter Quast <g.quast@kit.edu>
"""

major = 1
minor = 2
revision = 7


def _get_version_tuple():
    """
    version as a tuple
    """
    return (major, minor, revision)


def _get_version_string():
    """
    version as a string
    """
    return "%d.%d.%d" % _get_version_tuple()

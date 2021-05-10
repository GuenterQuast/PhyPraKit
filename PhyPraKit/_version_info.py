'''
.. module:: _version_info
   :platform: python 2.7, >=3.5
   :synopsis: Version 1.1.2 of PhyPraKit, rel. Jan 2021

.. moduleauthor:: Guenter Quast <g.quast@kit.edu>
'''

major = 1
minor = 2
revision = 0

def _get_version_tuple():
  '''
  version as a tuple
  '''
  return (major, minor, revision)

def _get_version_string():
  '''
  version as a string
  '''
  return "%d.%d.%d" % _get_version_tuple()


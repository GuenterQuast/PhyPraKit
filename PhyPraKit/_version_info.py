'''
.. module:: _version_info
   :platform: python 2.7, >=3.4
   :synopsis: Version 1.0.2 of PhyPraKit, rel. Feb. 2019

.. moduleauthor:: Guenter Quast <g.quast@kit.edu>
'''

major = 1
minor = 0
revision = 3

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


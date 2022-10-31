'''
.. module:: _version_info
   :platform: >=3.6
   :synopsis: Version 1.2.3 of PhyPraKit, rel. Sept. 2022

.. moduleauthor:: Guenter Quast <g.quast@kit.edu>
'''

major = 1
minor = 2
revision = 4

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


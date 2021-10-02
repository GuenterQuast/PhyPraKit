"""

.. moduleauthor:: Guenter Quast <guenter.quast@online.de>

.. module PhyPraKit   
   :synopsis: a collection of tools for data analysis
     recommended for "Physikalisches Praktikum am KIT"

.. moduleauthor:: Guenter Quast <g.quast@kit.edu>

**PhyPraKit**  
    a collection of tools for data handling, visualisation and analysis 
    in Physics Lab Courses, recommended for "Physikalisches Praktikum am KIT"

"""


# Import version info
from . import _version_info

# Import main components
from .PhyPraKit import *

_version_suffix = ''  # for suffixes such as 'rc' or 'beta' or 'alpha'
__version__ = _version_info._get_version_string()
__version__ += _version_suffix

"""**PhyPraKit**
   a collection of tools for data handling, visualisation and analysis
   in Physics Lab Courses, recommended for "Physikalisches Praktikum am KIT"

   Sub-modules phyTools and phyFit

   type help(PhyPraKit.phyTools) or help(PhyPraKit.phyFit) for
   an overview of the functionality.

.. module PhyPraKit
   :synopsis: a collection of tools for data analysis
     recommended for "Physikalisches Praktikum am KIT"

.. moduleauthor:: Guenter Quast <guenter.quast@online.de>
"""

# Import version info
from . import _version_info

# Import main components
from .phyTools import *
from .phyFit import *

__version__ = _version_info._get_version_string()

__all__ = ["phyTools", "phyFit"]

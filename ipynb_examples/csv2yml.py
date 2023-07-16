#!/usr/bin/env python3
"""csv2yml.py 
  read floating-point column-data in very general .txt  formats 
  and write an output block in yaml format

  keys taken from 1st header line

  Usage:

    ./cvs2yml [options] <input file name>

  Input: 

    - file name 

  Output:

    - yaml data block 
  

.. moduleauthor:: Guenter Quast <g.quast@kit.edu>

"""

from PhyPraKit.csv2yaml import csv2yaml
if __name__ == "__main__": # -------------------------------------------
  csv2yaml()  

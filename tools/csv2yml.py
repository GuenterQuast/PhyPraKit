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

from PhyPraKit.csv2yml import csv2yml

if __name__ == "__main__":  # -------------------------------------------
    csv2yml()

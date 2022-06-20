#!/usr/bin/env python3
"""csv2yml.py 
  uses csv2yaml to read floating-point column-data in very general .txt 
  formats and writes an output block in yaml format

.. moduleauthor:: Guenter Quast <g.quast@kit.edu>

"""
from PhyPraKit import csv2yaml

if __name__ == "__main__": # -------------------------------------------  
  import sys, PhyPraKit as ppk, yaml

  # check for / read command line arguments
  if len(sys.argv)==2:
    fnam = sys.argv[1]
  else:
    print("!!! no input file given - exiting")
    sys.exit(1)
  
  # read data from file in .txt format
  fnam = sys.argv[1]
  nlhead = 1
  delim = ','
  f = open(fnam, 'r') 

  hlines, ylines = csv2yaml(f, nlhead, delim)

  # print to screen
  for l in ylines:
    print(l)

  # write to file
  with open(fnam.split('.')[0] + '.yaml', 'w') as f:
   for l in ylines:
     print(l, file=f)


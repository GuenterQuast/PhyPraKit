# -*- coding: utf-8 -*-
'''cassylabxParser.py

   read files in xml-format produced with the Leybold Cassy system
    
   uses PhyPraPit.labxParser()

.. moduleauthor:: Guenter Quast <g.quast@kit.edu>

'''

# -----example Code illustrating usage --------------------
if __name__ == "__main__":
  import sys, numpy as np, matplotlib.pyplot as plt
  from PhyPraKit import odFit, labxParser
  from scipy import interpolate

# check for / read command line arguments
  if len(sys.argv)==2:
    fname = sys.argv[1]
  else:
    fname = "CassyExample.labx"
  print('*==* script ' + sys.argv[0]+ ' executing \n',\
      '     processing file ' + fname) 


  names, values = labxParser(fname, prlevel=0)

  for i, tag in enumerate(names):
     print(tag, len(values[i]))

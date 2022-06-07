"""test_readPicoSocpe.py 
   read data exported by PicoScope usb-oscilloscope

.. moduleauthor:: Guenter Quast <g.quast@kit.edu>

"""

# -----example Code illustrating usage --------------------
if __name__ == "__main__":
  import numpy as np, matplotlib.pyplot as plt
  from PhyPraKit import readPicoScope
  from scipy import interpolate
  from scipy import signal
  import sys
  
  # check for / read command line arguments
  if len(sys.argv)==2:
    fname = sys.argv[1]
  else:
    fname = "PicoScopeData.txt"
  print(('\n*==* script ' + sys.argv[0]+ ' executing \n',\
      '     processing file ' + fname)) 

  # read data from PicoScope
  units, data = readPicoScope(fname, prlevel=2)
  ic = len(data)
 
  t = data[0]    
# make a plot
  fig=plt.figure(1, figsize=(5.,5.))
  ax1=fig.add_subplot(1, 1, 1)
  for i in range(1,ic):
    ax1.plot(t, data[i])

  plt.show()

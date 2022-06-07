"""test_readtxt.py 
   uses readtxt() to read floating-point column-data in very general 
   .txt formats, here the output from PicoTech 8 channel data logger,
   with '\t' separated values, 2 header lines,
   german decimal comma and special character '^@'

.. moduleauthor:: Guenter Quast <g.quast@kit.edu>

"""

# -----example Code illustrating usage of readtxt() -------------
if __name__ == "__main__":
  import numpy as np, matplotlib.pyplot as plt, PhyPraKit as ppk
  from scipy import interpolate, signal
  import sys
  
  # check for / read command line arguments
  if len(sys.argv)==2:
    fname = sys.argv[1]
  else:
    fname = "Temperaturen.txt"  # output of PicoTech 8 channel data logger
                                # '\t' separated values, 2 header lines
                                # german decimal comma, special char '^@'
  print('\n*==* script ' + sys.argv[0]+ ' executing \n',\
      '     processing file ' + fname) 

  # read data from file in .txt format
  hlines, data = ppk.readtxt(fname, delim='\t', nlhead=2)
  print(hlines)
  print(" --> number of columns", data.shape[0])
  print(" --> number of data points", data.shape[1])

# make a plots
  fig=plt.figure(1, figsize=(10, 7.5))
  fig.suptitle('Script: test_readtxt.py', size='x-large', color='b')
  fig.subplots_adjust(left=0.14, bottom=0.1, right=0.97, top=0.93,
                    wspace=None, hspace=.25)#

  ax1=fig.add_subplot(1, 1, 1)
  t = data[0]    
  ax1.plot(t, data[1], alpha=0.5, label='Kanal1')
  ax1.plot(t, data[2], alpha=0.5, label='Kanal2')
  ax1.plot(t, data[3], alpha=0.5, label='Kanal3')
  ax1.plot(t, data[4], alpha=0.5, label='Kanal4')
  ax1.plot(t, data[5], alpha=0.5, label='Kanal5')
  ax1.plot(t, data[6], alpha=0.5, label='Kanal6')
  ax1.plot(t, data[7], alpha=0.5, label='Kanal7')

  ax1.set_ylabel('Spannung (mV)', size='large')
  ax1.set_xlabel('Zeit (s)', size='large')
  ax1.legend(loc='best', numpoints=1, prop={'size':10})
  ax1.grid()
 
  plt.show()

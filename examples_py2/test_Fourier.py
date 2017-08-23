'''test_Fourier.py 

   Einlesen von Daten aus mit PicoScope erstellten Dateien
     am Beispiel der akustischen Anregung eines Stabes

   Fouriertransformation des Signals
  
.. moduleauthor:: Guenter Quast <g.quast@kit.edu>

'''


# -----example Code illustrating usage --------------------
if __name__ == "__main__":
  import numpy as np, matplotlib.pyplot as plt, PhyPraKit as ppk
  from scipy import interpolate, signal
  import sys
  
  # check for / read command line arguments
  if len(sys.argv)==2:
    fname = sys.argv[1]
  else:
    fname = "Wellenform.csv"
  print '\n*==* script ' + sys.argv[0]+ ' executing \n',\
      '     processing file ' + fname 

  # read data from PicoScope
  units, data = ppk.readPicoScope(fname, prlevel=2)
  t = data[0]    
  a = data[1]

  print "** Fourier Spectrum"
  freq, amp = ppk.FourierSpectrum(t, a, fmax=20)
 # freq, amp = ppk.Fourier_fft(t, a)  # use fast algorithm
  frequency = freq[np.where(amp==max(amp))]
  print " --> Frequenz mit max. Amplitude: ", frequency

# make  plots
  fig=plt.figure(1, figsize=(7.5, 7.5))
  fig.suptitle('Script: test_Fourier.py', size='x-large', color='b')
  fig.subplots_adjust(left=0.14, bottom=0.1, right=0.97, top=0.93,
                    wspace=None, hspace=.25)#
  ax1=fig.add_subplot(2, 1, 1)
  ax1.plot(t, a)
  ax1.set_xlabel('$time$ '+units[0], size='large')
  ax1.set_ylabel('$Amplitude$ '+units[1], size='large')
  ax1.grid()

  ax2=fig.add_subplot(2,1,2)
  ax2.plot(freq, amp, 'b-')
  ax2.set_xlabel('$Frequenz$ $f$ (kHz)', size='large')
  ax2.set_ylabel('$Amplitude$', size='large')
  ax2.set_yscale('log')
  ax2.grid()

  plt.show()

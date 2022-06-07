#! /usr/bin/env python
"""test_AutoCorrelation.py
   test function `autocorrelate()` in PhyPraKit; 
   determines the frequency of a periodic signal from maxima and minima
   of the autocorrelation function and performs statistical analysis
   of time between peaks/dips

   uses `readCSV()`, `autocorrelate()`, `convolutionPeakfinder()` 
   and `histstat()` from PhyPraKit

.. moduleauthor:: Guenter Quast <g.quast@kit.edu>

"""

# ---------------------------------------------------------------
if __name__ == "__main__":
  import sys, numpy as np, matplotlib.pyplot as plt
  from PhyPraKit import readCSV, autocorrelate, convolutionPeakfinder, histstat

# read signal file
  # check for / read command line arguments
  if len(sys.argv)==2:
    fname = sys.argv[1]
  else:
    fname = "AudioData.csv"
  print('\n*==* script ' + sys.argv[0]+ ' executing \n',\
      '     processing file ' + fname) 

  hlines, data= readCSV(fname, nlhead=1)
  print(hlines)
  print(" --> number of columns", data.shape[0])
  print(" --> number of data points", data.shape[1])
  t = data[0]    
  a = data[1]

# calculate autocorrelation function
  ac_a = autocorrelate(a)
  ac_t = t  

# find maxima and minima using convolution peak finder
  width= 3
  pidxac =  convolutionPeakfinder(ac_a, width, th=0.66)
  didxac =  convolutionPeakfinder(-ac_a, width, th=0.66)
  if len(pidxac) > 1:
    print(" --> %i auto-correlation peaks found"%(len(pidxac)))
    pidxac[0]=0 # first peak is at 0 by construction
    ac_tp, ac_ap= np.array(ac_t[pidxac]), np.array(ac_a[pidxac])
    ac_td, ac_ad= np.array(ac_t[didxac]), np.array(ac_a[didxac])
  else:
    print("*!!* not enough correlation peaks found")

# make plots
# 1. signal
  fig = plt.figure(1, figsize=(7.5, 9.))
  fig.subplots_adjust(left=0.1, bottom=0.1, right=0.98, top=0.98,
                      wspace=0.3, hspace=0.5)
  ax1 = fig.add_subplot(3, 1, 1)
  ax1.plot(t, a)
  ax1.set_xlabel('$time$ (ms)', size='large')
  ax1.set_ylabel('$amplitude$ (a.u.)', size='large')
  ax1.grid()

# 2. auto-correlation 
  ax2=fig.add_subplot(3,1,2)
  ax2.plot(ac_tp, ac_ap, 'bx', alpha=0.9, label='peaks')
  ax2.plot(ac_td, ac_ad, 'gx', alpha=0.9, label='dips')
  ax2.plot([0.,ac_t[-1]],[1., 0.],'m--', alpha=0.3) # maximum auto-correlation
  ax2.plot(ac_t, ac_a, 'k-')
  ax2.set_xlabel('$time \, displacement$ (ms) ', size='large')
  ax2.set_ylabel('$autocorrelation$', size='large')
  ax2.legend(loc='best', numpoints=1, prop={'size':10})
#  ax2.set_yscale('log')
  ax2.grid()

# 3. analysis of auto-correlation function
  ax3 = fig.add_subplot(3, 1, 3)
  ac_dtp = ac_tp[1:] - ac_tp[:-1] 
  ac_dtd = ac_td[1:] - ac_td[:-1] 
  bins=np.linspace(min(min(ac_dtp),min(ac_dtd)), max(max(ac_dtp), max(ac_dtd)), 100)
  bc, be, _ = ax3.hist([ac_dtp, ac_dtd], bins, stacked = True, 
                         color=['b','g'], label=['peaks','dips'], alpha=0.5)
  ax3.set_xlabel(r'$time \, difference \, of \, maxima/minima$ (ms)', size='large')
  ax3.set_ylabel(r'$frequency$', size='large')
  ax3.legend(loc='best', numpoints=1, prop={'size':10})
  ax3.grid()
# analyis of histogram
  m_dtp, s_dtp, sm_dtp = histstat(bc[0], be, pr=False)
  m_dtd, s_dtd, sm_dtd = histstat(bc[1], be, pr=False)
# if uncertainty is zero, take it from bin width
  if (sm_dtp == 0.): sm_dtp= (be[0][1]-be[0][0])/np.sqrt(12.)
  if (sm_dtd == 0.): sm_dtd= (be[1][1]-be[1][0])/np.sqrt(12.)

  print(" --> Time difference Auto-Correlation: (%.5g +/- %.2g) ms"%(m_dtp, sm_dtp)) 
  print(" -->                                   (%.5g +/- %.2g) ms"%(m_dtd, sm_dtd)) 
  ax3.text(0.55, 0.85,  "peaks: (%.5g$\pm$%.2g) ms"%(m_dtp, sm_dtp), 
     transform=ax3.transAxes )
  ax3.text(0.55, 0.75,  "dips: (%.5g$\pm$%.2g) ms"%(m_dtd, sm_dtd), 
     transform=ax3.transAxes )

  plt.show()

# -*- coding: utf-8 -*-
from __future__ import print_function  # for python2.7 compatibility
'''Beispiel_Drehpendel.py 

   Auswertung der Daten aus einer im CASSY labx-Format gespeicherten Datei
   am Beispiel des Drehpendels

   * Einlesen der Daten im .labx-Format

   * Säubern der Daten durch verschiedene Filterfunktionen:
     - offset-Korrektur
     - Glättung durch gleitenden Mittelwert
     - Zusammenfassung benachberter Daten durch Mittelung
  
   * Fourier-Transformation (einfach und fft) 

   * Suche nach Extrema (`peaks` und `dips`)

   * Anpassung von Funkionen an Einhüllende der Maxima und Minima

   * Interpolation durch Spline-Funktionen 

   * numerische Ableitung und Ableitung der Splines

   * Phasenraum-Darstellung (aufgezeichnete Wellenfunktion
     gegen deren Ableitung nach der Zeit)
     
.. moduleauthor:: Guenter Quast <g.quast@kit.edu>

'''

# last changed: Nov. 16
#  GQ 16-Nov-16: added more fit funtions for envelope
#                peakfinder tuned to new version in PhyPraKit
# -----------------------------------------------------------
import numpy as np
import PhyPraKit as ppk

# -----example Code illustrating usage --------------------
if __name__ == "__main__":
  import sys, numpy as np, matplotlib.pyplot as plt
  from PhyPraKit import odFit, labxParser
  from scipy import interpolate
  from scipy import signal

#  filename="CassyExample.labx"
  filename="Drehpendel.labx"
  names, values = labxParser(filename, prlevel=0)

# collect data we are interested in:
  print("\n *==* Data received:")
  for i, tag in enumerate(names):
    print(tag.split(':'))
    tnam=tag.split(':')[1]
    if tnam=='Zeit':   t = np.array(values[i][:])
    if tnam=='Weg':    s = np.array(values[i][:])
    if tnam=='Winkel': phi = np.array(values[i][:])
  
 # print "time=", len(t), t
 # print "phi_count=", len(s), s
 # print "phi=", len(phi), phi

# ----  some examples of data analysis -----

# some filtering and smoothing
  print("** filtering:")
  # * use, if there is an unwanted offset
  print("  offset correction")
  phi_c = ppk.offsetFilter(phi)          

  # * use , if there is high-frequency noise 
  #print "  smoothing with sliding average"
  #phi=ppk.meanFilter(phi_c, width=10) #    

  # * use, if sampling rate is too high  
  if len(phi)>2000: 
    print("  resampling")
    phi, t = ppk.resample(phi, t, n=int(len(phi)/1000)) # average n samples into 1

# numerical differentiation with numpy 
  print("** numerical derivative")
  dt = t[1]-t[0]
  omega = np.gradient(phi, dt)

# calculate fourier spectrum 
  print("** Fourier Spectrum")
  freq, amp = ppk.FourierSpectrum(t, phi, fmax=1.)
#  freq, amp = ppk.Fourier_fft(t, phi)  # fast algorithm
  frequency = freq[np.where(amp==max(amp))]
  print(" --> Frequenz: ",frequency)

# run a peak finder
  # first, determine width of typical peaks and dips 
  width=  0.5* len(t)  / (t[-1] - t[0]) / frequency
#  use convoluted template filter 
  peakind = ppk.convolutionPeakfinder(phi, width, th=0.53)
  dipind  = ppk.convolutionPeakfinder(-phi, width, th=0.53)
  if len(peakind) > 5:
    print(" --> %i peaks and %i dips found"%(len(peakind), len(dipind)))
    tp, phip = np.array(t[peakind]), np.array(phi[peakind])
    td, phid =np.array(t[dipind]), np.array(phi[dipind])
  else:
    print("*!!* not enough peaks found for envelope fit")
    print("     tune peakfinder parameters!")
    sys.exit(1)

# cubic spline interpolation with scipy.interpolate
  print("** spline interpolation")
  cs_phi=interpolate.UnivariateSpline(t, phi, s=0)
  cs_omega=cs_phi.derivative()  

# define functional forms for envelope curves ...
  def env_exp_p(t, A=1., tau=75.):
    return A * np.exp(-t/tau) 

  def env_exp_d(t, A=2., tau=75.):
    return -A * np.exp(-t/tau) 

  def env_quad_p(t, a=1., b=1., c=1.):
    return a*t**2+b*t+c

  def env_quad_d(t, a=-1., b=1., c=1.):
    return a*t**2+b*t+c

  def env_linexp_p(t, A=2., tau=75., a=0.001):
    return  A*(np.exp(-t/tau) - a*t) 

  def env_linexp_d(t, A=2., tau=75., a=0.001):
    return  -A*(np.exp(-t/tau) - a*t)

# select functions for envelope ...
#  !!! Achtung: nichlineare Anpassungen benötigen Startwerte,
#           also Schätzungen der Parameterwerte
  envp=env_linexp_p     # exp + linear
  p0p=[2., 100., 0.001] # initial parameters 
  envd=env_linexp_d
  p0d=[-2., 100., 0.001]  

# ... and fit parameters 
  print("** envelope fit")
  parp, parpe, corp, chi2p =\
    odFit(envp, tp, phip, 0., 0.1, p0=p0p)
  print("** fit of positive envelope, chi2/df= %.2g"%(chi2p/(len(tp)-len(parp))))
  np.set_printoptions(precision=3)
  print(" --> parameters:   ", parp)
  np.set_printoptions(precision=2)
  print(" --> uncertainties:", parpe) 
  print(" --> correlations:\n", corp)

  pard, parde, cord, chi2d =\
    odFit(envd, td, phid, 0., 0.03, p0=p0d)
  print("fit of negative envelope, chi2/df= %.2g"%(chi2d/(len(td)-len(pard))))
  np.set_printoptions(precision=3)
  print(" -> parameters:   ", pard)
  np.set_printoptions(precision=2)
  print(" -> uncertainties:", parde) 
  print(" -> correlations:\n", cord)

# plot data and analysis results
  fig=plt.figure("Amplitude", figsize=(7.5, 10.))
  fig.suptitle('Script: Beispiel_Drehpendel.py', size='x-large', color='b')
  fig.subplots_adjust(left=0.1, bottom=0.1, right=0.97, top=0.93,
                    wspace=None, hspace=.25)#

  ax1=fig.add_subplot(3,1,1)
  ax1.plot(t, phi)
  ax1.plot(tp, phip, 'rx', alpha=0.5, label='peak')
  ax1.plot(td, phid, 'gx', alpha=0.5, label='dip')
  x=np.linspace(0., max(tp), 100)
  ax1.plot(x, envp(x, *parp), 
    'r-', linewidth=1, label=u'positive Einhüllende')
  ax1.plot(x, envd(x, *pard), 
    'g-', linewidth=1, label=u'negative Einhüllende')
  ax1.set_xlabel('$Zeit$ $t$ (s)', size='large')
  ax1.set_ylabel('$Winkel$  $\phi$', size='large')
  ax1.legend(loc='best', numpoints=1, prop={'size':10})
  ax1.grid()

  ax2=fig.add_subplot(3,1,2)
  ax2.plot(t , omega)
  ax2.set_xlabel('$Zeit$ $t$ (s)', size='large')
  ax2.set_ylabel('$Winkelgeschwindigkeit$  $\omega$ (1/s)', size='large')

  ax3=fig.add_subplot(3,1,3)
  ax3.plot(freq, amp, 'b.')
  ax3.set_xlabel('$Frequenz$ $f$ (Hz)', size='large')
  ax3.set_ylabel('$Amplitude$', size='large')
  ax3.set_yscale('log')
  ax3.grid()

# Phasenraumdiagramme mit interpolierten Daten
  fig2=plt.figure("PhaseSpace", figsize=(10, 5.))
  fig2.subplots_adjust(left=0.1, bottom=0.1, right=0.97, top=0.93,
                    wspace=.25, hspace=None)
  axa=fig2.add_subplot(1,2,1)
  tplt=np.linspace(t[0], t[-1], 50000)
  axa.plot(cs_phi(tplt), cs_omega(tplt), 'b-', alpha=0.5, linewidth=1)
  axa.set_xlabel('$\phi$', size='large')
  axa.set_ylabel('$\omega$ (1/s)', size='large')
  axa.set_title("Phasenraum-Diagramm")

  axb=fig2.add_subplot(1,2,2)
  tplt=np.linspace(t[int(0.75*len(t))], t[-1], 20000)
  axb.plot(cs_phi(tplt), cs_omega(tplt), 'b-', alpha=0.5, linewidth=1)
  axb.set_xlabel('$\phi$', size='large')
  axb.set_ylabel('$\omega$ (1/s)', size='large')
  axb.set_title("Phasenraum-Diagramm (Ausklingphase)")

  plt.show()

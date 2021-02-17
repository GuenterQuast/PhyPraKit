#! /usr/bin/env python3
"""Beispeil_Diodenkennlinie.py
   Messung einer Strom-Spannungskennlinie und Anpassung der Schockley-Gleichung. 

   - Konstruktion der Kovarianzmatrix für ein reales Messinstrument mit Signalrauschen,
     Anzeigeunsicherheiten und korrelierten, realtiven Kalibratsionsunsicherheiten
     für die Strom- und Spannungsmessung

   - Generierung von (simulierten) Daten gemäß Shockley-Gleichung

   - Ausführen der Anpassung mit *mFit* aus dem Paket *PhyPraKit.iminuitFit*
   
.. moduleauthor:: Guenter Quast <g.quast@kit.edu>
"""

import numpy as np, matplotlib.pyplot as plt
from PhyPraKit import generateXYdata
from PhyPraKit.iminuitFit import mFit

# define the model function to fit
def Shockley(U, I_s = 0.5, U0 = 0.03):
  """Parametrisierung einer Diodenkennlinie

  Args:
    - U: Spannung (V)
    - I_s: Sperrstrom (nA)
    - U0: Temperaturspannung (V) * Emissionskoeffizient
    
  Returns:
    - float I: Strom (mA)
  """
  return 1e-6 * I_s * np.exp( (U/U0) - 1.)


if __name__ == "__main__": # --------------------------------------  

#
# Anpassung einer Diodenkennlinie (Beispiel für PhyPraKit.mFit())
#

# --- settings for general fit
#
# Schockleygleichung als Fitfunktion
  model = Shockley 

# Zahl der Messungen
  nd = 20 
  
# Komponenten der Messunsicherheit
# - Genauigkeit Spannungsmessung: 4000 Counts, +/-(1.0% + 4 digits)
#    - Messbereich 2V
  crel_U = 0.010
  Udigits = 4
  Urange = 2
  Ucounts = 4000
# - Genauigkeit Strommessung: 2000 Counts, +/-(1.5% + 3 digits) 
#    - Messbereiche 200µA, 20mA und 200mA 
  crel_I = 0.015
  Idigits = 3
  Icounts = 2000
  Irange1 = 0.2
  Irange2 = 20
  Irange3 = 200
# - Rauschanteil (aus Fluktuationen der letzen Stelle)
#   - delta U = 0.005 V
  deltaU = 0.005
#   - delta I = 0.025 mA
  deltaI = 0.025

# Anzeigegenauigkeit der Spannung (V)  
  sx = Udigits * Urange / Ucounts
  sabsx = np.sqrt(deltaU**2 + sx**2)
# korrelierte Kalibrationsunsicherheit  
  crelx = crel_U
  
# Anzeigegenauigkeit des Stroms (mA), 3 Messbereiche
  sy = np.asarray(   nd//4 * [Idigits * Irange1 / Icounts] + \
                      2*nd//4 * [Idigits * Irange2 / Icounts] + \
                        nd//4 * [Idigits * Irange3 / Icounts])  
  sabsy = np.sqrt(deltaI**2 + sy**2)
# korrelierte Kalibrationsunsicherheit  
  crely = crel_I 
  
# generate pseudo-data
# - initialize random generator
  np.random.seed(31415)  # initialize random generator
# - set range and x-data
  xmin =   0.4
  xmax1 =  0.55
  xmax2 =  0.67
  data_x = np.concatenate( (np.linspace(xmin, xmax1, nd//4, endpoint=False),
                  np.linspace(xmax1, xmax2, 3*nd//4)) ) # x of data points
# -  set true model values  
  mpardict = { 'I_s' : 0.25, 'U0' : 0.033 }
# - generate the data    
  xt, yt, data_y = generateXYdata(data_x, model, sabsx, sabsy,
                                      xabscor = 0.,
                                      xrelcor = crelx,
                                      yabscor = 0.,
                                      yrelcor = crely,
                                      mpar=mpardict.values() )

# perform fit to data with function mFit from package iminuitFit
  parvals, parerrs, cor, chi2 = mFit(model,
      data_x, data_y,      # data x and y coordinates
      sx=sabsx,            # indep x
      sy=sabsy,            # indel y
      xrelcor=crelx,       # correlated rel. x
      yrelcor=crely,       # correlated rel. y
##      srelx=srelx,         # indep. rel. x
##      srely=srely,         # indep. rel. y
##      xabscor=cabsx,       # correlated x
##      yabscor=cabsy,       # correlated y
      ref_to_model=True,   # reference of rel. uncert. to model
      p0=(1.e-10, 0.05),   # initial guess for parameter values 
#      constraints=['I_s', 0.1, 0.03], # constraints within errors
      use_negLogL=True,    # full -2log(L) if parameter dep. uncertainties
      plot=True,           # plot data and model
      plot_band=True,      # plot model confidence-band
      plot_cor=False,      # plot profiles likelihood and contours
      showplots = False,   # plt.show() in user code                       
      quiet=False,         # suppress informative printout
      axis_labels=['U (V)', '$I_D$ (mA)   \  Shockley-Gl.'], 
      data_legend = 'Generierte Daten',    
      model_legend = 'Shockley-Gleichung'
  )

# Print results to illustrate how to use output
  print('\n*==* Fit Result:')
  print(" chi2: {:.3g}".format(chi2))
  print(" parameter values:      ", parvals)
  print(" neg. parameter errors: ", parerrs[:,0])
  print(" pos. parameter errors: ", parerrs[:,1])
  print(" correlations : \n", cor)

# set final options for plots and show them on screen  
  plt.figure(num=1)    # activate first figure ...
  plt.ylim(-1., 65.)  # and set y-limit
#  plt.ylim(1.e-4, 150.)  # and set y-limit
#  plt.yscale('log')    # try log
  # plt.xlim(0.05, 0.68) # and set y-limit
  plt.show()           # show all figures

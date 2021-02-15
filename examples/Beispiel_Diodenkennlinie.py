#! /usr/bin/env python3
"""Kennlinie.py
   Messung einer Strom-Spannungskennlinie
   und Anpassung der Schockley-Gleichung. 

   - Konstruktion der Kovarianzmatrix für ein reales Messinstrument
   - Generierung der (simulierten) Daten
   - Ausführen der Anpassung mit *mFit* aus dem Paket *iminuitFit*
   
.. moduleauthor:: Guenter Quast <g.quast@kit.edu>
"""

import numpy as np, matplotlib.pyplot as plt
from PhyPraKit import generateXYdata
from PhyPraKit.iminuitFit import mFit


# --- Helper functions
import inspect
def get_functionSignature(f):
  # get arguments and keyword arguments passed to a function
  pars = inspect.signature(f).parameters
  args = []
  kwargs = {}
  for p in pars.values():
    if p.default is p.empty:
      args.append(p.name)
    else:
      kwargs[p.name]=p.default
  return args, kwargs

# --- end helper functions ----

if __name__ == "__main__": # --------------------------------------  
  #
  # Example of an application of PhyPraKit.mFit()
  #
  # define the model function to fit

  def Shockley(U, I_s = 0.15, U0 = 0.031):
    """Diodenkennlinie

    Args:
 
    - U: Spannung (V)
    - I_s: Sperrstrom (nA)
    - U0: Temperaturspannung (V) * Emissionskoeffizient
    
    Returns:

    - float I: Strom (mA)
    """

    return 1e-6 * I_s * np.exp( (U/U0) - 1.)   

# --- settings for general fit
#
# Schockleygleichung als Fitfunktion
  model = Shockley 
# Zahl der Messungen
  nd = 20 
  
# Komponenten der Messunsicherheit
# - Genauigkeit Spannungsmessung: 2000 Counts, +/-(1,5% + 3 digits)
# - Genauigkeit Strommessung: 2000 Counts, +/-(2% + 4 digits) 

# Anzeigegenauigkeit der Spannung (V),  Messbereich 2V DC
  sabsx = 3 * 2. / 2000

# Anzeigegenauigkeit des Stroms (mA), Messbereiche 200µA, 20mA und 200mA 
  sabsy = np.asarray(   nd//4 * [4*0.2/2000] + \
                      2*nd//4 * [4*20./2000] +  
                        nd//4 * [4*200/2000])  

# Kalibartionsunsicherheit der Geräte
  crely = 0.015  # korrelierte, relative Messunsicherheit
  crelx = 0.020
  
# keine unabhängigen, relativen Unsicherheiten
  srely = 0.          
  srelx = 0.  
  
# generate pseudo data
# - initialize random generator
  np.random.seed(3141)  # initialize random generator
# - set range and x-data
  xmin =  -0.1
  xmax1 =  0.5
  xmax2 =  0.675
  data_x = np.concatenate( (np.linspace(xmin, xmax1, nd//4, endpoint=False),
                  np.linspace(xmax1, xmax2, 3*nd//4)) ) # x of data points
# - extract model parameters and defaults
  mpardict = get_functionSignature(model)[1] # keyword arguments of model
# - calculate independent uncertainty from absolute and relative components 
  sigy = np.sqrt(sabsy * sabsy + (srely*model(data_x, **mpardict))**2)
  sigx = np.sqrt(sabsx * sabsx + (srelx * data_x)**2)
# - generate the data   
  xt, yt, data_y = generateXYdata(data_x, model, sigx, sigy,
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
      plot_cor=True,       # plot profiles likelihood and contours
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
  plt.figure(num=1)   # activate first figure ...
  plt.ylim(-1., 180.) # and set y-limit
  plt.show()          # show all figures

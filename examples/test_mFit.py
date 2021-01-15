#! /usr/bin/env python3
"""test_mFit.py
   Fitting example with iminiut

   Uses function PhyPraKit.mFit, which in turn uses iminuitFit

   This is a rahter complete example showing 
   independente and correlated,
   absolute and relative uncertainties 
   in x and y direction. 
   
.. moduleauthor:: Guenter Quast <g.quast@kit.edu>

"""

import numpy as np, matplotlib.pyplot as plt 
from PhyPraKit import generateXYdata, mFit

# ------------------------ end of iminuitFit ----------------------
      
if __name__ == "__main__": # --------------------------------------  
  #
  # Example of an application of iminuitFit.mFit()
  #
  # define the model function to fit
  def model(x, A=1., x0=1.):
    return A*np.exp(-x/x0)
  mpardict = {'A':1., 'x0':0.5}  # model parameters

# set error components 
  sabsy = 0.07
  srely = 0.05 # 5% of model value
  cabsy = 0.04
  crely = 0.03 # 3% of model value
  sabsx = 0.05
  srelx = 0.04 # 4%
  cabsx = 0.03
  crelx = 0.02 # 2%

# generate pseudo data
  np.random.seed(314)      # initialize random generator
  nd=10
  data_x = np.linspace(0, 1, nd)       # x of data points
  sigy = np.sqrt(sabsy * sabsy + (srely*model(data_x, **mpardict))**2)
  sigx = np.sqrt(sabsx * sabsx + (srelx * data_x)**2)
  xt, yt, data_y = generateXYdata(data_x, model, sigx, sigy,
                                      xabscor=cabsx,
                                      xrelcor=crelx,
                                      yabscor=cabsy,
                                      yrelcor=crely,
                                      mpar=mpardict.values() )

# perform fit to data with function mFit using iminuitFit class
  parvals, parerrs, cor, chi2 = mFit(model, data_x, data_y,
                                     sx=sabsx,
                                     sy=sabsy,
                                     srelx=srelx,
                                     srely=srely,
                                     xabscor=cabsx,
                                     xrelcor=crelx,
                                     yabscor=cabsy,
                                     yrelcor=crely,
                                     p0=(1., 0.5),
#                                     constraints=['A', 1., 0.03],
#                                     constraints=[0, 1., 0.03],
                                     plot=True,
                                     plot_band=True,
                                     plot_cor=True,
                                     quiet=False,
                                     axis_labels=['x', 'y   \  f(x, *par)'], 
                                     data_legend = 'random data',    
                                     model_legend = 'exponential model')

# Print results to illustrate how to use output
  print('\n*==* Fit Result:')
  print(f" chi2: {chi2:.3g}")
  print(f" parameter values:      ", parvals)
  print(f" neg. parameter errors: ", parerrs[:,0])
  print(f" pos. parameter errors: ", parerrs[:,1])
  print(f" correlations : \n", cor)
  
  plt.show()

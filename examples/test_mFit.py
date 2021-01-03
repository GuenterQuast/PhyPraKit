#! /usr/bin/env python3
'''test_iminuit.py
   Fitting with iminiut

   This example illustrates the special features of iminuit:
    - definition of a custom cost function 
         used to implement least squares method with correlated errors   
    - profile likelihood for asymmetric errors
    - plotting of profile likeliood and confidence contours

    supports iminuit vers. < 2.0 and >= 2.0

.. moduleauthor:: Guenter Quast <g.quast@kit.edu>
'''

from PhyPraKit import generateXYdata, mFit
import numpy as np, matplotlib.pyplot as plt

if __name__ == "__main__": # --------------------------------------  
  #
  # Example of an application
  # 
  # the model function to fit
  def model(x, A=1., x0=1.):
    return A*np.exp(-x/x0)
  mpardict = {'A':1., 'x0':0.5}  # model parameters
    
# generate pseudo data
  np.random.seed(314159)      # initialize random generator
  nd=10
  data_x = np.linspace(0, 1, nd)       # x of data points
  sigy_abs = np.sqrt(0.05*model(data_x, **mpardict) + 0.07*0.07) 
  xt, yt, data_y = generateXYdata(data_x, model, 0., sigy_abs,
                                      mpar=mpardict.values() )

# perform fit to data with iminuit
  parvals, parerrs, cor, chi2 = mFit(model, data_x, data_y, sigy_abs,
                                     p0=(2., 1.),
                                     run_minos=True, plot=True, plot_cor=True)

# Print results to illustrate how to use output
  print('\n*==* Fit Result:')
  print(f" chi2: {chi2:.3g}")
  print(f" parameter values:      ", parvals)
  print(f" neg. parameter errors: ", parerrs[:,0])
  print(f" pos. parameter errors: ", parerrs[:,1])
  print(f" correlations : \n", cor)
  
  plt.show()

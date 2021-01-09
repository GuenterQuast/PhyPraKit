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
  # Example of an application of mFit()
  #  supprting 8 different error categories:
  #  - indpendent absolulte and relative uncertainties on x and y
  #  - correlated absolulte and relative uncertainties on x and y
  # relative uncertainties are calculated with reference to model
  #  in an iterated fit
  # x-uncertainties are projected on y in an iterative fit
  # covariance matrix is updated dynamically during fitting
  
  # define the model function to fit
  def model(x, A=1., x0=1.):
    return A*np.exp(-x/x0)
  mpardict = {'A':1., 'x0':0.5}  # model parameters

# set error components for x and y data
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
  sigy = np.sqrt(sabsy**2+ (srely*model(data_x, **mpardict))**2)
  sigx = np.sqrt(sabsx**2 + (srelx * data_x)**2)
  xt, yt, data_y = generateXYdata(data_x, model, sigx, sigy,
                                      xabscor=cabsx,
                                      xrelcor=crelx,
                                      yabscor=cabsy,
                                      yrelcor=crely,
                                      mpar=mpardict.values() )

# perform fit to data with mFit based in imiunit
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
                                  #   constraints=['A', 1., 0.03],
                                  #   constraints=[0, 1., 0.03],
                                     plot=True, plot_cor=False)

# Print results to illustrate how to use output
  print('\n*==* Fit Result:')
  print(f" chi2: {chi2:.3g}")
  print(f" parameter values:      ", parvals)
  print(f" neg. parameter errors: ", parerrs[:,0])
  print(f" pos. parameter errors: ", parerrs[:,1])
  print(f" correlations : \n", cor)

# show figures   
  plt.show()

  

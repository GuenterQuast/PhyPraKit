#! /usr/bin/env python3
"""test_xyFit.py
   Fitting example for x-y data with iminiut

   Uses function PhyPraKit.xmFit, which in turn uses mnFit from phyFit

   This is a rather complete example showing a fit to
   data with independent and correlated, absolute and 
   relative uncertainties in the x and y directions. 
   
.. moduleauthor:: Guenter Quast <g.quast@kit.edu>

"""

import numpy as np, matplotlib.pyplot as plt
from PhyPraKit import generateXYdata, xyFit

if __name__ == "__main__": # --------------------------------------  
  #
  # Example of an application of PhyPraKit.xyFit()
  #
  # define the model function to fit
  def model(x, A=1., x0=1.):
    return A*np.exp(-x/x0)
  mpardict = {'A':1., 'x0':1.}  # model parameters

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
  np.random.seed(314159)  # initialize random generator
  nd=14
  xmin = 0.
  xmax = 2.6
  data_x = np.linspace(xmin, xmax, nd)       # x of data points
  sigy = np.sqrt(sabsy * sabsy + (srely*model(data_x, **mpardict))**2)
  sigx = np.sqrt(sabsx * sabsx + (srelx * data_x)**2)
  xt, yt, data_y = generateXYdata(data_x, model, sigx, sigy,
                                      xabscor=cabsx,
                                      xrelcor=crelx,
                                      yabscor=cabsy,
                                      yrelcor=crely,
                                      mpar=mpardict.values() )

# perform fit to data with function xyFit using mnFit class from phyFit
  rdict = xyFit(model,
      data_x, data_y,      # data x and y coordinates
      sx=sabsx,            # indep x
      sy=sabsy,            # indel y
      srelx=srelx,         # indep. rel. x
      srely=srely,         # indep. rel. y
      xabscor=cabsx,       # correlated x
      xrelcor=crelx,       # correlated rel. x
      yabscor=cabsy,       # correlated y
      yrelcor=crely,       # correlated rel. y
      ref_to_model=True,   # reference of rel. uncert. to model
      p0=(1., 0.5),        # initial guess for parameter values 
#       constraints=['A', 1., 0.03], # constraints within errors
      use_negLogL=True,    # full -2log(L) if parameter dep. uncertainties
      plot=True,           # plot data and model
      plot_band=True,      # plot model confidence-band
      plot_cor=True,       # plot profiles likelihood and contours
      quiet=False,         # suppress informative printout
      axis_labels=['x', 'y   \  f(x, *par)'], 
      data_legend = 'random data',    
      model_legend = 'exponential model' )

# print results to illustrate how to use output
  print('\n*==* Fit Result:')

  pvals, perrs, cor, chi2, pnams= rdict.values()
  print(" chi2: {:.3g}".format(chi2))
  print(" parameter names:       ", pnams)
  print(" parameter values:      ", pvals)
  np.set_printoptions(precision=3)
  print(" neg. parameter errors: ", perrs[:,0])
  print(" pos. parameter errors: ", perrs[:,1])
  print(" correlation matrix : \n", cor)

#- alternative: print dictionary
#  for key in rdict:
#    print("{}\n".format(key), rdict[key])

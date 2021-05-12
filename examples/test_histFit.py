#! /usr/bin/env python3
"""test_histFit.py
   histogram fit  with iminiut

.. moduleauthor:: Guenter Quast <g.quast@kit.edu>

"""

import numpy as np, matplotlib.pyplot as plt
from PhyPraKit.phyFit import mnFit

if __name__ == "__main__": # --------------------------------------  
  #
  # Example of a histogram fit
  #

  # define the model function to fit
  def model(x, A=1., mu = 3., sigma = 2., s = 0.5):
    '''pdf of a Gaussian signal on top of flat background
    '''
    normal = np.exp(-0.5*((x-mu)/sigma)**2)/np.sqrt(2.*np.pi*sigma**2)
    flat = 1./(max-min) 
    return A*(s * normal + (1-s) * flat)

  #
  # generate Histogram Data
  #
  # parameters of data sample, signal and background parameters
  N = 200  # number of entries
  min=0.   # range of data, mimimum
  max= 10. # maximum
  s = 0.8  # signal fraction 
  pos = 6.66 # signal position
  width = 0.33 # signal width
  
  # fix random generator seed 
  #!np.random.seed(314159)  # initialize random generator

  def generate_data(N, min, max, p, w, s):
    '''generate a random dataset: 
       Gaussian signal at position p with width w and signal fraction s 
       on top of a flat background between min and max
     '''
     # signal sample
    data_s = np.random.normal(loc=pos, scale=width, size=int(s*N) )
     # background sample
    data_b = np.random.uniform(low=min, high=max, size=int((1-s)*N) )
    return np.concatenate( (data_s, data_b) )

  # generate a histogram data sample ...
  SplusB_data = generate_data(N, min, max, pos, width, s)  

# ... and create the histogram
  bc, be = np.histogram(SplusB_data, bins=40)
  
  Fit = mnFit("hist")
  Fit.init_hData(bc, be)
  Fit.init_hFit(model)
  # perform fit to histogram data
  fitResult = Fit.do_hFit()
  print(fitResult[0])
  print(fitResult[1])

  # plot graph
  Fit.plotModel(data_legend="Binned data", model_legend="Signal + Background Fit")
  
  plt.show()


# Print results to illustrate how to use output
#  print('\n*==* Fit Result:')
#  print(" chi2: {:.3g}".format(chi2))
#  print(" parameter values:      ", parvals)
#  print(" neg. parameter errors: ", parerrs[:,0])
#  print(" pos. parameter errors: ", parerrs[:,1])
#  print(" correlations : \n", cor)
  

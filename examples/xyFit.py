#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# script xyFit.py

"""Perform fit with data and model from yaml file 

    Uses function PhyPraKit.phyfit.xyFit, a wrapper for phyFit.mnFit

    This code performs fits to x-y data with 
    independent and correlated, absolute and relative 
    uncertainties in the x and y directions. 

    usage:
      ./xyFit.py <file>

   Input:
     - file in yaml format
     - output options
"""


from PhyPraKit.phyFit import xyFit_from_file                
plot=True           # plot data and model
plot_band=True      # plot model confidence-band
plot_cor=False      # plot profiles likelihood and contours
quiet=True          # suppress informative printout

if __name__ == "__main__": # --------------------------------------  
  #
  # xyFit.py: Example of an application of PhyPraKit.phyFit.xyFit_from_file()
  #

  # package imports
  import sys, yaml, numpy as np
  
  # load file
  # check for / read command line arguments
  if len(sys.argv) == 2:
    filen = sys.argv[1]
  else:
    print("!!! no file name given")
    sys.exit(1)

  # open and read input yaml file    
  try:
    with open(filen) as f:
      fd = yaml.load(f, Loader=yaml.Loader)
  except (OSError, yam.YAMLError) as exception:
    print('!!! failed to read configuration file ' + filen)
    print(str(exception))
    sys.exit(1)
      
  # another check
  if len(fd.keys()) == 0:
    print("!!! data file is empty!")
    sys.exit(1)

  print("*==*", sys.argv[0], "received valid yaml data for fit:")
  # pprint.pprint(fd)

  rdict = xyFit_from_file(fd,                 # the input dictionary defining the fit 
                 plot=plot,           # plot data and model
                 plot_band=plot_band, # plot model confidence-band
                 plot_cor=plot_cor,   # plot profiles likelihood and contours
                 quiet=quiet          # suppress informative printout
                 ) 

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

  #- alternative: print result dictionary
  #  for key in rdict:
  #    print("{}\n".format(key), rdict[key])

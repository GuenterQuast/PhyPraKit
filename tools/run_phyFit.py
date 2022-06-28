#!/usr/bin/env python3
"""**run_phyFit.py** [options] <input file name>

   Perform fit with data and model from yaml file 

   Uses functions xyFit and hFit from PhyPraKit.phyFit

   This code performs fits 

      - to x-y data with independent and correlated, absolute 
        and relative uncertainties in the x and y directions 
 
      - and to histogram data with a binned likelihood fit.

   usage:

      ./run_phyFit.py [options] <input file name>

      ./run_phyFit.py --help for help


   Input:

      - input file in yaml format
      - output depending on options

"""

from PhyPraKit.phyFit import xyFit_from_yaml, hFit_from_yaml
from pprint import pprint

if __name__ == "__main__": # --------------------------------------  
  #
  # xyFit.py: Example of an application of PhyPraKit.phyFit.xyFit_from_yaml()
  #

  # package imports
  import sys, argparse, yaml, numpy as np, matplotlib.pyplot as plt

  # - - - Parse command-line arguments
  parser = argparse.ArgumentParser(description = \
    "Perform a fit with PhyPraKit.phyFit package driven by input file")
  parser = argparse.ArgumentParser(usage=__doc__)

  parser.add_argument('filename', type=str,
      help="name of fit input file in yaml format")
  parser.add_argument('-v', '--verbose', 
      action='store_const', const=True, default=False,
      help="suppress ouput of plots on screen")
  parser.add_argument('-r', '--result_to_file', 
      action='store_const', const=True, default=False,
      help="store results to file")
  parser.add_argument('-n', '--noplot', 
      action='store_const', const=True, default=False,
      help="suppress ouput of plots on screen")
  parser.add_argument('-s', '--saveplot', 
      action='store_const', const=True, default=False,
      help="save plot(s) in file(s)")
  parser.add_argument('-c', '--contour', 
      action='store_const', const=True, default=False,
      help="plot contours and profiles")
  parser.add_argument('--noband', 
      action='store_const', const=True, default=False,
      help="suppress 1-sigma band around function")
  parser.add_argument('-f','--format', 
      type=str, default='pdf',
      help="graphics output format, default=pdf")

  if len(sys.argv)==1:  # print help message if no input given
    parser.print_help()
    sys.exit(1)

  # collect input from ArgumentParser
  args = parser.parse_args()
  fname=args.filename
  quiet_flg = not args.verbose
  store_result = args.result_to_file
  plt_flg= not args.noplot
  sav_flg=args.saveplot
  cont_flg=args.contour
  band_flg=not args.noband
  pltfmt=args.format

  #  - - - End: Parse command-line arguments
    
  # open and read input yaml file
  f = open(fname, 'r')
  try:
    ymldata = yaml.load_all(f, Loader=yaml.Loader)
  except (OSError, yaml.YAMLError) as exception:
    print('!!! failed to read configuration file ' + fname)
    print(str(exception))
    sys.exit(1)
      
  fitType = 'xy'
  ddata = []
  for d in ymldata:
    if 'type' in d:
      fit_type = d['type']
    ddata.append(d)
  f.close()

  # select appropriate wrapper
  if fitType == 'xy':
    fit = xyFit_from_yaml
  elif fitType == 'histogram':
    fit = hFit_from_yaml
  else:
    print('!!! unsupported type of fit:', fitType)
    sys.exit(1)

  for fd in ddata:  
    if 'type' in fd.keys():
      fitType = fd['type']
    print("*==*", sys.argv[0], "received valid yaml data for fit:")
    if 'parametric_model' in fd: # for complex kafe2go format
      pprint(fd, compact=True)
    else:  # "nice" printout for simple xyFit format
      print(' **  Type of Fit:', fitType)
      for key in fd:
        if type(fd[key]) is not type([]):     # got a scalar or string
          print(key + ': ', fd[key])
        elif type(fd[key][0]) is not type({}): # got list of scalars
              print(key + ': ', fd[key])
        else:  # got list of uncertainty dictionaries
          print(key+':')
          for d in fd[key]:
            for k in d:
              print('  '+ k +': ', d[k], end=' ') 
            print()
  # run fit    
    rdict = fit(fd,                     # the input dictionary defining the fit 
              plot=plt_flg,           # show plot of data and model
              plot_band=band_flg,     # plot model confidence-band
              plot_cor=cont_flg,      # plot profiles likelihood and contours
              showplots= False,        # show plots on screen 
              quiet=quiet_flg,        # suppress informative printout
              return_fitObject=False
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

    if store_result:
      outfile = fname.split('.')[0]+'.result'
      with open(outfile, 'a') as outf:
        for key in rdict:
          print("{}\n".format(key), rdict[key], file=outf)

  if (sav_flg):
    # save all figures to file(s)
    n_fig = 0
    tag = ''
    for n in plt.get_fignums():
      plt.figure(n)      
      plt.savefig( (fname.split('.')[0] + '%s.' + pltfmt) %(tag))
      n_fig += 1
      tag = '_'+str(n_fig)
  else:
    # show on screen
    plt.show()
    

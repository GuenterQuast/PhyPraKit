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
     
  output:
      
     - text and/or file, graph depending on options


  **yaml format for x-y fit:**

  .. code-block:: yaml

    label: <str data-set name>

    x_label: <str name x-data>
    x_data: [  list of float ]   

    y_label: <str name y-data>  
    y_data: [ list of float ]

    x_errors: <float>, [list of floats], or {dictionary/ies}
    y_errors:  <float>, [list of floats], or {dictionary/ies}

    # optionally, add Gaussian constraints on parameters
    parameter_constraints: 
      <parameter name>:
        value: <value>
        uncertaintiy: <value>

    model_label: <str model name>
    model_function: |
      <Python code>

    format of uncertainty dictionary: 
    - error_value: <float> or [list of floats]
    - correlation_coefficient: 0. or 1.
    - relative: true or false
    relative errors may be spcified as <float>%

  
  Simple example of *yaml* input:

  .. code-block:: yaml

    label: 'Test Data'

    x_data: [.05,0.36,0.68,0.80,1.09,1.46,1.71,1.83,2.44,2.09,3.72,4.36,4.60]
    x_errors: 3%
    x_label: 'x values'

    y_data: [0.35,0.26,0.52,0.44,0.48,0.55,0.66,0.48,0.75,0.70,0.75,0.80,0.90]
    y_errors: [.06,.07,.05,.05,.07,.07,.09,.1,.11,.1,.11,.12,.1]
    y_label: 'y values'

    model_label: 'Parabolic Fit'
    model_function: |
      def quadratic_model(x, a=0., b=1., c=0. ):
        return a * x*x + b*x + c


  **Example of yaml input for histogram fit:**

  .. code-block:: yaml

    # Example of a fit to histogram data
    type: histogram

    label: example data
    x_label: 'h' 
    y_label: 'pdf(h)'

    # data:
    raw_data: [ 79.83,79.63,79.68,79.82,80.81,79.97,79.68,80.32,79.69,79.18,
            80.04,79.80,79.98,80.15,79.77,80.30,80.18,80.25,79.88,80.02 ]

    n_bins: 15
    bin_range: [79., 81.]
    # alternatively an array for the bin edges can be specified
    #bin_edges: [79., 79.5, 80, 80.5, 81.]

    model_density_function: |
      def normal_distribution(x, mu=80., sigma=1.):
        return np.exp(-0.5*((x - mu)/sigma)** 2)/np.sqrt(2.*np.pi*sigma** 2)


  *Remark*: more than one input data sets are also possible. 
  Data sets and models can be overlayed in one plot if option 
  `showplots = False` ist specified. Either provide more than
  one input file, or use yaml syntax, as shown here:

  .. code-block:: yaml

    # several input sets to be separated by 
    ...
    ---   
"""

from PhyPraKit.phyFit import xyFit_from_yaml, hFit_from_yaml
from pprint import pprint

# --- helper function
def wexit(code):
  # keep Python window open on MS Windows 
  import os, sys
  if os.name == 'nt':
    _ = input('\n      ==> type <ret> to end > ')
  sys.exit(code)

if __name__ == "__main__": # --------------------------------------  
  #
  # xyFit.py: Example of an application of PhyPraKit.phyFit.xyFit_from_yaml()
  #

  # package imports
  import os, sys, argparse, yaml, numpy as np, matplotlib.pyplot as plt
  if os.name == 'nt': # interactive mode on windows if error occurs
    os.environ['PYTHONINSPECT']='x'

  # - - - Parse command-line arguments
  parser = argparse.ArgumentParser(description = \
    "Perform a fit with PhyPraKit.phyFit package driven by input file")
  # parser = argparse.ArgumentParser(usage=__doc__)

  parser.add_argument('filename', type=str, nargs='+',
      help="name(s) of fit input file(s) in yaml format")
  parser.add_argument('-v', '--verbose', 
      action='store_const', const=True, default=False,
      help="full printout to screen")
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
    print(" \n !!! no input file given - exiting \n")
    wexit(1)

  # collect input from ArgumentParser
  args = parser.parse_args()
  fnames=args.filename
  quiet_flg = not args.verbose
  store_result = args.result_to_file
  plt_flg= not args.noplot
  sav_flg=args.saveplot
  cont_flg=args.contour
  band_flg=not args.noband
  pltfmt=args.format

  #  - - - End: Parse command-line arguments

  ddata = []
  # open and read input yaml file
  for fnam in fnames:
    f = open(fnam, 'r')
    try:
      ymldata = yaml.load_all(f, Loader=yaml.Loader)
    except (OSError, yaml.YAMLError) as exception:
      print('!!! failed to read configuration file ' + fnam)
      print(str(exception))
      wexit(1)
      
    fitType = 'xy'
    for d in ymldata:
      if 'type' in d:
        fitType = d['type']
      ddata.append(d)
    f.close()
  
  # select appropriate wrapper
  if fitType == 'xy':
    fit = xyFit_from_yaml
  elif fitType == 'histogram':
    fit = hFit_from_yaml
  else:
    print('!!! unsupported type of fit:', fitType)
    wexit(1)

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
      outfile = (fnames[0].split('.')[0]+ '.result')
      with open(outfile, 'a') as outf:
        for key in rdict:
          print("{}\n".format(key), rdict[key], file=outf)
      print(' -> result saved to file ', outfile)

  if (sav_flg):
    # save all figures to file(s)
    n_fig = 0
    tag = ''
    for n in plt.get_fignums():
      plt.figure(n)
      oname = (fnames[0].split('.')[0] + '%s.' + pltfmt) %(tag)
      plt.savefig( oname)
      print(' -> figure saved to file ', oname)
      n_fig += 1
      tag = '_'+str(n_fig)
  else:
    # show on screen
    plt.show()

  wexit(0)  

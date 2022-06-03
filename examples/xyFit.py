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
   
.. moduleauthor:: Guenter Quast <g.quast@kit.edu>
"""

# package imports
import sys, yaml
import numpy as np
from PhyPraKit.phyFit import xyFit

# Helper functions
def parse_code(code_string):
    """Watch out for "dangerous" commands in Python code and extract function name

      Args: 
       - user-defined code
      Returns:
       - function name
       - code
    """

    FORBIDDEN_TOKENS = ['import', 'exec', 'global', 'execfile']

    for s in FORBIDDEN_TOKENS:
      contains_forbidden_token = False
      if code_string.find(s) >=0:
        _e = "!!! Encountered forbidden token '%s' in user-entered code" % (s)
        print(_e)
        contains_forbidden_token = True
      if(contains_forbidden_token): sys.exit(1)           

    function_name=''  
    words_in_code = code_string.split()
    for i, w in enumerate(words_in_code):
      if w == 'def':
        fn = words_in_code[i+1]
        function_name=fn[0:fn.find( '(' )]
        break
    if function_name is '':
        _e = "No function name in user entered code."         
        print(_e)
        sys.exit(1)
    return function_name, code_string 

                
def decode_uDict(uDict):
  """Decode dictionary with uncertainties

    yaml format:
    y-errors:
    - type:                     "simple" or "matrix"
      error_value:              number or array
      correlation_coefficient:  0. or 1.
      relative:                 false or true

    Args:
      - uDict : dictionary with uncertainties
     
    Returns:  
      - s       # independent errors
      - srel    # independent relative errors
      - sabscor # correlated absolute errors
      - srelcor # correlated relative errors
  """
  
  def decode_rel(e):
    """Decode numbers with %-sign: interpreted as relative errors
    """
    if type(e) is type(''):
      if '%' in e:
        v = float(e[0:e.find('%')])/100.
    else:
        v = e
    return v

  s = None        # independent errors
  srel = None     # independent relative errors
  sabscor = None  # correlated absolute errors
  srelcor = None  # correlated relative errors

  if type(uDict) is not type([]):      # got a scalar, one error for all
    if type(uDict) is type(''):
      if '%' in uDict:
        srel=decode_rel(uDict)
    else:
      s = uDict
    return s, srel, sabscor, srelcor

  if type(uDict[0]) is not type({}): # got an array of uncertainties
    for i, v in enumerate(uDict):
      if type(v) is type(''):
        uDict[i]=decode_rel(v) 
        srel= uDict
      else:                          
        s = uDict
    return s, srel, sabscor, srelcor
        
  # decode complex error dictionary
  for ed in uDict:
    if 'type' in ed:
      typ = ed['type']
    else:
      typ = "simple"
      if 'relative' in ed:
        rel = ed['relative']
      else:
        rel = False   
    if 'correlation_coefficient' in ed:
      cor = ed['correlation_coefficient'] 
    else:
      cor = 0.
    if 'error_value' in ed:
      e = ed['error_value']
      if type(e) is type([]):
        for i,v in enumerate(e):
          if type(v) is type(''):  
            if '%' in v:
              rel = True
              e[i]=decode_rel(v)
      else:
        if type(e) is type(''):  
          if '%' in e:
            rel = True
            e=decode_rel(e)          
    else:
      e = None
    #    
    if cor == 0. and not rel:
      # independent absolute error
        if s is None:
          s = e
        else:
          s = [s]
          s.append(e)
    elif cor == 0. and rel:
      # independent relative error
        if srel is None:
          srel = e
        else:
          srel = [srel]
          srel.append(e)
    elif cor !=0 and not rel:
      # correlated absolute error
        if sabscor is None:
          sabscor = e
        else:
          sabscor = [sabscor]
          sabscor.append(e)
    elif cor !=0 and rel:
      # correlated relative error
        if srelcor is None:
          srelcor = e
        else:
          srelcor = [srelcor]
          srelcor.append(e)
   # -- end for 
  return s, srel, sabscor, srelcor  

if __name__ == "__main__": # --------------------------------------  
  #
  # Example of an application of PhyPraKit.xyFit()
  #

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


# extract contents   
  if 'label' in fd:
    data_label = fd['label']
  else:
    data_label = 'data'

# - data
  data_x= fd['x_data']
  data_y = fd['y_data']
  if 'x_label' in fd:
    x_label = fd['x_label']
  else:
    x_label = 'x'
  if 'y_label' in fd:
    y_label = fd['y_label']
  else:
    y_label = 'y'
    
# - uncertainties
#   x errors
  if('x_errors') in fd:
    sx, srelx, sabscorx, srelcorx = decode_uDict(fd['x_errors'])
  else:
    sx = None
    srelx = None
    sabscorx = None
    srelcorx = None
  
#  y errors 
  try:      # look at to two different places
    uDict = fd['parametric_model']['y_errors']
  except:
    try:
      uDict = fd['y_errors']
    except:
      print("!!! no y-errors found !")
      sys.exit(1)  # must have uncertainties in y !
  sy, srely, sabscory, srelcory = decode_uDict(uDict)
          
# - model name and python code
  if 'model_label' in fd:
    model_label = fd['model_label']
  else:
    model_label = 'model'
  try:
    code_str = fd['model_function']['python_code']
  except:
    try:
      code_str = fd['model_function']
    except:
      print("!!! no code to fit found !")
      sys.exit(1)
  fitf_name, code = parse_code(code_str)    

# print input and model
  print('x-data:', data_x)
  print('+/- abs', sx)
  print('    rel', srelx)
  print('   cabs', sabscorx)
  print('   crel', srelcorx)
  print('y-data:', data_y)
  print('+/- abs', sy)
  print('    rel', srely)
  print('   cabs', sabscory)
  print('   crel', srelcory)
  print("\n*==* model to fit: %s" %(fitf_name))
  print('%s' %(code) )    

  scope = dict()
  header= 'import numpy as np\n' + 'import scipy\n' 
  exec(header + code, scope)
  
# perform fit to data with function xyFit using mnFit class from phyFit
  rdict = xyFit(scope[fitf_name],
      data_x, data_y,    # data x and y coordinates
      sx=sx,             # indep x
      sy=sy,             # indel y
      srelx=srelx,       # indep. rel. x
      srely=srely,       # indep. rel. y
      xabscor=sabscorx,   # correlated x
      xrelcor=srelcorx,   # correlated rel. x
      yabscor=sabscory,   # correlated y
      yrelcor=srelcory,   # correlated rel. y
      axis_labels=[x_label, y_label], 
      data_legend =  data_label,    
      model_legend = model_label, 
      plot=True,          # plot data and model
      plot_band=True,     # plot model confidence-band
      plot_cor=False,     # plot profiles likelihood and contours
      quiet=True          # suppress informative printout
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

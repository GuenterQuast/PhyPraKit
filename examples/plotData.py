#!/usr/bin/env python3
"""plotData.py [options] <input file name>

   Plot (several) data set(s) with error bars in x- and y- directions from
   file in yaml format

   usage: 

     ./plotData.py [options] <input file name>

  Input: 

    - input file in yaml format

  Output:

    - figure
 
  yaml-format:

  .. code-block:: yaml

     title: <title of plot>
     x_label: <label for x-axis>
     y_label: <label for y-axis>

     label: <name of data set>
     x_data: [ x values ]
     y_data: [ y values ]
     x_errors: x-uncertainty or [x-uncertainties]
     y_errors: y-uncertainty or [y-uncertainties]

     several input sets to be separated by 
     ...
     ---   
"""

from pprint import pprint
import numpy as np, matplotlib.pyplot as plt
import sys, yaml

def ymlplot(d):
  """plot data from yaml file
  """

  def plot(x, y, ex, ey, title=None,
           label='data', x_label = 'x', y_label = 'y'):
    """return figure with xy data and uncertainties
    """
    # draw data
    plt.plot(x, y, marker='x', linestyle='', color='grey', alpha=0.5)
    if (ex is not None) and (ey is not None):
      plt.errorbar(x, y, xerr=ex, yerr=ey, fmt='.', label=label)
    elif ey is not None:
      plt.errorbar(x, y, yerr=ey, fmt='.', label=label)
    elif ex is not None:
      plt.errorbar(x, y, xerr=ex, fmt='.', label=label)
    else:      
      plt.errorbar(x, y, ey, fmt=".", label=label)

    plt.legend()
    if x_label is not None: plt.xlabel(x_label, size='x-large')
    if y_label is not None: plt.ylabel(y_label, size='x-large')
    if title is not None:
      plt.title(title, size='xx-large')
    return plt.gcf()
  # -- end plot function 

  if 'title' in d:
    title = d['title']
  else:
    title = None

  if 'label' in d:
    data_label = d['label']
  else:
    data_label = None

  x_dat = d['x_data']
  if 'x_label' in d:
    x_label = d['x_label']
  else:
    x_label = None
  if 'x_errors' in d:
    x_err = d['x_errors']
  else:
    x_err = None

  y_dat = d['y_data']
  if 'y_errors' in d:
    y_err = d['y_errors']
  else:
    y_err = None
  if 'y_label' in d:
    y_label = d['y_label']
  else:
    y_label = None
    
  fig = plot(x_dat, y_dat, x_err, y_err, title=title,
         label=data_label, x_label = x_label, y_label = y_label)

if __name__ == "__main__": # --------------------------------------  

  import argparse

  # - - - Parse command-line arguments
  parser = argparse.ArgumentParser(usage=__doc__)

  parser.add_argument('filename', type=str,
      help="name of fit input file in yaml format")
  parser.add_argument('-s', '--saveplot', 
      action='store_const', const=True, default=False,
      help="save plot(s) in file(s)")
  parser.add_argument('-f','--format', 
      type=str, default='pdf',
      help="graphics output format, default=pdf")
  parser.add_argument('-n', '--noplot', 
      action='store_const', const=True, default=False,
      help="suppress ouput of plots on screen")
  
  if len(sys.argv)==1:  # print help message if no input given
    parser.print_help()
    sys.exit(1)

  # collect input from ArgumentParser
  args = parser.parse_args()
  fnam = args.filename
  sav_flg = args.saveplot
  pltfmt = args.format
  plt_flg = not args.noplot

  f = open(fnam,'r')
  try:
    ymldata = yaml.load_all(f, Loader=yaml.Loader)
  except (OSError, yaml.YAMLError) as exception:
    print('!!! failed to read configuration file ' + fname)
    print(str(exception))
    sys.exit(1)
    
  # create a figure
  num = 'plotData'
  figsize = (7.5, 6.5)
  figure = plt.figure(num=num, figsize=figsize)

  # decode yaml input and plot data for each yaml file
  for d in ymldata:
    ymlplot(d)
  f.close()

  # output to file or screen
  if (sav_flg):
    plt.savefig( (fnam.split('.')[0] + '.'+pltfmt) )
  # show plot on screen
  if plt_flg:
    plt.show()

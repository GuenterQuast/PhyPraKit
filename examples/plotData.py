#!/usr/bin/env python3
"""**plotData.py** [options] <input file name>

  Plot (several) data set(s) with error bars in x- and y- directions 
  or histograms from file in yaml format

  usage: 

    ./plotData.py [options] <input file name>

  Input: 

    - input file in yaml format

  Output:

    - figure
 
  yaml-format for (x-y) data:

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

  yaml-format for histogram:

  .. code-block:: yaml

     title: <title of plot>
     x_label: <label for x-axis>
     y_label: <label for y-axis>

     label: <name of data set>
     raw_data: [x1, ... , xn]
     # define binning
     n_bins: n
     bin_range: [x_min, x_max]
     #   alternatively: 
     # bin edges: [e0, ..., en]

     several input sets to be separated by 
     ...
     ---   
"""

def plot_xy_from_yaml(d):
  """plot (xy) data from yaml file

     Input: 

         dictionary from yaml input
     
     Output: 

         matplotlib figure

  yaml-format of input:

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

  import numpy as np, matplotlib.pyplot as plt

  def plot(x, y, ex, ey, title=None,
           label='data', x_label = 'x', y_label = 'y',
           marker='x', color='grey'):
    """return figure with (x,y) data and uncertainties
    """
    # draw data
    plt.plot(x, y, marker=marker, linestyle='', color=color, alpha=0.5)
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
             label=data_label, x_label = x_label, y_label = y_label,
             marker='x', color='grey')

def plot_hist_from_yaml(d):
  """plot histogram data from yaml file

     Input: 

         dictionary from yaml input
     
     Output: 

         matplotlib figure

  yaml-format of input:

  .. code-block:: yaml

     title: <title of plot>
     x_label: <label for x-axis>
     y_label: <label for y-axis>

     label: <name of data set>
     raw_data: [x1, ... , xn]
     # define binning
     n_bins: n
     bin_range: [x_min, x_max]
     #   alternatively: 
     # bin edges: [e0, ..., e_n]

     several input sets to be separated by 
     ...
     ---   
  """
  # trick to generate a global variable for accumulated statistics
  import numpy as np, matplotlib.pyplot as plt
  from PhyPraKit import histstat

  global statinfo
  try: statinfo
  except NameError: statinfo = []

  def plot(bconts, bedges, title=None,
           label='histogram', x_label = 'x', y_label = 'y',
           grid=True, statistics=True):
    """return figure with histogram data
    """
    # global variable for statistics information
    global statinfo

    # draw data
    w = 0.9*(be[1:] - be[:-1])
    
    plt.bar(bedges[:-1],bconts,
            align='edge', width = w, alpha=0.5,
#            facecolor='cadetblue',
            edgecolor='grey', 
            label = label)
    
    # get statistical information
    if statistics:
      mean, sigma, sigma_m = histstat(bconts, bedges, pr=False)
      if len(statinfo) == 0:
        statinfo.append('Statistics:')
      else:      
        statinfo.append('  - - - - - - - ')
      statinfo.append('  $<>$:  {:.3g}'.format(mean))
      statinfo.append('     $\sigma$   : {:.3g}'.format(sigma))
    plt.legend(loc='best', title="\n".join(statinfo))
    
    if x_label is not None: plt.xlabel(x_label, size='x-large')
    if y_label is not None: plt.ylabel(y_label, size='x-large')
    if title is not None:
      plt.title(title, size='xx-large')
    if grid: plt.grid()
    return plt.gcf()
  # -- end plot function 

  # get data
  hdata = d['raw_data']
  bins = 10
  if 'n_bins' in d:
    bins = d['n_bins']
  if 'bin_edges' in d:
    bins = d['bin_edges']
  bin_range = None
  if 'bin_range' in d:
    bin_range = d['bin_range']

  if 'title' in d:
    title = d['title']
  else:
    title = None

  if 'label' in d:
    data_label = d['label']
  else:
    data_label = None
  
  if 'x_label' in d:
    x_label = d['x_label']
  else:
    x_label = 'x'
  if 'y_label' in d:
    y_label = d['y_label']
  else:
    y_label = 'y'

  bc, be = np.histogram(hdata, bins=bins,  range=bin_range)
  
  fig = plot(bc, be, title=title,
             label=data_label, x_label = x_label, y_label = y_label,
             grid=True)

  
if __name__ == "__main__": # --------------------------------------  

  import sys, yaml, argparse, matplotlib.pyplot as plt


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
    
  data_type = 'xy'
  ddata = []
  for d in ymldata:
    if 'type' in d:
      data_type = d['type']
    ddata.append(d)

  # create a figure
  if data_type == 'xy':
     fignam = 'plotxyData'
  elif data_type == 'histogram':
     fignam = 'plothistData'
  else:
     print('!!! invalid data type', data_type)
     sys.exit(1)

  # create figure
  figsize = (7.5, 6.5)
  figure = plt.figure(num=fignam, figsize=figsize)

  # decode yaml input and plot data for each yaml file
  for d in ddata:
    if data_type == 'xy':
      plot_xy_from_yaml(d)
    elif data_type == 'histogram':
      plot_hist_from_yaml(d)
    f.close()

  # output to file or screen
  if (sav_flg):
    plt.savefig( (fnam.split('.')[0] + '.'+pltfmt) )
  # show plot on screen
  if plt_flg:
    plt.show()

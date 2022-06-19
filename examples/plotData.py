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

  In case a model function is also supplied, it is overlayed in the 
  output graph. The corresponding *yaml* block looks as follows:

  .. code-block:: yaml

    # optional model specification
    model_label: <model name>
    model_function: |
    <Python code of model function>

"""

from PhyPraKit import plot_xy_from_yaml,plot_hist_from_yaml

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

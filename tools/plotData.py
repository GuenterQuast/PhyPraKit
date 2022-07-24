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

  *Remark*: more than one input data sets are also possible. 
  Data sets and models can be overlayed in one plot if option 
  `showplots = False` ist specified. Either provide more than
  one input file, or use yaml syntax, as shown here:

  .. code-block:: yaml

    # several input sets to be separated by 
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

  In case a model function is supplied, it is overlayed in the 
  output graph. The corresponding *yaml* block looks as follows:

  .. code-block:: yaml

    # optional model specification
    model_label: <model name>
    model_function: |
      <Python code of model function>

  If no `y_data` or `raw_data` keys are provided, only the model function 
  is shown. Note that minimalistic `x_data` and `bin_range` or `bin_edges`
  information must be given to define the x-range of the graph. 

"""

from PhyPraKit import plot_xy_from_yaml,plot_hist_from_yaml

# --- helper function
def wexit(code):
  # keep Python window open on MS Windows 
  import os, sys
  if os.name == 'nt':
    _ = input('\n      ==> type <ret> to end > ')
  sys.exit(code)

if __name__ == "__main__": # --------------------------------------  

  import os, sys, yaml, argparse, matplotlib.pyplot as plt
  if os.name == 'nt': # interactive mode on windows if error occurs
    os.environ['PYTHONINSPECT']='x'

  # - - - Parse command-line arguments
  parser = argparse.ArgumentParser(description = \
   "Plot data with error bars or a histrogram from file in yaml format")
  # parser = argparse.ArgumentParser(usage=__doc__)

  parser.add_argument('filename', type=str, nargs='+',
      help="name(s) of input file(s) in yaml format")
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
    print(" \n !!! no input file given - exiting \n")
    wexit(1)

  # collect input from ArgumentParser
  args = parser.parse_args()
  fnames = args.filename
  sav_flg = args.saveplot
  pltfmt = args.format
  plt_flg = not args.noplot

  ddata = []
  for fnam in fnames:
    f = open(fnam,'r')
    try:
      ymldata = yaml.load_all(f, Loader=yaml.Loader)
    except (OSError, yaml.YAMLError) as exception:
      print('!!! failed to read configuration file ' + fnam)
      print(str(exception))
      wexit(1)
    
    data_type = 'xy'
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
     wexit(1)

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
    oname = fnames[0].split('.')[0] + '.'+pltfmt
    plt.savefig( oname )
    print('  -> graph saved to file ', oname) 
  # show plot on screen
  if plt_flg:
    plt.show()

  wexit(0)

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

from PhyPraKit.plotData import plotData

if __name__ == "__main__":  # --------------------------------------
    plotData()

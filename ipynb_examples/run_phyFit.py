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

from PhyPraKit.run_phyFit import run_phyFit

if __name__ == "__main__":  # --------------------------------------
    run_phyFit()

#!/usr/bin/python3

"""plotCSV.py 
   uses PhyPraKkt.readtxt() to read floating-point column-data in csv format 


  usage: 

    ./plotCSV.py [options] <input file name>

  Input: 

    - input file in csv format

  Options:

    - s : character used as field separator, default ','
    - H : number of header lines, default 1

  Output:

    - figure
 


"""

from PhyPraKit import readtxt

# --- helper function
def wexit(code):
  # keep Python window open on MS Windows 
  import os, sys
  if os.name == 'nt':
    _ = input('\n      ==> type <ret> to end > ')
  sys.exit(code)

if __name__ == "__main__":
  import sys, argparse, numpy as np, matplotlib.pyplot as plt

    # - - - Parse command-line arguments
  parser = argparse.ArgumentParser(description = \
    "plot contents of CSV file")

  parser.add_argument('filename', type=str, nargs='+',
      help="name of csv file")
  parser.add_argument('-v', '--verbose', 
      action='store_const', const=True, default=False,
      help="full printout to screen")
  parser.add_argument('-s', '--separator',
      type=str, default=',',
      help="character used as field separator ")
  parser.add_argument('-H', '--Headerlines',
      type=int, default = 1,
      help="number of header lines")
  
  if len(sys.argv)==1:  # print help message if no input given
    parser.print_help()
    print(" \n !!! no input file given - exiting \n")
    wexit(1)

  # collect input from ArgumentParser
  args = parser.parse_args()
  fname = args.filename[0]
  quiet_flg = not args.verbose
  separator = args.separator    # field separator: , or \t, or \_ or ; etc.
  nHlines = args.Headerlines   # number of header lines 
  # print(args)
  
  # end parsing input ------------------------------------------
  
  # read data from file in .txt format
  hlines, data = readtxt(fname, delim=separator, nlhead=nHlines)
  nColumns = data.shape[0]
  nRows = data.shape[1]
  keys = hlines[0].split(separator)[:]
  # output to terminal
  print(hlines)
  print(" --> number of columns", nColumns)
  print(" --> number of data points", nRows)
  
# make plots - columns 1, 2, 3, ...  vs. column 0
  fig=plt.figure(1, figsize=(10, 2.25*nColumns))
  fig.tight_layout()
  fig.suptitle('contents of file '+fname, size='x-large', color='b')
  fig.subplots_adjust(left=0.14, bottom=0.1, right=0.97, top=0.93,
                      wspace=None, hspace=.33)#
  x = data[0]
  axes = []
  ncol = nColumns - 1 
  for i in range(1, ncol+1):
    axes.append(fig.add_subplot(ncol, 1, i))
    ax = axes[i-1]
    ax.plot(x, data[i], alpha=0.3, label='Column' + str(i))
    ax.set_ylabel(keys[i], size='large')
    ax.set_xlabel(keys[0], size='large')
    ax.legend(loc='best', numpoints=1, prop={'size':10})
    ax.grid()
 
  plt.show()

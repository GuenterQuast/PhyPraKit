#!/usr/bin/env python3
"""kfitf.py  
   Perform a fit with the kafe package driven by input file  

   usage: kfitf.py [-h] [-n] [-s] [-c] [--noinfo] [-f FORMAT] filename

   positional arguments:
     filename                       name of fit input file

   optional arguments:
     -h, --help           show this help message and exit
     -n, --noplot         suppress ouput of plots on screen
     -s, --saveplot       save plot(s) in file(s)
     -c, --contour        plot contours and profiles
     --noinfo             suppress fit info on plot
     --noband             suppress 1-sigma band around function
     --format FMT         graphics output format, default FMT = pdf
.. moduleauthor:: Guenter Quast <g.quast@kit.edu>
"""

## Author:      G. Quast   Oct. 2016, adapted from erlier version
##
## dependencies: PYTHON v2.7, sys, numpy, matplotlib.pyplot, kafe
## last changed:  GQ Oct-23-16

import sys, argparse 
import kafe, matplotlib.pyplot as plt
from kafe.file_tools import buildFit_fromFile
#
# ---------------------------------------------------------
#
def kfitf():
# - - - Parse command-line arguments
  parser = argparse.ArgumentParser(description = \
    "Perform a fit with the kafe package driven by input file")
#  parser = argparse.ArgumentParser(usage=__doc__)

  parser.add_argument('filename', type=str,
      help="name of fit input file")
  parser.add_argument('-n', '--noplot', 
      action='store_const', const=True, default=False,
      help="suppress ouput of plots on screen")
  parser.add_argument('-s', '--saveplot', 
      action='store_const', const=True, default=False,
      help="save plot(s) in file(s)")
  parser.add_argument('-c', '--contour', 
      action='store_const', const=True, default=False,
      help="plot contours and profiles")
  parser.add_argument('--noinfo', 
      action='store_const', const=True, default=False,
      help="suppress fit info on plot")
  parser.add_argument('--noband', 
      action='store_const', const=True, default=False,
      help="suppress 1-sigma band around function")
  parser.add_argument('-f','--format', 
      type=str, default='pdf',
      help="graphics output format, default=pdf")

  if len(sys.argv)==1:  # print help message if no input given
    parser.print_help()
    sys.exit(1)
  args = parser.parse_args()

# collect input from ArgumentParser
  fname=args.filename
  nplt_flg=args.noplot
  sav_flg=args.saveplot
  cont_flg=args.contour
  noinfo_flg=args.noinfo
  noband_flg=args.noband
  pltfmt=args.format
#  - - - End: Parse command-line arguments

# - - - Perform fit as specified on command line

# initialize fit object from file
  theFit = buildFit_fromFile(fname)
  theFit.fit_name = theFit.dataset.basename
# perform fit
  theFit.do_fit()
# produce desired output
  if (not nplt_flg or sav_flg):
  # make the plot(s)
    thePlot = kafe.Plot(theFit)
    if noinfo_flg:
      info=None
    else:
      info= 'all'
    if noband_flg:
      band=None
    else:
      band= 'meaningful'
    thePlot.plot_all( show_info_for=info, 
                      show_data_for='all',
                      show_function_for='all', 
                      show_band_for=band)
# eventually, store figure
  if (sav_flg):
    thePlot.save(fname.split('.')[0]+'.'+pltfmt)

# eventually, plot contours and profiles
  if (cont_flg):
    corrFig = theFit.plot_correlations()
    if (sav_flg):
      corrFig.savefig(fname.split('.')[0]+'_corr.'+pltfmt)

# show everything on screen
  if (not nplt_flg):
    thePlot.show()


# ------------ begin execution --------------------------------
if __name__ == "__main__":
    kfitf()

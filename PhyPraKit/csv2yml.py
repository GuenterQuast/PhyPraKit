#!/usr/bin/env python3
"""csv2yml.py 
  read floating-point column-data in very general .txt  formats 
  and write an output block in yaml format

  keys taken from 1st header line

  Usage:

    ./cvs2yml [options] <input file name>

  Input: 

    - file name 

  Output:

    - yaml data block 
  

.. moduleauthor:: Guenter Quast <g.quast@kit.edu>

"""

if __name__ == "__main__": # -------------------------------------------

  import sys, os, yaml, argparse
  from PhyPraKit import csv2yaml
  if os.name == 'nt': # interactive mode on windows if error occurs
    os.environ['PYTHONINSPECT']='x'

  # - - - Parse command-line arguments
  # parser = argparse.ArgumentParser(usage=__doc__)
  parser = argparse.ArgumentParser(
    description="convert csv to yaml format")

  parser.add_argument('filename', type=str,
      help="name of input file in csv format")
  parser.add_argument('-s', '--save', 
      action='store_const', const=True, default=False,
      help="save result in file")
  parser.add_argument('-H', '--Header', 
      action='store_const', const=True, default=False,
      help="print csv Header lines")
  parser.add_argument('-q', '--quiet', 
      action='store_const', const=True, default=False,
      help="quiet - no output to screen")
  parser.add_argument('-d','--delimiter', 
                      type=str, default=',',
           help="delimiter, default=','")
  parser.add_argument('-n','--header_lines', 
                      type=int, default=1,
           help="numer of header lines, default=1")
  
  if len(sys.argv)==1:  # print help message if no input given
    parser.print_help()
    raise ValueError('!!! not input given -exit !')

  # collect input from ArgumentParser
  args = parser.parse_args()
  fnam = args.filename

  nlhead = args.header_lines
  delim = args.delimiter
  showHeader = args.Header
  sav_flg = args.save
  quiet = args.quiet
  if quiet: sav_flg = True

  # ---- end argument parsing -------------------
  
  f = open(fnam, 'r') 
  hlines, ylines = csv2yaml(f, nlhead, delim)

  if not quiet:
    print('-->', fnam, 'to yaml', end='')

  if showHeader:
    print('  csv header:')
    for l in hlines:
      print(l)

  # print results to screen
  if not quiet:
    print('  yaml block:')
    for l in ylines:
      print(l)

  # write to file
  if sav_flg:
    ymlfn = fnam.split('.')[0] + '.yaml'
    with open(ymlfn, 'w') as f:
     for l in ylines:
       print(l, file=f)
    print('   -> yaml saved in file',ymlfn) 

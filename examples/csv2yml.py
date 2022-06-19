#!/usr/bin/env python3
"""csv2yml.py 
   uses readtxt() to read floating-point column-data in very general 
   .txt formats and writes an output block in yaml format
.. moduleauthor:: Guenter Quast <g.quast@kit.edu>

"""

def csv2yaml(file, nlhead=1, delim='\t'):
  """ read floating point data in general csv format
  skip header lines, replace decimal comma, remove special characters,
  and ouput as yaml data block
  
  Args:
    * file: open file
    * nhead: number of header lines to skip
    * delim: column separator
  Returns: 
    * hlines: list of string, header lines
    * ymltxt: list of text lines with yaml key and data
  """
  
# --------------------------------------------------------------------

# -- helper function to filter input lines
  def specialCharFilter(f, delim):
    """a generator to filter lines read from file
    replace German ',' by '.', remove special characters 

    Args:
      * string f:  file name
    Yields:
      * a valid line with numerical data
    """
    while True:
      l=f.readline()      # read one line
      if (not l): break   # end-of-file reached, exit

    # remove white spaces and control characters, fix German floats 
        # remove leading and trailing white spaces
      l=l.strip()         
      # remove ascii control characters (except delimiter) 
      for i in range(32):
        if delim != chr(i) : l=l.replace(chr(i),'') 
      if l=='': continue        # skip empty lines 
        # replace German decimal comma (if not CSV format)
      if delim != ',' :
        filtline=l.replace(',', '.')
      else:
        filtline=l      
      yield filtline           # pass filtered line to loadtxt()
#   -- end specialCharFilter

  # open file for read (if necessary)
  if type(file)==type(' '): f = open(file, 'r') # file is a file name
  else: f = file        # assume input is file handle of an open file 

  # set-up generator for text lines from file 
  lfilt = specialCharFilter(f, delim)

  # read header
  hlines=[]
  for i in range (nlhead):
    hlines.append(next(lfilt))  # header line(s)

  dlines=[]
  while True:
    try:
      dlines.append(next(lfilt).split(delim))  # data line(s)
    except StopIteration:
      break
    
  Nlin = len(dlines)
  Ncol = len(dlines[0])
  #print(" --> number of columns", Ncol)
  #print(" --> number of data points", Nlin)

  # interpret strings header[0] as keys
  keys = hlines[0].split(delim)
  Nkey = len(keys)
  if Nkey != Ncol:
    print("!!! number of keys{} not equal number of columns - exit")
    print('Nkey=', Nkey, ' Ncol=', Ncol)
    sys.exit(1)  

  # construct string in yaml format
  #   transpose list with number strings
  dlinesT = [[row[i] for row in dlines] for i in range(Ncol)]
  ylines=[]
  for i, k in enumerate(keys):
    yl = "{0}: [{1}]".format(k.strip(), ','.join(dlinesT[i]))
    ylines.append(yl)
  return hlines, ylines

if __name__ == "__main__": # -------------------------------------------  
  import sys, PhyPraKit as ppk, yaml

  # check for / read command line arguments
  if len(sys.argv)==2:
    fnam = sys.argv[1]
  else:
    print("!!! no input file given - exiting")
    sys.exit(1)
  
  # read data from file in .txt format
  fnam = sys.argv[1]
  nlhead = 1
  delim = ','
  f = open(fnam, 'r') 

  hlines, ylines = csv2yaml(f, nlhead, delim)

  # print to screen
  for l in ylines:
    print(l)

  # write to file
  with open(fnam.split('.')[0] + '.yaml', 'w') as f:
   for l in ylines:
     print(l, file=f)


#!/usr/bin/env python3

"""plotCSV.py 
   uses PhyPraKit.readtxt() to read floating-point column-data in csv format

   replaces decimal comma by dot if needed. 


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

from PhyPraKit.plotCSV import plotCSV

if __name__ == "__main__":  # --------------------------------------
    plotCSV()

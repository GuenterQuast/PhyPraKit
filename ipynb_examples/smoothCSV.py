#!/usr/bin/env python3

"""smoothCSV.py 
   uses PhyPraKit.readtxt() to read floating-point column-data in csv format 
   and provides a smoothed and resampled version

   replaces decimal comma by dot, if needed.

  usage: 

    ./smoothCSV.py [options] <input file name>

  Input: 

    - input file in csv format

  Options:

    - w : window size
    - H : number of header lines, default 1
    - s : character used as field separator, default ','
    - n : no graphical output

  Output:

    - figure
    - new csv file  

"""

from PhyPraKit.smoothCSV import smoothCSV

if __name__ == "__main__":
    smoothCSV()

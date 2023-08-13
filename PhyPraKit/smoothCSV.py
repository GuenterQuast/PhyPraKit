#!/usr/bin/python3

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

from PhyPraKit import readtxt, meanFilter, resample


# --- helper function
def wexit(code):
    # keep Python window open on MS Windows
    import os, sys

    if os.name == "nt":
        _ = input("\n      ==> type <ret> to end > ")
    sys.exit(code)


def smoothCSV():
    import sys, argparse, numpy as np, matplotlib.pyplot as plt

    # - - - Parse command-line arguments
    parser = argparse.ArgumentParser(description="smooth CSV file")

    parser.add_argument("filename", type=str, nargs="+", help="name of csv file")
    parser.add_argument(
        "-n",
        "--noplot",
        action="store_const",
        const=True,
        default=False,
        help="suppress figure",
    )
    parser.add_argument(
        "-r",
        "--resample",
        action="store_const",
        const=True,
        default=False,
        help="down-sampling of smoothed data",
    )
    parser.add_argument(
        "-s",
        "--separator",
        type=str,
        default=",",
        help="character used as field separator ",
    )
    parser.add_argument(
        "-H", "--Headerlines", type=int, default=1, help="number of header lines"
    )
    parser.add_argument("-w", "--WindowSize", type=int, default=10, help="window size")
    parser.add_argument(
        "-k",
        "--key_line",
        type=int,
        default=0,
        help="header line containing keys, default=0",
    )

    if len(sys.argv) == 1:  # print help message if no input given
        parser.print_help()
        print(" \n !!! no input file given - exiting \n")
        wexit(1)

    # collect input from ArgumentParser
    args = parser.parse_args()
    fname = args.filename[0]
    delim = args.separator  # field separator: , or \t, or \_ or ; etc.
    nHlines = args.Headerlines  # number of header lines
    l_key = args.key_line  # line containing keys
    downsample = args.resample  # down-sampling of smoothed data
    noplot = args.noplot
    nW = args.WindowSize  # size of window for sliding average and resampling
    # print(args)

    # end parsing input ------------------------------------------

    # read data from file in .txt format
    hlines, rawdata = readtxt(fname, delim=delim, nlhead=nHlines)
    nColumns = rawdata.shape[0]
    nRows = rawdata.shape[1]
    keys = hlines[l_key].split(delim)[:]
    # output to terminal
    print(hlines)
    print(" --> number of columns", nColumns)
    print(" --> number of data points", nRows)

    # smooth data ("sliding average")
    #  - parameters nav and nr to adjust to given problem !
    nav = nW  # window width
    nr = nW  # resampling factor
    x = resample(rawdata[0], n=nr) if downsample else rawdata[0]
    data = np.zeros((nColumns, len(x)), dtype=np.float32)
    data[0] = x
    for i in range(1, nColumns):
        d = meanFilter(rawdata[i], width=nav)
        data[i] = resample(d, n=nr) if downsample else d
    nc = data.shape[0]
    nr = data.shape[1]
    print(" --> smoothed data")
    print("     columns", nc)
    print("     data points", nr)

    fn = fname.split(".")[0] + "_smooth_" + str(nW) + ".csv"
    f = open(fn, "w")
    for l in hlines:
        print(l, file=f)
    f.close()
    f = open(fn, "a")
    np.savetxt(f, data.transpose(), delimiter=delim, fmt="%.6g")
    print(" -->  smoothed data written to file ", fn)
    f.close()

    if noplot:
        wexit(0)

    # make plots - columns 1, 2, 3, ...  vs. column 0
    fig = plt.figure(1, figsize=(10, 2.25 * nColumns))
    fig.tight_layout()
    fig.suptitle("smoothed contents in " + fn, size="x-large", color="b")
    fig.subplots_adjust(
        left=0.14, bottom=0.1, right=0.97, top=0.93, wspace=None, hspace=0.33
    )  #
    axes = []
    ncol = nColumns - 1
    for i in range(1, ncol + 1):
        axes.append(fig.add_subplot(ncol, 1, i))
        ax = axes[i - 1]
        ax.plot(x, data[i], alpha=0.3, label=keys[i])
        ax.set_ylabel(keys[i], size="large")
        ax.set_xlabel(keys[0], size="large")
        ax.legend(loc="best", numpoints=1, prop={"size": 10})
        ax.grid()

    plt.show()


if __name__ == "__main__":  # ------------------------------------------
    smoothCSV()

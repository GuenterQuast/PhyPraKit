#!/bin/sh

# run all tests and examples using PhyPraKit modules

echo ""
echo "running PhyPraKit tests and examples"
echo ""

set -e

# reading data
echo " *** testing data reading"
python3 test_readtxt.py
python3 test_readColumnData.py
python3 test_readPicoScope.py
python3 test_readCassy.py
python3 test_labxParser.py

# statistics
echo " *** testing error propagation"
python3 test_propagatedError.py

# signal processing
echo " *** testing signal processing"
python3 test_AutoCorrelation.py
python3 test_convolutionFilter.py
python3 test_Fourier.py

# histogramming
echo " *** testing historgram"
python3 test_Histogram.py

# regression and fititing
echo " *** testing simple fit"
python3 test_linRegression.py
echo " *** testing fits"
python3 test_odFit.py
python3 test_xyFit.py
python3 test_hFit.py
python3 test_mlFit.py
python3 test_xFit.py
echo " *** testing kafe2 fits"
python3 test_simplek2Fit.py
python3 test_k2Fit.py
python3 test_k2hFit.py

# toy data
echo " *** testing toy data generation"
python3 test_generateData.py
# multiple fits with toy data
python3 toyMC_Fit.py 100

# more complex examples
echo " *** running examples"
python3 Beispiel_Diodenkennlinie.py
python3 Beispiel_Drehpendel.py
python3 Beispiel_Hysterese.py
python3 Beispiel_Wellenform.py
python3 Beispiel_MultiFit.py
python3 Beispiel_GeomOptik.py
python3 Beispiel_GammaSpektroskopie.py

# check stand-alone tools
echo " *** testing stand-alone tools"
cd ../tools
plotData.py data.ydat
run_phyFit.py simpleFit.yfit

rm -rf .kafe

echo ""
echo "all tests done"
echo ""


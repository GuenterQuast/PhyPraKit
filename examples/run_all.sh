#!/bin/sh

# run all tests and examples using PhyPraKit modules

echo ""
echo "running PhyPraKit tests and examples"
echo ""

set -e

# reading data
python3 test_readtxt.py
python3 test_readColumnData.py
python3 test_readPicoScope.py
python3 test_readCassy.py
python3 test_labxParser.py

# statistics
python3 test_propagatedError.py

# signal processing
python3 test_AutoCorrelation.py
python3 test_convolutionFilter.py
python3 test_Fourier.py

# regression and fititing
python3 test_linRegression.py
python3 test_odFit.py
python3 test_xyFit.py
python3 test_simplek2Fit.py
python3 test_k2Fit.py
python3 test_k2hFit.py
python3 test_hFit.py
python3 test_mlFit.py
python3 test_xFit.py

# toy data
python3 test_generateData.py
# multiple fits with toy data
python3 toyMC_Fit.py 100

# histogramming
python3 test_Histogram.py

# more complex examples
python3 Beispiel_Diodenkennlinie.py
python3 Beispiel_Drehpendel.py
python3 Beispiel_Hysterese.py
python3 Beispiel_Wellenform.py
python3 Beispiel_MultiFit.py
python3 Beispiel_GeomOptik.py
python3 Beispiel_GammaSpektroskopie.py

rm -rf .kafe
echo ""
echo "all tests done"
echo ""


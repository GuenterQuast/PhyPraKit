#! /usr/bin/env python3
"""Beispiel_Diodenkennlinie.py
   Messung einer Strom-Spannungskennlinie und Anpassung der Schockley-Gleichung. 

   - Konstruktion der Kovarianzmatrix für reale Messinstrumente mit 
     Signalrauschen, Anzeigeunsicherheiten und korrelierten, realtiven 
     Kalibratsionsunsicherheiten für die Strom- und Spannungsmessung.

   - Ausführen der Anpassung der Shockley-Gleichung mit *k2Fit* oder *mFit* 
     aus dem Paket *PhyPraKit*. Wichtig: die Modellfunktion ist nicht 
     nach oben beschränkt, sondern divergiert sehr schnell. Daher muss der 
     verwendete numerische Optimierer Parameterlimits unterstützen.
   
.. moduleauthor:: Guenter Quast <g.quast@kit.edu>
"""

import numpy as np, matplotlib.pyplot as plt
from PhyPraKit import xyFit, k2Fit

# define the model function to fit
def Shockley(U, I_s = 0.5, U0 = 0.03):
  """Parametrisierung einer Diodenkennlinie

  U0 sollte während der Anpassung auf einen solchen Wert beschränkt 
  werden, dass U/U0<150 bleibt, um Überscheitungen des  mit 64 Bit 
  Genauigkeit darstellbaren Zahlenraums zu verhindern

  Args:
    - U: Spannung (V)
    - I_s: Sperrstrom (nA)
    - U0: Temperaturspannung (V) * Emissionskoeffizient
    
  Returns:
    - float I: Strom (mA)
  """
  return 1e-6 * I_s * np.exp( (U/U0) - 1.)


if __name__ == "__main__": # --------------------------------------  

#
# Anpassung der Shockley-Gleichung an eine Diodenkennlinie 
#
  

# Schockleygleichung als Fitfunktion setzen
  model = Shockley 

# Messdaten:
#  - Spannung im Messbereich 2V
  data_x = [0.450, 0.470, 0.490, 0.510, 0.530, 
      0.550, 0.560, 0.570, 0.580, 0.590, 0.600, 0.610, 0.620, 0.630,
            0.640, 0.645, 0.650, 0.655, 0.660, 0.665 ]
# - Strommessungen: 2 im Bereich 200µA, 12 mit 20mA und  6 mit 200mA
  data_y = [0.056, 0.198, 0.284, 0.404, 0.739, 1.739, 1.962,
            2.849, 3.265, 5.706, 6.474, 7.866, 11.44, 18.98,
            23.35, 27.96, 38.61, 46.73, 49.78, 57.75]
  
# Komponenten der Messunsicherheit
#  - Genauigkeit Spannungsmessung: 4000 Counts, +/-(0.5% + 3 digits)
#     - Messbereich 2V
  crel_U = 0.005
  Udigits = 3
  Urange = 2
  Ucounts = 4000
#  - Genauigkeit Strommessung: 2000 Counts, +/-(1.0% + 3 digits) 
#     - Messbereiche 200µA, 20mA und 200mA 
  crel_I = 0.010
  Idigits = 3
  Icounts = 2000
  Irange1 = 0.2
  Irange2 = 20
  Irange3 = 200
# - Rauschanteil (aus Fluktuationen der letzen Stelle)
#   - delta U = 0.005 V
  deltaU = 0.005
#   - delta I = 0.025 mA
  deltaI = 0.025

# - Anzeigegenauigkeit der Spannung (V)  
  sx = Udigits * Urange / Ucounts
  sabsx = np.sqrt(deltaU**2 + sx**2) # Rauschanteil addieren
# - korrelierte Kalibrationsunsicherheit  
  crelx = crel_U
  
# - Anzeigegenauigkeit des Stroms (mA), 3 Messbereiche
  sy = np.asarray(   2 * [Idigits * Irange1 / Icounts] + \
                    12 * [Idigits * Irange2 / Icounts] + \
                     6 * [Idigits * Irange3 / Icounts])  
  sabsy = np.sqrt(deltaI**2 + sy**2) # Rauschanteil addieren
# - korrelierte Kalibrationsunsicherheit  
  crely = crel_I
      
# Anpassung ausführen (mit Fit-Funktionen aus Paket PhyPraKit)
  thisFit = xyFit    # Alternativen: xyFit oder k2fit
  parvals, parerrs, cor, chi2 = thisFit(model,
 # - data and uncertainties
      data_x, data_y,      # data x and y coordinates
      sx=sabsx,            # indep x
      sy=sabsy,            # indel y
      xrelcor=crelx,       # correlated rel. x
      yrelcor=crely,       # correlated rel. y
      ref_to_model=True,   # reference of rel. uncert. to model
 # - fit control
      p0=(0.2, 0.05),   # initial guess for parameter values 
      limits=('U0', 0.005, None), # parameter limits
# - output options
      plot=True,           # plot data and model
      plot_cor=False,      # plot profiles likelihood and contours
      showplots = False,   # plt.show() in user code                       
      quiet=False,         # suppress informative printout
      axis_labels=['U (V)', '$I_D$ (mA)   \  Shockley-Gl.'], 
      data_legend = 'Messwerte mit Unsicherheiten',    
      model_legend = 'Shockley-Gleichung'
  )

# Ausgabe der Ergebnisse in Textform:
  print('\n*==* Fit Result:')
  print(" chi2: {:.3g}".format(chi2))
  print(" parameter values:      ", parvals )
  print(" parameter uncertainties: ", parerrs )
  np.set_printoptions(precision=3)
  print(" correlations : \n", cor )

# Anpassung der Optionen für grafische Darstellung
  plt.figure(num=1)    # activate first figure ...
  plt.ylim(-1., 60.)   # ... set y-limit ...
  plt.show()           # .. and show all figures

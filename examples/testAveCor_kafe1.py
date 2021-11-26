# Mittelung korrelierter Messwerte mit kafe2 / k2Fit

import numpy as np, matplotlib.pyplot as plt
import PhyPraKit as ppk

# das Modell
def fitf(x, c):
   # das Ergebnis muss ein Vektor der Länge von x sein
   if len(np.shape(x))==0:
     return c
   else:
     return c*np.ones(len(x)) 

# Daten und Unsicherheiten
x = np.arange(6)+1.
m = np.array([0.82, 0.81, 1.32, 1.44, 0.93, 0.99])
sig_u = 0.1 # unabhängig
sig_s = 0.15 # korreliert für Messungen (1,2), (3,4) und (5,6)
sig_t = 0.05 # korreliert für alle Messungen

# Konstruktion der Komponenten für die Kovarianz-Matrix
sys_y= [[sig_t, sig_t, sig_t, sig_t, sig_t, sig_t],
        [sig_s, sig_s, 0. ,0. ,0. ,0.], 
        [0., 0., sig_s, sig_s, 0., 0.], 
        [0., 0., 0., 0., sig_s, sig_s] ]

# Fit ausführen
par, pare, cor, chi2 = ppk.k2Fit(fitf, x , m, sx=0., sy=sig_u, yabscor = sys_y)

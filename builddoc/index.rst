.. PhyPraKit documentation master file, created by
   sphinx-quick-start on Sat Oct 15 18:03:17 2016. You
   can adapt this file completely to your liking, but it
   should at least contain the root `toctree` directive.

.. meta:
   :description lang=en: PhyPraKit - a collection of python modules
   for data visualization and analysis in experimental laboratory
   courses in Physics, developed at the faculty of physics at
   Karlsruhe Institute of Technology (KIT)
   :description lang=de: PhyPraKit - eine Sammlung von Funktionen in der 
   Sprache `Python` zur Visualisierung und Auswertung von Daten in den 
   physikalischen Praktika, entwickelt an der Fakultät für Physik am 
   Karlsruher Institut für Physik (KIT) 
   :robots: index, follow
   :keywords: Datenauswertung, Praktikum, Regression, Anpassung 

.. moduleauthor:
   Günter Quast <g.quast@kit.edu>


===========================
**PhyPraKit Documentation**
===========================


About
=====

     Version |release|, Date |date|

|

**PhyPraKit** is a collection of python modules
for data visualization and analysis in experimental laboratory
courses in physics, in use at the faculty of physics at
Karlsruhe Institute of Technology (KIT). As the modules are
intended primarily for use by undergraduate students in
Germany, the documentation is partly in German language,
in particular the description of the examples.

Created by: 

* Guenter Quast <guenter (dot) quast (at) online (dot) de>


A pdf version of this documentation is available here: PhyPraKit.pdf_.

.. _PhyPraKit.pdf: PhyPraKit.pdf



Installation:
-------------

To use PhyPraKit, it is sufficient to place the the directory
`PhyPraKit` and all the files in it in the same directory as the
python scripts importing it.

Installation via `pip` is also supported. After downloading, execute: 

``pip install --user .`` 

in the main directory of the *PhyPraKit* package (where *setup.py*
is located) to install in user space.  

The installation via the *whl*-package provided
in the subdirectory `dist` may also be used:

   ``pip install --user --no-cache PhyPraKit<version>.whl``

Installation via the PyPI Python Package Index is also available, simply
execute:
   
   ``pip install --user PhyPraKit``

|

|

**German Description:**

**PhyPraKit** ist eine Sammlung nützlicher Funktionen in der Sprache 
`Python (>=3.6, die meisten Module laufen auch noch mit der inzwischen
veralteten Verson 2.7)` zum Aufnehmen, zur Bearbeitung, 
Visualisierung und Auswertung von Daten in Praktika zur Physik.
Die Anwendung der verschiedenen Funktionen des Pakets   
werden jeweils durch Beispiele illustriert.
			     

.. toctree::
   :maxdepth: 2

.. |date| date:: 



**Visualization and Analysis of Measurement Data**
==================================================
Methods for recording, processing, visualization and analysis of
measurement data are required in all laboratory courses in Physics.

This collection of tools in the package `PhyPraKit` contains
functions for reading data from various sources, for data
visualization, signal processing and statistical data analysis and
model fitting as well as tools for generation of simulated data.
Emphasis was put on simple implementations, illustrating the
principles of the underlining coding.

The class *mnFit* in the module *phyFit* offers a light-weight
implementation for fitting model functions to data with uncorrelated
and/or correlated absolute and/or relative uncertainties in ordinate
and/or abscissa directions. Support for likelihood fits to binned
data (histograms) and to unbinned data is also provided.

For complex kinds of uncertainties, there are hardly any are easy-to-use
program packages. Most of the existing applications use presets aiming
at providing a parametrization of measurement data, whereby the validity
of the parametrization is assumed and the the parameter uncertainties are
scaled so that the data is well described. In physics applications, on the
contrary, testing the validity of model hypothesis is of central importance
before extracting any model parameters. Therefore, uncertainties must be
understood, modeled correctly and incorporated in the fitting procedure.

*PhyPraKit* offers adapted interfaces to the fit modules in the package
*scipy* (*optimize.curve_fit* and *ODR*) to perform fits with a test of
the validity of the hypothesis. A very lean implementation, relying on
the mimimization and uncertainty-analysis tool minuit, is also provided
in the sub-package *phyFit* for the above-mentioned use cases.
*PhyPraKit* also contains a simplified interface to the very
function-rich fitting package *kafe2*.

|

|

  **German: Darstellung und Auswertung von Messdaten**

In allen Praktika zur Physik werden Methoden zur Aufnahme, 
Bearbeitung, Darstellung und Auswertung von Messdaten benötigt.

Die vorliegende Sammlung im Paket `PhyPraKit` enthält 
Funktionen zum Einlesen von Daten aus diversen Quellen, zur 
Datenvisualisierung, Signalbearbeitung und zur statistischen
Datenauswertung und Modellanpassung sowie Werkzeuge zur Erzeugung
simulierter Daten. 
Dabei wurde absichtlich Wert auf eine einfache, die Prinzipien 
unterstreichende Codierung gelegt und nicht der möglichst effizienten 
bzw. allgemeinsten Implementierung der Vorzug gegeben.

Das Modul *phyFit* bietet mit der Klasse *mnFit* eine schlanke
Implementierung zur Anpassung von Modellfunktionen an Daten,
die mit unkorrelierten und/oder korrelierten absoluten
und/oder relativen Unsicherheiten in Ordinaten- und/oder
Abszissenrichtung behaftet sind. Anpassungen an gebinnte Daten
(Histogramme) und Maxium-Likelihood-Anassungen zur Bestimmung der
Parameter der Verteilung von Daten werden ebenfalls unterstützt.
Für solche in der Physik häufig auftretenden komplexen Formen von
Unsicherheiten gibt es kaum andere, einfach zu verwendende
Programmpakete. Viele Pakte sind als Voreinstellung auf
die Parametrisierung von Messdaten ausgelegt, wobei die
Parameterunsicherheiten unter Annahme der Gültigkeit der
Parametrisierung so skaliert werden, dass die Daten gut
repräsentiert werden. *PhyPraKit* bietet daher entsprechend 
angepasste Interfaces zu den Fitmodulen im Paket *scipy*
(*optimize.curve_fit* und *ODR*), um Anpassungen mit Test der 
Gültigkeit der Modellhypothese durchzuführen. *PhyPraKit* enthält 
ebenfalls ein vereinfachtes Interface zum sehr funktionsreichen 
Anpassungspaket *kafe2*.

In der Vorlesung "Computergestützte Datenauswertung" an der Fakultät
für Physik am Karlsruher Institut für Physik 
(http://www.etp.kit.edu/~quast/CgDA)
werden die in *PhyPraKit* verwendeten Methoden eingeführt und beschrieben.
Hinweise zur Installation der empfohlenen Software finden sich unter den Links
http://www.etp.kit.edu/~quast/CgDA/CgDA-SoftwareInstallation-html und 
http://www.etp.kit.edu/~quast/CgDA/CgDA-SoftwareInstallation.pdf     

Speziell für das "Praktikum zur klassischen Physik" am KIT gibt es 
eine  kurze Einführung in die statistischen Methoden und Werkzeuge
(http://www.etp.kit.edu/~quast/CgDA/PhysPrakt/CgDA_APraktikum.pdf). 



Dokumentation der Beispiele
===========================

**`PhyPraKit.py`** ist ein Paket mit nützlichen Hilfsfunktionen
zum import in eigene Beispiele mittels::

     import PhyPraKit as ppk

oder::

     from PhyPraKit import ... 


**PhyPraKit** enthält folgende **Funktionen**:

      1. Daten-Ein und -Ausgabe
    
        - readColumnData() Daten und Meta-Daten aus Textdatei lesen
        - readCSV()        Daten im csv-Format aus Datei mit Header lesen
        - readtxt()        Daten im Text-Format aus Datei mit Header lesen
        - readPicoScope()  mit PicoScope exportierte Daten einlesen 
        - readCassy()      mit CASSY im .txt-Format exportierte Dateien einlesen
        - labxParser()     mit CASSY im .labx-Format exportierte Dateien einlesen
        - writeCSV()       Daten csv-Format schreiben (optional mit Header)
        - writeTexTable()  Daten als LaTeX-Tabelle exportieren
        - round_to_error() Runden von Daten mit Präzision wie Unsicherheit 
        - ustring()        korrekt gerundete Werte v +/- u als Text;   
          alternativ: der Datentyp *ufloat(v, u)* im Paket *uncertainties*
	  unterstützt die korrekte Ausgabe von Werten *v* mit Unsicherheiten *u*. 

	  
      2. Signalprozessierung:
    
        - offsetFilter()      Abziehen eines off-sets 
        - meanFilter()        gleitender Mittelwert zur Glättung
        - resample()          Mitteln über n Datenwerte
        - simplePeakfinder()  Auffinden von Maxima (Peaks) und Minima 
          (`Empfehlung: convolutionPeakfinder nutzen`)
        - convolutionPeakfinder() Finden von  Maxima 
        - convolutionEdgefinder() Finden von Kanten
        - Fourier_fft()       schnelle Fourier-Transformation (FFT)
        - FourierSpectrum()   Fourier-Transformation
          `(langsam, vorzugsweise FFT-Version nutzen)`
        - autocorrelate()     Autokorrelation eines Signals
    
      3. Statistik:
    
        - wmean()                  Berechnen des gewichteten Mittelwerts
        - BuildCovarianceMatrix()  Kovarianzmatrix aus Einzelunsicherheiten
        - Cov2Cor()                Konversion Kovarianzmatrix -> Korrelationsmatrix
        - Cor2Cov()                Konversion Korrelationsmatrix +
	  Unsicherheiten -> Kovarianzmatrix
        - chi2prob()               Berechnung der chi^2-Wahrscheinlichkeit 
        - propagatedError()        Numerische Fehlerfortpflanzung;  
          Hinweis: der Datentyp *ufloat(v, u)* im Paket *uncertainties*
	  unterstützt Funktionen von Werten *v* mit Unsicherheiten *u* und
	  die korrekte Fehlerfortpflanzung  
        - getModelError()          Numerische Fehlfortpflanzung für 
	  parameterabhängige Funktionswerte 
    
      4. Histogramm:
    
        - barstat()   statistisch Information aus Histogramm (Mittelwert,
	  Standardabweichung, Unsicherheit des Mittelwerts)
        - nhist()    Histogramm-Grafik mit np.historgram() und plt.bar()  
          `(besser matplotlib.pyplot.hist() nutzen)`
        - histstat() statistische Information aus 1d-Histogram
        - nhist2d()  2d-Histogramm mit np.histrogram2d, plt.colormesh()  
          `(besser matplotlib.pyplot.hist2d() nutzen)`  
        - hist2dstat() statistische Information aus 2d-histogram
        - profile2d()  "profile plot" für 2d-Streudiagramm
        - chi2p_indep2d() chi^2-Test auf Unabhängigkeit zweier Variabler
        - plotCorrelations() Darstellung von Histogrammen und Streudiagrammen
	  von Variablen bzw. Paaren von Variablen eines multivariaten
	  Datensatzes
	  
      5. Lineare Regression und Anpassen von Funktionen:

        - linRegression()    lineare Regression, y=ax+b, mit analytische Formel
        - odFit()            Funktionsanpassung mit x- und y-Unsicherheiten (scipy ODR)
        - xyFit()            Funktionsanpassung mit (korrelierten) x- und y-Unsicherheiten mit *phyFit*
        - hFit()             Anpassung einer Verteilungsdichte an Histogramm-Daten
        - k2Fit()            Funktionsanpassung mit (korrelierten) x- und y-Unsicherheiten
          mit dem Paket *kafe2*
        - hFit()             Maximum-Likelihood-Anpassung einer Verteilungsdichte an Histogramm-Daten
        - mFit()             Anpassung einer Nutzerdefinierten Kostenfunktion, oder einer
	  Verteilungsdichte an ungebinnete Daten mit der maximum-likelood Methode

      6. Erzeugung simulierter Daten mit MC-Methode:
    
        - smearData()          Addieren von zufälligen Unsicherheiten auf Eingabedaten
        - generateXYdata()     Erzeugen simulierter Datenpunkte (x+Delta_x, y+Delta_y)


Die folgenden **Beispiele** illustrieren die Anwendung:

  * `test_readColumnData.py` ist ein Beispiel zum
    Einlesen von Spalten aus Textdateien; die zugehörigen 
    *Metadaten* können ebenfalls an das Script übergeben 
    werden und stehen so bei der Auswertung zur Verfügung.

  * `test_readtxt.py` liest Ausgabedateien im allgemeinem `.txt`-Format; 
    ASCII-Sonderzeichen außer dem Spalten-Trenner werden ersetzt,
    ebenso wie das deutsche Dezimalkomma durch den Dezimalpunkt

  * `test_readPicoScope.py` liest Ausgabedateien von USB-Oszillographen 
    der Marke PicoScope im Format `.csv` oder `.txt`.

  * `test_labxParser.py` liest Ausgabedateien von Leybold
    CASSY im `.labx`-Format. Die Kopfzeilen und Daten von Messreihen 
    werden als Listen in *Python* zur Verfügung gestellt. 

  * `test_convolutionFilter.py` liest die Datei `Wellenform.csv` und 
    bestimmt Maxima und fallende Flanken des Signals.
    
  * `test_AutoCorrelation.py` liest die Datei `AudioData.csv` und führt 
    eine Analyse der Autokorrelation zur Frequenzbestimmung durch.
    
  * `test_Fourier.py` illustriert die Durchführung einer 
    Fourier-Transfomation eines periodischen Signals, das in 
    der PicoScope-Ausgabedatei `Wellenform.csv` enthalten ist.
    
  * `test_propagatedError.py` illustriert die Anwendung von numerisch
    berechneter Fehlerfortpflanzung und korrekter Rundung von Größen
    mit Unsicherheit
    
  * `test_linRegression.py` ist eine einfachere Version mit
    `python`-Bordmitteln zur Anpassung einer Geraden an
    Messdaten mit Fehlern in Ordinaten- und Abszissenrichtung. 
    Korrelierte Unsicherheiten werden nicht unterstützt.
    
  * `test_xyFit` dient zur Anpassung einer beliebigen Funktion an
    Messdaten mit Fehlern in Ordinaten- und Abszissenrichtung und mit
    allen Messpunkten gemeinsamen (d. h. korrelierten) relativen oder
    absoluten systematischen Fehlern. Dazu wird das Paket imunit
    verwendet, das den am CERN entwickelten Minimierer MINUIT nutzt.
    Da die Kostenfunktion frei definiert und auch während der Anpassung
    dynamisch aktualisiert werden kann, ist die Implementierung von
    Parameter-abhängigen Unsicherheiten möglich. Ferner unterstützt
    iminuit die Erzeugung und Darstellung von Profil-Likelihood-Kurven
    und Konfidenzkonturen, die so mit `xyFit` ebenfalls dargestellt
    werden können.
    
  * `test_k2Fit.py` verwendet die Version *kafe2* zur Anpassung einer
    Funktion an Messdaten mit unabhängigen oder korrelierten relativen oder
    absoluten Unsicherheiten in Ordinaten- und Abszissenrichtung.
    
  * `test_simplek2Fit.py` illustriert die Durchführung einer einfachen
    linearen Regression mit *kafe2* mit einer minimalen Anzahl eigener
    Codezeilen.
    
  * `test_hFit` illustriert die Anpassung einer Verteilungsdichte an
    histogrammierte Daten. Die Kostenfunktion für die Minimierung ist das
    zweifache der negativen log-Likelihood-Funktion der Poisson-Verteilung,
    Poiss(k; lam), oder - optional - ihrer Annäherung durch eine
    Gauß-Verteilung mit Gauss(x, mu=lam, sig**2=lam). Die Unsicherheiten
    werden aus der Modellvorhersage bestimmt, um auch Bins mit null Einträgen
    korrekt behandeln zu können. Grundsätzlich wird eine normierte
    Verteilungsdichte angepasst; es ist aber optional auch möglich, die
    Anzahl der Einträge mit zu berücksichtigen, um so z. B. ggf. die
    Poisson-Unsicherheit der Gesamtanzahl der Histogrammeinträge zu
    berücksichtigen.
    
  * `test_mlFit` illustriert die Anpassung einer Verteilungsdichte an
    ungebinnte Daten mit der maximum-likelihood Methode. Die Kostenfunktion
    für die Minimierung ist der negative natürliche Logarithmus der vom
    Nutzer agegebenen Verteilungsdichte (oder, optional, deren Zweifaches).

  * `test_Histogram.py` ist ein Beispiel zur Darstellung und 
    statistischen Auswertung von Häufigkeitsverteilungen (Histogrammen) 
    in ein oder zwei Dimensionen.

  * `test_generateXYata.py` zeigt, wie man mit Hilfe von Zufallszahlen 
    "künstliche Daten" zur Veranschaulichung oder zum Test von Methoden
    zur Datenauswertung erzeugen kann. 

  * `toyMC_Fit.py` führt eine große Anzahl Anpassungen an simulierte
    Daten durch. Durch Vergleich der wahren Werte mit den aus der
    Anpassung bestimmten Werten lassen sich Verzerrungen der
    Parameterschätzungen, Korrelationen der Parameter oder die Form
    der Verteilung der Chi2-Wahrscheinlichkeit überprüfen, die im
    Idealfall eine Rechteckverteilung im Intervall [0,1] sein sollte. 

  Die folgenden *python*-Skripte sind etwas komplexer und illustrieren 
  typische Anwendungsfälle der Module in `PhyPraKit`:

  * `Beispiel_Diodenkennlinie.py` demonstriert die Analyse einer
    Strom-Spannungskennlinie am Beispiel von (künstlichen) Daten,
    an die die Shockley-Gleichung angepasst wird. Typisch für
    solche Messungen über einen weiten Bereich von Stromstärken
    ist die Änderung des Messbereichs und damit der Anzeigegenauigkeit
    des verwendeten Messgeräts. Im steil ansteigenden Teil der
    Strom-Spannungskennlinie ist es außerdem wichtig, auch die Unsicherheit
    der auf der x-Achse aufgetragen Spannungsmessungen zu berücksichtigen.
    Eine weitere Komponente der Unsicherheit ergibt sich aus der
    Kalibrationsgenauigkeit des Messgeräts, die als relative,
    korrelierte Unsicherheit aller Messwerte berücksichtigt werden
    muss. Das Beispiel zeigt, wie man in diesem Fall die Kovarianzmatrix
    aus Einzelunsicherheiten aufbaut. Die Funktionen *k2Fit()* und
    *mfit()* bieten dazu komfortable und leicht zu verwendende
    Interfaces, deren Anwendung zur Umsetzung des komplexen Fehlermodells
    in diesem Beispiel gezeigt wird. 
    
  * `Beispiel_Drehpendel.py` demonstriert die Analyse von am Drehpendel
    mit CASSY aufgenommenen Daten. Enthalten sind einfache Funktionen zum
    Filtern und Bearbeiten der Daten, zur Suche nach Extrema und Anpassung 
    einer Einhüllenden, zur diskreten Fourier-Transformation und zur 
    Interpolation von Messdaten mit kubischen Spline-Funktionen. 

  * `Beispiel_Hysterese.py` demonstriert die Analyse von Daten,
    die mit einem USB-Oszilloskop der Marke `PicoScope` am
    Versuch zur Hysterese aufgenommen wurden. Die aufgezeichneten Werte 
    für Strom und B-Feld werden in einen Zweig für steigenden und 
    fallenden Strom aufgeteilt, mit Hilfe von kubischen Splines 
    interpoliert und dann integriert. 

  * `Beispiel_Wellenform.py`  zeigt eine typische Auswertung
    periodischer Daten am Beispiel der akustischen Anregung eines 
    Metallstabs. Genutzt werden Fourier-Transformation und
    eine Suche nach charakteristischen Extrema. Die Zeitdifferenzen
    zwischen deren Auftreten im Muster werden bestimmt, als
    Häufgkeitsverteilung dargestellt und die Verteilungen statistisch
    ausgewertet.

  * `Beispiel_GammaSpektroskopie.py` liest mit dem Vielkanalanalysator
    des CASSY-Systems im `.labx` -Format gespeicherten Dateien ein
    (Beispieldatei `GammaSpektra.labx`).


Module Documentation 
====================

..  automodule:: PhyPraKit
     :imported-members:
     :members:

..  automodule:: PhyPraKit.phyFit
     :members:
	
..  automodule:: test_readColumnData 

..  automodule:: test_readtxt

..  automodule:: test_readPicoScope

..  automodule:: test_labxParser

..  automodule:: test_Histogram

..  automodule:: test_convolutionFilter

..  automodule:: test_AutoCorrelation

..  automodule:: test_Fourier

..  automodule:: test_propagatedError  

..  automodule:: test_odFit

..  automodule:: test_xyFit

..  automodule:: test_k2Fit

..  automodule:: test_generateData

..  automodule:: toyMC_Fit

..  automodule:: Beispiel_Diodenkennlinie

..  automodule:: Beispiel_Drehpendel

..  automodule:: Beispiel_Hysterese

..  automodule:: Beispiel_Wellenform

..  automodule:: Beispiel_GammaSpektroskopie

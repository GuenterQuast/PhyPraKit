.. PhyPraKit documentation master file, created by
   sphinx-quick-start on Sat Oct 15 18:03:17 2016. You
   can adapt this file completely to your liking, but it
   should at least contain the root `toctree` directive.

.. meta:
   :description lang=en: PhyPraKit - a collection of python modules
   for data visualization and analysis in experimental laboratory
   courses in Physics, developed in the Department of Physics at
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


**PhyPraKit** is a collection of python modules
for data visualization and analysis in experimental laboratory
courses in physics and is in use in the Department of Physics
at Karlsruhe Institute of Technology (KIT).
As the modules are intended primarily for use by undergraduate
students in Germany, the documentation is partly in German language,
in particular the description of the examples.

Created by: 

* Guenter Quast <guenter (dot) quast (at) online (dot) de>


A pdf version of this documentation is available here: PhyPraKit.pdf_.

.. _PhyPraKit.pdf: PhyPraKit.pdf



Installation:
-------------

To use PhyPraKit, it is sufficient to place the directory
`PhyPraKit` and all the files in it in the same directory as the
python scripts importing it.

Installation via `pip` is also supported.
After downloading, execute: 

``pip install --user .`` 

in the main directory of the *PhyPraKit* package (where *setup.py*
is located) to install in user space.  

Comfortable installation via the PyPI Python Package Index
is also possible by executing
   
   ``pip install --user PhyPraKit``

The installation via the *whl*-package provided
in the subdirectory `dist` may alternatively be used:

   ``pip install --user --no-cache PhyPraKit<version>.whl``

*python* scripts and *Jupyter* notebook versions illustrate common
use cases of the package and provide examples of typical applications. 


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
model fitting as well as tools for the generation of simulated data.
Emphasis was put on simple implementations, illustrating the
principles of the underlying algorithms.

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
*scipy* (*optimize.curve_fit* and *ODR*) to perform fits including a test
of the validity of the model hypothesis. A very lean implementation, relying
on the mimimization and uncertainty-analysis tool *MINUIT*, is also provided
in the sub-package *phyFit* for the above-mentioned use cases.
*PhyPraKit* also contains a simplified interface to the very
function-rich fitting package *kafe2*.


  **German: Darstellung und Auswertung von Messdaten**

In allen Praktika zur Physik werden Methoden zur Aufnahme, 
Bearbeitung, Darstellung und Auswertung von Messdaten benötigt.

Die vorliegende Sammlung im Paket `PhyPraKit` enthält 
Funktionen zum Einlesen von Daten aus diversen Quellen,
zur Signalbearbeitung und Datenvisualisierung und zur
statistischen Datenauswertung und Modellanpassung sowie
Werkzeuge zur Erzeugung simulierter Pseudo-Daten. 
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
repräsentiert werden. Um den besonderen Anforderungen in
der Physik Rechnung zu tragen, bietet *PhyPraKit* deshalb
entsprechend  angepasste Interfaces zu den Fitmodulen im
Paket *scipy* (*optimize.curve_fit* und *ODR*), um Anpassungen
mit Test der  Gültigkeit der Modellhypothese durchzuführen.
*PhyPraKit* enthält ebenfalls ein vereinfachtes Interface zum sehr
funktionsreichen Anpassungspaket *kafe2*.

In der Vorlesung "Computergestützte Datenauswertung" an der Fakultät
für Physik am Karlsruher Institut für Physik 
(http://www.etp.kit.edu/~quast/CgDA)
werden die in *PhyPraKit* verwendeten Methoden eingeführt und
beschrieben.
Hinweise zur Installation der empfohlenen Software finden sich
unter den Links
http://www.etp.kit.edu/~quast/CgDA/CgDA-SoftwareInstallation-html und 
http://www.etp.kit.edu/~quast/CgDA/CgDA-SoftwareInstallation.pdf .

Speziell für das "Praktikum zur klassischen Physik" am KIT gibt es 
eine  kurze Einführung in die statistischen Methoden und Werkzeuge
unter dem Link
http://www.etp.kit.edu/~quast/CgDA/PhysPrakt/CgDA_APraktikum.pdf .

Über den Link
http://www.etp.kit.edu/~quast/jupyter/jupyterTutorial.html
werden eine Einführung in die Verwendung von Jupyter Notebooks
sowie Tutorials für verschiedene Aspekte der statistischen
Datenauswertung mit Beispielen zum Einsatz von Modulen aus
*PhyPraKit* bereit gestellt.


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
        - chi2prob()               Berechnung der :math:`\chi^2`-Wahrscheinlichkeit 
        - propagatedError()        Numerische Fehlerfortpflanzung;  
          Hinweis: der Datentyp *ufloat(v, u)* im Paket *uncertainties*
	  unterstützt Funktionen von Werten *v* mit Unsicherheiten *u* und
	  die korrekte Fehlerfortpflanzung  
        - getModelError()          Numerische Fehlfortpflanzung für 
	  parameterabhängige Funktionswerte 
    
      4. Histogramme:
    
        - barstat()   statistisch Information aus Histogramm (Mittelwert,
	  Standardabweichung, Unsicherheit des Mittelwerts)
        - nhist()    Histogramm-Grafik mit np.historgram() und plt.bar()  
          `(besser matplotlib.pyplot.hist() nutzen)`
        - histstat() statistische Information aus 1d-Histogram
        - nhist2d()  2d-Histogramm mit np.histrogram2d, plt.colormesh()  
          `(besser matplotlib.pyplot.hist2d() nutzen)`  
        - hist2dstat() statistische Information aus 2d-histogram
        - profile2d()  "profile plot" für 2d-Streudiagramm
        - chi2p_indep2d() :math:`\chi^2`-Test auf Unabhängigkeit von zwei Variablen
        - plotCorrelations() Darstellung von Histogrammen und Streudiagrammen
	  von Variablen bzw. Paaren von Variablen eines multivariaten
	  Datensatzes
	  
      5. Lineare Regression und Anpassen von Funktionen:

        - linRegression()    lineare Regression, y=ax+b, mit analytische Formel
        - odFit()            Funktionsanpassung mit x- und y-Unsicherheiten (scipy ODR)
        - xyFit()            Funktionsanpassung an Datenpunkte (x_i, y_i=f(x_i)) mit
	  (korrelierten) x- und	y-Unsicherheiten mit *phyFit* 
        - hFit()             maximum-likelihood-Anpassung einer
	  Verteilungsdichte an Histogramm-Daten mit *phyFit*
        - mFit()             Anpassung einer Nutzerdefinierten Kostenfunktion oder einer
	  Verteilungsdichte an ungebinnete Daten mit der maximum-likelood
	  Methode (mit *phyFit*)
	- xFit()            Anpassung eines Modells an indizierte Daten
	  x_i=x_i(x_j, \*par) mit *phyFit*
        - k2Fit()            Funktionsanpassung mit (korrelierten) x- und y-Unsicherheiten
          mit dem Paket *kafe2* an Datenpunkte (x_i , y_i=f(x_i))


      6. Erzeugung simulierter Daten mit MC-Methode:
    
        - smearData()          Addieren von zufälligen Unsicherheiten auf Eingabedaten
        - generateXYdata()     Erzeugen simulierter Datenpunkte (x+Delta_x, y+Delta_y)


Die folgenden **Beispiele** dienen der Illustration der Anwendung der
zahlreichen Funktionen. 
Eine direkt im Browser ausführbare Installation von *PhyPraKit* gibt es auf 
`mybinder.org
<https://mybinder.org/v2/git/https%3A%2F%2Fgit.scc.kit.edu%2Fyh5078%2FPhyPraKit/master>`_.

**Beispiele zur Anwendung der Module aus PhyPraKit**

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
    Messdaten mit Unsicherheiten in Ordinaten- und Abszissenrichtung. 
    Korrelierte Unsicherheiten werden nicht unterstützt.
    
  * `test_xyFit` dient zur Anpassung einer beliebigen Funktion an
    Messdaten mit Unsicherheiten in Ordinaten- und Abszissenrichtung
    und mit allen Messpunkten gemeinsamen (d. h. korrelierten) relativen
    oder absoluten systematischen Fehlern. Dazu wird das Paket *imunit*
    verwendet, das den am CERN entwickelten Minimierer MINUIT nutzt.
    Da die Kostenfunktion frei definiert und auch während der Anpassung
    dynamisch aktualisiert werden kann, ist die Implementierung von
    Parameter-abhängigen Unsicherheiten möglich. Ferner unterstützt
    *iminuit* die Erzeugung und Darstellung von Profil-Likelihood-Kurven
    und Konfidenzkonturen, die so mit `xyFit` ebenfalls dargestellt
    werden können.
    
  * `test_k2Fit.py` verwendet das funktionsreiche Anpassungspaket *kafe2*
    zur Anpassung einer Funktion an Messdaten mit unabhängigen oder
    korrelierten relativen oder absoluten Unsicherheiten in Ordinaten-
    und Abszissenrichtung.
    
  * `test_simplek2Fit.py` illustriert die Durchführung einer einfachen
    linearen Regression mit *kafe2* mit einer minimalen Anzahl eigener
    Codezeilen.
    
  * `test_k2hFit.py` führt eine Anpassung einer Verteilungsdichte an
    Histogrammdaten mit *kafe2* durch. Die Kostenfunktion ist das
    zweifache der negativen log-Likelihood-Funktion der Poisson-Verteilung,
    Poiss(k; lam), oder - optional - ihrer Annäherung durch eine
    Gauß-Verteilung mit Gauss(x, mu=lam, sig**2=lam). Die Unsicherheiten
    werden aus der Modellvorhersage bestimmt, um auch Bins mit wenigen oder
    sogar null Einträgen korrekt zu behandeln.
    
  * `test_hFit` illustriert die Anpassung einer Verteilungsdichte an
    histogrammierte Daten. Die Kostenfunktion für die Minimierung ist das
    zweifache der negativen log-Likelihood-Funktion der Poisson-Verteilung,
    Poiss(k; lam), oder - optional - ihrer Annäherung durch eine
    Gauß-Verteilung mit Gauss(x, mu=lam, sig**2=lam). Die Unsicherheiten
    werden aus der Modellvorhersage bestimmt, um auch Bins mit wenigen oder
    sogar null Einträgen korrekt zu behandeln. Grundsätzlich wird eine normierte
    Verteilungsdichte angepasst; es ist aber optional auch möglich, die
    Anzahl der Einträge mit zu berücksichtigen, um so z. B. die
    Poisson-Unsicherheit der Gesamtanzahl der Histogrammeinträge zu
    berücksichtigen.
    
  * `test_mlFit` illustriert die Anpassung einer Verteilungsdichte an
    ungebinnte Daten mit der maximum-likelihood Methode. Die Kostenfunktion
    für die Minimierung ist der negative natürliche Logarithmus der vom
    Nutzer agegebenen Verteilungsdichte (oder, optional, deren Zweifaches).

  * `test_xFit` ist ein Beispiel für eine Anpassung einer Modellvorhersage
    an allgemeine Eingabedaten ("indizierte Daten" *x_1, ..., x_n*). Dabei
    sind die x_i Funktionen der Parameter p_i einer Modellvorhersage, und ggf.
    auch von Elementen x_j der Eingabedaten: x_i(x_j, \*par). In diesem
    Beispiel werden zwei Messungen eines Ortes in Polarkoordinaten gemittelt
    und in kartesische Koordinaten umgerechnet. Bei dieser nicht-linearen
    Transformation weden sowohl die Zentralwerte als auch Konfidenzkonturen
    korrekt bestimmt. 
    
  * `test_Histogram.py` ist ein Beispiel zur Darstellung und 
    statistischen Auswertung von Häufigkeitsverteilungen (Histogrammen) 
    in ein oder zwei Dimensionen.

  * `test_generateXYata.py` zeigt, wie man mit Hilfe von Zufallszahlen 
    "künstliche Daten" zur Veranschaulichung oder zum Test von Methoden
    zur Datenauswertung erzeugen kann. 

  * `toyMC_Fit.py` führt eine große Anzahl Anpassungen an simulierte
    Daten durch. Durch Vergleich der wahren Werte mit den aus der
    Anpassung bestimmten Schätzwerte und deren Unsicherheiten lassen
    sich Verzerrungen der Parameterschätzungen, die korrekte Überdeckung
    der in der Anpassung geschätzen Konfidenzbereiche für die Parameter,
    Korrelationen der Parameter oder die Form der Verteilung der
    :math:`\chi^2`-Wahrscheinlichkeit überprüfen, die im Idealfall
    eine Rechteckverteilung im Intervall [0,1] sein sollte. 

**Komplexere Beispiele für konkrete Anwendungen in Praktika**
  
  Die folgenden *python*-Skripte sind etwas komplexer und illustrieren 
  typische Anwendungsfälle der Module in `PhyPraKit`:

  * `Beispiel_Diodenkennlinie.py` demonstriert die Analyse einer
    Strom-Spannungskennlinie am Beispiel von (künstlichen) Daten,
    an die die Shockley-Gleichung angepasst wird. Typisch für
    solche Messungen über einen weiten Bereich von Stromstärken
    ist die Änderung des Messbereichs und damit der Anzeigegenauigkeit
    des verwendeten Messgeräts. Im steil ansteigenden Teil der
    Strom-Spannungskennlinie dominieren die Unsicherheiten
    der auf der x-Achse aufgetragen Spannungsmesswere. 
    Eine weitere Komponente der Unsicherheit ergibt sich aus der
    Kalibrationsgenauigkeit des Messgeräts, die als relative,
    korrelierte Unsicherheit aller Messwerte berücksichtigt werden
    muss. Das Beispiel zeigt, wie man in diesem Fall die Kovarianzmatrix
    aus Einzelunsicherheiten aufbaut. Die Funktionen *k2Fit()* und
    *xyfit()* bieten dazu komfortable und leicht zu verwendende
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

  * `Beispiel_Multifit.py` illustiert die simultane Anpassung von Parametern
    an mehrere, gleichartige Messreihen, die mit *kafe2* möglich ist.
    Ein Anwendungsfall sind mehrere Messreihen mit der gleichen Apparatur,
    um die Eigenschaften von Materialien in Proben mit unterschiedlicher
    Geometrie zu bestimmen, wie z. B. die Elastizität oder den spezifischen
    Widerstand an Proben mit unterschiedlichen Querschnitten und Längen.
    Auf die Apparatur zurückzuführende Unsicherheiten sind in allen Messreihen
    gleich, auch die interessierende Materialeigenschaft ist immer die
    gleiche, lediglich die unterschiedlichen Gemoetrie-Parameter und die
    jeweils bestimmten Werte der Messreihen haben eigene, unabhängige
    Unsicherheiten.

  * `Beispiel_GeomOptik.py` zeigt, wie man mittels Parametertransformation
    die Einzelbrennweiten der beiden Linsen eines Zwei-Linsensystems
    aus der Systembrennweite und den Hauptebenenlagen bestimmen kann.
    Dabei wird neben der Transformation auf den neuen Parametersatz
    auch eine Mittelung über mehrere Messreihen durchgeführt, deren
    Ergebnisse ihrerseits aus Anpassungen gewonnen wurden. Die
    Paramtertransformation wird als Anpassungsproblem mit einer
    :math:`\chi^2` Kostenfunktion behandelt und so auch die Konfidenzkonturen
    der neuen Parameter bestimmt. 
  
  * `Beispiel_GammaSpektroskopie.py` liest mit dem Vielkanalanalysator
    des CASSY-Systems im `.labx` -Format gespeicherten Dateien ein
    (Beispieldatei `GammaSpektra.labx`).

    
**Daten darstellen mit dem Skript plotData.py**

  Mitunter ist eine einfache und unkomplizierte Darstellung von Daten
  erwünscht, ohne speziellen *Python*-Code zu erstellen. Damit das
  funktioniert, müssen die Daten in einem Standardformat vorliegen.
  Dazu empfiehlt sich die Datenbeschreibungssprache *yaml*, mit der
  auch die notwendigen "Meta-Daten" wie Titel, Art des Datensatzes und
  der auf der x- und y-Achse darzustellenden Daten angegeben werden
  können. Die Daten und deren Unsicherheiten werden als Liste von durch
  Kommata getrennten Dezimalzahlen (mit Dezimalpunkt!) angegeben.

  Das Skript *plotData.py* unterstützt die Darstellun von Datenpunkten
  (x,y) mit Unsicherheiten und Histogramme. Die Beispieldateien 'data.yaml'
  und hData.yaml erläutern das unterstützte einfache Datenformat.

  Für (x,y)-Daten:
  
  .. code-block:: yaml

     title: <title of plot>
     x_label: <label for x-axis>
     y_label: <label for y-axis>

     label: <name of data set>
     x_data: [ x1, x2, ... , xn]
     y_data: [ y1, y2, ... , yn ]
     x_errors: x-uncertainty or [ex1, ex2, ..., exn]
     y_errors: y-uncertainty or [ey1, ey2, ..., eyn]

     Bei Eingabe von mehreren Datensätzen werden diese getrennt durch
     ...
     ---  
     label: <name of 2nd data set>
     x_data: [ 2nd set of x values ]
     y_data: [ 2nd set of y values ]
     x_errors: x-uncertainty or [x-uncertainties]
     y_errors: y-uncertainty or [y-uncertainties]

     
  und für Histogrammdaten:

  .. code-block:: yaml

    title: <title of plot>
    x_label: <label for x-axis>
    y_label: <label for y-axis>
    label: <name of data set>
    raw_data: [x1, ... , xn]
    # define binning
    n_bins: n
    bin_range: [x_min, x_max]
    #   alternatively: 
    # bin edges: [e0, ..., en]

    # wie oben ist Eingabe von mehreren Datensätzen möglich, getrennt durch
    ...
    ---  

    
  Zur Ausführung dient die Eingabe von
  
  ``python3 plotData.py [option] <yaml.datei>``

  auf der Kommandozeile.
  ``python3 plotData.py -h`` gibt die unterstützen Optionen aus. 


 
**Einfache Anpassungen mit run_phyFit.py**
  
  Die notwendigen Informationen zur Durchführung von Anpassungen
  können ebenfalls als Datei angegeben werden, die in der
  Datenbeschreibungssprache *yaml* erstellt wurden. 

  Zur Ausführung dient die Eingabe von
  
  ``python3 run_phyFit.py [option] <yaml.datei>``

  auf der Kommandozeile.

  * `simpleFit.fit` zeigt am Beispiel der Anpassung einer Parabel,
    wie mit ganz wenigen Eingaben eine Anpassung durchgeführt werden kann.
    
  * `xyFit.fit` ist ein komplexeres Beispiel, das alle *phyFit*
    unterstützten Arten von Unsicherheiten (d.h. x/y, absolut/relativ und
    unabhängig/korreliert) enthält; relative Unsicherheiten werden dabei
    auf den Modellwert und nicht auf die gemessenen Datenpunkte bezogen. 

  * `hFit.fit` zeigt die Anpassung einer Gaußverteilug an histogrammierte
    Daten.
    
**Anpassungen mit kafe2go**
    
  Alternativ kann auch das Skript *kafe2go* aus dem Paket *kafe2*, verwendet
  werden, mit dem ebenfalls Anpassungen von Modellen an Messdaten ohne eigenen
  *Python*-Code erstellt weden können. Ausgeführt wird die Anpassung durch
  Eingabe von
  
  ``kafe2go [option] <yaml.datei>``

  auf der Kommandozeile.

  Für Nutzer von MS-Windows gibt es eine ausführbare Datei, `kafe2go.exe`.
  Zur Anwendung wird eine der Beispieldateien mit Rechtsklick angewählt 
  und mit der Anwendung `kafe2go.exe` geöffnet.
  
  * `kafe2go_simleFit.fit` zeigt am Beispiel der Anpassung einer Parabel,
    wie mit ganz wenigen Eingaben eine Anpassung durchgeführt werden kann.
    
  * `kafe2go_xyFit.fit` ist ein komplexeres Beispiel, das alle von *kafe2*
    unterstützten Arten von Unsicherheiten (d.h. x/y, absolut/relativ und
    unabhängig/korreliert) enthält; relative Unsicherheiten werden dabei
    auf den Modellwert und nicht auf die gemessenen Datenpunkte bezogen. 
  
    
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

..  automodule:: test_xFit

..  automodule:: test_k2Fit

..  automodule:: test_simplek2Fit

..  automodule:: test_k2hFit
		 		 
..  automodule:: test_generateData

..  automodule:: toyMC_Fit

..  automodule:: Beispiel_MultiFit

..  automodule:: Beispiel_Diodenkennlinie

..  automodule:: Beispiel_Drehpendel

..  automodule:: Beispiel_Hysterese

..  automodule:: Beispiel_Wellenform

..  automodule:: Beispiel_GeomOptik

..  automodule:: Beispiel_GammaSpektroskopie

..  automodule:: run_phyFit

..  automodule:: plotData
		 

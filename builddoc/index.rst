.. PhyPraKit documentation master file, created by
   sphinx-quickstart on Sat Oct 15 18:03:17 2016. You
   can adapt this file completely to your liking, but it
   should at least contain the root `toctree` directive.

.. meta:
   :description lang=en: PhyPraKit - a collection of pyhton modules
   for data visialisation and analysis in experimental laboratory
   cources in Physics, developed at the faculty of physics at
   Karlsruhe Institute of Technology (KIT)
   :description lang=de: PhyPraKit - eine Sammlung von Funktionen in der 
   Sprache `python` zur Visualisierung und Auswertung von Daten in den 
   physikalischen Praktika, entwickelt an der Fakultät für Physik am 
   Karlsruher Institut für Physik (KIT) 
   :robots: index, follow
   :keywords: Datenauswertung, Praktikum, Regression, Anpassung 

.. moduleauthor:
   Günter Quast <g.quast@kit.edu>


**PhyPraKit Documentation**
===========================

                                    `Version` |date|

=====
About
=====

**PhyPraKit** is a collection of python modules
for data visialisation and analysis in experimental laboratory
cources in physics, in use at the faculty of physics at
Karlsruhe Institute of Technology (KIT). As the modules are
intended primarily for use by undertraduate students in
Germany, the documentation is partly in German language,
in particular the desctiption of the examples.

Cerated by: 

* Guenter Quast <guenter (dot) quast (at) online (dot) de>


A pdf version of this documentation is available here: PhyPraKit.pdf_.

.. _PhyPraKit.pdf: PhyPraKit.pdf



Installation:
-------------

To use PhyPraKit, it is sufficient to place the the direcotory
`PhyPraKit` and all the files in it in the same directory as the
python scripts importing it.

Installation via `pip` is also supported. The recommendation is
to use the installation package in the subdirectory `dist` and
install in user space:

   ``pip install --user --no-cache PhyPraKit<vers.>``


Übersicht:
----------

**PhyPraKit** ist eine Sammlung nützlicher Funktionen in der Sprache 
`Python (>=3.6, die meisten Module laufen auch noch mit der inzwischen
veralteten Verson 2.7)` zum Aufnehmen, zur Bearbeitung, 
Visualisierung und Auswertung von Daten in den physikalischen 
Praktika. Die Anwendung der verschiedenen Funktionen des Pakets   
werden jeweils durch Beispiele illustriert.
			     

.. toctree::
   :maxdepth: 2

.. |date| date:: 


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

**Darstellung und Auswertung von Messdaten**
============================================

In allen Praktika zur Physik werden Methoden zur Aufnahme, 
Bearbeitung, Darstellung und Auswertung von Messdaten benötigt. 
Die Script- und Programmiersprache `python` mit den Zusatzpaketen 
`numpy` und `matplotlib` ist ein universelles Werkzeug, um die 
Wiederholbarkeit von Datenauswertungen und die Reprodzierbarkeit 
der Ergebnisse zu gewährleiseten.

In der Veranstaltung "Computergestützte Datenauswertung" 
(http://www.ekp.kit.edu/~quast/CgDA), die im Studienplan
für den Bachelorstudiengang Physik am KIT seit dem Sommersemester
2016 angeboten wird, werden Methoden und Software zur 
grafischen Darstellung von Daten, deren Auswertung und
Modellierung eingeführt. Die Installation der empfohlenen
Software ist unter dem foltenden Link beschrieben:

      * Dokumentation in html: 
        http://www.ekp.kit.edu/~quast/CgDA/CgDA-SoftwareInstallation-html
      * Dokumentation in pdf:  
        http://www.ekp.kit.edu/~quast/CgDA/CgDA-SoftwareInstallation.pdf     

Speziell für das "Praktikum zur klassischen Physik" finden sich eine 
kurze Einführung  
(http://www.ekp.kit.edu/~quast/CgDA/PhysPrakt/CgDA_APraktikum.pdf) 
sowie die hier dokumentierten einfachen Beispiele als Startpunkt für 
eigene Auswertungen 
(http://www.ekp.kit.edu/~quast/CgDA/PhysPrakt/).

Die vorliegende Sammlung im Paket `PhyPraKit` enthält 
Funktionen zum Einlesen von Daten aus diversen Quellen, zur 
Datenvisualisierung, Signalbearbeitung und zur statistischen
Datenauswertung und Modellanpassung sowie Werkzeuge zur Erzeugung
simulierter Daten. 
Dabei wurde absichtlich Wert auf eine einfache, die Prinzipien 
unterstreichende Codierung gelegt und nicht der möglichst effizienten 
bzw. allgemeinensten Implementierung der Vorzug gegeben. 


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
        - BuildCovarianceMatrix()  Koravianzmatrix aus Einzelunsicherheiten
        - Cov2Cor()                Konversion Kovarianzmatrix -> Korrelationsmatrix
        - Cor2Cov()                Konversion Korrelationsmatrix +
	  Unsicherheiten -> Kovarianzmatrix
        - chi2prob()               Berechnung der chi^2-Wahrscheinlichkeit 
        - propagatedError()        Numerische Fehlerfortpflanzung
        - getModelError()          Numerische Fehlefortpflanzung für 
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
    
      5. Lineare Regression und Anpassen von Fuktionen:

        - linRegression()    lineare Regression, y=ax+b, mit analytische Formel
        - linRegressionXY()  lineare Regression, y=ax+b, mit x- und y-Unsicherheiten   
          ``! veraltet, `odFit` mit linearem Model verwenden``  
	- kRegression()      lineare Regression, y=ax+b, mit (korrelierten) x-
	  und y-Unsicherheiten   
          ``! veraltet, `k2Fit` mit linearem Modell verwenden``  	    
        - odFit()            Funktionsanpassung with x- und y-Unsicherheiten (scipy ODR)
        - mFit()             Funktionsanpassung mit with iminuit,
	  (korrelierte) x- und y-Unsicherheiten 
        - kFit()             Funktionsanpassung mit (korrelierten) x- und
	  y-Unsicherheiten mit kafe, ``! veraltet, `k2Fit` verwenden`` 
        - k2Fit()            Funktionsanpassung mit (korrelierten) x- und
	  y-Unsicherheiten mit kafe2

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
    werden als Listen in *python* zur Verfügung gestellt. 
  * `test_convolutionFilter.py` liest die Datei `Wellenform.csv` und 
    bestimmt Maxima und fallende Flanken des Signals 
  * `test_AutoCorrelation.py` liest die Datei `AudioData.csv` und führt 
    eine Analyse der Autokorrelation zur Frequenzbestimmung durch. 
  * `test_Fourier.py` illustriert die Durchführung einer 
    Fourier-Transfomation eines periodischen Signals, das in 
    der PicoScope-Ausgabedatei `Wellenform.csv` enthalten ist.
  * `test_propagatedError.py` illustriert die Anwendung von numerisch
    berechneter Fehlerfortpflanzung
  * `test_linRegression.py` ist eine einfachere Version mit
    `python`-Bordmitteln zur Anpassung einer Geraden an
    Messdaten mit Fehlern in Ordinaten- und Abszissenrichtung. 
    Korrelierte Unsicherheiten werden nicht unterstützt.
  * `test_mFit` dient zur Anpassung einer beliebigen Funktion an
    Messdaten mit Fehlern in Ordinaten- und Abszissenrichtung und mit
    allen Messpunkten gemeinsamen (d. h. korrelierten) relativen oder
    absoluten systematischen Fehlern. Dazu wird das Paket imunit
    verwendet, das den am CERN entwicklten Minimierer MINUIT nutzt.
    Da die Kostenfunktion frei definiert und auch während der Anpassung
    dynamisch aktualisiert werden kann, ist die Implementierung von
    Parameter-abhängigen Unsicherheiten möglich. Ferner unterstützt
    iminuit die Erzeugung und Darstellung von Profil-Likelihood-Kurven
    und Konfidenzkonturen, die so mit mFit ebenfalls dargestellt
    werden können. 
  * `test_kFit.py` ist mittlerweile veraltet und dient ebenfalls
    zur Anpassung einer beliebigen Funktion an Messdaten mit Fehlern
    in Ordinaten- und Abszissenrichtung und mit allen Messpunkten
    gemeinsamen (d. h. korrelierten) relativen oder absoluten
    systematischen Fehlern mit dem Paket `kafe`.
  * `test_k2Fit.py` verwendet die Version *kafe2* zur Anpassung einer
    Funktion an Messdaten mit unabhängigen oder korrelierten relativen oder
    absoluten Unsicherheiten in Ordinaten- und Abszissenrichtung.
  * `test_simplek2Fit.py` illustriert die Durchführung einer einfachen
    linearen Regression mit *kafe2* mit einer minimalen Anzal eigener
    Codezeilen. 
  * `test_Histogram.py` ist ein Beispiel zur Darstellung und 
    statistischen Auswertung von Häufigkeitsverteilungen (Histogrammen) 
    in ein oder zwei Dimensionen.
  * `test_generateXYata.py` zeigt, wie man mit Hilfe von Zufallszahlen 
    "künstliche Daten" zur Veranschaulichung oder zum Test von Methoden
    zur Datenauswertung erzeugen kann. 
  * `toyMC_Fit.py` führt eine große Anzahl Anpassungen an simulierte
    Daten durch. Durch Vergleich der wahren Werte mit den aus der
    Anpassung bestimmten Werten lassen sich Verzerrungen der
    Parameterschätzungen oder die Form der Verteilung der
    Chi2-Wahrscheinlichkeit überprüfen, die im Idealfall eine
    Rechteckverteilung im Intervall [0,1] sein sollte. 

  Die folgenden *python*-Skripte sind etwas komplexer und illustrieren 
  typische Anwendungsfälle der Module in `PhyPraKit`:

  * `kfitf.py` ist ein Kommandozeilen-Werkzeug, mit dem man komfortabel
    Anpassungen ausführen kann, bei denen Daten und Fit-Funktion in
    einer einzigen Datei angegeben werden. Beispiele finden sich
    in den Dateien mit der Endung `.fit`. 

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
    korrelierte Unsicherheit aller Messwerte berücksichtig weden
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

..  automodule:: PhyPraKit.iminuitFit
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

..  automodule:: test_kRegression

..  automodule:: test_odFit

..  automodule:: test_mFit

..  automodule:: test_kFit

..  automodule:: test_k2Fit

..  automodule:: test_generateData

..  automodule:: toyMC_Fit

..  automodule:: kfitf

..  automodule:: Beispiel_Diodenkennlinie

..  automodule:: Beispiel_Drehpendel

..  automodule:: Beispiel_Hysterese

..  automodule:: Beispiel_Wellenform

..  automodule:: Beispiel_GammaSpektroskopie


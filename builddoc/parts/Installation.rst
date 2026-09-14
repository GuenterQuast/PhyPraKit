Installation
============

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
   :caption: Contents:

   parts/PhyPraKit.rst
   parts/phyTools-api.rst
   parts/phyFit-api.rst
   parts/examples.rst

.. |date| date:: 



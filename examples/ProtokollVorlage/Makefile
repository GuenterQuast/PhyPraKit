# Vorschrift ("Makefile") zur Erzeugung der pdf-Datei
#  aus einer LaTeX-Quelldatei (Protokollvorlage.tex) und
#  verschiedenen Bild- und LaTeX-Dateien im Verzeichnis \include

# Haupt-Dokument für LaTeX
MAIN=ProtokollVorlage
# Analyse-Script
ANALYSIS=generate_ProtokollData.py

# Befehle
LATEX=pdflatex
      # Latex-Befehl
BIBTEX=bibtex
      # zum Einlesen des Literaturverzeichnisses
ANALYSIS=generate_ProtokollData.py 
      # Ausfuehren der Analyse

# Abhaengigkeiten
TEXFILES=$(wildcard *.tex)
                          #  alle .tex-Dateien im Hauptverzeichnis
INCLUDEFILES=$(wildcard include/*.tex) $(wildcard include/*.png)
                          # alle Bild- und tex-Dateien im include-Vereichnis

# Targets: 
all: $(MAIN).pdf  
clean:
	rm *.aux *.log *.out *.bbl *.blg *.toc *.lof *.lot
clean-all:
	rm *.aux *.log *.out *.bbl *.blg *.toc *.lof *.lot *.pdf include/*

# ------------------------------------------ 
# Abhaengigkeiten und Aktionen spezifizieren

$(INCLUDEFILES): $(ANALYSIS)
	@./$(ANALYSIS)

# use double-colon rules to distingish 2 cases:
#  1) analysis script is newer than target
$(MAIN).pdf:: $(ANALYSIS)
	@./$(ANALYSIS)

# 2) any LaTeX input is newer than target
$(MAIN).pdf:: $(TEXFILES) $(INCLUDEFILES)
	@$(LATEX) $(MAIN)
#	@$(BIBTEX) $(MAIN)
#	@$(LATEX) $(MAIN)
	@$(LATEX) $(MAIN)



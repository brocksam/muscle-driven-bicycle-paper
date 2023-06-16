pdf: paper.pdf
paper.html: paper.tex references.bib
	pandoc --mathjax --standalone -o paper.html paper.tex --citeproc --bibliography references.bib
paper.pdf: paper.tex references.bib
	pdflatex paper.tex
	bibtex paper.aux
	pdflatex paper.tex
	pdflatex paper.tex
clearpdf:
	rm paper.pdf
clean:
	(rm -rf *.ps *.log *.dvi *.aux *.*% *.lof *.lop *.lot *.toc *.idx *.ilg *.ind *.bbl *.blg *.cpt *.out)
	rm figures/*.png
export-env:
	mamba env export --no-builds | grep -v "^prefix: " > environment.yml

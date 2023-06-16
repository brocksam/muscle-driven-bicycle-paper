pdf: paper.pdf
paper.html: paper.tex references.bib
	pandoc --mathjax --standalone -o paper.html paper.tex --citeproc --bibliography references.bib
paper.pdf: paper.tex references.bib
	pdflatex paper.tex
clean:
	rm -rf paper.pdf *.ps *.log *.dvi *.aux *.*% *.lof *.lop *.lot *.toc *.idx *.ilg *.ind *.bbl *.blg *.cpt *.out *.bcf *.run.xml
export-env:
	mamba env export --no-builds | grep -v "^prefix: " > environment.yml

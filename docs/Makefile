# Makefile for the MNESIS documentation.
#
# The site is built *in place* inside docs/ so that GitHub Pages can be
# configured to serve the /docs folder of the main branch:
#
#   pip install -r docs/requirements.txt
#   make -C docs html        # symlink notebooks + build the site here
#   git add docs && git commit && git push   # publishes to GitHub Pages
#
# Doctree caches go to docs/_build/ (git-ignored).

SPHINXBUILD ?= sphinx-build
SPHINXOPTS  ?=
BUILDDIR     = _build

NOTEBOOKS := $(notdir $(wildcard ../src/*.ipynb))

.PHONY: help notebooks html clean all

help:
	@echo "Available targets:"
	@echo "  notebooks  (re)create symlinks to ../src/*.ipynb into notebooks/"
	@echo "  html       build the HTML site in place (docs/) from the notebooks"
	@echo "  clean      remove generated artifacts (keeps conf.py, Makefile, *.md)"
	@echo "  all        notebooks + html (default)"

notebooks:
	@mkdir -p notebooks
	@rm -f notebooks/*.ipynb
	@for nb in $(NOTEBOOKS); do \
		ln -sf ../../src/$$nb notebooks/$$nb ; \
	done
	@printf '*\n!.gitignore\n!*.html\n' > notebooks/.gitignore

html: notebooks
	$(SPHINXBUILD) -b html $(SPHINXOPTS) -d $(BUILDDIR)/doctrees . .

clean:
	rm -rf $(BUILDDIR) _static _sources notebooks
	rm -f index.html installation.html api.html experiments.html \
		.genindex genindex.html search.html objects.inv .buildinfo \
		searchindex.js mods.json py-modindex.html

all: html

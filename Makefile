# Makefile for minihmm
#
# Run in project folder
date := $(shell date +%Y-%m-%d)

help:
	@echo "minihmm make help"
	@echo " "
	@echo "Please use \`make <target>\`, choosing <target> from the following:"
	@echo "    dist        to make HTML documentation and eggs for distribution"
	@echo "    docs        to make HTML documentation"
	@echo "    eggs        to make 2.7 and 3.x egg distributions"
	@echo "    dev_egg     to make development release"
	@echo "    cleandoc    to remove previous generated documentation components"
	@echo "    clean       to remove everything previously built"
	@echo " "

docs/source/substitutions.txt :
	mkdir -p docs/source
	get_class_substitutions minihmm minihmm
	mv minihmm_substitutions.txt docs/source/class_substitutions.txt

docs/build/html : docs/source/substitutions.txt | docs/source/generated
	$(MAKE) html -C docs

docs/source/generated :
	sphinx-apidoc -e -o docs/source/generated minihmm
	fix_package_template test minihmm docs/source/generated

docs : | docs/build/html docs/source/generated

dev_egg :
	python setup.py egg_info -rbdev$(date) bdist_egg

python27 :
	python2.7 setup.py bdist_egg

python3x :
	python3 setup.py bdist_egg

eggs: python27 python3x

dist: docs dev_egg python27 python3x

cleandoc : 
	rm -rf docs/build
	rm -rf docs/source/generated
	rm -rf docs/class_substitutions.txt

clean : cleandoc
	rm -rf dist
	rm -rf build
	
.PHONY : docs dist clean cleandoc dev_egg eggs help python27 python3x

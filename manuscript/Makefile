SHELL := /bin/sh

.PHONY: pdf clean prose-lint prose-sync

pdf:
	./build.sh

prose-sync:
	@if [ -x "./tools/vale" ]; then VALE=./tools/vale; \
	elif command -v vale >/dev/null 2>&1; then VALE=$$(command -v vale); \
	else echo "Install Vale via ./install_vale.sh or ensure 'vale' is on PATH."; exit 1; fi; \
	$$VALE sync

prose-lint:
	@if [ -x "./tools/vale" ]; then VALE=./tools/vale; \
	elif command -v vale >/dev/null 2>&1; then VALE=$$(command -v vale); \
	else echo "Install Vale via ./install_vale.sh or ensure 'vale' is on PATH."; exit 1; fi; \
	$$VALE sync && $$VALE main.tex sections/*.tex

clean:
	rm -rf build
	rm -f main.aux main.bbl main.blg main.fdb_latexmk main.fls main.log main.out main.pdf main.synctex.gz

# Triton 讲义（CS336 assignment 版式）。在本目录执行 make。
# 中文必须 XeLaTeX。缺 TeX Live 时请先：
#   sudo apt install texlive-xetex texlive-lang-chinese latexmk \
#     texlive-science texlive-fonts-recommended texlive-latex-extra

MAIN := main
LATEXMK := latexmk
ENGINE := xelatex

.PHONY: all pdf clean help

all: pdf

help:
	@echo "make        编译 PDF"
	@echo "make clean  清辅助文件"
	@echo "需要：xelatex、latexmk、ctex"

pdf:
	@if ! command -v $(ENGINE) >/dev/null 2>&1; then \
	  echo "缺少 $(ENGINE)。请安装 TeX Live（见 Makefile 顶部注释）。"; \
	  exit 1; \
	fi
	@if ! command -v $(LATEXMK) >/dev/null 2>&1; then \
	  echo "缺少 latexmk。请安装：sudo apt install latexmk"; \
	  exit 1; \
	fi
	$(LATEXMK) -$(ENGINE) -interaction=nonstopmode -file-line-error $(MAIN).tex || true
	@if [ ! -f $(MAIN).pdf ] && [ -f $(MAIN).xdv ]; then xdvipdfmx -o $(MAIN).pdf $(MAIN).xdv; fi
	@test -f $(MAIN).pdf && echo "OK: $(MAIN).pdf"

clean:
	$(LATEXMK) -C $(MAIN).tex
	rm -f $(MAIN).xdv $(MAIN).bbl $(MAIN).run.xml

include config.mk
include src/variants/$(VARIANT)/flags.mk

# ==========================================================================
# Configuration
# ==========================================================================
CXX         = g++
PYTHON      = python3
BASE_FLAGS  = -std=c++20 -Wall -Wextra -I./src
BUILD_DIR   = build
RESULTS_DIR = results
TEST_INPUT  = $(BUILD_DIR)/test_input.bin
PIN        := setarch $(shell uname -m) -R taskset -c $(CORE)

COMMON_SRC = src/common/parser.cpp
LOAD_MAX  ?= 1.0

# bench.csv schema. Must match exactly what main.cpp prints in --bench
# mode (F, S, trial, then per-call averages of every event in
# DEFAULT_PAPI_EVENTS). PAPI_TOT_CYC is the cycle measurement; the
# old RDTSC `cycles` column is gone.
BENCH_HEADER = F,S,trial,papi_tot_cyc,papi_tot_ins,papi_l1_dcm,papi_l2_dcm,papi_l3_tcm,papi_res_stl,papi_br_ins,papi_br_msp,papi_sp_ops,papi_lst_ins

.DEFAULT_GOAL := help
.DELETE_ON_ERROR:

# ==========================================================================
# Help
# ==========================================================================
.PHONY: help
help: ## show this help
	@printf "Usage: make <target> [VAR=value ...]\n\n"
	@grep -hE '^[a-zA-Z0-9_-]+:.*##' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ==========================================================================
# Build
# ==========================================================================
$(BUILD_DIR)/%: src/main.cpp src/variants/%/comp.cpp src/variants/%/flags.mk $(COMMON_SRC) Makefile config.mk | $(BUILD_DIR)
	$(CXX) $(BASE_FLAGS) $(VARIANT_FLAGS) src/main.cpp src/variants/$*/comp.cpp $(COMMON_SRC) -o $@ -lpapi

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

_rebuild:
	@rm -f $(BUILD_DIR)/$(VARIANT)

# ==========================================================================
# Local targets
# ==========================================================================
.PHONY: gen-input compile run check bench clean

gen-input: | $(BUILD_DIR) ## regenerate build/test_input.bin
	$(PYTHON) tools/gen_input.py --out $(TEST_INPUT) --seed $(SEED) \
		--F $(F) --S $(S) --interval $(INTERVAL) --data $(DATA)

compile: _rebuild $(BUILD_DIR)/$(VARIANT) ## compile the variant binary

run: _rebuild $(BUILD_DIR)/$(VARIANT) gen-input ## run the variant on the current input
	./$(BUILD_DIR)/$(VARIANT) $(TEST_INPUT)

check: _rebuild $(BUILD_DIR)/$(VARIANT) gen-input ## run + compare top-K against NumPy reference
	./$(BUILD_DIR)/$(VARIANT) $(TEST_INPUT) > $(BUILD_DIR)/cpp_out.txt
	$(PYTHON) tools/reference.py $(TEST_INPUT) $(BUILD_DIR)/cpp_out.txt

bench: BASE_FLAGS += -DMICROBENCH
bench: _rebuild $(BUILD_DIR)/$(VARIANT) gen-input ## bench with per-phase cycle counts
	@echo "$(BENCH_HEADER)"
	@$(PIN) ./$(BUILD_DIR)/$(VARIANT) $(TEST_INPUT) --bench --rep $(REP)

clean: ## remove the build/ directory
	rm -rf $(BUILD_DIR)

# ==========================================================================
# Sweep
# ==========================================================================
.PHONY: sweep

# The sweep iterates over every variant in $(SWEEP_VARIANTS) by default.
# Override on the command line for ad-hoc subsets, e.g.:
#   make sweep VARIANTS_LIST="0_baseline 7_multi_accumulator"
VARIANTS_LIST ?= $(SWEEP_VARIANTS)

# All sweep work lives in tools/sweep.sh. The Makefile only collects
# parameters and forwards them as env vars so the per-cell loop has a
# single source of truth.
SWEEP_ENV = \
    VARIANTS="$(VARIANTS_LIST)" \
    SWEEP_F="$(SWEEP_F)" SWEEP_S="$(SWEEP_S)" \
    SWEEP_INTERVAL="$(SWEEP_INTERVAL)" SWEEP_DATA="$(SWEEP_DATA)" \
    REP="$(REP)" SEED="$(SEED)" CORE="$(CORE)" \
    PYTHON="$(PYTHON)" BUILD_DIR="$(BUILD_DIR)" RESULTS_DIR="$(RESULTS_DIR)" \
    LOAD_MAX="$(LOAD_MAX)" \
    BASE_FLAGS="$(BASE_FLAGS)" CXX="$(CXX)" MAKE="$(MAKE)" \
    BENCH_HEADER="$(BENCH_HEADER)"

sweep: | $(BUILD_DIR) ## sweep VARIANTS_LIST × SWEEP_F × SWEEP_S × SWEEP_INTERVAL × SWEEP_DATA (always with perf)
	@$(SWEEP_ENV) bash tools/sweep.sh

# ==========================================================================
# Report
# ==========================================================================
.PHONY: report report-figures report-open report-clean

REPORT_DIR       = report
REPORT_BUILD_DIR = $(REPORT_DIR)/build
REPORT_PDF       = $(REPORT_DIR)/report.pdf
REPORT_FIG_DIR   = $(REPORT_DIR)/figures/generated
LATEXMK_FLAGS    = -pdf -bibtex -interaction=nonstopmode -outdir=build

report-figures: ## (re)generate report figures from results/ into figures/generated/
	$(PYTHON) tools/report_plots.py --data GAUSSIAN --outdir $(REPORT_FIG_DIR)
	$(PYTHON) tools/report_plots.py --data BLOCK_CORR --outdir $(REPORT_FIG_DIR)

report: report-figures ## regenerate figures, then build report/report.pdf
	mkdir -p $(REPORT_BUILD_DIR)
	cd $(REPORT_DIR) && latexmk $(LATEXMK_FLAGS) report.tex
	cp $(REPORT_BUILD_DIR)/report.pdf $(REPORT_PDF)

report-open: report ## build and open the report
	@if command -v xdg-open >/dev/null 2>&1; then xdg-open $(REPORT_PDF); \
	elif command -v open >/dev/null 2>&1; then open $(REPORT_PDF); fi

report-clean: ## remove report/build/
	rm -rf $(REPORT_BUILD_DIR)

# ==========================================================================
# Remote
# ==========================================================================
.PHONY: push pull-results remote-setup remote-restore remote-shell \
        remote-compile remote-run remote-check remote-bench \
        remote-sweep remote-sweep-attach remote-sweep-peek remote-sweep-kill

RSYNC_FLAGS   = -a --no-times --delete --checksum
PUSH_INCLUDES = --include='/src/***' --include='/tools/***' \
                --include='/Makefile' --include='/config.mk' --exclude='*'
TMUX_SESSION  = sweep

push: ## rsync source to remote
	@ssh $(REMOTE_HOST) 'mkdir -p $(REMOTE_DIR)'
	rsync $(RSYNC_FLAGS) $(PUSH_INCLUDES) ./ $(REMOTE_HOST):$(REMOTE_DIR)/

pull-results: ## rsync results/ back from remote
	@mkdir -p $(RESULTS_DIR)
	rsync -az $(REMOTE_HOST):$(REMOTE_DIR)/$(RESULTS_DIR)/ $(RESULTS_DIR)/

remote-setup: push ## pin CPU freq, turbo off, sibling offline
	ssh -t $(REMOTE_HOST) 'cd $(REMOTE_DIR) && tools/bench.sh setup $(BENCH_SIBLING)'

remote-restore: push ## undo remote-setup
	ssh -t $(REMOTE_HOST) 'cd $(REMOTE_DIR) && tools/bench.sh restore $(BENCH_SIBLING)'

remote-shell: ## ssh into remote project dir
	ssh -t $(REMOTE_HOST) 'cd $(REMOTE_DIR) && exec $$SHELL -l'

remote-compile: push ## compile on remote
	ssh $(REMOTE_HOST) 'cd $(REMOTE_DIR) && make compile $(MAKEOVERRIDES)'

remote-run: push ## run on remote
	ssh $(REMOTE_HOST) 'cd $(REMOTE_DIR) && make run $(MAKEOVERRIDES)'

remote-check: push ## check on remote
	ssh $(REMOTE_HOST) 'cd $(REMOTE_DIR) && make check $(MAKEOVERRIDES)'

remote-bench: push ## bench on remote
	ssh $(REMOTE_HOST) 'cd $(REMOTE_DIR) && make bench $(MAKEOVERRIDES)'

remote-sweep: remote-setup ## start sweep in remote tmux
	@ssh $(REMOTE_HOST) 'tmux has-session -t $(TMUX_SESSION) 2>/dev/null' && \
	  { echo "tmux session '$(TMUX_SESSION)' already running — use peek/attach/kill" >&2; exit 1; } || true
	@ssh $(REMOTE_HOST) "tmux new -d -s $(TMUX_SESSION) -c $(REMOTE_DIR) \
	    'make sweep $(MAKEOVERRIDES); echo \"[done] Press enter.\"; read'"
	@echo "[remote-sweep] launched in tmux. Use: peek / attach / pull-results / kill"

remote-sweep-attach: ## attach to remote sweep tmux
	ssh -t $(REMOTE_HOST) 'tmux attach -t $(TMUX_SESSION)'

remote-sweep-peek: ## show last 30 lines of remote sweep
	@ssh $(REMOTE_HOST) 'tmux capture-pane -pt $(TMUX_SESSION) 2>/dev/null | tail -30 || echo "(no session)"'

remote-sweep-kill: ## kill remote sweep tmux session
	ssh $(REMOTE_HOST) 'tmux kill-session -t $(TMUX_SESSION) 2>/dev/null || echo "(no session)"'
	@echo "reminder: run 'make remote-restore' to undo bench setup"

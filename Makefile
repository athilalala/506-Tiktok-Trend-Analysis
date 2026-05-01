# ============================================================
#  TikTok Virality Prediction — CS 506
#  Usage:
#    make install   → install all Python dependencies
#    make run       → execute baseline notebook end-to-end
#    make test      → run unit tests
#    make all       → install + run + test
#    make clean     → remove generated outputs
# ============================================================

PYTHON      := python3
PIP         := $(PYTHON) -m pip
NOTEBOOK    := baseline.ipynb
OUTPUT_NB   := baseline_executed.ipynb
DATA_DIR    := data
VIZ_DIR     := visualizations
TEST_DIR    := tests

.PHONY: all install run test clean help

# ── Default target ────────────────────────────────────────────
all: install run test

# ── Install dependencies ──────────────────────────────────────
install:
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip --quiet
	$(PIP) install -r requirements.txt --quiet
	@echo "✅ Dependencies installed"

# ── Run baseline notebook ─────────────────────────────────────
run:
	@echo "Running baseline notebook..."
	@mkdir -p $(VIZ_DIR)
	jupyter nbconvert \
		--to notebook \
		--execute $(NOTEBOOK) \
		--ExecutePreprocessor.timeout=600 \
		--output $(OUTPUT_NB)
	@echo "✅ Baseline complete → see $(OUTPUT_NB)"
	@echo "   Transformer model: open transformer.ipynb in Google Colab (T4 GPU)"

# ── Run unit tests ────────────────────────────────────────────
test:
	@echo "Running unit tests..."
	$(PYTHON) -m pytest $(TEST_DIR)/ -v --tb=short
	@echo "✅ All tests passed"

# ── Clean generated outputs ───────────────────────────────────
clean:
	@echo "Cleaning generated files..."
	rm -f $(OUTPUT_NB)
	rm -rf $(VIZ_DIR)/*.png $(VIZ_DIR)/*.html
	rm -rf __pycache__ .pytest_cache
	find . -name "*.pyc" -delete
	find . -name ".DS_Store" -delete
	@echo "✅ Cleaned"

# ── Help ──────────────────────────────────────────────────────
help:
	@echo ""
	@echo "TikTok Virality Prediction — CS 506"
	@echo "------------------------------------"
	@echo "  make install  → install all dependencies"
	@echo "  make run      → run baseline notebook"
	@echo "  make test     → run unit tests"
	@echo "  make all      → install + run + test"
	@echo "  make clean    → remove generated outputs"
	@echo ""

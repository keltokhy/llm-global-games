# Makefile — rebuild all paper assets from raw experiment data.
#
# Usage:
#   make          Full rebuild (stats → tables → figures → paper)
#   make stats    Recompute verified_stats.json
#   make tables   Regenerate LaTeX tables
#   make figures  Regenerate all figures
#   make paper    Compile LaTeX (×2 for references)
#   make clean    Remove generated paper assets

.PHONY: all stats tables figures paper lint clean

all: stats tables figures paper

# ── Statistics ──────────────────────────────────────────────────────
stats:
	uv run python analysis/verify_paper_stats.py

# ── Tables ──────────────────────────────────────────────────────────
tables: stats
	uv run python analysis/render_paper_tables.py
	uv run python analysis/agent_regressions.py
	uv run python analysis/classifier_baselines.py

# ── Figures ─────────────────────────────────────────────────────────
figures: stats
	uv run python analysis/make_figures.py
	uv run python analysis/make_diagrams.py
	uv run python analysis/construct_validity.py

# ── Paper ───────────────────────────────────────────────────────────
paper: tables figures
	cd paper && pdflatex -interaction=nonstopmode paper.tex
	cd paper && pdflatex -interaction=nonstopmode paper.tex

# ── Lint (not in all chain) ────────────────────────────────────────
lint:
	uv run python analysis/check_paper_numbers.py

# ── Clean ───────────────────────────────────────────────────────────
clean:
	rm -f paper/paper.aux paper/paper.log paper/paper.out paper/paper.bbl paper/paper.blg

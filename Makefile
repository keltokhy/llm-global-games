# Makefile — rebuild all paper assets from raw experiment data.
#
# Usage:
#   make          Full rebuild (stats → tables → figures → paper)
#   make stats    Recompute verified_stats.json
#   make tables   Regenerate LaTeX tables
#   make figures  Regenerate all figures
#   make paper    Compile LaTeX and bibliography
#   make anonymous  Compile double-blind PDF
#   make audit    Run submission-package checks
#   make jebo-audit  Run JEBO-specific submission checks
#   make jebo-final-audit  Run JEBO audit plus author-supplied portal checks
#   make clean    Remove generated paper assets

.PHONY: all stats tables figures paper anonymous lint audit jebo-cover-letter jebo-highlights jebo-package jebo-audit jebo-final-audit clean

all: stats tables figures paper

# ── Statistics ──────────────────────────────────────────────────────
stats:
	uv run python analysis/verify_paper_stats.py

# ── Tables ──────────────────────────────────────────────────────────
tables: stats
	uv run python analysis/agent_regressions.py
	uv run python analysis/rationale_theta_checks.py
	uv run python analysis/nested_campaign_checks.py
	uv run python analysis/classifier_baselines.py
	uv run python analysis/render_paper_tables.py

# ── Figures ─────────────────────────────────────────────────────────
figures: stats
	uv run python analysis/make_figures.py
	uv run python analysis/make_diagrams.py
	uv run python analysis/construct_validity.py

# ── Paper ───────────────────────────────────────────────────────────
paper: tables figures
	cd paper && pdflatex -interaction=nonstopmode paper.tex
	cd paper && bibtex paper
	cd paper && pdflatex -interaction=nonstopmode paper.tex
	cd paper && pdflatex -interaction=nonstopmode paper.tex
	cd paper && pdflatex -interaction=nonstopmode paper.tex

# ── Double-blind manuscript ───────────────────────────────────────
anonymous:
	cd paper && pdflatex -interaction=nonstopmode paper_anonymous.tex
	cd paper && bibtex paper_anonymous
	cd paper && pdflatex -interaction=nonstopmode paper_anonymous.tex
	cd paper && pdflatex -interaction=nonstopmode paper_anonymous.tex
	cd paper && pdflatex -interaction=nonstopmode paper_anonymous.tex

# ── Lint (not in all chain) ────────────────────────────────────────
lint:
	uv run python analysis/check_paper_numbers.py

# ── Submission Audit ───────────────────────────────────────────────
audit: lint
	uv run pytest
	uv run python analysis/check_manuscript_text.py
	uv run python analysis/check_data_manifest.py
	uv run python analysis/check_submission_package.py

# ── JEBO Submission Audit ─────────────────────────────────────────
jebo-cover-letter:
	cd paper && pdflatex -interaction=nonstopmode cover_letter_jebo.tex

jebo-highlights:
	uv run python analysis/make_jebo_highlights_docx.py

jebo-package: jebo-highlights jebo-cover-letter
	uv run python analysis/build_jebo_submission_package.py

jebo-audit: audit jebo-package
	uv run python analysis/check_jebo_submission.py

jebo-final-audit: jebo-audit
	uv run python analysis/check_jebo_final_inputs.py

# ── Revision experiments (require OPENROUTER_API_KEY; issue live API calls) ──
revision-experiments:
	uv run python analysis/contamination_audit.py
	uv run python analysis/decode_messages.py

# ── Clean ───────────────────────────────────────────────────────────
clean:
	rm -f paper/paper.aux paper/paper.log paper/paper.out paper/paper.bbl paper/paper.blg
	rm -f paper/paper_anonymous.aux paper/paper_anonymous.log paper/paper_anonymous.out paper/paper_anonymous.bbl paper/paper_anonymous.blg
	rm -f paper/cover_letter_jebo.aux paper/cover_letter_jebo.log

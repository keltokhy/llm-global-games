Replication README
==================

Paper
-----
Speaking in Code: Surveillance and Coordinated Dissent in a Language-Based Global Game with LLM Agents

This repository contains the simulation code, raw experiment outputs,
analysis scripts, generated tables and figures, and LaTeX source needed
to rebuild the paper.


Software Requirements
---------------------
- Python 3.12 or newer
- uv
- A TeX distribution with pdflatex and bibtex
- Poppler is useful for visual PDF checks, but is not required to build
  the paper

Install the Python environment from the repository root:

    uv sync


Main Rebuild
------------
The paper rebuild is controlled by the Makefile. From the repository
root:

    make

This runs the full pipeline:

    analysis/verify_paper_stats.py
    analysis/agent_regressions.py
    analysis/rationale_theta_checks.py
    analysis/nested_campaign_checks.py
    analysis/classifier_baselines.py
    analysis/render_paper_tables.py
    analysis/make_figures.py
    analysis/make_diagrams.py
    analysis/construct_validity.py
    analysis/contamination_audit.py   (live API; make revision-experiments)
    analysis/decode_messages.py       (live API; make revision-experiments)
    pdflatex/bibtex on paper/paper.tex

The main output is:

    paper/paper.pdf

Useful partial targets:

    make stats
    make tables
    make figures
    make paper
    make anonymous
    make lint
    make audit

For double-blind review, build the anonymized manuscript:

    make anonymous

This writes:

    paper/paper_anonymous.pdf


Verification
------------
Run the test suite:

    uv run pytest

Run the paper-number lint:

    make lint

Run the target-neutral submission audit:

    make audit

At the time this README was added, the current rebuilt paper passed:

    make paper
    make lint
    make audit
    uv run pytest

The pytest suite includes checks that paper citations have bibliography
entries, generated assets listed in paper/asset_manifest.tsv exist, the
asset manifest matches the compiled paper, and compiled LaTeX labels do
not contain duplicate or missing references.
The submission audit also checks manuscript text for repeated words,
common typo patterns, and stale drafting phrases; verifies that
paper/asset_manifest.tsv inputs and DATA_MANIFEST.txt output families
resolve to real local artifacts; scans built LaTeX logs; checks normal
and anonymous PDF metadata; verifies the public replication URL in the
normal manuscript; and verifies that the anonymous manuscript does not
expose author or repository identifiers.


Data and Generated Artifacts
----------------------------
Raw and intermediate experiment outputs are under:

    output/

The data inventory is:

    DATA_MANIFEST.txt

The paper asset lineage file is:

    paper/asset_manifest.tsv

The canonical model roster used by analysis scripts is:

    analysis/models.py

The main verified statistics cache is:

    analysis/verified_stats.json

Generated LaTeX tables are under:

    paper/tables/

Generated figures are under:

    paper/figures/


Regenerating Missing Experiment Data
------------------------------------
The checked-in output directory is the data source for the paper rebuild.
If experiment outputs are missing, inspect the declared data targets:

    uv run python scripts/make_data.py

Print commands without running them:

    uv run python scripts/make_data.py --dry-run

Run missing targets:

    uv run python scripts/make_data.py --run

The runnable manifest includes a `centering` group that creates the
per-model briefing-centering JSON files used by the checked experiment
outputs.  Full reruns use the legacy `--load-calibrated` flag so
regenerated outputs match the stored provenance; when running a narrow
group from an empty output directory, run the corresponding centering
target first.

Experiment generation calls hosted LLM APIs through OpenRouter unless a
local compatible API base is supplied. Set the required API key before
running new model calls:

    export OPENROUTER_API_KEY=...

Optional cache location:

    export GGC_LLM_CACHE_DIR=/path/to/cache

Use the checked-in outputs for exact paper reproduction. Running new
LLM experiments can change realized stochastic outputs unless the same
model versions, seeds, prompts, API behavior, and cached responses are
available.


Provenance
----------
Per-run provenance is stored in JSON manifests under output/*/manifests/.
LLM responses are cached by request hash where available. The analysis
pipeline recomputes paper statistics from raw CSV and JSON experiment
outputs rather than relying on hand-entered paper numbers.


Expected Build Directory State
------------------------------
After a full rebuild, expected generated files include:

    analysis/verified_stats.json
    analysis/classifier_results.json
    analysis/regression_results.json
    analysis/construct_validity_results.json
    paper/tables/*.tex
    paper/figures/*.pdf
    paper/paper.pdf

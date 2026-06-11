"""Fast integrity checks for the paper build and citation layer."""

import csv
from collections import Counter
import importlib.util
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT / "paper"


def _cited_keys() -> set[str]:
    tex = (PAPER_DIR / "paper.tex").read_text()
    keys: set[str] = set()
    for match in re.finditer(r"\\cite\w*(?:\[[^\]]*\])*\{([^}]*)\}", tex):
        keys.update(key.strip() for key in match.group(1).split(",") if key.strip())
    return keys


def _bib_keys() -> set[str]:
    bib = (PAPER_DIR / "references.bib").read_text()
    return set(re.findall(r"@\w+\{([^,]+),", bib))


def _bib_entries() -> dict[str, tuple[str, set[str]]]:
    bib = (PAPER_DIR / "references.bib").read_text()
    starts = list(re.finditer(r"@(\w+)\s*\{\s*([^,]+),", bib))
    entries: dict[str, tuple[str, set[str]]] = {}

    for idx, match in enumerate(starts):
        entry_type = match.group(1).lower()
        key = match.group(2).strip()
        end = starts[idx + 1].start() if idx + 1 < len(starts) else len(bib)
        body = bib[match.end() : end]
        fields = {
            field_match.group(1).lower()
            for field_match in re.finditer(r"\n\s*(\w+)\s*=", body)
        }
        entries[key] = (entry_type, fields)

    return entries


def _paper_assets() -> set[str]:
    tex = (PAPER_DIR / "paper.tex").read_text()
    assets: set[str] = set()
    assets.update(re.findall(r"\\input\{((?:tables|figures)/[^}]+)\}", tex))
    assets.update(
        re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{((?:tables|figures)/[^}]+)\}", tex)
    )
    return assets


def _compiled_tex() -> str:
    """Paper source plus files reached through \\input directives."""

    seen: set[Path] = set()

    def collect(path: Path) -> str:
        if path in seen:
            return ""
        seen.add(path)
        text = path.read_text()
        parts = [text]
        for raw_path in re.findall(r"\\input\{([^}]+)\}", text):
            input_path = PAPER_DIR / raw_path
            if not input_path.suffix:
                input_path = input_path.with_suffix(".tex")
            if input_path.exists():
                parts.append(collect(input_path))
        return "\n".join(parts)

    return collect(PAPER_DIR / "paper.tex")


def _strip_tex(text: str) -> str:
    text = re.sub(r"\\cite\w*(?:\[[^\]]*\])*\{[^}]*\}", " ", text)
    text = re.sub(r"\$[^$]*\$", " ", text)
    text = re.sub(r"``|''", " ", text)
    text = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?(?:\{([^{}]*)\})?", r" \1 ", text)
    text = re.sub(r"[{}]", " ", text)
    return text


def _asset_manifest_rows() -> list[dict[str, str]]:
    with (PAPER_DIR / "asset_manifest.tsv").open(newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_paper_citations_have_bibliography_entries():
    assert _cited_keys() - _bib_keys() == set()


def test_cited_bibliography_entries_have_submission_metadata():
    entries = _bib_entries()
    missing: dict[str, set[str]] = {}

    for key in _cited_keys():
        entry_type, fields = entries[key]
        required = {"author", "title", "year"}
        if entry_type == "article":
            required.add("journal")
        elif entry_type in {"incollection", "inproceedings"}:
            required.add("booktitle")
        elif entry_type == "techreport":
            required.add("institution")
        absent = required - fields
        if absent:
            missing[key] = absent

    assert missing == {}


def test_compiled_paper_has_no_broken_or_duplicate_labels():
    tex = _compiled_tex()
    labels = re.findall(r"\\label\{([^}]+)\}", tex)
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", tex))

    duplicates = {label for label, count in Counter(labels).items() if count > 1}

    assert duplicates == set()
    assert refs - set(labels) == set()


def test_corrected_citation_keys_do_not_regress():
    stale_keys = {"carlini2025", "huang2024", "petrov2025", "larooij2025"}
    assert stale_keys.isdisjoint(_cited_keys())
    assert stale_keys.isdisjoint(_bib_keys())

    required_keys = {"cheung2025", "ouyang2024", "jia2025", "larooij2026"}
    assert required_keys <= _cited_keys()
    assert required_keys <= _bib_keys()


def test_cited_paper_archive_has_valid_pdfs_for_available_sources():
    allowed_missing = {"shadmehr2011", "roberts2018"}  # roberts2018 is a book
    archive_keys = {path.stem for path in (PAPER_DIR / "cited_papers").glob("*.pdf")}

    assert _cited_keys() - archive_keys == allowed_missing

    for pdf_path in (PAPER_DIR / "cited_papers").glob("*.pdf"):
        assert pdf_path.read_bytes().startswith(b"%PDF-"), f"{pdf_path} is not a PDF"


def test_makefile_paper_target_runs_bibtex():
    makefile = (ROOT / "Makefile").read_text()
    assert "cd paper && bibtex paper" in makefile


def test_makefile_has_anonymous_submission_target():
    makefile = (ROOT / "Makefile").read_text()
    anonymous_tex = (PAPER_DIR / "paper_anonymous.tex").read_text()

    assert "anonymous:" in makefile
    assert "pdflatex -interaction=nonstopmode paper_anonymous.tex" in makefile
    assert "cd paper && bibtex paper_anonymous" in makefile
    assert r"\def\anonymous{1}" in anonymous_tex
    assert r"\input{paper.tex}" in anonymous_tex


def test_makefile_has_submission_audit_target():
    makefile = (ROOT / "Makefile").read_text()

    assert "audit:" in makefile
    assert "uv run pytest" in makefile
    assert "uv run python analysis/check_manuscript_text.py" in makefile
    assert "uv run python analysis/check_data_manifest.py" in makefile
    assert "uv run python analysis/check_submission_package.py" in makefile
    assert (ROOT / "analysis" / "check_manuscript_text.py").exists()
    assert (ROOT / "analysis" / "check_data_manifest.py").exists()
    assert (ROOT / "analysis" / "check_submission_package.py").exists()


def test_generative_ai_declaration_is_before_references():
    tex = (PAPER_DIR / "paper.tex").read_text()
    title = (
        r"\section*{Declaration of Generative AI and AI-Assisted Technologies "
        r"in the Writing Process}"
    )

    declaration_pos = tex.find(title)
    references_pos = tex.find("%% REFERENCES")
    conclusion_pos = tex.find(r"\section{Conclusion}")

    assert declaration_pos != -1
    assert conclusion_pos < declaration_pos < references_pos
    assert "OpenAI Codex" in tex[declaration_pos:references_pos]
    assert "reviewed and edited" in tex[declaration_pos:references_pos]


def test_submission_front_matter_is_complete_and_concise():
    tex = (PAPER_DIR / "paper.tex").read_text()
    abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    jel_match = re.search(r"\\textbf\{JEL:\}\s*([^\\]+)", tex)
    keywords_match = re.search(r"\\textbf\{Keywords:\}\s*([^\n]+)", tex)

    assert abstract_match is not None
    assert jel_match is not None
    assert keywords_match is not None

    words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", _strip_tex(abstract_match.group(1)))
    jel_codes = re.findall(r"[A-Z]\d{2}", jel_match.group(1))
    keywords = [word.strip() for word in keywords_match.group(1).split(",") if word.strip()]

    assert 100 <= len(words) <= 250
    assert len(jel_codes) >= 3
    assert 4 <= len(keywords) <= 8


def test_submission_text_has_no_placeholders_or_todos():
    searched_files = [
        PAPER_DIR / "paper.tex",
        ROOT / "REPLICATION_README.txt",
        ROOT / "DATA_MANIFEST.txt",
    ]
    forbidden = re.compile(r"\b(TODO|FIXME|TBD|XXX|PLACEHOLDER|to be added)\b", re.I)
    matches = {
        str(path.relative_to(ROOT)): forbidden.findall(path.read_text())
        for path in searched_files
    }
    assert {path: hits for path, hits in matches.items() if hits} == {}


def test_main_and_anonymous_manuscript_pdfs_exist():
    for pdf_name in ["paper.pdf", "paper_anonymous.pdf"]:
        pdf_path = PAPER_DIR / pdf_name
        assert pdf_path.exists()
        assert pdf_path.read_bytes().startswith(b"%PDF-")


def test_asset_manifest_matches_paper_assets():
    manifest_assets = {row["asset"] for row in _asset_manifest_rows()}
    assert _paper_assets() - manifest_assets == set()
    assert manifest_assets - _paper_assets() == set()

    for asset in manifest_assets:
        assert (PAPER_DIR / asset).exists(), f"{asset} is listed but missing"


def test_asset_manifest_generators_exist():
    for row in _asset_manifest_rows():
        generator = row["generator"]
        assert generator
        assert (ROOT / generator).exists(), f"{generator} is listed but missing"


def test_data_manifest_mentions_known_output_families():
    manifest = (ROOT / "DATA_MANIFEST.txt").read_text()
    required_tokens = {
        "prompt-isolation-surveillance/",
        "prompt-isolation-surveillance-placebo/",
        "prompt-isolation-surveillance-anonymous/",
        "fixed-messages-surv/",
        "no-messages-llama/",
        "no-messages-qwen30b/",
        "cross-task-placebo-baseline/",
        "cross-task-placebo-surveillance/",
        "xmodel-source-llama-baseline/",
        "xmodel-source-qwen-surveillance/",
        "xmodel-matched-llama-writes-qwen-reads-baseline/",
        "xmodel-matched-qwen-writes-llama-reads-surveillance/",
        "revision-beliefs-*/",
        "punishment-risk*/",
        "calibrated_params_*.json",
        "autocalibrate_history.csv",
        "temperature-robustness*/",
        "cross-generator*/",
        "mistralai--mistral-small-creative-n*/",
        "network-k8/",
        "mixed-5model-pure/",
        "mixed-5model-comm/",
        "mixed-mistral-gptoss-comm/",
        "holdout-validation/",
        "group-size-info/",
        "mistralai--mistral-small-creative-infodesign-comm/",
        "surveillance-x-censorship/",
        "bandwidth-005/",
        "bandwidth-030/",
        "z-centered/",
    }
    missing = sorted(token for token in required_tokens if token not in manifest)
    assert missing == []


def test_replication_readme_mentions_required_rebuild_artifacts():
    readme = (ROOT / "REPLICATION_README.txt").read_text()
    required_tokens = {
        "uv sync",
        "make",
        "make anonymous",
        "make lint",
        "make audit",
        "uv run pytest",
        "DATA_MANIFEST.txt",
        "paper/asset_manifest.tsv",
        "analysis/models.py",
        "analysis/verified_stats.json",
        "scripts/make_data.py",
        "OPENROUTER_API_KEY",
    }

    missing = sorted(token for token in required_tokens if token not in readme)
    assert missing == []


def test_june_feedback_comment_responses_do_not_regress():
    tex = _compiled_tex()
    models = _load_module(ROOT / "analysis" / "models.py")

    required_fragments = [
        "actionability cues",
        'rather than a mediation result',
        "not standardized by the prior-predictive signal standard deviation",
        r"(1+\sigma^2)/\sigma^2",
        "the comparison is external in the sense",
        "cross-period briefing permutation breaking the link between",
        "the briefing received in that period",
        "Equal-weighted model averages are close to zero under both summaries",
        "Table~\\ref{tab:main_results}",
        "matched-cell estimator",
        'The full-scale evidence is the primary-model result',
        "the codedness control above randomizes the coded form itself",
        "separate dictionary counts",
        "That comparison uses its own live-message rerun",
        "The no-message control uses the main communication baseline",
        "separate 500-cell factorial rerun",
        "rather than a mediation result",
        "the rotation-matched Llama subset is smaller",
        "Inconclusive",
        "core benchmark suites",
        r"$1 - d$, where $d$ is the briefing-generator direction slider",
        "not surveillance-channel ablations",
    ]
    missing = [fragment for fragment in required_fragments if fragment not in tex]
    assert missing == []

    forbidden_fragments = [
        "Llama 3.370",
        "Llama 70B",
        "Qwen 235B",
        "full sender information object",
        "prior-predictive z-score with variance",
        r"Figure~\ref{fig:communication}). Simple text features",
        "theta$-to-signal link",
        "Not tested",
        "surveillance and pure overlap; the factorial",
    ]
    stale = [fragment for fragment in forbidden_fragments if fragment in tex]
    assert stale == []

    assert models.CONSTRUCT_VALIDITY_SLUGS == models.PART1_SLUGS


def test_second_june_feedback_comment_responses_do_not_regress():
    tex = _compiled_tex()

    required_fragments = [
        "sign checks on only 20 matched cells each",
        "20 matched cells",
        "grids were not nested",
        "weather terms are organic peer-message metaphors",
        "action-prediction diagnostic",
        "how agents map politically loaded prose into action",
        "does not isolate a structurally non-coordination component",
        "evaluated at each realized period-level",
        "simple post-decision rationalization would tend",
        "rotation-matched Llama subset is smaller and closer",
        "modestly outperforms a one-feature sentiment baseline on average",
        "two-decimal rounding",
        "agent 2, period 1007",
    ]
    missing = [fragment for fragment in required_fragments if fragment not in tex]
    assert missing == []

    forbidden_fragments = [
        r"computed separately from realized $\theta$",
        "outperforms a one-feature sentiment baseline across the seven paper models",
        "generic message degradation is part of the effect",
        "Table~\\ref{tab:classifiers} compares text-classifier accuracy on the intelligence briefings under baseline and surveillance conditions",
        "agent 8, period 124",
    ]
    stale = [fragment for fragment in forbidden_fragments if fragment in tex]
    assert stale == []

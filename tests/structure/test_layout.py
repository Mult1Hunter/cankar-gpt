"""Structure law (ADR 0007): the entropy surface of this repo is this file.

Adding a root entry or a stage is a conscious act: edit the allowlist/tuple
here AND cite an ADR in the same PR. A rule without a check is a wish.
"""

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

# Rule 1: the root is frozen. Future entries arrive WITH their phase and an ADR:
# apps/ (Ph4). Ph2.25 eval set lives under registry/evals/ as a provenance-
# stamped partition, not a root evalsets/ dir (ADR 0013 supersedes that line).
ROOT_ALLOWLIST = {
    ".claude",
    ".editorconfig",
    ".env.example",
    ".github",
    ".gitignore",
    ".gitleaks.toml",  # secrets-scan config: manifest sha256 allowance (ADR 0003)
    ".pre-commit-config.yaml",
    "CLAUDE.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "ROADMAP.md",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",  # MIT attribution for vendored nanochat code (ADR 0016)
    "apps",  # deployable frontends: landing-page (ADR 0019); web/api reserved (ADR 0002)
    "cankar",
    "configs",
    "data",
    "docs",
    "ops",
    "pyproject.toml",
    "registry",
    "tests",
    "uv.lock",
}

# Rule 4: a stage exists everywhere or nowhere. New stage = edit this tuple.
STAGES = ("corpus", "tokenizer", "evals", "train")  # train: Phase 2.5/3 (ADR 0016)
NON_STAGE_PACKAGES = {"core", "model"}  # model: vendored GPT, bottom layer (ADR 0016)

BANNED_BASENAMES = {"utils.py", "helpers.py", "common.py", "misc.py"}
BANNED_PATTERNS = ("_v2.", "_new.", "_old.", "temp_", "stuff")

GENERATED_MARKER = "<!-- GENERATED"


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True)
    return out.stdout.splitlines()


def test_root_is_frozen() -> None:
    roots = {f.split("/", 1)[0] for f in tracked_files()}
    unexpected = roots - ROOT_ALLOWLIST
    assert not unexpected, (
        f"new root entries {sorted(unexpected)} - roots only change with an ADR "
        f"and an allowlist edit in the same PR (ADR 0007)"
    )


def test_all_python_lives_in_package_or_tests() -> None:
    """Rule 2: there is no scripts/ directory, ever."""
    bad = [
        f
        for f in tracked_files()
        if f.endswith(".py") and not f.startswith(("cankar/", "tests/", "apps/"))
    ]
    assert not bad, f"python outside cankar/tests/apps: {bad}"


def test_stage_mirror() -> None:
    cankar_pkgs = {
        p.name for p in (REPO / "cankar").iterdir() if p.is_dir() and p.name != "__pycache__"
    }
    assert cankar_pkgs == set(STAGES) | NON_STAGE_PACKAGES, (
        f"cankar/ packages {sorted(cankar_pkgs)} must be exactly stages {STAGES} "
        f"+ {sorted(NON_STAGE_PACKAGES)} - new stage = edit STAGES here + ADR"
    )
    for stage in STAGES:
        assert (REPO / "tests" / stage).is_dir(), f"tests/{stage}/ missing (stage mirror)"
    config_dirs = {p.name for p in (REPO / "configs").iterdir() if p.is_dir()}
    assert config_dirs <= set(STAGES), f"configs/ dirs {config_dirs} not all stages"
    fixture_dirs = {p.name for p in (REPO / "tests" / "fixtures").iterdir() if p.is_dir()}
    assert fixture_dirs <= set(STAGES), f"fixture dirs {fixture_dirs} not all stages"


def test_no_banned_basenames() -> None:
    """Rule 10: junk-drawer names are banned."""
    bad = [
        f
        for f in tracked_files()
        if Path(f).name in BANNED_BASENAMES or any(pat in Path(f).name for pat in BANNED_PATTERNS)
    ]
    assert not bad, f"banned junk-drawer names: {bad}"


def test_no_tracked_working_data() -> None:
    """Rule: data/ and checkpoints/ are never tracked (force-add guard)."""
    bad = [
        f
        for f in tracked_files()
        if f.startswith(("data/", "checkpoints/")) and f != "data/README.md"
    ]
    assert not bad, f"tracked working data: {bad}"


def test_reports_carry_generated_marker() -> None:
    """Rule 5: machine-owned files are marked and never hand-edited."""
    for report in (REPO / "registry" / "reports").glob("*.md"):
        if report.name == "README.md":  # the directory contract, not a report
            continue
        first = report.read_text().splitlines()[0]
        assert first.startswith(GENERATED_MARKER), f"{report.name} missing GENERATED marker"


def test_library_code_has_no_exit_or_print() -> None:
    """ADR 0008 rules 4+5, mechanized: SystemExit and print() live ONLY in
    cli.py modules. AST-based - prose mentioning them does not trip this."""
    import ast

    offenders: list[str] = []
    for path in (REPO / "cankar").rglob("*.py"):
        if path.name == "cli.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
                if getattr(node.exc.func, "id", "") == "SystemExit":
                    offenders.append(f"{path.name}:{node.lineno} raise SystemExit")
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "print":
                offenders.append(f"{path.name}:{node.lineno} print()")
    assert not offenders, f"library code must raise domain errors and log: {offenders}"


def test_directory_contracts_exist() -> None:
    """Rule 7: every governed directory declares its contract in <=30 lines."""
    governed = [
        "cankar",
        "cankar/core",
        "cankar/model",  # non-stage package, README contract enforced (ADR 0016)
        "configs",
        "docs",
        "ops",
        "registry",
        "registry/works",
        "registry/reports",
        "registry/datasets",
        "registry/evals",
        "tests",
    ]
    governed += [f"cankar/{s}" for s in STAGES]
    for d in governed:
        readme = REPO / d / "README.md"
        assert readme.exists(), f"{d}/README.md missing (30-second stranger test)"
        assert len(readme.read_text().splitlines()) <= 30, f"{d}/README.md over 30 lines"

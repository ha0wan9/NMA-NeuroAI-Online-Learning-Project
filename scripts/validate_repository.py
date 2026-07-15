#!/usr/bin/env python3
"""Validate the repository using only the Python standard library."""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
import py_compile
import re
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlsplit


REQUIRED_FILES = (
    ".github/workflows/ci.yml",
    ".gitignore",
    "AGENTS.md",
    "README.md",
    "agents/ci-cd.md",
    "agents/delegation.md",
    "agents/execution-rules.md",
    "agents/memory-writeback-check.md",
    "agents/pre-commit-delivery.md",
    "agents/project-artifacts.md",
    "agents/research-project-contract.md",
    "scripts/validate_repository.py",
)

PROVENANCE_FILES = (
    "agents/delegation.md",
    "agents/execution-rules.md",
    "agents/pre-commit-delivery.md",
    "agents/memory-writeback-check.md",
    "agents/project-artifacts.md",
)

PROVENANCE_FIELDS = (
    "artifact_name",
    "instantiated_from",
    "source_reference",
    "project_scope",
    "owner",
    "review_policy",
    "last_reviewed",
)

MANIFEST_FIELDS = (
    "artifact_name",
    "instantiated_from",
    "source_reference",
    "owner",
    "review_policy",
    "last_reviewed",
)

SKIP_PARTS = {
    ".cache",
    ".git",
    ".harness",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "env",
    "node_modules",
    "venv",
}

INLINE_LINK_RE = re.compile(r"!?\[[^\]\n]*\]\((?P<target><[^>\n]+>|[^)\n]+)\)")
REFERENCE_LINK_RE = re.compile(
    r"^\s{0,3}\[[^\]\n]+\]:\s*(?P<target><[^>\n]+>|\S+)", re.MULTILINE
)
KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (default: parent of this script's directory)",
    )
    return parser.parse_args()


def iter_files(root: Path, suffix: str) -> list[Path]:
    matches: list[Path] = []
    for current, directories, filenames in os.walk(root):
        directories[:] = sorted(
            name
            for name in directories
            if name not in SKIP_PARTS and not name.endswith(".egg-info")
        )
        current_path = Path(current)
        for filename in sorted(filenames):
            path = current_path / filename
            if path.suffix.lower() == suffix and not path.is_symlink():
                matches.append(path)
    return matches


def check_required_files(root: Path, errors: list[str]) -> None:
    for relative in REQUIRED_FILES:
        if not (root / relative).is_file():
            errors.append(f"missing required file: {relative}")


def without_code_fences(text: str) -> str:
    output: list[str] = []
    fence: str | None = None
    for line in text.splitlines():
        stripped = line.lstrip()
        marker = stripped[:3]
        if marker in {"```", "~~~"}:
            if fence is None:
                fence = marker
            elif marker == fence:
                fence = None
            continue
        if fence is None:
            output.append(re.sub(r"`[^`\n]*`", "", line))
    return "\n".join(output)


def link_path(raw_target: str) -> str | None:
    target = raw_target.strip()
    if target.startswith("<"):
        closing = target.find(">")
        if closing == -1:
            return None
        target = target[1:closing]
    else:
        target = target.split(maxsplit=1)[0]
    target = target.replace(r"\ ", " ")
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc:
        return None
    path = unquote(parsed.path)
    if not path or path.startswith("/"):
        return None
    return path


def link_exists(source: Path, relative_target: str) -> tuple[bool, list[Path]]:
    target = source.parent / relative_target
    candidates = [target]
    if not target.suffix:
        candidates.extend(
            (target.with_suffix(".md"), target / "README.md", target / "index.md")
        )
    return any(candidate.exists() for candidate in candidates), candidates


def check_markdown_links(root: Path, markdown_files: list[Path], errors: list[str]) -> None:
    for source in markdown_files:
        try:
            text = source.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            errors.append(f"cannot read Markdown file {source.relative_to(root)}: {exc}")
            continue
        visible_text = without_code_fences(text)
        matches = list(INLINE_LINK_RE.finditer(visible_text))
        matches.extend(REFERENCE_LINK_RE.finditer(visible_text))
        for match in matches:
            raw_target = match.group("target")
            relative_target = link_path(raw_target)
            if relative_target is None:
                continue
            exists, candidates = link_exists(source, relative_target)
            if not exists:
                tried = ", ".join(
                    str(candidate.relative_to(root))
                    if candidate.is_relative_to(root)
                    else str(candidate)
                    for candidate in candidates
                )
                errors.append(
                    f"broken link in {source.relative_to(root)}: "
                    f"{raw_target!r} (tried: {tried})"
                )


def scalar_value(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'\"', "'"}:
        value = value[1:-1]
    return value.strip("`")


def parse_frontmatter(path: Path, root: Path, errors: list[str]) -> dict[str, str]:
    relative = path.relative_to(root)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        errors.append(f"cannot read provenance artifact {relative}: {exc}")
        return {}
    if not lines or lines[0].strip() != "---":
        errors.append(f"{relative}: provenance frontmatter must be the first block")
        return {}
    try:
        closing = next(index for index in range(1, len(lines)) if lines[index].strip() == "---")
    except StopIteration:
        errors.append(f"{relative}: provenance frontmatter has no closing delimiter")
        return {}

    values: dict[str, str] = {}
    for line_number, line in enumerate(lines[1:closing], start=2):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if line[:1].isspace() or ":" not in line:
            errors.append(f"{relative}:{line_number}: frontmatter must contain scalar key-value pairs")
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if not KEY_RE.fullmatch(key):
            errors.append(f"{relative}:{line_number}: invalid frontmatter key {key!r}")
            continue
        if key in values:
            errors.append(f"{relative}:{line_number}: duplicate frontmatter key {key!r}")
            continue
        if not value or value in {"|", ">"} or value[0] in "[{&*":
            errors.append(f"{relative}:{line_number}: {key!r} must have a scalar value")
            continue
        values[key] = scalar_value(value)
    return values


def check_provenance(root: Path, errors: list[str]) -> dict[str, dict[str, str]]:
    artifacts: dict[str, dict[str, str]] = {}
    for relative in PROVENANCE_FILES:
        path = root / relative
        if not path.is_file():
            continue
        values = parse_frontmatter(path, root, errors)
        artifacts[relative] = values
        for field in PROVENANCE_FIELDS:
            if not values.get(field):
                errors.append(f"{relative}: missing required provenance field {field!r}")
        instantiated_from = values.get("instantiated_from")
        if instantiated_from and not instantiated_from.startswith("project-meta/templates/"):
            errors.append(
                f"{relative}: instantiated_from must begin with "
                "'project-meta/templates/'"
            )
        owner = values.get("owner")
        if owner and owner not in {"agent-facing", "shared-user-facing", "local-user"}:
            errors.append(f"{relative}: invalid owner {owner!r}")
        reviewed = values.get("last_reviewed")
        if reviewed:
            try:
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", reviewed):
                    raise ValueError
                dt.date.fromisoformat(reviewed)
            except ValueError:
                errors.append(f"{relative}: last_reviewed must be an ISO date (YYYY-MM-DD)")
    return artifacts


def parse_manifest(path: Path) -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    in_artifacts = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "Artifacts:":
            in_artifacts = True
            continue
        if not in_artifacts:
            continue
        start = re.match(r"^\s*-\s+path:\s*(.+?)\s*$", line)
        if start:
            current = {"path": scalar_value(start.group(1))}
            entries[current["path"]] = current
            continue
        field = re.match(r"^\s{2,}([A-Za-z_][A-Za-z0-9_-]*):\s*(.+?)\s*$", line)
        if current is not None and field:
            current[field.group(1)] = scalar_value(field.group(2))
            continue
        if current is not None and line.strip() and not line[:1].isspace():
            break
    return entries


def check_manifest(
    root: Path, artifacts: dict[str, dict[str, str]], errors: list[str]
) -> None:
    manifest_path = root / "agents/project-artifacts.md"
    if not manifest_path.is_file():
        return
    try:
        entries = parse_manifest(manifest_path)
    except (OSError, UnicodeError) as exc:
        errors.append(f"cannot read artifact manifest: {exc}")
        return
    if not entries:
        errors.append("agents/project-artifacts.md: no parseable 'Artifacts:' entries")
        return

    governed = [relative for relative in PROVENANCE_FILES if relative != "agents/project-artifacts.md"]
    for relative in governed:
        if relative not in artifacts:
            continue
        entry = entries.get(relative)
        if entry is None:
            errors.append(f"agents/project-artifacts.md: missing entry for {relative}")
            continue
        for field in MANIFEST_FIELDS:
            artifact_value = artifacts[relative].get(field)
            if artifact_value and entry.get(field) != artifact_value:
                errors.append(
                    f"agents/project-artifacts.md: {relative} field {field!r} "
                    f"does not agree with artifact frontmatter"
                )
        if not entry.get("refresh_trigger"):
            errors.append(f"agents/project-artifacts.md: {relative} is missing refresh_trigger")

    manifest_entry = entries.get("agents/project-artifacts.md")
    if manifest_entry is not None:
        for field in MANIFEST_FIELDS:
            artifact_value = artifacts.get("agents/project-artifacts.md", {}).get(field)
            if artifact_value and manifest_entry.get(field) != artifact_value:
                errors.append(
                    f"agents/project-artifacts.md: self-entry field {field!r} "
                    f"does not agree with frontmatter"
                )


def check_tracked_preferences(root: Path, errors: list[str]) -> None:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        if (root / ".git").exists():
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            errors.append(f"unable to inspect tracked files with git: {detail}")
        return
    tracked = result.stdout.decode("utf-8", errors="surrogateescape").split("\0")
    forbidden = sorted(
        path for path in tracked if Path(path).name in {"USER.md", "USER.template.md"}
    )
    for path in forbidden:
        errors.append(f"local preference file must not be tracked: {path}")


def check_python(python_files: list[Path], root: Path, errors: list[str]) -> None:
    with tempfile.TemporaryDirectory(prefix="repository-validation-") as temporary:
        output = Path(temporary)
        for index, source in enumerate(python_files):
            try:
                py_compile.compile(
                    str(source),
                    cfile=str(output / f"{index}.pyc"),
                    doraise=True,
                )
            except py_compile.PyCompileError as exc:
                errors.append(f"Python compile failed for {source.relative_to(root)}: {exc.msg}")


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    errors: list[str] = []
    if not root.is_dir():
        print(f"Repository root does not exist or is not a directory: {root}", file=sys.stderr)
        return 2

    check_required_files(root, errors)
    markdown_files = iter_files(root, ".md")
    python_files = iter_files(root, ".py")
    check_markdown_links(root, markdown_files, errors)
    artifacts = check_provenance(root, errors)
    check_manifest(root, artifacts, errors)
    check_tracked_preferences(root, errors)
    check_python(python_files, root, errors)

    if errors:
        print(f"Repository validation failed with {len(errors)} error(s):", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(
        "Repository validation passed: "
        f"{len(REQUIRED_FILES)} required files, "
        f"{len(markdown_files)} Markdown files, "
        f"{len(python_files)} Python files, and "
        f"{len(artifacts)} provenance artifacts checked."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

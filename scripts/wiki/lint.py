#!/usr/bin/env python3
"""Structural linter for the local docs-as-code wiki."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WIKI_DIR = ROOT / "wiki"
INDEX = WIKI_DIR / "index.md"

ALLOWED_TYPES = {
    "analysis",
    "concept",
    "control",
    "entity",
    "project",
    "question",
    "source",
    "tool",
}
ALLOWED_STATUS = {"draft", "reviewed", "stable", "deprecated"}
REQUIRED_KEYS = {"title", "type", "status", "updated", "sources", "tags"}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
KEBAB_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*\.md$")
WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")

REQUIRED_SECTIONS = {
    "analysis": {"Summary", "Details", "Evidence", "Links", "Open Questions"},
    "concept": {"Summary", "Details", "Evidence", "Links", "Open Questions"},
    "control": {"Summary", "Details", "Evidence", "Links"},
    "entity": {"Summary", "Details", "Evidence", "Links", "Open Questions"},
    "project": {"Summary", "Details", "Evidence", "Links", "Open Questions"},
    "question": {"Question", "Current State", "Evidence", "Links"},
    "source": {"Summary", "Key Points", "Evidence", "Links"},
    "tool": {"Summary", "Usage", "Evidence", "Links"},
}


def wiki_pages() -> list[Path]:
    return sorted(
        path
        for path in WIKI_DIR.rglob("*.md")
        if "templates" not in path.relative_to(WIKI_DIR).parts
    )


def parse_frontmatter(path: Path) -> tuple[dict[str, object], str, list[str]]:
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text, ["missing opening frontmatter fence"]

    end_index = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_index = index
            break
    if end_index is None:
        return {}, text, ["missing closing frontmatter fence"]

    frontmatter_lines = lines[1:end_index]
    body = "\n".join(lines[end_index + 1 :])
    data: dict[str, object] = {}
    current_list_key: str | None = None

    for raw_line in frontmatter_lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("- "):
            if current_list_key is None:
                errors.append(f"list item without key: {stripped}")
                continue
            value = stripped[2:].strip()
            assert isinstance(data[current_list_key], list)
            data[current_list_key].append(value)
            continue

        if ":" not in line:
            errors.append(f"frontmatter line is not key/value: {line}")
            current_list_key = None
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            errors.append(f"empty frontmatter key: {line}")
            current_list_key = None
            continue

        if value == "":
            data[key] = []
            current_list_key = key
        elif value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            data[key] = [item.strip().strip("\"'") for item in inner.split(",") if item.strip()]
            current_list_key = None
        else:
            data[key] = value.strip("\"'")
            current_list_key = None

    return data, body, errors


def body_headings(body: str) -> set[str]:
    headings: set[str] = set()
    for line in body.splitlines():
        match = re.match(r"^##+\s+(.+?)\s*$", line)
        if match:
            headings.add(match.group(1).strip())
    return headings


def has_h1(body: str) -> bool:
    return any(re.match(r"^#\s+\S", line) for line in body.splitlines())


def check_page(path: Path, stems: set[str], index_text: str) -> list[str]:
    rel = path.relative_to(ROOT)
    errors: list[str] = []

    if path.name != "README.md" and not KEBAB_RE.match(path.name):
        errors.append(f"{rel}: filename must be lowercase kebab-case")

    metadata, body, parse_errors = parse_frontmatter(path)
    errors.extend(f"{rel}: {error}" for error in parse_errors)
    if parse_errors:
        return errors

    missing_keys = REQUIRED_KEYS - set(metadata)
    if missing_keys:
        errors.append(f"{rel}: missing frontmatter keys: {', '.join(sorted(missing_keys))}")

    page_type = metadata.get("type")
    if page_type not in ALLOWED_TYPES:
        errors.append(f"{rel}: invalid type {page_type!r}")

    status = metadata.get("status")
    if status not in ALLOWED_STATUS:
        errors.append(f"{rel}: invalid status {status!r}")

    updated = metadata.get("updated")
    if not isinstance(updated, str) or not DATE_RE.match(updated):
        errors.append(f"{rel}: updated must be an ISO date")

    for list_key in ("sources", "tags"):
        if not isinstance(metadata.get(list_key), list):
            errors.append(f"{rel}: {list_key} must be a YAML list")

    sources = metadata.get("sources")
    if isinstance(sources, list):
        if not sources:
            errors.append(f"{rel}: sources must not be empty")
        for source in sources:
            if not isinstance(source, str) or not source:
                errors.append(f"{rel}: source entries must be non-empty strings")
                continue
            if re.match(r"^[a-z]+://", source):
                errors.append(f"{rel}: source must be a local path, not a URL: {source}")
                continue
            source_path = (ROOT / source).resolve()
            try:
                source_path.relative_to(ROOT)
            except ValueError:
                errors.append(f"{rel}: source escapes repository: {source}")
                continue
            if not source_path.exists():
                errors.append(f"{rel}: source path does not exist: {source}")

    if not has_h1(body):
        errors.append(f"{rel}: missing H1")

    if isinstance(page_type, str) and page_type in REQUIRED_SECTIONS:
        missing_sections = REQUIRED_SECTIONS[page_type] - body_headings(body)
        if missing_sections:
            errors.append(
                f"{rel}: missing required sections: {', '.join(sorted(missing_sections))}"
            )

    for target in WIKILINK_RE.findall(body):
        if target not in stems:
            errors.append(f"{rel}: dangling wikilink [[{target}]]")

    if path.name not in {"index.md", "README.md"}:
        stem = path.stem
        if f"[[{stem}]]" not in index_text:
            errors.append(f"{rel}: missing [[{stem}]] coverage in wiki/index.md")

    return errors


def main() -> int:
    if not WIKI_DIR.exists():
        print("wiki-lint failed: wiki/ does not exist", file=sys.stderr)
        return 1
    if not INDEX.exists():
        print("wiki-lint failed: wiki/index.md does not exist", file=sys.stderr)
        return 1

    pages = wiki_pages()
    stems: dict[str, list[Path]] = {}
    for path in pages:
        stems.setdefault(path.stem, []).append(path)

    errors: list[str] = []
    for stem, paths in stems.items():
        if len(paths) > 1:
            rel_paths = ", ".join(str(path.relative_to(ROOT)) for path in paths)
            errors.append(f"duplicate wiki stem {stem!r}: {rel_paths}")

    index_text = INDEX.read_text(encoding="utf-8")
    stem_set = set(stems)
    for path in pages:
        errors.extend(check_page(path, stem_set, index_text))

    if errors:
        print("wiki-lint failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"wiki-lint passed: {len(pages)} pages checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

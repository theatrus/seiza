#!/usr/bin/env python3
"""Reject release diffs that strand published crates on incompatible pins."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
Version = tuple[int, int, int]


@dataclass(frozen=True)
class Package:
    name: str
    version: Version
    dependencies: frozenset[str]
    publishable: bool


@dataclass(frozen=True)
class Violation:
    dependency: Package
    old_dependency: Package
    dependent: Package


def parse_version(value: str) -> Version:
    core = value.split("+", 1)[0].split("-", 1)[0]
    parts = core.split(".")
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise ValueError(f"expected a three-part semver version, got {value!r}")
    return tuple(int(part) for part in parts)  # type: ignore[return-value]


def format_version(version: Version) -> str:
    return ".".join(str(part) for part in version)


def caret_allows(old: Version, new: Version) -> bool:
    """Return whether Cargo's implicit ^old requirement can select new."""
    if new < old:
        return False
    if old[0] != 0:
        return new[0] == old[0]
    if old[1] != 0:
        return new[:2] == old[:2]
    return new == old


def effective_version(package: dict, workspace_version: Version) -> Version:
    value = package.get("version")
    if isinstance(value, str):
        return parse_version(value)
    if isinstance(value, dict) and value.get("workspace") is True:
        return workspace_version
    raise ValueError(f"package {package.get('name', '<unknown>')} has no usable version")


def dependency_names(manifest: dict) -> frozenset[str]:
    names: set[str] = set()

    def add_table(table: object) -> None:
        if not isinstance(table, dict):
            return
        for key, value in table.items():
            if isinstance(value, dict) and isinstance(value.get("package"), str):
                names.add(value["package"])
            else:
                names.add(key)

    add_table(manifest.get("dependencies"))
    add_table(manifest.get("build-dependencies"))
    for target in manifest.get("target", {}).values():
        if isinstance(target, dict):
            add_table(target.get("dependencies"))
            add_table(target.get("build-dependencies"))
    return frozenset(names)


def snapshot(load_manifest: Callable[[str], dict]) -> dict[str, Package]:
    root = load_manifest("Cargo.toml")
    workspace_version = parse_version(root["workspace"]["package"]["version"])
    packages: dict[str, Package] = {}
    for member in root["workspace"]["members"]:
        manifest = load_manifest(f"{member}/Cargo.toml")
        package = manifest["package"]
        publish = package.get("publish", True)
        item = Package(
            name=package["name"],
            version=effective_version(package, workspace_version),
            dependencies=dependency_names(manifest),
            publishable=publish is not False and publish != [],
        )
        packages[item.name] = item
    return packages


def find_violations(
    base: dict[str, Package], current: dict[str, Package]
) -> list[Violation]:
    incompatible: dict[str, tuple[Package, Package]] = {}
    for name, package in current.items():
        old = base.get(name)
        if old is None or old.version == package.version:
            continue
        if not caret_allows(old.version, package.version):
            incompatible[name] = (old, package)

    violations: list[Violation] = []
    for dependent in current.values():
        old_dependent = base.get(dependent.name)
        if not dependent.publishable or old_dependent is None:
            continue
        if old_dependent.version != dependent.version:
            continue
        for dependency_name in dependent.dependencies:
            changed = incompatible.get(dependency_name)
            if changed is not None:
                old_dependency, dependency = changed
                violations.append(
                    Violation(
                        dependency=dependency,
                        old_dependency=old_dependency,
                        dependent=dependent,
                    )
                )
    return sorted(violations, key=lambda item: (item.dependency.name, item.dependent.name))


def load_worktree_manifest(path: str) -> dict:
    with (REPO_ROOT / path).open("rb") as handle:
        return tomllib.load(handle)


def git_manifest_loader(base_ref: str) -> Callable[[str], dict]:
    def load(path: str) -> dict:
        content = subprocess.check_output(
            ["git", "show", f"{base_ref}:{path}"], cwd=REPO_ROOT
        )
        return tomllib.loads(content.decode())

    return load


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {Path(argv[0]).name} <base-git-ref>", file=sys.stderr)
        return 2

    base_ref = argv[1]
    try:
        base = snapshot(git_manifest_loader(base_ref))
        current = snapshot(load_worktree_manifest)
        violations = find_violations(base, current)
    except (KeyError, OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"release version-chain check could not run: {error}", file=sys.stderr)
        return 2

    if violations:
        print("release version-chain check failed:", file=sys.stderr)
        for violation in violations:
            old = format_version(violation.old_dependency.version)
            new = format_version(violation.dependency.version)
            dependent = format_version(violation.dependent.version)
            print(
                f"  - {violation.dependency.name} {old} -> {new} crosses its caret "
                f"compatibility boundary, but published dependent "
                f"{violation.dependent.name} remains at {dependent}",
                file=sys.stderr,
            )
        print(
            "bump every listed dependent and its workspace pin before releasing",
            file=sys.stderr,
        )
        return 1

    print(f"release version-chain check passed against {base_ref}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

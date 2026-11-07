#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("check_release_version_chain.py")
SPEC = importlib.util.spec_from_file_location("release_version_chain", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def package(name, version, dependencies=(), publishable=True):
    return MODULE.Package(name, version, frozenset(dependencies), publishable)


class ReleaseVersionChainTests(unittest.TestCase):
    def test_removed_dependency_still_requires_published_dependent_bump(self):
        base = {
            "core": package("core", (0, 1, 0)),
            "host": package("host", (1, 0, 0), ("core",)),
        }
        current = {
            "core": package("core", (0, 2, 0)),
            "host": package("host", (1, 0, 0)),
        }

        violations = MODULE.find_violations(base, current)
        self.assertEqual(
            [(item.dependency.name, item.dependent.name) for item in violations],
            [("core", "host")],
        )
        current["host"] = package("host", (1, 0, 1))
        self.assertEqual(MODULE.find_violations(base, current), [])

    def test_workspace_aliases_resolve_in_all_dependency_tables(self):
        for table in ("dependencies", "build-dependencies"):
            for target in (False, True):
                with self.subTest(table=table, target=target):
                    dependency_table = {table: {"engine": {"workspace": True}}}
                    manifest = {"package": {"name": "host", "version": "1.0.0"}}
                    if target:
                        manifest["target"] = {"cfg(unix)": dependency_table}
                    else:
                        manifest.update(dependency_table)
                    manifests = {
                        "Cargo.toml": {"workspace": {
                            "package": {"version": "1.0.0"},
                            "members": ["host"],
                            "dependencies": {"engine": {
                                "package": "core", "version": "0.1.0",
                            }},
                        }},
                        "host/Cargo.toml": manifest,
                    }
                    packages = MODULE.snapshot(manifests.__getitem__)
                    self.assertEqual(packages["host"].dependencies, frozenset({"core"}))

    def test_dependency_resolution_preserves_local_names_and_ignores_dev_dependencies(self):
        manifest = {
            "dependencies": {
                "plain": "1.0.0",
                "local": {"package": "local-core", "version": "1.0.0"},
                "inherited": {"workspace": True},
            },
            "dev-dependencies": {"test-only": "1.0.0"},
            "target": {"cfg(unix)": {
                "dev-dependencies": {"target-test-only": "1.0.0"},
            }},
        }
        workspace_dependencies = {
            # A local dependency with the same key does not inherit this rename.
            "local": {"package": "other-core", "version": "1.0.0"},
            "inherited": "1.0.0",
        }
        self.assertEqual(
            MODULE.dependency_names(manifest, workspace_dependencies),
            frozenset({"plain", "local-core", "inherited"}),
        )

    def test_missing_inherited_dependency_is_rejected(self):
        with self.assertRaises(KeyError):
            MODULE.dependency_names(
                {"dependencies": {"missing": {"workspace": True}}}, {}
            )

    def test_pre_one_minor_requires_dependent_bump(self):
        base = {
            "seiza": package("seiza", (0, 12, 2)),
            "seiza-satellites": package(
                "seiza-satellites", (0, 4, 2), ("seiza",)
            ),
        }
        current = {
            "seiza": package("seiza", (0, 13, 0)),
            "seiza-satellites": package(
                "seiza-satellites", (0, 4, 2), ("seiza",)
            ),
        }

        violations = MODULE.find_violations(base, current)

        self.assertEqual(
            [(item.dependency.name, item.dependent.name) for item in violations],
            [("seiza", "seiza-satellites")],
        )

    def test_bumped_dependent_passes(self):
        base = {
            "seiza": package("seiza", (0, 12, 2)),
            "seiza-satellites": package(
                "seiza-satellites", (0, 4, 2), ("seiza",)
            ),
        }
        current = {
            "seiza": package("seiza", (0, 13, 0)),
            "seiza-satellites": package(
                "seiza-satellites", (0, 4, 3), ("seiza",)
            ),
        }

        self.assertEqual(MODULE.find_violations(base, current), [])

    def test_compatible_patch_does_not_require_dependent_bump(self):
        base = {
            "seiza": package("seiza", (0, 12, 1)),
            "seiza-satellites": package(
                "seiza-satellites", (0, 4, 2), ("seiza",)
            ),
        }
        current = {
            "seiza": package("seiza", (0, 12, 2)),
            "seiza-satellites": package(
                "seiza-satellites", (0, 4, 2), ("seiza",)
            ),
        }

        self.assertEqual(MODULE.find_violations(base, current), [])

    def test_any_incompatible_workspace_dependency_is_guarded(self):
        base = {
            "seiza-stacking": package("seiza-stacking", (0, 1, 1)),
            "seiza-cli": package("seiza-cli", (0, 13, 0), ("seiza-stacking",)),
        }
        current = {
            "seiza-stacking": package("seiza-stacking", (0, 2, 0)),
            "seiza-cli": package("seiza-cli", (0, 13, 0), ("seiza-stacking",)),
        }

        violations = MODULE.find_violations(base, current)

        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].dependent.name, "seiza-cli")

    def test_unpublished_dependents_are_ignored(self):
        base = {
            "seiza": package("seiza", (0, 12, 2)),
            "internal": package("internal", (0, 1, 0), ("seiza",), False),
        }
        current = {
            "seiza": package("seiza", (0, 13, 0)),
            "internal": package("internal", (0, 1, 0), ("seiza",), False),
        }

        self.assertEqual(MODULE.find_violations(base, current), [])


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Keep the core crates, the C ABI (seiza-cabi) and Python (seiza-py) in step.

Against `.github/binding-parity.toml`, this checks that:

0. every public field of a core type the C ABI carries (a `[rust.<C type>]`
   entry: `seiza-parallax::VideoOptions` for `VideoRequest`) reaches the C
   type, and every function a core crate exports from its root or a public
   module, with every public method of the objects hosts hold
   (`[core_functions] objects`), is called by the C ABI; or an entry says
   why not;

1. every exported C function maps to a Python name that `seiza-py/seiza.pyi`
   declares, or says why Python has none; and every Python function,
   constructor and method is some C export's twin, or says why it has none;
2. every field of the C ABI's JSON types (its serde structs) has a Python
   name, after the naming conventions, or an entry saying why not;
3. each Python call driven by keyword arguments accepts exactly the
   keywords `seiza.pyi` declares for it;
4. with `--base SHA`, a change to one binding's source comes with a change
   to the other's, or a commit says why not with a `Binding-parity:` line.

Entries are `python` (Python has it under that name), `c` (the C type has
it under that name) or `none` (no counterpart by design, with the reason).
There is no entry for a missing counterpart: add it to the binding that
lacks it.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(".github/binding-parity.toml")
C_SOURCES = Path("seiza-cabi/src")
STUB = Path("seiza-py/seiza.pyi")
PY_SOURCES = Path("seiza-py/src")
TRAILER = "Binding-parity:"
BINDING_CRATES = ("seiza-cabi", "seiza-cli", "seiza-py")

EXPORT = re.compile(r'extern\s+"C"\s+fn\s+(seiza_\w+)')
SERDE_STRUCT = re.compile(
    r"#\[derive\(([^)]*)\)\]\s*((?:#\[[^\]]*\]\s*)*)"
    r"(?:pub(?:\([a-z]+\))?\s+)?struct\s+(\w+)(?:<[^>]*>)?\s*\{(.*?)\n\}",
    re.S,
)
FIELD = re.compile(r"\s*(?:pub(?:\([a-z]+\))?\s+)?(?:r#)?([a-z_][a-z0-9_]*)\s*:")
ANY_STRUCT = re.compile(r"struct\s+(\w+)(?:<[^>]*>)?\s*\{(.*?)\n\}", re.S)
PUB_FIELD = re.compile(r"^\s*pub\s+(?:r#)?([a-z_][a-z0-9_]*)\s*:", re.M)
RAW_STRING = re.compile(r'r(#*)"')
CHAR_LITERAL = re.compile(r"'(?:\\.|[^\\'])'")
NOT_IN_JSON = re.compile(r"\bskip\b|\bskip_serializing\b(?!_if)|\bskip_deserializing\b|\bflatten\b")


@dataclass
class Report:
    problems: list[str] = field(default_factory=list)

    def problem(self, message: str) -> None:
        self.problems.append(message)


def block_end(source: str, start: int) -> int:
    """The index just past the `}` closing the block whose `{` is at
    `start`, stepping over strings, characters and comments."""
    depth, at = 0, start
    while at < len(source):
        if source.startswith("//", at):
            at = source.find("\n", at)
            at = len(source) if at < 0 else at
            continue
        if source.startswith("/*", at):
            at = source.find("*/", at) + 2
            continue
        # Matched in place: a slice per character makes long files quadratic.
        raw = RAW_STRING.match(source, at)
        if raw and (at == 0 or not (source[at - 1].isalnum() or source[at - 1] == "_")):
            closing = '"' + raw.group(1)
            at = source.find(closing, raw.end()) + len(closing)
            continue
        character = source[at]
        if character == '"':
            at += 1
            while at < len(source) and source[at] != '"':
                at += 2 if source[at] == "\\" else 1
            at += 1
            continue
        if character == "'":
            literal = CHAR_LITERAL.match(source, at)
            if literal:
                at = literal.end()
                continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return at + 1
        at += 1
    return len(source)


def without_tests(source: str) -> str:
    """Rust source without its `#[cfg(test)]` modules, wherever they sit."""
    while True:
        match = re.search(r"#\[cfg\(test\)\]\s*(?:pub(?:\([a-z]+\))?\s+)?mod\s+\w+\s*\{", source)
        if match is None:
            return source
        source = source[: match.start()] + source[block_end(source, match.end() - 1):]


def c_source_paths(root: Path) -> list[Path]:
    """The C ABI's Rust sources: every module in seiza-cabi/src."""
    return sorted((root / C_SOURCES).glob("*.rs"))


def c_exports(sources: list[str]) -> set[str]:
    return {name for source in sources for name in EXPORT.findall(without_tests(source))}


def json_types(sources: list[str]) -> dict[str, list[str]]:
    """Each serde struct's JSON fields, by their Rust (snake_case) names."""
    types: dict[str, list[str]] = {}
    for source in sources:
        for derive, _attrs, name, body in SERDE_STRUCT.findall(without_tests(source)):
            if "Serialize" not in derive and "Deserialize" not in derive:
                continue
            fields, attributes = [], ""
            for line in body.split("\n"):
                stripped = line.strip()
                if stripped.startswith("#["):
                    attributes += stripped
                    continue
                if stripped.startswith("//") or not stripped:
                    continue
                match = FIELD.match(line)
                if match and not NOT_IN_JSON.search(attributes):
                    fields.append(match.group(1))
                attributes = ""
            types[name] = fields
    return types


@dataclass
class PythonSurface:
    """What Python declares and uses: qualified names for functions, and
    every name a field may go by."""

    qualified: set[str]
    names: set[str]
    signatures: dict[str, list[str]]


def python_surface(stub: str, rust_sources: list[str]) -> PythonSurface:
    tree = ast.parse(stub)
    qualified: set[str] = set()
    names: set[str] = set()
    signatures: dict[str, list[str]] = {}

    def parameters(function: ast.FunctionDef) -> list[str]:
        arguments = function.args
        every = arguments.posonlyargs + arguments.args + arguments.kwonlyargs
        return [argument.arg for argument in every if argument.arg not in ("self", "cls")]

    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            qualified.add(node.target.id)
        elif isinstance(node, ast.Assign):
            qualified |= {target.id for target in node.targets if isinstance(target, ast.Name)}
        elif isinstance(node, ast.FunctionDef):
            qualified.add(node.name)
            signatures[node.name] = parameters(node)
        elif isinstance(node, ast.ClassDef):
            qualified.add(node.name)
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    qualified.add(f"{node.name}.{item.name}")
                    signatures[f"{node.name}.{item.name}"] = parameters(item)
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    qualified.add(f"{node.name}.{item.target.id}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    for source in rust_sources:
        names |= set(re.findall(r'set_item\(\s*"([A-Za-z_0-9]+)"', source))
        names |= set(re.findall(r'"([a-z_0-9]+)"\s*(?:\|\s*"[a-z_0-9]+"\s*)*=>', source))
        names |= set(re.findall(r'#\[pyo3\(get[^\]]*\)\]\s*(?:pub\s+)?([a-z_0-9]+)', source))
        names |= set(re.findall(r'name\s*=\s*"([A-Za-z_0-9]+)"', source))
    return PythonSurface(qualified, names, signatures)


def python_name(field_name: str, conventions: dict[str, str]) -> str:
    """A C field's name as Python would spell it."""
    for c_suffix, python_suffix in conventions.items():
        if field_name.endswith(c_suffix):
            return field_name[: -len(c_suffix)] + python_suffix
    return field_name


def entry_kind(entry: object, counterpart: str = "python") -> tuple[str, str]:
    """An entry's kind and value: (`counterpart`, name) or ("none", why)."""
    if isinstance(entry, str) and counterpart == "python":
        return "python", entry
    if isinstance(entry, dict) and len(entry) == 1:
        kind, value = next(iter(entry.items()))
        if kind == "gap":
            raise ValueError(
                "gap entries are no longer allowed: add the missing counterpart to the binding that lacks it"
            )
        if kind in (counterpart, "none") and isinstance(value, str) and value.strip():
            return kind, value
    raise ValueError(f"expected {counterpart} or none, with text, got {entry!r}")


def c_structs(sources: list[str]) -> dict[str, list[str]]:
    """Every struct with named fields in the C ABI's source, JSON or not."""
    structs: dict[str, list[str]] = {}
    for source in sources:
        for name, body in ANY_STRUCT.findall(without_tests(source)):
            fields = [match.group(1) for line in body.split("\n") if (match := FIELD.match(line))]
            structs.setdefault(name, fields)
    return structs


def core_struct(root: Path, spec: str) -> tuple[list[str] | None, str]:
    """The public fields of the core type `crate::Type`, or None and why."""
    crate, _, name = spec.partition("::")
    sources = sorted((root / crate / "src").rglob("*.rs"))
    if not sources:
        return None, f"no crate {crate}"
    found = []
    for path in sources:
        source = without_tests(path.read_text())
        for match in re.finditer(rf"\bpub\s+struct\s+{re.escape(name)}\b(?:<[^>]*>)?\s*\{{", source):
            body = source[match.end(): block_end(source, match.end() - 1)]
            found.append(PUB_FIELD.findall(body))
    if len(found) != 1:
        return None, f"{len(found)} definitions of pub struct {name} in {crate}"
    return found[0], ""


def crate_functions(root: Path, crate: str) -> list[str]:
    """The functions a core crate exports from its root: `pub fn` in its
    lib.rs, and functions it `pub use`s from its own modules."""
    sources = {path: without_tests(path.read_text()) for path in sorted((root / crate / "src").rglob("*.rs"))}
    lib = sources.get(root / crate / "src" / "lib.rs", "")
    names = set(re.findall(r"^pub\s+(?:const\s+)?(?:async\s+)?fn\s+(\w+)", lib, re.M))
    used: set[str] = set()
    for group in re.findall(r"^pub\s+use\s+(?:crate::|self::)?[\w:]+::\{([^}]*)\}", lib, re.M):
        used |= {item.strip().split(" as ")[-1].strip() for item in group.split(",")}
    used |= set(re.findall(r"^pub\s+use\s+(?:crate::|self::)?[\w:]+::([a-z_][a-z0-9_]*)\s*;", lib, re.M))
    for name in used:
        if name and name[0].islower() and any(
            re.search(rf"^pub\s+(?:const\s+)?(?:async\s+)?fn\s+{re.escape(name)}\b", source, re.M) for source in sources.values()
        ):
            names.add(name)
    return sorted(names)


def module_functions(root: Path, crate: str) -> list[str]:
    """The functions a core crate's public modules hold: `module::name` for
    each `pub fn` in a module reached through `pub mod` declarations from
    lib.rs. A public module's functions are as public as the root's."""
    found: list[str] = []

    def module_file(directory: Path, name: str) -> Path | None:
        for candidate in (directory / f"{name}.rs", directory / name / "mod.rs"):
            if candidate.exists():
                return candidate
        return None

    def walk(path: list[str], file: Path) -> None:
        source = without_tests(file.read_text())
        if path:
            found.extend(
                "::".join(path + [name])
                for name in re.findall(r"^pub\s+(?:const\s+)?(?:async\s+)?fn\s+(\w+)", source, re.M)
            )
        directory = file.parent if file.name in ("lib.rs", "mod.rs") else file.parent / file.stem
        for name in re.findall(r"^pub\s+mod\s+(\w+)\s*;", source, re.M):
            child = module_file(directory, name)
            if child is not None:
                walk(path + [name], child)

    lib = root / crate / "src" / "lib.rs"
    if lib.exists():
        walk([], lib)
    return sorted(found)


def object_methods(root: Path, spec: str) -> tuple[list[str] | None, str]:
    """The public methods of core type `crate::Type`, from its inherent impls."""
    crate, _, name = spec.partition("::")
    sources = sorted((root / crate / "src").rglob("*.rs"))
    if not sources:
        return None, f"no crate {crate}"
    methods: set[str] = set()
    for path in sources:
        source = without_tests(path.read_text())
        for match in re.finditer(rf"^impl(?:<[^>]*>)?\s+{re.escape(name)}(?:<[^>]*>)?\s*\{{", source, re.M):
            body = source[match.end() - 1: block_end(source, match.end() - 1)]
            methods |= set(re.findall(r"^\s{4}pub\s+(?:const\s+)?(?:async\s+)?fn\s+(\w+)", body, re.M))
    if not methods:
        return None, f"no public methods of {name} in {crate}"
    return sorted(methods), ""


def check_core_functions(report: Report, manifest: dict, root: Path, c_source: str) -> None:
    config = manifest.get("core_functions", {})
    entries = {key: value for key, value in config.items() if key != "objects"}
    wanted: dict[str, str] = {}
    for lib in sorted(root.glob("seiza*/src/lib.rs")):
        crate = lib.parent.parent.name
        if crate in BINDING_CRATES:
            continue
        at_root = crate_functions(root, crate)
        for name in at_root:
            wanted[f"{crate}::{name}"] = rf"(?<![\w.]){re.escape(name)}\s*(?:::<[^>]*>)?\("
        for path in module_functions(root, crate):
            name = path.rsplit("::", 1)[1]
            if name not in at_root:
                wanted[f"{crate}::{path}"] = rf"(?<![\w.]){re.escape(name)}\s*(?:::<[^>]*>)?\("
    for spec in config.get("objects", []):
        methods, why = object_methods(root, spec)
        if methods is None:
            report.problem(f"[core_functions] objects: {spec}: {why}")
            continue
        for name in methods:
            wanted[f"{spec}::{name}"] = rf"(?:\.|::){re.escape(name)}\s*(?:::<[^>]*>)?\("
    # Imports under another name: `use seiza_stacking::{checkpoint_depths as rust_checkpoint_depths}`.
    aliases: dict[str, set[str]] = {}
    for statement in re.findall(r"^\s*use\s+[^;]*;", c_source, re.M):
        for name, alias in re.findall(r"\b([a-z_][a-z0-9_]*)\s+as\s+([a-z_][a-z0-9_]*)\b", statement):
            aliases.setdefault(name, set()).add(alias)
    used: set[str] = set()
    for qualified, pattern in sorted(wanted.items()):
        # The entry for this function, or for the nearest module or crate
        # holding it: "seiza-imgproc::blur::*", then "seiza-imgproc::*".
        parts = qualified.split("::")
        candidates = [qualified] + ["::".join(parts[:end]) + "::*" for end in range(len(parts) - 1, 0, -1)]
        key = next((candidate for candidate in candidates if candidate in entries), None)
        name = qualified.rsplit("::", 1)[1]
        called = re.search(pattern, c_source) is not None or any(
            re.search(rf"(?<![\w.]){re.escape(alias)}\s*\(", c_source) for alias in aliases.get(name, ())
        )
        if key is None:
            if not called:
                report.problem(
                    f"{qualified} is not called by the C ABI: add a C export that calls it (and a "
                    f"Python twin), or add \"{qualified}\" to [core_functions]"
                )
            continue
        used.add(key)
        try:
            kind, value = entry_kind(entries[key], "c")
        except ValueError as error:
            report.problem(f"[core_functions] {key}: {error}")
            continue
        if kind == "c":
            report.problem(f"[core_functions] {key}: give none; a function the C ABI calls needs no entry")
        elif called and key == qualified:
            report.problem(f"[core_functions] {key} says the C ABI does not call it, but it does now: remove the entry")
    for key in sorted(entries.keys() - used):
        report.problem(f"[core_functions] lists {key}, which no core crate or public module exports: remove it")


def check_core(report: Report, manifest: dict, root: Path, structs: dict[str, list[str]]) -> None:
    conventions = manifest.get("conventions", {}).get("suffixes", {})
    for c_type, spec in sorted(manifest.get("rust", {}).items()):
        entries = {key: value for key, value in spec.items() if key != "core"}
        if c_type not in structs:
            report.problem(f"[rust.{c_type}]: the C ABI has no struct {c_type}: remove the entry")
            continue
        core_fields, why = core_struct(root, spec.get("core", ""))
        if core_fields is None:
            report.problem(f"[rust.{c_type}] core = {spec.get('core')!r}: {why}")
            continue
        carried = {python_name(name, conventions) for name in structs[c_type]}
        for name in core_fields:
            present = python_name(name, conventions) in carried
            if name not in entries:
                if not present:
                    report.problem(
                        f"{spec['core']}.{name} does not reach the C ABI's {c_type}: add it there "
                        f"(and to Python), or add {name} to [rust.{c_type}]"
                    )
                continue
            try:
                kind, value = entry_kind(entries[name], "c")
            except ValueError as error:
                report.problem(f"[rust.{c_type}] {name}: {error}")
                continue
            if kind == "c" and value not in structs[c_type]:
                report.problem(f"[rust.{c_type}] {name} says {c_type}.{value} carries it, but {c_type} has no {value}")
            elif kind != "c" and present:
                report.problem(f"[rust.{c_type}] {name} says the C ABI lacks it, but {c_type} has it now: remove the entry")
        for name in sorted(entries.keys() - set(core_fields)):
            report.problem(f"[rust.{c_type}] lists {name}, which {spec['core']} no longer has: remove it")


def check_functions(report: Report, manifest: dict, exports: set[str], surface: PythonSurface) -> None:
    entries = manifest.get("functions", {})
    for name in sorted(exports - entries.keys()):
        report.problem(
            f"{name} is exported but not in [functions]: map it to the Python name that does "
            f"the same (and add that to seiza-py), or say why Python has none"
        )
    for name in sorted(entries.keys() - exports):
        report.problem(f"[functions] lists {name}, which the C ABI no longer exports: remove it")
    for name in sorted(entries.keys() & exports):
        try:
            kind, value = entry_kind(entries[name])
        except ValueError as error:
            report.problem(f"[functions] {name}: {error}")
            continue
        if kind == "python" and value not in surface.qualified:
            report.problem(f"[functions] {name} maps to {value}, which seiza.pyi does not declare")


def python_callables(stub: str) -> list[str]:
    """seiza.pyi's functions, the classes it gives a constructor, and their
    methods (not properties): what Python callers can call."""
    def decorated(node: ast.FunctionDef, name: str) -> bool:
        return any(getattr(d, "id", getattr(d, "attr", None)) == name for d in node.decorator_list)

    names: set[str] = set()
    for node in ast.parse(stub).body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            members = [item for item in node.body if isinstance(item, ast.FunctionDef)]
            if any(item.name == "__init__" for item in members):
                names.add(node.name)
            names |= {
                f"{node.name}.{item.name}"
                for item in members
                if not item.name.startswith("_") and not decorated(item, "property")
            }
    return sorted(names)


def check_python_functions(report: Report, manifest: dict, exports: set[str], stub: str) -> None:
    targets: set[str] = set()
    for entry in manifest.get("functions", {}).values():
        try:
            kind, value = entry_kind(entry)
        except ValueError:
            continue
        if kind == "python":
            targets |= {value, value.removesuffix(".__init__")}
    entries = manifest.get("python_functions", {})
    callables = python_callables(stub)
    for name in callables:
        twinned = name in targets
        if name not in entries:
            if not twinned:
                report.problem(
                    f"Python's {name} is no C export's twin: add a C export that does the same (and "
                    f"map it in [functions]), or add {name} to [python_functions]"
                )
            continue
        try:
            kind, value = entry_kind(entries[name], "c")
        except ValueError as error:
            report.problem(f"[python_functions] {name}: {error}")
            continue
        if twinned:
            report.problem(f"[python_functions] {name}: [functions] maps a C export to it now: remove the entry")
        elif kind == "c" and value not in exports:
            report.problem(f"[python_functions] {name} gives {value}, which the C ABI does not export")
    for name in sorted(entries.keys() - set(callables)):
        report.problem(f"[python_functions] lists {name}, which seiza.pyi does not declare: remove it")


def check_fields(report: Report, manifest: dict, types: dict[str, list[str]], surface: PythonSurface) -> None:
    conventions = manifest.get("conventions", {}).get("suffixes", {})
    whole = manifest.get("types", {})
    fields = manifest.get("fields", {})
    used: set[str] = set()
    for type_name in sorted(whole):
        if type_name not in types:
            report.problem(f"[types] lists {type_name}, which is no longer a JSON type: remove it")
            continue
        try:
            kind, value = entry_kind(whole[type_name])
        except ValueError as error:
            report.problem(f"[types] {type_name}: {error}")
            continue
        if kind == "python":
            report.problem(f"[types] {type_name}: whole types take none, not a Python name")
    for type_name, names in sorted(types.items()):
        if type_name in whole:
            continue
        for name in names:
            key = f"{type_name}.{name}"
            entry_key = key if key in fields else f"*.{name}" if f"*.{name}" in fields else None
            present = name in surface.names or python_name(name, conventions) in surface.names
            if entry_key is None:
                if not present:
                    report.problem(
                        f"{key} has no Python name: add it to seiza-py (spelt "
                        f"{python_name(name, conventions)}), or add \"{key}\" to [fields]"
                    )
                continue
            used.add(entry_key)
            try:
                kind, value = entry_kind(fields[entry_key])
            except ValueError as error:
                report.problem(f"[fields] {entry_key}: {error}")
                continue
            if kind == "python":
                if value not in surface.names:
                    report.problem(f"[fields] {entry_key} maps to {value}, which Python does not use")
            elif present and not entry_key.startswith("*."):
                report.problem(f"[fields] {entry_key} says Python lacks it, but Python has it now: remove the entry")
    for key in sorted(fields.keys() - used):
        report.problem(f"[fields] lists {key}, which no JSON type has: remove it")


def rust_function(source: str, name: str) -> str:
    """The text of top-level Rust function `name`, to the next item."""
    start = re.search(rf"^(?:pub(?:\([a-z]+\))?\s+)?fn {re.escape(name)}\b", source, re.M)
    if start is None:
        return ""
    end = re.search(r"^(?:pub(?:\([a-z]+\))?\s+)?(?:fn|struct|enum|impl|const|static|#\[)", source[start.end():], re.M)
    return source[start.start(): start.end() + (end.start() if end else len(source))]


def outer_arms(body: str) -> set[str]:
    """The string patterns of the first `match` in `body` that has any, at
    its own indentation: arms of matches nested inside them are left out."""
    arms = re.findall(r'^(\s*)"([a-z_0-9]+)"\s*=>', body, re.M)
    if not arms:
        return set()
    indent = arms[0][0]
    return {name for spaces, name in arms if spaces == indent}


def check_keywords(report: Report, manifest: dict, root: Path, surface: PythonSurface) -> None:
    for qualified, spec in sorted(manifest.get("keywords", {}).items()):
        if qualified not in surface.signatures:
            report.problem(f"[keywords] {qualified} is not a function seiza.pyi declares")
            continue
        ignored = set(spec.get("ignore", []))
        declared = set(surface.signatures[qualified]) - ignored
        body = rust_function((root / spec["source"]).read_text(), spec["function"])
        if not body:
            report.problem(f"[keywords] {qualified}: no fn {spec['function']} in {spec['source']}")
            continue
        accepted = outer_arms(body) - ignored
        for name in sorted(declared - accepted):
            report.problem(f"{qualified} declares {name} in seiza.pyi, but {spec['function']}() does not accept it")
        for name in sorted(accepted - declared):
            report.problem(f"{spec['function']}() accepts {name}, but seiza.pyi does not declare it on {qualified}")


def binding_sides(paths: list[str]) -> tuple[bool, bool]:
    """Whether `paths` change the C ABI's source, and Python's."""
    c_side = any(path.startswith("seiza-cabi/src/") for path in paths)
    python_side = any(path.startswith("seiza-py/src/") or path == str(STUB) for path in paths)
    return c_side, python_side


def check_both_sides(report: Report, paths: list[str], messages: str) -> None:
    c_side, python_side = binding_sides(paths)
    if c_side == python_side:
        return
    if re.search(rf"^{re.escape(TRAILER)}\s*\S", messages, re.M):
        return
    changed, unchanged = ("the C ABI", "Python") if c_side else ("Python", "the C ABI")
    report.problem(
        f"this change touches {changed}'s source but not {unchanged}'s. Make the same change "
        f"there (see .github/binding-parity.toml for each function's twin), or add a line "
        f"'{TRAILER} <why the other binding needs nothing>' to a commit message"
    )


def git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    ).stdout


def run(root: Path, base: str | None) -> Report:
    report = Report()
    manifest = tomllib.loads((root / MANIFEST).read_text())
    c_sources = [path.read_text() for path in c_source_paths(root)]
    rust_sources = [path.read_text() for path in sorted((root / PY_SOURCES).glob("*.rs"))]
    surface = python_surface((root / STUB).read_text(), rust_sources)
    check_core(report, manifest, root, c_structs(c_sources))
    check_core_functions(report, manifest, root, "\n".join(without_tests(source) for source in c_sources))
    exports = c_exports(c_sources)
    check_functions(report, manifest, exports, surface)
    check_python_functions(report, manifest, exports, (root / STUB).read_text())
    check_fields(report, manifest, json_types(c_sources), surface)
    check_keywords(report, manifest, root, surface)
    if base:
        paths = git(root, "diff", "--name-only", f"{base}...HEAD").split()
        messages = git(root, "log", "--format=%B", f"{base}..HEAD")
        check_both_sides(report, paths, messages)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", help="also require both bindings to change together since this commit")
    arguments = parser.parse_args()
    report = run(REPO_ROOT, arguments.base)
    for problem in report.problems:
        print(f"binding parity: {problem}", file=sys.stderr)
    if report.problems:
        print(f"{len(report.problems)} binding parity problem(s); see .github/binding-parity.toml", file=sys.stderr)
        return 1
    print("binding parity check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

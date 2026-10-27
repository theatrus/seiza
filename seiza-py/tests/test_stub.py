"""seiza.pyi declares the module as it is.

The binding parity check (.github/scripts/check_binding_parity.py) compares
the C ABI with seiza.pyi, so the stub must be true to the module: every name
it declares exists, and every function takes the parameters it declares.
"""

import ast
import inspect
import pathlib
import typing

import pytest
import seiza

STUB = ast.parse((pathlib.Path(__file__).parents[1] / "seiza.pyi").read_text())


def decorated(node, name):
    return any(
        (isinstance(d, ast.Name) and d.id == name) or (isinstance(d, ast.Attribute) and d.attr == name)
        for d in node.decorator_list
    )


def declared():
    """Each (qualified name, runtime object, stub node) the stub declares."""
    for node in STUB.body:
        if isinstance(node, ast.FunctionDef):
            yield node.name, getattr(seiza, node.name, None), node
        elif isinstance(node, ast.ClassDef):
            owner = getattr(seiza, node.name, None)
            yield node.name, owner, node
            if typing.is_typeddict(owner):
                # These declare dictionary keys, not attributes of the class.
                continue
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    name = item.name
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    name = item.target.id
                else:
                    continue
                yield f"{node.name}.{name}", getattr(owner, name, None), item


DECLARED = list(declared())


def test_calibration_record_types_match_runtime_and_stub():
    source = pathlib.Path(__file__).parents[1] / "src/library_types.py"
    definitions = ast.parse(source.read_text())
    stub_types = {node.name: node for node in STUB.body if isinstance(node, ast.ClassDef)}
    for node in definitions.body:
        if isinstance(node, ast.ClassDef):
            assert ast.dump(node) == ast.dump(stub_types[node.name]), node.name
            value = getattr(seiza, node.name)
            assert typing.is_typeddict(value)
            keys = {item.target.id for item in node.body if isinstance(item, ast.AnnAssign)}
            assert set(value.__annotations__) == keys


@pytest.mark.parametrize("name, value, node", DECLARED, ids=[name for name, _, _ in DECLARED])
def test_every_declared_name_exists(name, value, node):
    assert value is not None, f"seiza.pyi declares {name}, which the module lacks"


def parameter_names(node):
    arguments = node.args
    names = [a.arg for a in arguments.posonlyargs + arguments.args + arguments.kwonlyargs]
    if arguments.vararg:
        names.append("*" + arguments.vararg.arg)
    if arguments.kwarg:
        names.append("**" + arguments.kwarg.arg)
    return [name for name in names if name not in ("self", "cls")]


def runtime_names(value):
    names = []
    for parameter in inspect.signature(value).parameters.values():
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            names.append("*" + parameter.name)
        elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
            names.append("**" + parameter.name)
        else:
            names.append(parameter.name)
    return [name for name in names if name not in ("self", "cls")]


CALLABLES = [
    (name, value, node)
    for name, value, node in DECLARED
    if isinstance(node, ast.FunctionDef)
    and value is not None
    and not decorated(node, "property")
    and not decorated(node, "overload")
    and not decorated(node, "staticmethod")
    and not name.endswith((".__iter__", ".__next__", ".__len__", ".__enter__", ".__exit__"))
]


@pytest.mark.parametrize("name, value, node", CALLABLES, ids=[name for name, _, _ in CALLABLES])
def test_declared_parameters_are_the_modules(name, value, node):
    target = value
    if name.endswith(".__init__"):
        # pyo3 gives a class's constructor signature on the class itself.
        target = getattr(seiza, name.split(".")[0])
    try:
        actual = runtime_names(target)
    except (TypeError, ValueError):
        pytest.skip(f"the module gives no signature for {name}")
    expected = parameter_names(node)
    if actual and actual[-1].startswith("**") and not any(n.startswith("**") for n in expected):
        # The stub spells out each keyword the module takes as **kwargs; the
        # binding parity check holds those to the Rust code that reads them.
        named = actual[:-1]
        assert [n for n in expected if n in named] == named, f"seiza.pyi's {name} differs from the module's"
        return
    assert expected == actual, f"seiza.pyi's {name} differs from the module's"

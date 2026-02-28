from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import yaml


def _load_yaml(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _resolve_object(path: str) -> object:
    module_path, attr_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


def _flatten_keys(payload: dict, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    for key, value in payload.items():
        full = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            keys |= _flatten_keys(value, prefix=full)
        else:
            keys.add(full)
    return keys


def _parse_expected_params(signature: str) -> list[tuple[str, str | None]]:
    inside = signature.strip()[1:-1].strip()
    if not inside:
        return []
    out: list[tuple[str, str | None]] = []
    for token in inside.split(","):
        item = token.strip()
        if "=" in item:
            name, default = item.split("=", 1)
            out.append((name.strip(), default.strip()))
        else:
            out.append((item, None))
    return out


def _assert_signature(path: str, name: str, expected_signature: str) -> None:
    cls = _resolve_object(path)
    member = getattr(cls, name)
    sig = inspect.signature(member)
    actual_params = list(sig.parameters.values())
    if actual_params and actual_params[0].name in {"self", "cls"}:
        actual_params = actual_params[1:]

    expected = _parse_expected_params(expected_signature)
    assert len(actual_params) == len(expected), f"{path}.{name} parameter count mismatch"

    for actual, (exp_name, exp_default) in zip(actual_params, expected, strict=False):
        assert actual.name == exp_name, f"{path}.{name} expected parameter '{exp_name}'"
        if exp_default is None:
            continue
        assert actual.default is not inspect._empty, f"{path}.{name}.{exp_name} missing default"
        assert repr(actual.default) == exp_default, (
            f"{path}.{name}.{exp_name} default mismatch: {repr(actual.default)} != {exp_default}"
        )


def test_parity_manifest_class_contracts() -> None:
    payload = _load_yaml("baseline/parity_manifest.yml")
    classes = payload["classes"]
    assert len(classes) == 16

    for row in classes:
        python_class = row["python_class"]
        cls = _resolve_object(python_class)
        assert inspect.isclass(cls), f"{python_class} must resolve to a class"

        for method in row["public_methods"]:
            name = method["name"]
            signature = method["signature"]
            assert hasattr(cls, name), f"{python_class} missing member '{name}'"
            if signature == "property":
                prop = inspect.getattr_static(cls, name)
                assert isinstance(prop, property), f"{python_class}.{name} should be a property"
            elif signature.startswith("attribute["):
                dataclass_fields = getattr(cls, "__dataclass_fields__", {})
                assert name in dataclass_fields, f"{python_class}.{name} should be a dataclass field"
            else:
                _assert_signature(python_class, name, signature)


def test_parity_manifest_workflow_links_and_tolerances() -> None:
    payload = _load_yaml("baseline/parity_manifest.yml")
    behavior_path = Path(payload["metadata"]["behavior_contracts_file"])
    assert behavior_path.exists()

    workflows = payload["workflows"]
    assert any(row["topic"] == "nSTATPaperExamples" for row in workflows)

    tolerances = _load_yaml("tests/parity/tolerances.yml")
    tol_keys = _flatten_keys(tolerances)

    for row in workflows:
        assert Path(row["notebook"]).exists()
        assert Path(row["help_page"]).exists()
        for metric in row["expected_metrics"]:
            key = metric["tolerance_key"]
            assert key in tol_keys, f"unknown tolerance key: {key}"

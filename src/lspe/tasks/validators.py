"""Deterministic validity gates; invalid outputs never receive VSD credit."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    failure_code: str | None
    payload: Any | None


def validate_response(validator: str, text: str, expected: Any = None) -> ValidationResult:
    match validator:
        case "divergent_words":
            return _divergent_words(text)
        case "alternative_uses":
            return _alternative_uses(text)
        case "exact_answer":
            return _exact_answer(text, expected)
        case "json":
            return _json_object(text, expected)
        case "cross_domain_bridge":
            return _cross_domain_bridge(text)
        case "constrained_creative":
            return _constrained_creative(text, expected)
        case "python_function":
            return _python_function(text, expected)
        case _:
            return ValidationResult(False, f"UNKNOWN_VALIDATOR:{validator}", None)


def _parse_json(text: str) -> tuple[Any | None, str | None]:
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        return None, "INVALID_JSON"


def _divergent_words(text: str) -> ValidationResult:
    payload, error = _parse_json(text)
    if error:
        return ValidationResult(False, error, None)
    if not isinstance(payload, list) or len(payload) != 10:
        return ValidationResult(False, "EXPECTED_TEN_ITEMS", None)
    words = [item.strip() for item in payload if isinstance(item, str)]
    if len(words) != 10 or any(
        not re.fullmatch(r"[A-Za-z][A-Za-z -]{0,30}", word) for word in words
    ):
        return ValidationResult(False, "INVALID_WORD_ITEM", None)
    if len({word.casefold() for word in words}) != 10:
        return ValidationResult(False, "DUPLICATE_WORD", None)
    return ValidationResult(True, None, words)


def _alternative_uses(text: str) -> ValidationResult:
    payload, error = _parse_json(text)
    if error:
        return ValidationResult(False, error, None)
    if not isinstance(payload, list) or not payload:
        return ValidationResult(False, "EXPECTED_NONEMPTY_ARRAY", None)
    required = {"idea", "mechanism", "feasibility"}
    if any(not isinstance(item, dict) or set(item) != required for item in payload):
        return ValidationResult(False, "INVALID_ALTERNATIVE_USE_SCHEMA", None)
    if any(
        not all(isinstance(item[field], str) and item[field].strip() for field in required)
        for item in payload
    ):
        return ValidationResult(False, "EMPTY_ALTERNATIVE_USE_FIELD", None)
    ideas = [item["idea"].strip().casefold() for item in payload]
    if len(ideas) != len(set(ideas)):
        return ValidationResult(False, "DUPLICATE_IDEA", None)
    return ValidationResult(True, None, payload)


def _exact_answer(text: str, expected: Any) -> ValidationResult:
    if expected is None:
        return ValidationResult(False, "MISSING_EXPECTED_ANSWER", None)
    normalized = text.strip()
    return ValidationResult(
        normalized == str(expected).strip(),
        None if normalized == str(expected).strip() else "WRONG_ANSWER",
        normalized,
    )


def _json_object(text: str, expected: Any = None) -> ValidationResult:
    payload, error = _parse_json(text)
    if error:
        return ValidationResult(False, error, None)
    if not isinstance(payload, dict):
        return ValidationResult(False, "EXPECTED_JSON_OBJECT", payload)
    if expected is not None and payload != expected:
        return ValidationResult(False, "JSON_SCHEMA_MISMATCH", payload)
    return ValidationResult(True, None, payload)


def _cross_domain_bridge(text: str) -> ValidationResult:
    payload, error = _parse_json(text)
    required = {"source_principle", "target_application", "mechanism"}
    if error:
        return ValidationResult(False, error, None)
    if not isinstance(payload, dict) or set(payload) != required:
        return ValidationResult(False, "INVALID_BRIDGE_SCHEMA", None)
    if not all(isinstance(payload[key], str) and payload[key].strip() for key in required):
        return ValidationResult(False, "EMPTY_BRIDGE_FIELD", None)
    return ValidationResult(True, None, payload)


def _constrained_creative(text: str, expected: Any) -> ValidationResult:
    lines = [line.strip() for line in text.strip().splitlines()]
    if len(lines) != 4:
        return ValidationResult(False, "EXPECTED_FOUR_LINES", None)
    if any(not 6 <= len(line.split()) <= 14 for line in lines):
        return ValidationResult(False, "LINE_WORD_COUNT", None)
    required = expected.get("required_words", []) if isinstance(expected, dict) else []
    joined = "\n".join(lines).casefold()
    if any(str(word).casefold() not in joined for word in required):
        return ValidationResult(False, "MISSING_REQUIRED_WORD", None)
    return ValidationResult(True, None, lines)


def _python_function(text: str, expected: Any) -> ValidationResult:
    """Run a tiny pure-function submission in a time-limited isolated subprocess."""

    if not isinstance(expected, dict):
        return ValidationResult(False, "MISSING_CODE_TESTS", None)
    function_name = expected.get("function_name")
    cases = expected.get("cases")
    if not isinstance(function_name, str) or not isinstance(cases, list) or not cases:
        return ValidationResult(False, "MALFORMED_CODE_TESTS", None)
    if len(text) > 8_000:
        return ValidationResult(False, "CODE_TOO_LONG", None)
    try:
        tree = ast.parse(text, mode="exec")
    except SyntaxError:
        return ValidationResult(False, "CODE_SYNTAX_ERROR", None)
    if any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)):
        return ValidationResult(False, "CODE_IMPORT_FORBIDDEN", None)
    forbidden = {"__import__", "open", "eval", "exec", "compile", "input", "breakpoint"}
    if any(isinstance(node, ast.Name) and node.id in forbidden for node in ast.walk(tree)):
        return ValidationResult(False, "CODE_UNSAFE_BUILTIN", None)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != function_name:
        return ValidationResult(False, "CODE_REQUIRED_FUNCTION_MISSING", None)
    harness = _code_harness(function_name, cases)
    try:
        with tempfile.TemporaryDirectory(prefix="lspe-code-") as directory:
            root = Path(directory)
            (root / "solution.py").write_text(text, encoding="utf-8")
            (root / "harness.py").write_text(harness, encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, "-I", "harness.py"],
                cwd=root,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=2,
                check=False,
                env={"PYTHONIOENCODING": "utf-8", "PATH": "/usr/bin:/bin"},
                preexec_fn=_restrict_code_resources,
            )
    except subprocess.TimeoutExpired:
        return ValidationResult(False, "CODE_TIMEOUT", None)
    if completed.returncode != 0:
        return ValidationResult(False, "CODE_TEST_FAILED", None)
    return ValidationResult(True, None, {"function_name": function_name, "case_count": len(cases)})


def _code_harness(function_name: str, cases: list[Any]) -> str:
    encoded_cases = json.dumps(cases, sort_keys=True)
    return (
        "import importlib.util\n"
        "import json\n"
        "import sys\n"
        "spec = importlib.util.spec_from_file_location('solution', 'solution.py')\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        f"function = getattr(module, {function_name!r}, None)\n"
        "if not callable(function): sys.exit(2)\n"
        f"for case in json.loads({encoded_cases!r}):\n"
        "    if function(*case['args']) != case['expected']: sys.exit(3)\n"
    )


def _restrict_code_resources() -> None:
    """Apply conservative CPU and address-space limits in the code-test child."""

    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
        resource.setrlimit(resource.RLIMIT_AS, (128 * 1024 * 1024, 128 * 1024 * 1024))
    except (ImportError, OSError, ValueError):
        # The outer wall-clock timeout remains a mandatory portable limit.
        return

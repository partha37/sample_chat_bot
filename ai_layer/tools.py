"""Small, dependency-free tools the chat graph can call.

These exist so the debugger has something interesting to trace: you can watch
the router decide to call a tool, then watch the tool node execute and feed
the result back into the answer node.
"""

from __future__ import annotations

import ast
import operator
from datetime import datetime, timezone

from langchain_core.tools import tool

# Only these AST node types are allowed inside the calculator.
_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Constant,
    ast.Tuple,
    ast.List,
)

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("only numeric literals are allowed")
    if isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        handler = _OPERATORS.get(type(node.op))
        if handler is None:
            raise ValueError(f"operator {type(node.op).__name__} is not allowed")
        return handler(left, right)
    if isinstance(node, ast.UnaryOp):
        value = _safe_eval(node.operand)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
        raise ValueError("unary operator is not allowed")
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_safe_eval(item) for item in node.elts]  # type: ignore[return-value]
    raise ValueError(f"expression element {type(node).__name__} is not allowed")


@tool
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression such as '12 * (3 + 4)'.

    Only + - * / // % and ** are supported, with numeric literals only.
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        return f"Could not parse expression: {exc.msg}"

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return f"Expression contains a disallowed element: {type(node).__name__}"

    try:
        result = _safe_eval(tree)
    except Exception as exc:  # noqa: BLE001 - surfaced back to the user
        return f"Could not evaluate expression: {exc}"

    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


@tool
def current_datetime() -> str:
    """Return the current UTC date and time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@tool
def text_stats(text: str) -> str:
    """Return character, word and line counts for the given text."""
    lines = text.splitlines()
    words = [word for word in text.split() if word]
    return (
        f"characters={len(text)} words={len(words)} lines={len(lines)}"
    )


ALL_TOOLS = [calculator, current_datetime, text_stats]

TOOLS_BY_NAME = {tool_.name: tool_ for tool_ in ALL_TOOLS}

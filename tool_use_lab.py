"""Lab | Give the Model Hands — native Gemini function calling.

Two tools (lookup_order, calculate) are described to the model with schemas.
The model only ever *asks* for a tool; this script validates the arguments,
runs the real function, and feeds the result back until the model answers.

Usage:
    export GOOGLE_API_KEY="..."   # never committed
    python tool_use_lab.py
"""

import ast
import json
import operator
import os
import re
import sys
import time

from google import genai
from google.genai import types

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
ORDERS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orders.json")

with open(ORDERS_PATH, encoding="utf-8") as f:
    ORDERS = json.load(f)


# --------------------------------------------------------------------------
# The real tools (plain Python — the model never runs these itself)
# --------------------------------------------------------------------------

def lookup_order(order_id: str) -> dict:
    """Return item, price, purchase date and warranty for an order id."""
    order = ORDERS.get(order_id)
    if order is None:
        return {"error": f"order not found: {order_id!r}"}
    return {"order_id": order_id, **order}


# Safe arithmetic evaluator: only numbers and + - * / // % ** are allowed.
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"unsupported expression element: {ast.dump(node)}")


def calculate(expression: str) -> dict:
    """Evaluate a simple arithmetic expression (no names, no calls)."""
    try:
        result = _safe_eval(ast.parse(expression, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError) as exc:
        return {"error": f"could not evaluate {expression!r}: {exc}"}
    return {"expression": expression, "result": result}


# --------------------------------------------------------------------------
# Tool schemas the model sees, plus argument validation
# --------------------------------------------------------------------------

TOOL_DECLARATIONS = [
    {
        "name": "lookup_order",
        "description": (
            "Look up an order in the order database by its id (e.g. 'A1001'). "
            "Returns the item name, price in USD, purchase date and warranty "
            "length in months, or an error if the order does not exist."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order id, e.g. 'A1001'."}
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "calculate",
        "description": (
            "Evaluate a simple arithmetic expression, e.g. '1200 * 3'. Supports "
            "+ - * / // % ** and parentheses. Use this for any exact math "
            "instead of computing it yourself."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Arithmetic expression to evaluate."}
            },
            "required": ["expression"],
        },
    },
]

REGISTRY = {"lookup_order": lookup_order, "calculate": calculate}


def validate_and_run(name: str, args: dict) -> dict:
    """Validate a model-requested call against the schema, then execute it."""
    decl = next((d for d in TOOL_DECLARATIONS if d["name"] == name), None)
    if decl is None:
        return {"error": f"unknown tool: {name!r}"}
    params = decl["parameters"]
    for required in params["required"]:
        if required not in args:
            return {"error": f"missing required argument {required!r} for {name}"}
    for key, value in args.items():
        if key not in params["properties"]:
            return {"error": f"unexpected argument {key!r} for {name}"}
        if params["properties"][key]["type"] == "string" and not isinstance(value, str):
            return {"error": f"argument {key!r} must be a string"}
    return REGISTRY[name](**args)


# --------------------------------------------------------------------------
# The model -> tool -> model loop
# --------------------------------------------------------------------------

def make_client() -> genai.Client:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("Set GOOGLE_API_KEY first.")
    return genai.Client(api_key=api_key)


def generate_with_retry(client, contents, config, attempts: int = 4):
    """Call the model, waiting out free-tier 429 rate limits if needed."""
    from google.genai import errors

    for attempt in range(attempts):
        try:
            return client.models.generate_content(
                model=MODEL, contents=contents, config=config
            )
        except errors.APIError as exc:
            if exc.code != 429 or attempt == attempts - 1:
                raise
            match = re.search(r"retry in (\d+(?:\.\d+)?)s", str(exc))
            delay = float(match.group(1)) + 2 if match else 35.0
            print(f"  .. rate limited, retrying in {delay:.0f}s")
            time.sleep(delay)


def run_conversation(client: genai.Client, question: str, max_rounds: int = 6) -> None:
    print(f"\n{'=' * 70}\nUSER: {question}\n{'=' * 70}")

    config = types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=TOOL_DECLARATIONS)],
        # Disable the SDK's auto-execution so the loop stays in *our* hands.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    contents = [types.Content(role="user", parts=[types.Part(text=question)])]

    for _ in range(max_rounds):
        response = generate_with_retry(client, contents, config)
        calls = response.function_calls
        if not calls:
            print(f"\nFINAL ANSWER: {response.text}")
            return
        contents.append(response.candidates[0].content)
        for call in calls:
            args = dict(call.args)
            print(f"\n  -> model requested tool: {call.name}({json.dumps(args)})")
            result = validate_and_run(call.name, args)
            print(f"  <- tool result: {json.dumps(result)}")
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_function_response(
                        name=call.name, response={"result": result}
                    )],
                )
            )
    print("\nStopped: model did not produce a final answer within the round limit.")


def main() -> None:
    client = make_client()
    # 1) Needs both tools in sequence: lookup_order for the price, calculate for the math.
    run_conversation(client, "For order A1001, what would the total be if I bought three of them?")
    # 2) Needs no tool at all — the model should just answer.
    run_conversation(client, "What can you help me with?")
    # 3) Stretch: bad argument on purpose — non-existent order.
    run_conversation(client, "What did I buy in order A9999 and how much was it?")


if __name__ == "__main__":
    main()

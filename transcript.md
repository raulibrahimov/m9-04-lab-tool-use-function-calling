# Transcript — Lab | Give the Model Hands

Real output of `python tool_use_lab.py` (model: `gemini-3-flash-preview`, native
Gemini function calling, manual tool-use loop with argument validation).
The `2.5-flash` / `2.5-flash-lite` free-tier daily quotas were exhausted at run
time, so the script was run with `GEMINI_MODEL=gemini-3-flash-preview`.

## 1) Question that needs BOTH tools in sequence

```
======================================================================
USER: For order A1001, what would the total be if I bought three of them?
======================================================================

  -> model requested tool: lookup_order({"order_id": "A1001"})
  <- tool result: {"order_id": "A1001", "item": "laptop", "price": 1200, "purchased": "2026-05-20", "warranty_months": 12}

  -> model requested tool: calculate({"expression": "1200 * 3"})
  <- tool result: {"expression": "1200 * 3", "result": 3600}

FINAL ANSWER: If you bought three of the items from order A1001 (a laptop priced at $1,200), the total would be $3,600.
```

The loop went round twice: `lookup_order` fetched the price, then `calculate`
did the exact math. The model only *requested* each call — the script validated
the arguments against the schema and executed the real Python functions.

## 2) Question that needs NO tool

```
======================================================================
USER: What can you help me with?
======================================================================

FINAL ANSWER: I can assist you with a few specific tasks:

1.  **Order Lookups**: If you provide an order ID (e.g., A1001), I can find details such as the item name, price, purchase date, and warranty information.
2.  **Calculations**: I can perform math operations for you, from simple arithmetic to more complex expressions.

How can I help you today?
```

No tool call was emitted — the model answered directly.

## 3) Stretch: bad argument on purpose (non-existent order)

```
======================================================================
USER: What did I buy in order A9999 and how much was it?
======================================================================

  -> model requested tool: lookup_order({"order_id": "A9999"})
  <- tool result: {"error": "order not found: 'A9999'"}
  .. rate limited, retrying in 4s
  .. rate limited, retrying in 60s

FINAL ANSWER: I'm sorry, but I couldn't find an order with the ID A9999. Please double-check the order number and let me know if there's anything else I can help you with.
```

`lookup_order` returned a structured `"order not found"` error instead of
raising, and the model turned it into a graceful answer. (The `rate limited`
lines are the script waiting out free-tier 429s — part of normal operation on
the free tier, not an error.)

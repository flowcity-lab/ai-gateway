"""Kosten-Berechnung Anthropic (USD pro 1M Tokens). Quelle: Anthropic Pricing 2026-04."""

_COSTS = {
    "claude-opus-4-7":   {"input": 15.00, "output": 75.00},
    "claude-opus-4-5":   {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-5": {"input":  3.00, "output": 15.00},
    "claude-sonnet-4":   {"input":  3.00, "output": 15.00},
    "claude-haiku-4-5":  {"input":  1.00, "output":  5.00},
    "claude-haiku-4":    {"input":  1.00, "output":  5.00},
}


def anthropic_cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    """Errechnet die Kosten in USD fuer einen Claude-Call. Unbekannte Modelle: 0."""
    rates = _COSTS.get(model)
    if not rates or (tokens_in <= 0 and tokens_out <= 0):
        return 0.0
    return (tokens_in / 1_000_000.0) * rates["input"] + (tokens_out / 1_000_000.0) * rates["output"]

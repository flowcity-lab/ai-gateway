"""
Phase 1.5 — Intent Extraction (Claude Haiku).

Nimmt freien Trainer-Hinweistext + Kontext, gibt strukturierte Intents zurueck:
  * display_text       — bereinigter Hinweistext fuer die Rechnung
  * recipient_override — abweichender Empfaenger (oder None)
  * tax_rule           — Steuerregel (z.B. {global_rate: 20})
  * layout_rules       — kurze Token-Liste fuer den Renderer

Fehlertoleranz: bei JSON-Parse-Fehlern faellt die Funktion auf einen sicheren
Default zurueck (display_text = original, alles andere None) — der User sieht
seinen Text dann unveraendert auf der Rechnung, statt dass der Render bricht.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional

from .cost import anthropic_cost_usd
from .prompts import INTENT_EXTRACTION_SYSTEM, intent_extraction_user

log = logging.getLogger("ai-gateway.invoice.intents")

# Whitelist akzeptierter Override-Felder. Was nicht hier steht, wird verworfen.
_RECIPIENT_KEYS = ("name", "street", "zip", "city", "country", "vat_id", "email", "phone", "contact_person")


def extract_intents(
    *,
    client,                       # anthropic.Anthropic
    model: str,
    notes: str,
    context: dict,
    max_tokens: int = 1024,
) -> dict:
    """
    Liefert immer ein Dict mit den Schluesseln display_text, recipient_override,
    tax_rule, layout_rules, model, cost_usd, duration_ms, usage. Bei Fehler
    fallen die Intent-Felder auf None/[] zurueck (display_text bleibt = notes).
    """
    started = time.time()
    fallback = _empty_result(notes, model)

    notes = (notes or "").strip()
    if notes == "":
        return fallback

    try:
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=INTENT_EXTRACTION_SYSTEM,
            messages=[{"role": "user", "content": intent_extraction_user(notes, context)}],
        )
    except Exception as e:
        log.warning("intent_extraction: anthropic-call failed: %s", e)
        return fallback

    raw_text = _flatten_text(msg)
    parsed = _parse_json(raw_text)
    usage_in  = getattr(msg.usage, "input_tokens",  0) or 0
    usage_out = getattr(msg.usage, "output_tokens", 0) or 0

    if parsed is None:
        log.warning("intent_extraction: konnte JSON nicht parsen: %r", raw_text[:300])
        out = fallback
    else:
        out = _sanitize(parsed, notes)

    out["model"]       = model
    out["usage"]       = {"input_tokens": usage_in, "output_tokens": usage_out}
    out["cost_usd"]    = round(anthropic_cost_usd(model, usage_in, usage_out), 6)
    out["duration_ms"] = int((time.time() - started) * 1000)
    return out


def _empty_result(notes: str, model: str) -> dict:
    return {
        "display_text":       notes or "",
        "recipient_override": None,
        "tax_rule":           None,
        "layout_rules":       [],
        "model":              model,
        "usage":              {"input_tokens": 0, "output_tokens": 0},
        "cost_usd":           0.0,
        "duration_ms":        0,
    }


def _flatten_text(msg) -> str:
    """Konkateniert alle Text-Blocks der Antwort."""
    parts: list[str] = []
    for block in (getattr(msg, "content", None) or []):
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", "") or "")
    return "\n".join(parts).strip()


def _parse_json(text: str) -> Optional[dict]:
    """Robust: entfernt evtl. Markdown-Fences und parst das erste JSON-Objekt."""
    if not text:
        return None
    cleaned = text.strip()
    # Fence entfernen, falls Claude sie trotz Anweisung setzt.
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Notfall: erstes {…}-Konstrukt extrahieren.
    m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _sanitize(parsed: dict, original_notes: str) -> dict:
    """Filtert/normalisiert Felder. Verworfen wird, was offensichtlich nicht passt."""
    display = str(parsed.get("display_text") or "").strip() or original_notes

    override_raw = parsed.get("recipient_override")
    override = None
    if isinstance(override_raw, dict):
        cleaned = {k: str(v).strip() for k, v in override_raw.items()
                   if k in _RECIPIENT_KEYS and v not in (None, "")}
        override = cleaned or None

    tax_rule = None
    tr_raw = parsed.get("tax_rule")
    if isinstance(tr_raw, dict):
        gr = tr_raw.get("global_rate")
        if isinstance(gr, (int, float)) and 0 <= float(gr) <= 100:
            tax_rule = {"global_rate": float(gr)}

    layout_rules: list[str] = []
    lr_raw = parsed.get("layout_rules")
    if isinstance(lr_raw, list):
        for item in lr_raw:
            s = str(item).strip()
            if s and len(s) <= 80:
                layout_rules.append(s)

    return {
        "display_text":       display,
        "recipient_override": override,
        "tax_rule":           tax_rule,
        "layout_rules":       layout_rules,
    }

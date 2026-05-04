"""
VGSE-Invoice-Pipeline (Vision-Grounded Structural Editing).

Module:
  * intent_extraction — Phase 1.5: freier Hinweistext -> strukturierte Intents (Haiku).
  * document_map      — DOCX -> deterministische Map editierbarer Knoten (Paragraph/Tabelle/Zelle).
  * edit_plan         — Pydantic-Schemas fuer Edit-Plaene (von Phase 2 erzeugt).
  * edit_applier      — Wendet Edit-Plaene deterministisch via python-docx an.
  * prompts           — System- und User-Prompts (Deutsch).
  * cost              — Kosten-Tabelle Anthropic (Token-zu-USD).
  * pipeline          — Orchestriert Phase 2 (Plan generieren) + Phase 5 (Verify) + Render.
"""

from .cost import anthropic_cost_usd
from .document_map import build_document_map
from .edit_applier import apply_edit_plan
from .edit_plan import EditPlan, validate_edit_plan
from .intent_extraction import extract_intents
from .pipeline import generate_invoice_vgse

__all__ = [
    "anthropic_cost_usd",
    "build_document_map",
    "apply_edit_plan",
    "EditPlan",
    "validate_edit_plan",
    "extract_intents",
    "generate_invoice_vgse",
]

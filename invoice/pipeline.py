"""
VGSE-Pipeline-Orchestrator (Phase 2 + 4).

Phase 2: Render DOCX -> Seitenbilder (via doc-processor), bauen Document-Map,
         Claude Sonnet generiert Edit-Plan aus Vision + Map + Daten.
Phase 4: Edit-Plan deterministisch auf DOCX anwenden (python-docx).

Ergebnis: (output_bytes, render_meta), wobei render_meta ai_warning, model,
usage, cost_usd, applier_report enthaelt.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from typing import Optional

import httpx

from .cost import anthropic_cost_usd
from .document_map import build_document_map
from .edit_applier import apply_edit_plan
from .edit_plan import EditPlan, validate_edit_plan
from .prompts import EDIT_PLAN_SYSTEM, edit_plan_user

log = logging.getLogger("ai-gateway.invoice.pipeline")


def generate_invoice_vgse(
    *,
    client,                        # anthropic.Anthropic
    model: str,
    template_bytes: bytes,
    document_data: dict,
    layout_rules: list[str],
    doc_processor_url: str,
    doc_processor_secret: str,
    max_tokens: int = 8192,
    max_pages: int = 10,
) -> tuple[bytes, dict]:
    """
    Fuehrt die VGSE-Pipeline aus und gibt (docx_bytes, meta) zurueck.

    Bei harten Fehlern (Map-Build, Render-Fehler) wirft die Funktion eine
    RuntimeError — der Caller in main.py macht daraus einen failed-Callback.
    """
    started = time.time()

    # Phase 1 — Document-Map aus dem DOCX-Template aufbauen.
    document_map = build_document_map(template_bytes)

    # Phase 2a — Vorlage in Seiten-PNGs rendern (doc-processor).
    pages = _render_pages_via_doc_processor(
        template_bytes, doc_processor_url, doc_processor_secret, max_pages
    )

    # Phase 2b — Claude Sonnet bauen den Edit-Plan.
    plan_raw, usage_in, usage_out = _call_claude_for_edit_plan(
        client=client,
        model=model,
        max_tokens=max_tokens,
        document_map=document_map,
        document_data=document_data,
        layout_rules=layout_rules,
        page_pngs=[p["png_b64"] for p in pages],
    )

    plan = validate_edit_plan(plan_raw)

    # Phase 4 — Edit-Plan auf das DOCX anwenden.
    output_bytes, applier_report = apply_edit_plan(template_bytes, plan)

    duration_ms = int((time.time() - started) * 1000)
    cost_usd = round(anthropic_cost_usd(model, usage_in, usage_out), 6)

    # Verification: wenn Operationen geskippt wurden, vermerken wir das als
    # ai_warning — die Rechnung wird trotzdem ausgeliefert (Decision M5).
    ai_warning: Optional[str] = None
    if applier_report.get("skipped"):
        n_skipped = len(applier_report["skipped"])
        ai_warning = (
            f"{n_skipped} Operation(en) im Edit-Plan konnten nicht angewendet werden — "
            f"das Ergebnis bitte vor dem Versand pruefen."
        )

    meta = {
        "model":           model,
        "usage":           {"input_tokens": usage_in, "output_tokens": usage_out},
        "cost_usd":        cost_usd,
        "duration_ms":     duration_ms,
        "applier_report":  applier_report,
        "ai_warning":      ai_warning,
        "pages_rendered":  len(pages),
        "operations_count": len(plan.operations),
    }
    return output_bytes, meta


def _render_pages_via_doc_processor(
    docx_bytes: bytes, base_url: str, secret: str, max_pages: int
) -> list[dict]:
    """POSTet die DOCX an /docx/render-pages und gibt die Seiten zurueck."""
    if not base_url or not secret:
        raise RuntimeError("doc_processor_url/secret nicht gesetzt — Vision-Phase nicht moeglich")
    payload = {
        "docx_b64":  base64.b64encode(docx_bytes).decode("ascii"),
        "max_pages": max_pages,
        "dpi":       110,
    }
    url = base_url.rstrip("/") + "/docx/render-pages"
    headers = {"Authorization": f"Bearer {secret}"}
    try:
        resp = httpx.post(url, json=payload, headers=headers, timeout=90.0)
    except httpx.HTTPError as e:
        raise RuntimeError(f"doc-processor unerreichbar: {e}") from e
    if resp.status_code >= 400:
        raise RuntimeError(f"doc-processor HTTP {resp.status_code}: {resp.text[:300]}")
    pages = resp.json().get("pages") or []
    if not pages:
        raise RuntimeError("doc-processor lieferte keine Seiten")
    return pages


def _call_claude_for_edit_plan(
    *, client, model: str, max_tokens: int,
    document_map: dict, document_data: dict, layout_rules: list[str],
    page_pngs: list[str],
) -> tuple[dict, int, int]:
    """Ruft Claude Sonnet mit Vision + Map + Data und liefert (plan_dict, in, out)."""
    image_blocks = [{
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": b64},
    } for b64 in page_pngs]

    user_text = edit_plan_user(document_map, document_data, layout_rules)

    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=EDIT_PLAN_SYSTEM,
        messages=[{
            "role": "user",
            "content": image_blocks + [{"type": "text", "text": user_text}],
        }],
    )

    raw = "\n".join(getattr(b, "text", "") for b in (msg.content or []) if getattr(b, "type", None) == "text").strip()
    plan_dict = _parse_plan_json(raw)
    if plan_dict is None:
        raise RuntimeError("Claude lieferte kein gueltiges Edit-Plan-JSON")

    usage_in  = getattr(msg.usage, "input_tokens",  0) or 0
    usage_out = getattr(msg.usage, "output_tokens", 0) or 0
    return plan_dict, usage_in, usage_out


def _parse_plan_json(text: str) -> Optional[dict]:
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None

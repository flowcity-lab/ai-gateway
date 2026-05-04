"""
Document-Map: deterministische Repraesentation der editierbaren Knoten in einer DOCX.

Liefert nummerierte IDs (P0, P1, T0.R0.C0, ...) plus den aktuellen Text je Knoten,
damit Claude in Phase 2 mit diesen IDs einen Edit-Plan erzeugen kann. Der Applier
in edit_applier.py konsumiert dieselbe Nummerierung.

Die Reihenfolge ist die natuerliche Body-Reihenfolge von python-docx:
  doc.element.body durchlaufen, Tabellen und Paragraphs in der Auftrittsreihenfolge.
"""

from __future__ import annotations

import io
from typing import Iterator

from docx import Document
from docx.document import Document as _DocumentT
from docx.oxml.ns import qn
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph

# Maximale Text-Laenge je Knoten in der Map. Verhindert dass Lorem-Ipsum-Bloecke
# das Context-Window sprengen. Originale Bytes bleiben unangetastet.
_MAX_TEXT_LEN = 240


def build_document_map(docx_bytes: bytes) -> dict:
    """
    Erzeugt die Map als JSON-serialisierbares Dict:
      {
        "paragraphs": [{ "id", "text", "style", "in_table": false }],
        "tables":     [{ "id", "rows", "cols", "cells": [{ "id", "text" }] }],
      }
    """
    doc: _DocumentT = Document(io.BytesIO(docx_bytes))

    paragraphs: list[dict] = []
    tables:     list[dict] = []

    p_idx = 0
    t_idx = 0

    for block in _iter_body_blocks(doc):
        if isinstance(block, Paragraph):
            text = block.text or ""
            tab_count = text.count("\t")
            # Tabstop-Paragraphs sind „Pseudo-Tabellenzeilen" — die Vorlage benutzt
            # Tab-Stops statt echter <w:tbl>. Claude muss das erkennen, sonst
            # versucht er pro Spalte ein eigenes set_text und das Layout kollabiert.
            has_tabs = tab_count > 0 or _pPr_has_tabs(block)
            entry = {
                "id":       f"P{p_idx}",
                "text":     _truncate(text),
                "style":    _style_name(block),
                "in_table": False,
            }
            if has_tabs:
                entry["has_tabs"] = True
                entry["tab_count"] = tab_count
            paragraphs.append(entry)
            p_idx += 1
        elif isinstance(block, Table):
            tables.append(_table_to_map(block, t_idx))
            t_idx += 1

    return {"paragraphs": paragraphs, "tables": tables}


def _iter_body_blocks(doc: _DocumentT) -> Iterator[object]:
    """Iteriert ueber Paragraphs und Tabellen im Body in der Auftrittsreihenfolge."""
    body = doc.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            yield Table(child, doc)
        # sectPr und andere ignorieren — die werden vom Applier nicht angefasst.


def _table_to_map(table: Table, t_idx: int) -> dict:
    rows = table.rows
    n_rows = len(rows)
    n_cols = max((len(r.cells) for r in rows), default=0)

    cells: list[dict] = []
    for r_idx, row in enumerate(rows):
        for c_idx, cell in enumerate(row.cells):
            cells.append({
                "id":   f"T{t_idx}.R{r_idx}.C{c_idx}",
                "text": _truncate(_cell_text(cell)),
            })

    return {
        "id":       f"T{t_idx}",
        "rows":     n_rows,
        "cols":     n_cols,
        "cells":    cells,
    }


def _cell_text(cell: _Cell) -> str:
    """Konkateniert alle Paragraph-Texte einer Zelle mit '\\n'."""
    return "\n".join((p.text or "") for p in cell.paragraphs).strip()


def _style_name(p: Paragraph) -> str:
    try:
        return (p.style.name if p.style is not None else "Normal") or "Normal"
    except Exception:
        return "Normal"


def _pPr_has_tabs(p: Paragraph) -> bool:
    """True, wenn der Paragraph in seinen Properties Tab-Stops definiert hat
    (auch ohne aktuell vorhandene \\t-Zeichen — z.B. leere Vorlagenzeile)."""
    try:
        pPr = p._p.find(qn("w:pPr"))
        if pPr is None:
            return False
        return pPr.find(qn("w:tabs")) is not None
    except Exception:
        return False


def _truncate(s: str) -> str:
    s = (s or "").strip()
    if len(s) <= _MAX_TEXT_LEN:
        return s
    return s[: _MAX_TEXT_LEN - 1] + "…"

"""
Pydantic-Schemas fuer den Edit-Plan, den Claude in Phase 2 erzeugt.

Vier Operationen werden unterstuetzt:
  * set_text           — Paragraph-Text im Body ersetzen (Format-Erhalt durch
                          Run-Inplace-Replacement im Applier).
  * set_cell_text      — Zellen-Text einer Tabelle ersetzen.
  * rebuild_table_rows — Posten-Tabelle: Daten-Zeilen N-fach klonen + befuellen.
  * delete_paragraph   — Paragraph komplett entfernen (z.B. uebrig gebliebene
                          Mustache-Variablen ohne Wert).

Der Validator wirft `ValueError` bei Schema-Verletzungen — der Caller protokolliert
das und schickt einen Gateway-Fehler an Laravel (status: failed).
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, ValidationError, field_validator


class SetTextOp(BaseModel):
    op:     Literal["set_text"]
    target: str
    value:  str

    @field_validator("target")
    @classmethod
    def _check_target(cls, v: str) -> str:
        if not v.startswith("P"):
            raise ValueError("set_text.target muss mit 'P' beginnen")
        return v


class SetCellTextOp(BaseModel):
    op:     Literal["set_cell_text"]
    target: str
    value:  str

    @field_validator("target")
    @classmethod
    def _check_target(cls, v: str) -> str:
        if not v.startswith("T") or v.count(".") != 2:
            raise ValueError("set_cell_text.target muss Format 'T<i>.R<r>.C<c>' haben")
        return v


class RebuildTableRowsOp(BaseModel):
    op:                 Literal["rebuild_table_rows"]
    target:             str
    header_rows:        int = Field(ge=0, le=10, default=1)
    footer_rows:        int = Field(ge=0, le=10, default=0)
    row_template_index: int = Field(ge=0, le=99)
    data_rows:          list[list[str]] = Field(default_factory=list)

    @field_validator("target")
    @classmethod
    def _check_target(cls, v: str) -> str:
        if not v.startswith("T") or "." in v:
            raise ValueError("rebuild_table_rows.target muss Format 'T<i>' haben")
        return v

    @field_validator("data_rows")
    @classmethod
    def _check_rows(cls, v: list[list[str]]) -> list[list[str]]:
        if len(v) > 200:
            raise ValueError("data_rows: max. 200 Zeilen erlaubt (Hard-Limit)")
        for row in v:
            if len(row) > 32:
                raise ValueError("data_rows: max. 32 Zellen pro Zeile (Hard-Limit)")
        return v


class DeleteParagraphOp(BaseModel):
    op:     Literal["delete_paragraph"]
    target: str

    @field_validator("target")
    @classmethod
    def _check_target(cls, v: str) -> str:
        if not v.startswith("P"):
            raise ValueError("delete_paragraph.target muss mit 'P' beginnen")
        return v


Operation = Annotated[
    Union[SetTextOp, SetCellTextOp, RebuildTableRowsOp, DeleteParagraphOp],
    Field(discriminator="op"),
]


class EditPlan(BaseModel):
    operations: list[Operation] = Field(default_factory=list)


def validate_edit_plan(raw: dict) -> EditPlan:
    """
    Akzeptiert das Roh-JSON von Claude und gibt ein validiertes EditPlan-Objekt
    zurueck. Bei Validierungs-Fehlern: ValueError mit aussagekraeftiger Message.
    """
    if not isinstance(raw, dict):
        raise ValueError("Edit-Plan: Top-Level muss ein Objekt sein")
    ops = raw.get("operations")
    if not isinstance(ops, list):
        raise ValueError("Edit-Plan: 'operations' muss eine Liste sein")
    if len(ops) > 500:
        raise ValueError("Edit-Plan: max. 500 Operationen (Hard-Limit)")

    # Pydantic kann die Discriminated-Union ueber 'op' aufloesen.
    try:
        return EditPlan.model_validate({"operations": ops})
    except ValidationError as e:
        raise ValueError(f"Edit-Plan-Validierung fehlgeschlagen: {e}") from e

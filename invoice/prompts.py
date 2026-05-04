"""System- und User-Prompts fuer die VGSE-Pipeline (Deutsch, imperativ)."""

import json

# ── Phase 1.5: Intent Extraction (Claude Haiku) ────────────────────────

INTENT_EXTRACTION_SYSTEM = (
    "Du extrahierst aus freiem Hinweistext eines Trainers strukturierte Intents fuer eine "
    "Rechnung. Rechne NICHT — die Summen werden in PHP berechnet. Formuliere display_text "
    "als sauberen, hoeflichen Hinweis fuer den Rechnungsempfaenger (Anweisungen wie "
    "'Adresse aenderen' nicht in display_text uebernehmen). Gib AUSSCHLIESSLICH gueltiges "
    "JSON exakt nach Schema zurueck — keine Erklaerungen, keine Markdown-Fences."
)


def intent_extraction_user(notes: str, context: dict) -> str:
    ctx_recipient = context.get("recipient") or {}
    ctx_totals    = context.get("totals") or {}
    ctx_currency  = context.get("currency") or "EUR"
    schema = {
        "display_text": "string  // gereinigter Hinweistext fuer die Rechnung (deutsch, hoeflich)",
        "recipient_override": {
            "name": "string?", "street": "string?", "zip": "string?",
            "city": "string?", "country": "string?", "vat_id": "string?",
            "email": "string?", "phone": "string?", "contact_person": "string?",
        },
        "tax_rule": {"global_rate": "number? // Prozent, z.B. 20"},
        "layout_rules": ["string  // z.B. 'mwst_zeile_pro_satz_anzeigen'"],
    }
    return (
        "AUFGABE: Analysiere den TRAINER-HINWEIS und extrahiere strukturierte Intents.\n\n"
        "REGELN:\n"
        "- recipient_override NUR setzen wenn im Hinweis EINDEUTIG eine andere Empfaengeradresse genannt wird.\n"
        "- tax_rule.global_rate NUR setzen wenn der Hinweis einen klaren Steuersatz fordert (z.B. '20% MwSt').\n"
        "- layout_rules: kurze snake_case-Tokens. Erlaubte Beispiele: "
        "'mwst_zeile_pro_satz_anzeigen', 'rechnungsnummer_in_verwendungszweck', 'iban_groesser_anzeigen'.\n"
        "- display_text: bereinigter Text fuer den Rechnungs-Hinweisblock. Anweisungen wie "
        "'bitte richtige Adresse ist X' NICHT in display_text uebernehmen — die landet in recipient_override.\n"
        "- Fehlende Felder als null lassen, nicht raten.\n\n"
        f"KONTEXT (nur zur Orientierung, NICHT veraendern):\n"
        f"  Empfaenger laut System: {json.dumps(ctx_recipient, ensure_ascii=False)}\n"
        f"  Summen laut System:     {json.dumps(ctx_totals,    ensure_ascii=False)} {ctx_currency}\n\n"
        f"TRAINER-HINWEIS:\n\"\"\"\n{notes.strip()}\n\"\"\"\n\n"
        f"AUSGABE-SCHEMA (gueltiges JSON, exakt diese Keys, keine zusaetzlichen):\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )


# ── Phase 2: Edit-Plan Generation (Claude Sonnet, Vision + Map + Data) ─

EDIT_PLAN_SYSTEM = (
    "Du erzeugst einen deterministischen Edit-Plan, der eine DOCX-Vorlage in eine fertige "
    "Rechnung verwandelt. Du SIEHST die Vorlage als gerenderte Seitenbilder UND erhaeltst die "
    "Document-Map mit eindeutigen Knoten-IDs. Du veraenderst NICHTS am Layout, nur Texte und "
    "Tabellenzeilen-Anzahl. Du rechnest NICHT — alle Summen sind bereits in den Daten enthalten.\n\n"
    "Gib AUSSCHLIESSLICH ein JSON-Objekt mit Schluessel 'operations' zurueck. Keine Erklaerungen."
)


def edit_plan_user(document_map: dict, document_data: dict, layout_rules: list[str]) -> str:
    rules_block = ""
    if layout_rules:
        rules_block = (
            "\nZUSAETZLICHE LAYOUT-REGELN (vom User verlangt):\n- "
            + "\n- ".join(layout_rules) + "\n"
        )
    schema = {
        "operations": [
            {"op": "set_text", "target": "P<n>", "value": "string"},
            {"op": "set_cell_text", "target": "T<i>.R<r>.C<c>", "value": "string"},
            {
                "op": "rebuild_table_rows",
                "target": "T<i>",
                "header_rows": "int",
                "footer_rows": "int",
                "row_template_index": "int  // 0-basiert, Index der Daten-Beispielzeile",
                "data_rows": [["string", "..."]],
            },
            {"op": "delete_paragraph", "target": "P<n>"},
        ]
    }
    return (
        "AUFGABE: Erzeuge einen Edit-Plan, der die Vorlage in eine fertige Rechnung verwandelt.\n\n"
        "VORGEHEN:\n"
        "1. Vergleiche die Seitenbilder mit der Document-Map und identifiziere Platzhalter.\n"
        "2. Ersetze Mustache-Variablen ({{recipient.name}} etc.) mit den DATEN.\n"
        "3. Ersetze Beispieltexte (Lorem ipsum, Mock-Adressen) durch echte Daten.\n"
        "4. Fuer die Posten-Tabelle: nutze rebuild_table_rows. Beobachte header_rows und data_rows.\n"
        "5. Felder ohne Daten: leeren Text setzen (Platzhalter entfernen).\n\n"
        "REGELN:\n"
        "- Verwende AUSSCHLIESSLICH IDs aus der Document-Map.\n"
        "- Aendere KEINE Style-Knoten, keine Bilder, keine Header/Footer-Layouts.\n"
        "- Datums-Format: TT.MM.JJJJ. Geld: '1.234,56 EUR' (deutsches Format).\n"
        "- Rechne NICHT — alle Summen sind in totals enthalten.\n"
        f"{rules_block}\n"
        "DOCUMENT-DATA (echte Werte):\n```json\n"
        f"{json.dumps(document_data, ensure_ascii=False, indent=2)}\n```\n\n"
        "DOCUMENT-MAP (editierbare Knoten):\n```json\n"
        f"{json.dumps(document_map, ensure_ascii=False, indent=2)}\n```\n\n"
        f"AUSGABE-SCHEMA:\n```json\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n```"
    )

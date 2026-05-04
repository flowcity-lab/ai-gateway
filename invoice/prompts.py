"""System- und User-Prompts fuer die VGSE-Pipeline (Deutsch, imperativ)."""

import json

# ── Phase 1.5: Intent Extraction (Claude Haiku) ────────────────────────

INTENT_EXTRACTION_SYSTEM = (
    "Du extrahierst aus freiem Hinweistext eines Trainers strukturierte Intents fuer eine "
    "Rechnung. Rechne NICHT — die Summen werden in PHP berechnet. display_text ist ein "
    "WICHTIGES Output-Feld: alles was an den Rechnungsempfaenger kommuniziert werden soll "
    "(Zahlungs-Hinweise, Verwendungszweck-Bitten, Bankdetails, Danksagungen) MUSS dort "
    "stehen — nur strukturelle Anweisungen an den Trainer selbst (Adresse aenderen, "
    "Steuersatz setzen) wandern in recipient_override / tax_rule und werden aus display_text "
    "entfernt. Gib AUSSCHLIESSLICH gueltiges JSON exakt nach Schema zurueck — keine "
    "Erklaerungen, keine Markdown-Fences."
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
        "- layout_rules: kurze snake_case-Tokens. Aktuell unterstuetzt: "
        "'mwst_zeile_pro_satz_anzeigen' (eigene Zeile pro Steuersatz im Footer), "
        "'iban_groesser_anzeigen'. Andere Tokens NICHT erfinden.\n"
        "- display_text: hier landet ALLES was der Empfaenger lesen soll. Insbesondere:\n"
        "    * Zahlungs- und Verwendungszweck-Hinweise (z.B. 'Bitte bei Ueberweisung "
        "die Rechnungsnummer angeben.')\n"
        "    * Hinweise auf Bankverbindung, Frist, Skonto\n"
        "    * Danksagungen, Schlussworte\n"
        "  Diese Hinweise NICHT als layout_rule kodieren, NICHT weglassen.\n"
        "- Anweisungen die strukturell sind (Adresse aenderen, Steuersatz) gehoeren NICHT "
        "in display_text — sie landen in recipient_override / tax_rule.\n"
        "- display_text in vollstaendigen, hoeflichen deutschen Saetzen formulieren — kein "
        "Stichwort-Stil, keine Aufzaehlungszeichen.\n"
        "- Fehlende Felder als null lassen, nicht raten.\n\n"
        f"KONTEXT (nur zur Orientierung, NICHT veraendern):\n"
        f"  Empfaenger laut System: {json.dumps(ctx_recipient, ensure_ascii=False)}\n"
        f"  Summen laut System:     {json.dumps(ctx_totals,    ensure_ascii=False)} {ctx_currency}\n\n"
        f"TRAINER-HINWEIS:\n\"\"\"\n{notes.strip()}\n\"\"\"\n\n"
        f"AUSGABE-SCHEMA (gueltiges JSON, exakt diese Keys, keine zusaetzlichen):\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )


# ── Layout-Rule-Uebersetzung (snake_case → konkrete Anweisung) ─────────

_LAYOUT_RULE_TRANSLATIONS = {
    "mwst_zeile_pro_satz_anzeigen": (
        "Im Footer (vor der Gesamtsumme) eine eigene Zeile pro Steuersatz "
        "anzeigen, im Format 'zzgl. {satz}% MwSt: {betrag} {waehrung}'. "
        "Die Werte stehen in DOCUMENT-DATA.totals.tax_by_rate."
    ),
    "iban_groesser_anzeigen": (
        "Die IBAN im Footer optisch hervorheben (z.B. groessere Schrift "
        "oder Fettdruck), falls die Vorlage diese Stelle erlaubt."
    ),
}


def translate_layout_rules(tokens: list[str]) -> list[str]:
    """Wandelt snake_case-Tokens in konkrete Anweisungen fuer den Edit-Plan-Prompt."""
    out: list[str] = []
    for tok in tokens or []:
        tok = (tok or "").strip()
        if not tok:
            continue
        out.append(_LAYOUT_RULE_TRANSLATIONS.get(tok, tok))
    return out


# ── Phase 2: Edit-Plan Generation (Claude Sonnet, Vision + Map + Data) ─

EDIT_PLAN_SYSTEM = (
    "Du erzeugst einen deterministischen Edit-Plan, der eine DOCX-Vorlage in eine fertige "
    "Rechnung verwandelt. Du SIEHST die Vorlage als gerenderte Seitenbilder UND erhaeltst die "
    "Document-Map mit eindeutigen Knoten-IDs. Du veraenderst NICHTS am Layout, nur Texte und "
    "Tabellenzeilen-Anzahl. Du rechnest NICHT — alle Summen sind bereits in den Daten enthalten.\n\n"
    "GRUNDPRINZIP: Lieber zu wenig aendern als zu viel. Statische Trainer-Texte (Sprueche, "
    "Anrede-Floskeln, Zahlungs-Hinweise, rechtliche Hinweise, Schlussworte, Werbung) "
    "GEHOEREN ZUR VORLAGE und werden NIE entfernt. Nur klare Platzhalter werden ersetzt.\n\n"
    "Gib AUSSCHLIESSLICH ein JSON-Objekt mit Schluessel 'operations' zurueck. Keine Erklaerungen."
)


def edit_plan_user(document_map: dict, document_data: dict, layout_rules: list[str]) -> str:
    rules_block = ""
    if layout_rules:
        rules_block = (
            "\nZUSAETZLICHE LAYOUT-ANWEISUNGEN (vom Trainer fuer DIESE Rechnung verlangt):\n- "
            + "\n- ".join(layout_rules) + "\n"
        )

    notes_block = ""
    notes_value = (document_data.get("notes") or "").strip()
    if notes_value:
        notes_block = (
            "\nHINWEISTEXT FUER DEN EMPFAENGER (DOCUMENT-DATA.notes — MUSS auf der Rechnung erscheinen):\n"
            f"\"\"\"\n{notes_value}\n\"\"\"\n"
            "Platziere diesen Text WORTWOERTLICH in einem Paragraph zwischen Posten-Bereich "
            "und Footer/Bankdaten. Falls die Vorlage einen passenden Mustache-Platzhalter "
            "({{notes}}, {{hinweis}}, {{verwendungszweck}}) hat, fuelle diesen. Sonst nutze "
            "einen leeren Paragraph in dieser Region oder fuege den Text einem geeigneten "
            "vorhandenen Hinweis-Paragraph hinzu (NICHT bestehende Texte ueberschreiben — "
            "anhaengen). Formuliere ihn NICHT um.\n"
        )

    schema = {
        "operations": [
            {"op": "set_text", "target": "P<n>", "value": "string  // \\t fuer Tab-Stops"},
            {"op": "set_cell_text", "target": "T<i>.R<r>.C<c>", "value": "string"},
            {
                "op": "rebuild_table_rows",
                "target": "T<i>",
                "header_rows": "int",
                "footer_rows": "int",
                "row_template_index": "int  // 0-basiert, Index der Daten-Beispielzeile",
                "data_rows": [["string", "..."]],
            },
            {
                "op": "clone_paragraph_rows",
                "target": "P<n>  // Vorlagen-Zeile (hat has_tabs:true in der Map)",
                "data_rows": [["string", "..."]],
            },
            {"op": "delete_paragraph", "target": "P<n>"},
        ]
    }
    return (
        "AUFGABE: Erzeuge einen Edit-Plan, der die Vorlage in eine fertige Rechnung verwandelt.\n\n"
        "VORGEHEN:\n"
        "1. Vergleiche die Seitenbilder mit der Document-Map und identifiziere Platzhalter.\n"
        "2. Ersetze Mustache-Variablen ({{recipient.name}}, {{date}}, {{number}} etc.) mit den DATEN.\n"
        "3. Ersetze offensichtliche Mock-Daten (Lorem ipsum, 'Max Mustermann', "
        "'Beispielstrasse', '12345 Beispielstadt', Beispiel-IBANs/UIDs) durch echte Daten.\n"
        "4. Fuer die Posten-Liste:\n"
        "   a) Vorlage hat eine echte <w:tbl>? → rebuild_table_rows mit T<i>.\n"
        "   b) Vorlage hat KEINE Tabelle, sondern Tabstop-Paragraphs (Map-Knoten "
        "mit \"has_tabs\": true in der Posten-Region)? → clone_paragraph_rows auf "
        "die Vorlagen-Datenzeile (target = die P-ID dieser Beispiel-Zeile). "
        "Jede data_row ist eine Liste aus Strings; sie werden mit echten Tabs "
        "verknuepft, sodass die definierten Tab-Stops weiterhin greifen.\n"
        "5. Mustache-Platzhalter ohne Wert (z.B. {{steuernummer}} bei Kleinunternehmer): "
        "delete_paragraph nur dann, wenn der KOMPLETTE Paragraph nur aus diesem "
        "Platzhalter besteht. Sonst leere String einsetzen.\n\n"
        "ABSOLUTE REGELN:\n"
        "- Verwende AUSSCHLIESSLICH IDs aus der Document-Map.\n"
        "- Aendere KEINE Style-Knoten, keine Bilder, keine Header/Footer-Layouts.\n"
        "- Datums-Format: TT.MM.JJJJ. Geld: '1.234,56 EUR' (deutsches Format).\n"
        "- Rechne NICHT — alle Summen sind in totals enthalten.\n"
        "- ERHALTE alle Paragraphs, die KEINE Platzhalter sind. Statische Trainer-Texte "
        "(Sprueche, Zitate, Anrede-Floskeln, 'Vielen Dank fuer den Auftrag', "
        "Steuerbefreiungs-Hinweise wie '§ 6 Abs. 1 Z 27 UStG', Schlussworte, Werbung, "
        "Disclaimer) NIEMALS loeschen, NIEMALS leeren, NIEMALS umformulieren — auch "
        "wenn sie keine Daten aus DOCUMENT-DATA enthalten.\n"
        "- Bei Unsicherheit ob ein Paragraph Platzhalter oder Boilerplate ist: NICHT anfassen.\n"
        "- KEINE neuen Paragraphs erfinden ausser ueber clone_paragraph_rows.\n"
        f"{notes_block}"
        f"{rules_block}\n"
        "DOCUMENT-DATA (echte Werte):\n```json\n"
        f"{json.dumps(document_data, ensure_ascii=False, indent=2)}\n```\n\n"
        "DOCUMENT-MAP (editierbare Knoten — beachte 'has_tabs' fuer Pseudotabellen):\n```json\n"
        f"{json.dumps(document_map, ensure_ascii=False, indent=2)}\n```\n\n"
        f"AUSGABE-SCHEMA:\n```json\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n```"
    )

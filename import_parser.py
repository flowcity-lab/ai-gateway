"""
import_parser.py — Intelligenter Import: Datei-Parser, Telefon-Normalisierung und KI-Spaltenanalyse

Endpoints:
  POST /import/parse             — CSV / XLSX / XLS / ODS parsen
  POST /import/normalize-phones  — Telefonnummern normalisieren + Länder erkennen
  POST /import/analyze-columns   — GPT-4o analysiert Spalten anhand von Werten + schlägt Custom Fields vor
"""

import io
import csv
import json
import base64
import logging
import os
from typing import Optional

import chardet
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from openai import OpenAI as _OpenAI

log = logging.getLogger("ai-gateway.import")
router = APIRouter()

# Eigener OpenAI-Client (vermeidet circular import aus main.py)
_oai = _OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
_ANALYZE_MODEL = os.environ.get("IMPORT_ANALYZE_MODEL", "gpt-4o-mini")  # mini reicht hier vollständig

# ── Pydantic Models ───────────────────────────────────────────────────────────

class ParseRequest(BaseModel):
    file_content_base64: str
    file_name: str
    file_type: Optional[str] = None

class NormalizePhoneRequest(BaseModel):
    phones: list[str]
    fallback_country: Optional[str] = None
    candidate_countries: list[str] = []

class AnalyzeColumnsRequest(BaseModel):
    headers: list[str]                          # Spaltennamen aus Zeile 1 (können Datenwerte sein wenn kein Header)
    sample_rows: list[list[str]]                # 5-10 Beispielzeilen
    existing_custom_field_names: list[str] = [] # bereits vorhandene eigene Felder (nicht doppelt vorschlagen)

# ── Bekannte Export-Formate (Auto-Erkennung via Header-Patterns) ─────────────

KNOWN_FORMATS = {
    "mailchimp": ["email address", "member_rating", "optin_time"],
    "brevo":     ["email", "double opt-in"],
    "hubspot":   ["contact id", "hs_object_id", "createdate"],
}

EU_CANDIDATE_COUNTRIES = [
    "AT", "DE", "CH", "FR", "IT", "ES", "NL", "BE", "PL", "CZ",
    "SK", "HU", "RO", "BG", "HR", "SI", "LU", "PT", "SE", "DK",
    "FI", "NO", "GR", "IE", "LT", "LV", "EE",
]

# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def detect_format(headers: list[str]) -> str:
    """Erkennt bekannte Export-Formate anhand der Spaltenbezeichnungen."""
    lower_headers = {h.lower().strip() for h in headers}
    for fmt_name, signature in KNOWN_FORMATS.items():
        if all(sig in lower_headers for sig in signature):
            return fmt_name
    return "generic"


def parse_csv_bytes(raw_bytes: bytes) -> dict:
    """CSV parsen: Encoding und Delimiter auto-erkennen."""
    # Encoding erkennen
    detected = chardet.detect(raw_bytes)
    encoding = detected.get("encoding") or "utf-8"
    if encoding.lower() in ("ascii", "iso-8859-1", "latin-1", "windows-1252"):
        encoding = "utf-8-sig"  # BOM-tolerant

    try:
        text = raw_bytes.decode(encoding, errors="replace")
    except Exception:
        text = raw_bytes.decode("utf-8", errors="replace")

    # BOM entfernen
    text = text.lstrip("\ufeff")

    # Delimiter erkennen: Sniffer + Fallback-Liste
    delimiter = ","
    try:
        sample = text[:4096]
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        # Fallback: zählen welcher Delimiter am häufigsten vorkommt
        first_line = text.split("\n")[0] if "\n" in text else text[:200]
        counts = {d: first_line.count(d) for d in [",", ";", "\t", "|"]}
        delimiter = max(counts, key=counts.get)

    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows_raw = list(reader)

    # Leere Zeilen entfernen
    rows_raw = [r for r in rows_raw if any(cell.strip() for cell in r)]

    if not rows_raw:
        return {"headers": [], "rows": [], "row_count": 0, "has_data": False}

    headers = [h.strip() for h in rows_raw[0]]
    data_rows = [[cell.strip() for cell in row] for row in rows_raw[1:]]

    return {
        "name": "CSV",
        "headers": headers,
        "rows": data_rows,
        "row_count": len(data_rows),
        "has_data": len(data_rows) > 0,
        "detected_format": detect_format(headers),
        "delimiter_detected": delimiter,
        "encoding_detected": encoding,
    }


def parse_excel_bytes(raw_bytes: bytes) -> list[dict]:
    """Excel (XLSX/XLS/ODS) parsen: alle Sheets extrahieren."""
    import openpyxl

    wb = openpyxl.load_workbook(io.BytesIO(raw_bytes), read_only=True, data_only=True)
    sheets = []

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        all_rows = []
        for row in ws.iter_rows(values_only=True):
            # None-Werte zu leeren Strings, alles zu String konvertieren
            cleaned = [str(cell).strip() if cell is not None else "" for cell in row]
            if any(cell for cell in cleaned):  # nur nicht-leere Zeilen
                all_rows.append(cleaned)

        if not all_rows:
            sheets.append({
                "name": sheet_name,
                "headers": [],
                "rows": [],
                "row_count": 0,
                "has_data": False,
            })
            continue

        headers = all_rows[0]
        data_rows = all_rows[1:]

        sheets.append({
            "name": sheet_name,
            "headers": headers,
            "rows": data_rows,
            "row_count": len(data_rows),
            "has_data": len(data_rows) > 0,
            "detected_format": detect_format(headers),
        })

    wb.close()
    return sheets


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/import/parse")
async def parse_import_file(payload: ParseRequest, request: Request):
    """
    Parst eine hochgeladene Datei (CSV oder Excel) und gibt strukturierte
    Sheet/Zeilen-Daten zurück. Kein LLM beteiligt — rein deterministisch.
    """
    try:
        raw_bytes = base64.b64decode(payload.file_content_base64)
    except Exception:
        raise HTTPException(400, "Ungültiger Base64-Inhalt")

    # Dateitype ermitteln
    file_type = (payload.file_type or "").lower()
    if not file_type:
        ext = payload.file_name.rsplit(".", 1)[-1].lower() if "." in payload.file_name else ""
        file_type = ext

    try:
        if file_type == "csv" or file_type in ("txt", "tsv"):
            sheet = parse_csv_bytes(raw_bytes)
            return {
                "file_type": "csv",
                "sheets": [sheet],
                "delimiter_detected": sheet.get("delimiter_detected"),
                "encoding_detected": sheet.get("encoding_detected"),
            }

        elif file_type in ("xlsx", "xls", "ods", "xlsm"):
            sheets = parse_excel_bytes(raw_bytes)
            return {
                "file_type": file_type,
                "sheets": sheets,
            }

        else:
            raise HTTPException(415, f"Nicht unterstütztes Dateiformat: {file_type}. Unterstützt: csv, xlsx, xls, ods")

    except HTTPException:
        raise
    except Exception as e:
        log.error("Import parse error: %s", str(e), exc_info=True)
        raise HTTPException(500, f"Fehler beim Parsen der Datei: {str(e)}")


@router.post("/import/normalize-phones")
async def normalize_phones(payload: NormalizePhoneRequest, request: Request):
    """
    Normalisiert Telefonnummern nach E.164 und erkennt das Ursprungsland.

    Rückgabe pro Nummer:
      - e164: "+436641234567" oder null wenn nicht normalisierbar
      - country: "AT" oder null
      - confidence: "high" (1 Land), "medium" (2-3 Länder), "low" (viele/keine)
      - possible_countries: alle möglichen Länder
      - national_format: lesbares Format
    """
    import phonenumbers

    candidates = payload.candidate_countries or EU_CANDIDATE_COUNTRIES
    fallback = (payload.fallback_country or "").upper() or None

    results = []
    for raw in payload.phones:
        result = _normalize_single(raw, fallback, candidates, phonenumbers)
        results.append({"raw": raw, **result})

    return {"results": results}


_ANALYZE_SYSTEM_PROMPT = """
Du bist ein Experte für Datenimport von Kontaktlisten. Deine Aufgabe: Spalten aus einer Excel- oder CSV-Datei
eines Trainers oder Coaches den richtigen Kontaktfeldern zuordnen.

Verfügbare System-Felder:
  first_name        — Vorname
  last_name         — Nachname
  full_name         — Vollständiger Name (wird automatisch in Vor- + Nachname gesplittet)
  email             — E-Mail-Adresse
  phone             — Telefonnummer (Mobil oder Festnetz)
  date_of_birth     — Geburtsdatum
  organization_name — Firmenname / Organisation / Arbeitgeber
  organization_role — Position / Funktion / Berufsbezeichnung in der Firma
  notes             — Notizen, Freitext, sonstige Bemerkungen
  doi_confirmed_at  — DOI-Bestätigungsdatum (Double Opt-In / Einwilligung)
  status            — Kontaktstatus: lead, interessent, kunde, inaktiv
  _ignore           — Spalte NICHT importieren

Regeln:
1. Schau dir SOWOHL den Spaltennamen ALS AUCH die Beispielwerte an. Die Werte sind wichtiger.
2. Wenn eine Spalte sowohl Vor- als auch Nachnamen enthält (z.B. "Mag. Stefan Holzer"), nutze full_name.
3. has_no_header_row = true wenn die "Spaltenheader" wie echte Daten aussehen (Namen, Firmen, E-Mails).
4. _ignore verwenden für: interne IDs, Zeilennummern, leere Spalten, technische Felder, Land-Kürzel als Prefix.
5. custom_field_suggestions NUR für Daten die für einen Trainer/Coach wertvoll sind:
   - Sinnvoll: Interessen, Beruf, Region/Ort, Spezialisierung, Sprache, Notizen-Kategorie
   - NICHT sinnvoll: interne Nummern, Zeilen-IDs, Timestamps, technische Flags, Spalten mit < 10% Befüllung
6. Antworte ausschließlich mit validem JSON, kein Markdown, keine Erklärungen außerhalb des JSON.

Ausgabeformat (strikt):
{
  "has_no_header_row": false,
  "language_detected": "de",
  "mappings": { "0": "organization_name", "1": "full_name", "2": "_ignore" },
  "custom_field_suggestions": [
    { "column_index": 5, "column_header": "Region", "suggested_name": "Region", "type": "text",
      "reason": "Enthält Bundesland-Kürzel – nützlich für regionale Segmentierung" }
  ],
  "notes": "Optionale Anmerkung zur Datei"
}
""".strip()


@router.post("/import/analyze-columns")
async def analyze_columns(payload: AnalyzeColumnsRequest, request: Request):
    """
    GPT-4o analysiert Spalten anhand von Namen UND Beispielwerten.
    Erkennt fehlende Header-Zeilen und schlägt sinnvolle Custom Fields vor.
    """
    if not payload.headers:
        raise HTTPException(400, "headers darf nicht leer sein")

    # Kompakte Tabelle für den Prompt aufbauen
    table_lines = []
    for i, header in enumerate(payload.headers):
        samples = [row[i] for row in payload.sample_rows if i < len(row) and row[i].strip()][:5]
        samples_str = " | ".join(samples) if samples else "(leer)"
        table_lines.append(f"  Spalte {i}: Header=«{header}»  Beispiele: {samples_str}")

    existing_note = ""
    if payload.existing_custom_field_names:
        existing_note = f"\nBereits vorhandene eigene Felder (nicht nochmals vorschlagen): {', '.join(payload.existing_custom_field_names)}"

    user_msg = f"Analysiere diese {len(payload.headers)} Spalten:{existing_note}\n\n" + "\n".join(table_lines)

    try:
        response = _oai.chat.completions.create(
            model=_ANALYZE_MODEL,
            messages=[
                {"role": "system", "content": _ANALYZE_SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=1500,
        )
        raw_json = response.choices[0].message.content or "{}"
        result = json.loads(raw_json)
    except json.JSONDecodeError as e:
        log.error("analyze-columns: Ungültiges JSON von GPT: %s", e)
        raise HTTPException(500, "GPT hat kein valides JSON zurückgegeben")
    except Exception as e:
        log.error("analyze-columns: GPT-Fehler: %s", e, exc_info=True)
        raise HTTPException(500, f"KI-Analyse fehlgeschlagen: {str(e)}")

    # Mapping: Keys sicherstellen dass sie Integers sind (GPT gibt manchmal Strings zurück)
    raw_mappings = result.get("mappings", {})
    mappings = {str(k): str(v) for k, v in raw_mappings.items()}

    return {
        "has_no_header_row":        bool(result.get("has_no_header_row", False)),
        "language_detected":        result.get("language_detected", "unknown"),
        "mappings":                 mappings,
        "custom_field_suggestions": result.get("custom_field_suggestions", []),
        "notes":                    result.get("notes", ""),
        "model_used":               _ANALYZE_MODEL,
    }


def _normalize_single(raw: str, fallback: Optional[str], candidates: list[str], phonenumbers) -> dict:
    """Normalisiert eine einzelne Telefonnummer."""
    if not raw or not raw.strip():
        return {"e164": None, "country": None, "confidence": "none", "possible_countries": [], "national_format": None}

    raw = raw.strip()

    # Schritt 1: Direkt parsen (falls +XX vorhanden)
    try:
        parsed = phonenumbers.parse(raw, None)
        if phonenumbers.is_valid_number(parsed):
            country = phonenumbers.region_code_for_number(parsed)
            return {
                "e164": phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164),
                "country": country,
                "confidence": "high",
                "possible_countries": [country],
                "national_format": phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.NATIONAL),
            }
    except Exception:
        pass

    # Schritt 2: Fallback-Land direkt versuchen
    if fallback:
        try:
            parsed = phonenumbers.parse(raw, fallback)
            if phonenumbers.is_valid_number(parsed):
                country = phonenumbers.region_code_for_number(parsed)
                return {
                    "e164": phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164),
                    "country": country,
                    "confidence": "medium",
                    "possible_countries": [country],
                    "national_format": phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.NATIONAL),
                }
        except Exception:
            pass

    # Schritt 3: Alle Kandidaten-Länder durchprobieren
    possible = []
    for cc in candidates:
        try:
            parsed = phonenumbers.parse(raw, cc)
            if phonenumbers.is_valid_number(parsed):
                e164 = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
                possible.append({"country": cc, "e164": e164, "parsed": parsed})
        except Exception:
            continue

    if not possible:
        return {"e164": None, "country": None, "confidence": "low", "possible_countries": [], "national_format": raw}

    if len(possible) == 1:
        p = possible[0]
        return {
            "e164": p["e164"],
            "country": p["country"],
            "confidence": "high",
            "possible_countries": [p["country"]],
            "national_format": phonenumbers.format_number(p["parsed"], phonenumbers.PhoneNumberFormat.NATIONAL),
        }

    confidence = "medium" if len(possible) <= 3 else "low"
    # Bestes Ergebnis: Fallback-Land bevorzugen, sonst erstes
    best = next((p for p in possible if p["country"] == fallback), possible[0])
    return {
        "e164": best["e164"],
        "country": best["country"],
        "confidence": confidence,
        "possible_countries": [p["country"] for p in possible],
        "national_format": phonenumbers.format_number(best["parsed"], phonenumbers.PhoneNumberFormat.NATIONAL),
    }

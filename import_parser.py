"""
import_parser.py — Intelligenter Import: Datei-Parser, Telefon-Normalisierung und KI-Spaltenanalyse

Endpoints:
  POST /import/parse             — CSV / XLSX / XLS / ODS parsen
  POST /import/normalize-phones  — Telefonnummern normalisieren + Länder erkennen
  POST /import/analyze-columns   — GPT-4o analysiert Spalten anhand von Werten + schlägt Custom Fields vor
"""

import asyncio
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
    headers: list[str]
    sample_rows: list[list[str]]
    existing_custom_field_names: list[str] = []

class ExtractContactsRequest(BaseModel):
    file_content_base64: str
    file_name: str
    file_type: str                              # csv, xlsx, xls, ods, pdf, jpg, png, webp, heic
    exclude_identity: dict = {}                 # Trainer-eigene Daten: names, emails, phones, company_name
    available_fields: list[str] = []            # System + Custom Fields die der Trainer hat
    chunk_index: int = 0                        # Intern für chunked processing

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


_EXTRACT_SYSTEM_PROMPT = """
Du extrahierst Kontaktdaten aus beliebigen Dokumenten für einen Trainer/Coach.

Regeln:
1. Suche nach PERSONEN und ORGANISATIONEN — ignoriere KPIs, Summen, Metadaten, Diagramme, leere Zeilen, interne IDs, Zeilennummern.
2. Erkenne Beziehungen: welche Person gehört zu welcher Organisation?
3. SCHLIESSE AUS: Personen/Firmen die in exclude_identity stehen (das ist der Trainer selbst).
4. Bei Rechnungen: die Absender-Seite ist meist der Trainer (ausschließen), die Empfänger-Seite ist der Kunde (extrahieren).
5. Wenn ein Name Titel enthält (Mag., Dr., DI, Prof.) → in full_name übernehmen.
6. Gib nur echte Kontaktdaten zurück — keine Platzhalter, keine Beispieldaten.
7. confidence: 0.0-1.0 (wie sicher bist du dass das ein echter Kontakt ist)

Antworte ausschließlich mit validem JSON:
{
  "contacts": [
    {
      "first_name": "Stefan",
      "last_name": "Holzer",
      "full_name": null,
      "email": "stefan@example.at",
      "phone": "+43 664 123 45 67",
      "organization_name": "Ärztekammer Vorarlberg",
      "organization_role": null,
      "date_of_birth": null,
      "notes": null,
      "custom_fields": {},
      "confidence": 0.95,
      "source_hint": "Zeile 5"
    }
  ],
  "organizations": [
    {
      "name": "Ärztekammer Vorarlberg",
      "website": null,
      "address": "Schulgasse 17, 6850 Dornbirn"
    }
  ],
  "chunk_notes": "200 Zeilen verarbeitet, 45 Kontakte gefunden, 12 Zeilen waren KPI-Daten"
}
""".strip()


def _rows_to_text_table(rows: list[list]) -> str:
    """Konvertiert Zeilen in lesbaren Text für GPT."""
    lines = []
    for i, row in enumerate(rows):
        cells = [str(c).strip() if c is not None else "" for c in row]
        # Leere Zeilen überspringen
        if not any(cells):
            continue
        lines.append(f"Zeile {i+1}: " + " | ".join(cells))
    return "\n".join(lines)


def _extract_pdf_text(content_bytes: bytes) -> str:
    """Extrahiert Text aus PDF via pdfplumber."""
    try:
        import pdfplumber
        import io as _io
        with pdfplumber.open(_io.BytesIO(content_bytes)) as pdf:
            pages = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            return "\n\n--- Seite ---\n\n".join(pages)
    except Exception as e:
        log.warning("PDF-Extraktion fehlgeschlagen: %s", e)
        return ""


async def _call_gpt_extract(content_text: str | None, image_b64: str | None,
                             image_mime: str | None, exclude_identity: dict,
                             available_fields: list[str],
                             _max_tokens: int = 16000,
                             _attempt: int = 1) -> dict:
    """Ruft GPT-4o zur Kontakt-Extraktion auf. Entweder Text oder Bild.
    Bei Truncation (finish_reason=='length') wird automatisch mit höherem
    Token-Limit erneut versucht (max 2 Versuche)."""

    # Exclude-Hinweis aufbauen
    exclude_parts = []
    if exclude_identity.get("names"):
        exclude_parts.append("Namen: " + ", ".join(exclude_identity["names"]))
    if exclude_identity.get("emails"):
        exclude_parts.append("E-Mails: " + ", ".join(exclude_identity["emails"]))
    if exclude_identity.get("company_name"):
        exclude_parts.append("Firma: " + exclude_identity["company_name"])
    exclude_note = ("AUSSCHLIESSEN (das ist der Trainer selbst): " + " | ".join(exclude_parts)) if exclude_parts else ""

    fields_note = ("Verfügbare Custom-Felder des Trainers: " + ", ".join(available_fields)) if available_fields else ""

    user_content: list = []

    if image_b64 and image_mime:
        user_content.append({
            "type": "text",
            "text": f"Extrahiere alle Kontakte aus diesem Dokument/Bild.\n{exclude_note}\n{fields_note}"
        })
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{image_mime};base64,{image_b64}", "detail": "high"}
        })
    else:
        user_content.append({
            "type": "text",
            "text": f"{exclude_note}\n{fields_note}\n\nInhalt:\n{content_text}"
        })

    response = _oai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": _EXTRACT_SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=_max_tokens,
    )

    finish_reason = response.choices[0].finish_reason
    raw = response.choices[0].message.content or "{}"

    # Truncation-Erkennung: GPT wurde mitten im JSON abgeschnitten
    if finish_reason == "length":
        log.warning("GPT-Extraktion truncated (finish_reason=length, max_tokens=%d, attempt=%d)",
                     _max_tokens, _attempt)
        # Einmal mit doppeltem Limit retry (max 2 Versuche)
        if _attempt < 2:
            return await _call_gpt_extract(
                content_text=content_text, image_b64=image_b64,
                image_mime=image_mime, exclude_identity=exclude_identity,
                available_fields=available_fields,
                _max_tokens=min(_max_tokens * 2, 16384),
                _attempt=_attempt + 1,
            )
        # Nach 2 Versuchen: so viel wie möglich retten
        log.error("GPT-Extraktion nach %d Versuchen immer noch truncated — versuche partielle Rettung", _attempt)
        try:
            # Versuche das unvollständige JSON trotzdem zu parsen
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"contacts": [], "organizations": [],
                    "chunk_notes": f"Antwort wurde nach {_max_tokens} Tokens abgeschnitten und konnte nicht geparst werden."}

    return json.loads(raw)


# Semaphore: max 5 parallele GPT-Calls (Rate-Limit-Schutz)
_gpt_semaphore = asyncio.Semaphore(5)


def _log_gpt_result(result: dict, source: str, file_name: str) -> None:
    """Loggt GPT-Ergebnis direkt nach dem Call — bevor irgendetwas gefiltert wird."""
    contacts = result.get("contacts", [])
    orgs = result.get("organizations", [])
    notes = result.get("chunk_notes", "")
    if contacts:
        log.info("extract-contacts [%s] %s: %d Kontakte, %d Orgs gefunden. Notes: %s",
                 source, file_name, len(contacts), len(orgs), notes or "(keine)")
    else:
        log.warning("extract-contacts [%s] %s: 0 Kontakte gefunden. Notes: %s",
                    source, file_name, notes or "(keine)")


async def _extract_chunk(content_text: str | None, image_b64: str | None,
                          image_mime: str | None, exclude_identity: dict,
                          available_fields: list[str], label: str,
                          file_name: str) -> dict:
    """Einen Chunk mit Semaphore-Schutz extrahieren."""
    async with _gpt_semaphore:
        result = await _call_gpt_extract(
            content_text=content_text,
            image_b64=image_b64, image_mime=image_mime,
            exclude_identity=exclude_identity,
            available_fields=available_fields,
        )
        _log_gpt_result(result, label, file_name)
        return result


def _collect_results(results: list[dict]) -> tuple[list[dict], list[dict], list[str]]:
    """Sammelt Kontakte, Orgs und Notes aus mehreren Chunk-Ergebnissen."""
    contacts, orgs, notes = [], [], []
    for r in results:
        contacts.extend(r.get("contacts", []))
        orgs.extend(r.get("organizations", []))
        if r.get("chunk_notes"):
            notes.append(r["chunk_notes"])
    return contacts, orgs, notes


@router.post("/import/extract-contacts")
async def extract_contacts(payload: ExtractContactsRequest, request: Request):
    """
    GPT-4o extrahiert Kontakte + Organisationen aus beliebigen Dateien.
    CSV/Excel: chunked rows (50 Zeilen je Chunk), parallel (max 5).
    PDF: Text-Extraktion dann chunked parallel.
    Bilder (JPG/PNG/WebP): direkt GPT-4o Vision.
    """
    content_bytes = base64.b64decode(payload.file_content_base64)
    ft = payload.file_type.lower().lstrip(".")

    all_contacts: list[dict] = []
    all_orgs: list[dict] = []
    all_notes: list[str] = []

    try:
        # ── Bilder → GPT Vision ────────────────────────────────────────────
        IMAGE_TYPES = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                       "png": "image/png", "webp": "image/webp", "heic": "image/heic"}
        if ft in IMAGE_TYPES:
            result = await _extract_chunk(
                content_text=None,
                image_b64=payload.file_content_base64, image_mime=IMAGE_TYPES[ft],
                exclude_identity=payload.exclude_identity,
                available_fields=payload.available_fields,
                label="vision", file_name=payload.file_name,
            )
            all_contacts.extend(result.get("contacts", []))
            all_orgs.extend(result.get("organizations", []))
            if result.get("chunk_notes"):
                all_notes.append(result["chunk_notes"])

        # ── PDF → Text → chunked parallel ──────────────────────────────────
        elif ft == "pdf":
            pdf_text = _extract_pdf_text(content_bytes)
            if not pdf_text.strip():
                log.warning("extract-contacts [pdf] %s: pdfplumber lieferte keinen Text", payload.file_name)
                return {"contacts": [], "organizations": [], "notes": "PDF enthielt keinen lesbaren Text."}

            log.info("extract-contacts [pdf] %s: %d Zeichen Text extrahiert. Erste 500 Zeichen:\n%s",
                     payload.file_name, len(pdf_text), pdf_text[:500])

            chunk_size = 6000
            text_chunks = [pdf_text[i:i+chunk_size] for i in range(0, len(pdf_text), chunk_size)]
            total = len(text_chunks)

            # Alle PDF-Chunks parallel
            tasks = [
                _extract_chunk(
                    content_text=chunk, image_b64=None, image_mime=None,
                    exclude_identity=payload.exclude_identity,
                    available_fields=payload.available_fields,
                    label=f"pdf-chunk-{i+1}/{total}", file_name=payload.file_name,
                )
                for i, chunk in enumerate(text_chunks)
            ]
            results = await asyncio.gather(*tasks)
            c, o, n = _collect_results(results)
            all_contacts.extend(c)
            all_orgs.extend(o)
            all_notes.extend(n)

        # ── CSV / Excel / ODS → Zeilen → chunked parallel ─────────────────
        else:
            if ft in ("csv", "txt", "tsv"):
                sheet = parse_csv_bytes(content_bytes)
                sheets = [sheet]
            elif ft in ("xlsx", "xls", "ods", "xlsm"):
                sheets = parse_excel_bytes(content_bytes)
            else:
                raise HTTPException(415, f"Nicht unterstütztes Format: {ft}")

            for sheet in sheets:
                rows = sheet.get("rows", [])
                headers = sheet.get("headers", [])
                sheet_name = sheet.get("name", "")

                all_rows = [headers] + rows if headers else rows
                chunk_size = 50
                total_chunks = (len(all_rows) + chunk_size - 1) // chunk_size

                # Alle Chunks dieser Sheet parallel
                tasks = []
                for chunk_start in range(0, len(all_rows), chunk_size):
                    chunk_idx = chunk_start // chunk_size + 1
                    chunk = all_rows[chunk_start:chunk_start + chunk_size]
                    text = f"Sheet: {sheet_name}\n" + _rows_to_text_table(chunk)
                    tasks.append(_extract_chunk(
                        content_text=text, image_b64=None, image_mime=None,
                        exclude_identity=payload.exclude_identity,
                        available_fields=payload.available_fields,
                        label=f"{sheet_name}-chunk-{chunk_idx}/{total_chunks}",
                        file_name=payload.file_name,
                    ))

                results = await asyncio.gather(*tasks)
                c, o, n = _collect_results(results)
                all_contacts.extend(c)
                all_orgs.extend(o)
                for note in n:
                    all_notes.append(f"[{sheet_name}] {note}")

    except json.JSONDecodeError as e:
        log.error("extract-contacts: Ungültiges JSON von GPT: %s", e)
        raise HTTPException(500, "GPT hat kein valides JSON zurückgegeben")
    except Exception as e:
        log.error("extract-contacts: Fehler: %s", e, exc_info=True)
        raise HTTPException(500, f"Extraktion fehlgeschlagen: {str(e)}")

    log.info("extract-contacts FERTIG [%s] %s: %d Kontakte, %d Orgs total",
             ft, payload.file_name, len(all_contacts), len(all_orgs))

    return {
        "contacts":      all_contacts,
        "organizations": all_orgs,
        "notes":         " | ".join(all_notes),
        "total_found":   len(all_contacts),
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

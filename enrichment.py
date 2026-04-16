"""
Enrichment-Modul für das AI-Gateway.

Empfängt Enrichment-Requests von Laravel, crawlt Websites via Crawl4AI,
extrahiert Daten via LLM und sendet Ergebnisse per Callback zurück.

Analog zum import_parser.py Pattern.
"""

import os
import json
import time
import logging
import httpx
from typing import Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks

log = logging.getLogger("enrichment")

router = APIRouter(prefix="/enrich", tags=["enrichment"])

# ── Konfiguration ────────────────────────────────────────────────────

CRAWL4AI_URL = os.environ.get("CRAWL4AI_URL", "http://localhost:11235")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
AI_GATEWAY_SECRET = os.environ.get("AI_GATEWAY_SECRET", "")
ENRICHMENT_MODEL = os.environ.get("ENRICHMENT_MODEL", "gpt-4o-mini")
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

MAX_CONTENT_PER_PAGE = 6000
MAX_TOTAL_CONTENT = 20000
BRAVE_MAX_RESULTS = 10

# ── Bekannte Freemail-Domains (Fast-Path, kein Crawl nötig) ──────────

OBVIOUS_FREEMAIL = {
    "gmail.com", "googlemail.com", "yahoo.com", "yahoo.de", "yahoo.co.uk",
    "hotmail.com", "hotmail.de", "outlook.com", "outlook.de",
    "live.com", "live.de", "icloud.com", "me.com", "mac.com",
}

# ── Request/Response Models ──────────────────────────────────────────

class EnrichRequest(BaseModel):
    job_id: int
    callback_url: str
    entity_type: str = Field(..., pattern="^(organization|contact)$")

    # Organisation-Daten
    org_name: Optional[str] = None
    org_website: Optional[str] = None
    org_id: Optional[int] = None

    # Kontakt-Daten
    contact_name: Optional[str] = None
    contact_email: Optional[str] = None
    contact_city: Optional[str] = None
    contact_org_name: Optional[str] = None
    contact_org_website: Optional[str] = None

    # Benutzerdefinierte Felder (Kontakt)
    custom_field_defs: Optional[list] = None


def _verify(request: Request):
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer ") or auth[7:] != AI_GATEWAY_SECRET:
        raise HTTPException(401, "Unauthorized")


# ── Endpoint ─────────────────────────────────────────────────────────

@router.post("/process")
async def enrich_process(request: Request, bg: BackgroundTasks):
    _verify(request)
    body = await request.json()
    data = EnrichRequest(**body)
    log.info(f"Enrichment accepted: job_id={data.job_id}, type={data.entity_type}")
    bg.add_task(_enrich_pipeline, data)
    return {"status": "accepted", "job_id": data.job_id}


# ── Pipeline (im Background) ────────────────────────────────────────

def _enrich_pipeline(data: EnrichRequest):
    """Hauptpipeline: Crawl + LLM + Callback."""
    start = time.time()
    try:
        if data.entity_type == "organization":
            result = _enrich_organization(data)
        else:
            result = _enrich_contact(data)

        elapsed = round(time.time() - start, 1)
        log.info(f"Enrichment done: job_id={data.job_id}, {elapsed}s, fields={list(result.keys())}")

        _send_callback(data.callback_url, {
            "job_id": data.job_id,
            "status": "success",
            "entity_type": data.entity_type,
            "data": result,
            "elapsed_seconds": elapsed,
        })

    except Exception as e:
        log.error(f"Enrichment failed: job_id={data.job_id}, error={e}")
        _send_callback(data.callback_url, {
            "job_id": data.job_id,
            "status": "error",
            "entity_type": data.entity_type,
            "error": str(e),
        })


# ═══════════════════════════════════════════════════════════════════════
# ORGANISATION-ENRICHMENT
# ═══════════════════════════════════════════════════════════════════════

def _enrich_organization(data: EnrichRequest) -> dict:
    result = {}
    website = data.org_website

    # Schritt 1: Website finden wenn nicht vorhanden
    if not website and data.org_name:
        website = _discover_website(data.org_name)
        if website:
            result["discovered_website"] = website

    if not website:
        return result

    # Schritt 2: Deep Crawl
    content = _deep_crawl(website)
    if not content:
        return result

    # Schritt 3: LLM-Extraktion
    extracted = _llm_extract_org(content, website)
    if extracted:
        result.update(extracted)

    return result


# ═══════════════════════════════════════════════════════════════════════
# KONTAKT-ENRICHMENT
# ═══════════════════════════════════════════════════════════════════════

def _enrich_contact(data: EnrichRequest) -> dict:
    result = {}
    org_website = data.contact_org_website
    email = data.contact_email
    name = data.contact_name or ""

    # Pfad B: Firmen-E-Mail → Domain als Website
    if not org_website and email and not _is_freemail(email):
        domain = _extract_domain(email)
        domain_content = _crawl_single(f"https://{domain}")
        if domain_content:
            classification = _llm_classify_domain(domain, domain_content)
            if classification == "business":
                org_website = f"https://{domain}"
                result["discovered_org_website"] = org_website
                result["discovered_org_domain"] = domain

    # Pfad A: Org-Website vorhanden → Person dort suchen
    if org_website:
        content = _deep_crawl(org_website)
        if content:
            extracted = _llm_extract_contact(content, name, "org_website", data.custom_field_defs, email)
            if extracted:
                result.update(extracted)

    # Pfad C: Brave-Web-Research (wenn bisher nichts gefunden oder kein Org)
    needs_research = not result.get("organization_role") and not (result.get("social_links") or {}).get("linkedin")
    # Bei namenlosen Kontakten (nur E-Mail): auch suchen, damit first_name/last_name gefüllt werden
    has_search_anchor = bool(name) or bool(email)
    if needs_research and has_search_anchor:
        search_query = _build_search_query(name, email, data.contact_city)
        search_content = _brave_search_text(search_query)
        if search_content:
            extracted = _llm_extract_contact(search_content, name, "brave_search", data.custom_field_defs, email)
            if extracted:
                _merge_contact_result(result, extracted)

    return result


def _merge_contact_result(acc: dict, new: dict) -> None:
    """Fusioniert neue Extraktion in das Ergebnis. Flache Felder nur wenn leer,
    social_links mergen (bestehende nicht überschreiben)."""
    for k, v in new.items():
        if k == "social_links" and isinstance(v, dict):
            existing = acc.get("social_links") or {}
            for platform, url in v.items():
                if url and not existing.get(platform):
                    existing[platform] = url
            if existing:
                acc["social_links"] = existing
            continue
        if k == "custom_fields" and isinstance(v, dict):
            existing = acc.get("custom_fields") or {}
            for name, val in v.items():
                if val and not existing.get(name):
                    existing[name] = val
            if existing:
                acc["custom_fields"] = existing
            continue
        if k not in acc or not acc[k]:
            acc[k] = v



# ═══════════════════════════════════════════════════════════════════════
# CRAWL4AI HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _deep_crawl(url: str, max_depth: int = 2, max_pages: int = 12) -> str:
    """Deep Crawl einer Website mit BFS-Strategie via Crawl4AI."""
    url = _normalize_url(url)
    try:
        with httpx.Client(timeout=120) as client:
            resp = client.post(f"{CRAWL4AI_URL}/crawl", json={
                "urls": [url],
                "crawler_config": {
                    "type": "CrawlerRunConfig",
                    "params": {
                        "word_count_threshold": 30,
                        "deep_crawl_strategy": {
                            "type": "BFSDeepCrawlStrategy",
                            "params": {
                                "max_depth": max_depth,
                                "max_pages": max_pages,
                            },
                        },
                    },
                },
            })
            if resp.status_code != 200:
                log.warning(f"Deep crawl failed: {url}, status={resp.status_code}")
                return _crawl_single(url)
            return _extract_markdown(resp.json(), url)
    except Exception as e:
        log.warning(f"Deep crawl exception: {url}, {e}")
        return _crawl_single(url)


def _crawl_single(url: str) -> str:
    """Einzelne Seite crawlen."""
    url = _normalize_url(url)
    try:
        with httpx.Client(timeout=45) as client:
            resp = client.post(f"{CRAWL4AI_URL}/crawl", json={
                "urls": [url],
                "crawler_config": {
                    "type": "CrawlerRunConfig",
                    "params": {"word_count_threshold": 30},
                },
            })
            if resp.status_code != 200:
                return ""
            return _extract_markdown(resp.json(), url)
    except Exception as e:
        log.warning(f"Single crawl failed: {url}, {e}")
        return ""


def _brave_search(query: str, count: int = BRAVE_MAX_RESULTS, country: str = "DE") -> list[dict]:
    """Brave Web Search API.

    Liefert Liste von Dicts mit Keys: title, url, description, age (optional).
    Leere Liste wenn kein API-Key gesetzt oder Fehler.
    """
    if not BRAVE_API_KEY:
        log.warning("Brave search skipped: BRAVE_API_KEY not set")
        return []
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(BRAVE_SEARCH_URL, headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": BRAVE_API_KEY,
            }, params={
                "q": query,
                "count": min(max(count, 1), 20),
                "country": country,
                "safesearch": "moderate",
                "text_decorations": "false",
            })
            if resp.status_code != 200:
                log.warning(f"Brave search failed: {resp.status_code}, {resp.text[:200]}")
                return []
            data = resp.json()
            results = (data.get("web") or {}).get("results") or []
            hits = []
            for r in results:
                hits.append({
                    "title": r.get("title", "") or "",
                    "url": r.get("url", "") or "",
                    "description": r.get("description", "") or "",
                    "age": r.get("age", "") or "",
                })
            log.info(f"Brave search: q={query!r}, hits={len(hits)}")
            return hits
    except Exception as e:
        log.warning(f"Brave search exception: {e}")
        return []


def _brave_search_text(query: str, count: int = BRAVE_MAX_RESULTS) -> str:
    """Brave-Suchergebnisse als markdown-ähnlichen Text für LLM-Input."""
    hits = _brave_search(query, count=count)
    if not hits:
        return ""
    lines = [f"# Brave-Suchergebnisse: {query}\n"]
    for i, h in enumerate(hits, 1):
        lines.append(f"## {i}. {h['title']}")
        lines.append(f"URL: {h['url']}")
        if h['description']:
            lines.append(h['description'])
        lines.append("")
    return "\n".join(lines)


def _coerce_markdown(r: dict) -> str:
    """Robuste Markdown-Extraktion aus einem Crawl4AI-Result.

    Crawl4AI liefert je nach Version:
      - markdown als String (alt)
      - markdown als Dict mit raw_markdown/fit_markdown (neu, ab ~v0.4)
      - markdown_v2 als Dict mit raw_markdown (Übergangsformat)
      - cleaned_html / fit_html als finaler Fallback
    """
    raw = r.get("markdown")
    if isinstance(raw, dict):
        md = raw.get("raw_markdown") or raw.get("fit_markdown") or raw.get("markdown_with_citations") or ""
        if md:
            return md
    elif isinstance(raw, str) and raw:
        return raw

    v2 = r.get("markdown_v2")
    if isinstance(v2, dict):
        md = v2.get("raw_markdown") or v2.get("fit_markdown") or ""
        if md:
            return md

    return r.get("cleaned_html") or r.get("fit_html") or ""


def _extract_markdown(data: dict, source_url: str) -> str:
    """Extrahiert Markdown aus Crawl4AI-Response."""
    results = data.get("results", [])
    if not results and "result" in data:
        results = [data["result"]]
    if not results and isinstance(data, list):
        results = data

    log.info(f"Crawl4AI returned {len(results)} result(s) for {source_url}")

    parts = []
    total = 0
    skipped_short = 0
    for r in results:
        if not isinstance(r, dict):
            continue
        md = _coerce_markdown(r)
        if not md or len(md) < 50:
            skipped_short += 1
            continue
        page_url = r.get("url", source_url)
        if len(md) > MAX_CONTENT_PER_PAGE:
            md = md[:MAX_CONTENT_PER_PAGE] + "\n...[gekürzt]"
        if total + len(md) > MAX_TOTAL_CONTENT:
            break
        parts.append(f"=== SEITE: {page_url} ===\n{md}")
        total += len(md)

    if not parts and results:
        first = results[0] if isinstance(results[0], dict) else {}
        log.warning(
            f"_extract_markdown: 0 usable pages from {len(results)} result(s) "
            f"(skipped_short={skipped_short}, keys={list(first.keys())[:8]})"
        )

    return "\n\n".join(parts)


def _normalize_url(url: str) -> str:
    url = url.strip()
    if not url.startswith("http"):
        url = "https://" + url
    return url


# ═══════════════════════════════════════════════════════════════════════
# LLM HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _llm_call(system: str, user: str, max_tokens: int = 2000) -> dict:
    """Einfacher OpenAI-kompatibeler LLM-Call."""
    try:
        with httpx.Client(timeout=60) as client:
            resp = client.post("https://api.openai.com/v1/chat/completions", headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }, json={
                "model": ENRICHMENT_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.1,
                "max_tokens": max_tokens,
            })
            if resp.status_code != 200:
                log.error(f"LLM call failed: {resp.status_code}, {resp.text[:200]}")
                return {}
            content = resp.json()["choices"][0]["message"]["content"]
            return json.loads(content)
    except Exception as e:
        log.error(f"LLM exception: {e}")
        return {}


def _discover_website(company_name: str) -> str | None:
    """Offizielle Website einer Firma via Brave Search finden."""
    content = _brave_search_text(f'"{company_name}" offizielle website impressum')
    if not content:
        return None
    result = _llm_call(
        'Finde die offizielle Website der Firma (Hauptdomain, keine Unterseiten, keine Social-Media-URLs, keine Verzeichnis-Einträge). Antworte NUR als JSON: {"website": "https://..." oder null}.',
        f'Firma: "{company_name}"\n\nBrave-Suchergebnisse:\n{content}',
        max_tokens=100,
    )
    url = result.get("website")
    return url if url and isinstance(url, str) and url.startswith("http") else None


def _llm_classify_domain(domain: str, content: str) -> str:
    """Klassifiziert ob eine Domain eine Firma oder Freemail-Anbieter ist."""
    result = _llm_call(
        'Klassifiziere: Ist das eine Firmenwebsite oder ein E-Mail-Anbieter? Antworte NUR als JSON: {"type": "business"} oder {"type": "freemail"}',
        f"Domain: {domain}\n\nInhalt:\n{content[:3000]}",
        max_tokens=30,
    )
    return result.get("type", "unknown")


def _llm_extract_org(content: str, website: str) -> dict:
    """Extrahiert Organisations-Daten aus gecrawltem Content."""
    system = """Analysiere den Website-Inhalt und extrahiere Firmeninformationen.
Antworte als JSON mit optionalen Feldern (null wenn nicht erkennbar):
{
  "description": "Kurze Beschreibung (max 200 Zeichen)",
  "industry": "Branche",
  "employee_count": "1-10|11-50|51-200|201-500|501-1000|1001-5000|5001+",
  "founded_year": 2015,
  "legal_form": "GmbH|AG|e.V.|...",
  "vat_id": "USt-IdNr",
  "phone": "Haupttelefonnummer",
  "email": "Kontakt-E-Mail (info@, kontakt@)",
  "billing_street": "Straße + Nr",
  "billing_zip": "PLZ",
  "billing_city": "Stadt",
  "billing_country": "DE|AT|CH|...",
  "social_links": {
    "linkedin": "URL zum LinkedIn-Profil",
    "instagram": "URL",
    "website": "zusätzliche Website (nur wenn != Hauptdomain)",
    "youtube": "URL",
    "xing": "URL",
    "facebook": "URL",
    "tiktok": "URL",
    "x": "URL zu Twitter/X"
  },
  "team_members": [{"name":"Name","role":"Rolle","email":"Email","phone":"Tel","linkedin":"URL"}]
}
Wichtig:
- social_links: nur die 8 aufgelisteten Plattformen. Bei X/Twitter IMMER unter "x" speichern.
- Nur Fakten aus dem Text, keine Annahmen. Bei Unsicherheit: null oder Feld weglassen."""
    return _llm_call(system, f"Website {website}:\n\n{content}", max_tokens=2000)


def _llm_extract_contact(content: str, name: str, source: str, custom_fields: list | None, email: str | None = None) -> dict:
    """Extrahiert Kontakt-Daten."""
    ctx_map = {
        "org_website": "Suche auf der Firmenwebsite",
        "brave_search": "Suche in Brave-Web-Suchergebnissen",
        "google_search": "Suche in Web-Suchergebnissen",
    }
    ctx = ctx_map.get(source, "Suche in Web-Inhalten")

    custom_section = ""
    if custom_fields:
        custom_section = '\n\nZusätzlich unter "custom_fields": {\n'
        for f in custom_fields:
            custom_section += f'  "{f["name"]}": "{f["name"]} (null wenn nicht erkennbar)",\n'
        custom_section += "}"

    # Wenn kein Name bekannt ist: Person über E-Mail identifizieren + Namen mitliefern
    name_hint = f'"{name}"' if name else f"Person mit E-Mail \"{email or ''}\""
    name_rules = ""
    if not name:
        name_rules = "\n- Wenn Vor-/Nachname eindeutig zur E-Mail-Adresse gehören: first_name + last_name setzen."

    system = f"""{ctx} nach Infos über {name_hint}.
Antworte als JSON (null oder Feld weglassen wenn nicht erkennbar):
{{
  "first_name": "Vorname der Person (nur setzen wenn eindeutig)",
  "last_name": "Nachname der Person (nur setzen wenn eindeutig)",
  "organization_role": "Berufsbezeichnung/Titel der Person",
  "phone": "Direktnummer der Person (NICHT die Firmennummer)",
  "organization_name": "Firma (nur wenn eindeutig zugeordnet)",
  "billing_street": "Privatadresse: Straße + Nr (nur wenn eindeutig)",
  "billing_zip": "Privatadresse: PLZ",
  "billing_city": "Privatadresse: Stadt/Ort",
  "billing_country": "Privatadresse: DE|AT|CH|...",
  "social_links": {{
    "linkedin": "URL zum LinkedIn-Profil DIESER Person (keine Suchseite)",
    "instagram": "URL",
    "website": "eigene Website der Person",
    "youtube": "URL",
    "xing": "URL",
    "facebook": "URL",
    "tiktok": "URL",
    "x": "URL zu Twitter/X der Person"
  }}
}}{custom_section}
Wichtig:
- Keine Verwechslungen mit Namensvetter.
- social_links: nur die 8 aufgelisteten Keys. Keine Suchseiten, keine allgemeinen Firmen-URLs.
- Nur eindeutige Zuordnungen zur Person, keine Annahmen.{name_rules}"""
    user_msg = f'Infos über {name_hint}:\n\n{content}'
    return _llm_call(system, user_msg, max_tokens=900)


# ═══════════════════════════════════════════════════════════════════════
# UTILS
# ═══════════════════════════════════════════════════════════════════════

def _is_freemail(email: str) -> bool:
    domain = _extract_domain(email)
    return domain in OBVIOUS_FREEMAIL


def _extract_domain(email_or_url: str) -> str:
    s = email_or_url.strip()
    if "@" in s:
        s = s.split("@")[1]
    s = s.replace("https://", "").replace("http://", "").replace("www.", "")
    s = s.split("/")[0]
    return s.lower()


def _build_search_query(name: str, email: str | None, city: str | None) -> str:
    parts = []
    if name:
        parts.append(f'"{name}"')
    if email:
        parts.append(f'"{email}"')
    if city:
        parts.append(city)
    return " ".join(parts)


def _send_callback(url: str, payload: dict):
    """Ergebnis per POST an Laravel zurücksenden."""
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.post(url, json=payload, headers={
                "Authorization": f"Bearer {AI_GATEWAY_SECRET}",
                "Content-Type": "application/json",
            })
            if resp.status_code != 200:
                log.error(f"Callback failed: {url}, status={resp.status_code}, body={resp.text[:200]}")
            else:
                log.info(f"Callback sent: job_id={payload.get('job_id')}")
    except Exception as e:
        log.error(f"Callback exception: {url}, {e}")
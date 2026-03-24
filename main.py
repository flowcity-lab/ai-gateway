"""
AI-Gateway — FastAPI Service
Empfängt Chat-Requests von Laravel, orchestriert LLM-Calls,
und sendet Ergebnisse per Callback zurück.
"""

import os
import time
import json
import uuid
import base64
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import openai
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Konfiguration (Umgebungsvariablen) ────────────────────────────────

AI_GATEWAY_SECRET = os.environ.get("AI_GATEWAY_SECRET", "")

# Provider-Switch: 'openai' (Dev/Test) oder 'azure' (Produktion/DSGVO)
AI_PROVIDER = os.environ.get("AI_PROVIDER", "openai")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Azure OpenAI (nur wenn AI_PROVIDER=azure)
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_OPENAI_CHAT_ENDPOINT = os.environ.get("AZURE_OPENAI_CHAT_ENDPOINT", "")
AZURE_OPENAI_CHAT_API_KEY = os.environ.get("AZURE_OPENAI_CHAT_API_KEY", "")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")

# Laravel Callback-URLs (vom Gateway an Laravel)
LARAVEL_CALLBACK_URL = os.environ.get("LARAVEL_CALLBACK_URL", "")
LARAVEL_SKILL_URL = os.environ.get("LARAVEL_SKILL_URL", "")

# Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ai-gateway")

# ── Globaler State ────────────────────────────────────────────────────

oai_client = None


# ── Provider-Switch ───────────────────────────────────────────────────

def get_openai_client():
    """Erstellt OpenAI-Client basierend auf AI_PROVIDER Umgebungsvariable."""
    if AI_PROVIDER == "azure":
        from openai import AzureOpenAI
        return AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_CHAT_ENDPOINT,
            api_key=AZURE_OPENAI_CHAT_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
        )
    else:
        return openai.OpenAI(api_key=OPENAI_API_KEY)


def get_model_name(requested_model: str) -> str:
    """Gibt den richtigen Modell-/Deployment-Namen zurück."""
    if AI_PROVIDER == "azure":
        return AZURE_OPENAI_CHAT_DEPLOYMENT
    return requested_model or "gpt-4o"


# ── Lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global oai_client
    oai_client = get_openai_client()
    log.info("AI-Gateway ready. Provider=%s", AI_PROVIDER)
    yield


app = FastAPI(title="AI-Gateway", lifespan=lifespan)

# CORS für SSE-Streaming (Browser → Gateway direkt)
ALLOWED_ORIGINS = os.environ.get("CORS_ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Stream-Token Store (kurzlebig, in-memory) ────────────────────────
# Token → {data: ChatRequest-dict, created_at: timestamp}
stream_tokens: dict = {}
STREAM_TOKEN_TTL = 120  # Sekunden bis Token verfällt


# ── Auth ──────────────────────────────────────────────────────────────

def verify_auth(request: Request):
    """Bearer-Token gegen AI_GATEWAY_SECRET prüfen."""
    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(401, "Unauthorized")
    token = auth_header[7:]
    if not AI_GATEWAY_SECRET or token != AI_GATEWAY_SECRET:
        raise HTTPException(401, "Unauthorized")


# ── Pydantic Models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    user_id: int
    assistant_id: int
    chat_id: int
    message: str
    system_prompt: str = ""
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 2000
    conversation_history: list = []
    trainer_memories: list = []
    skills: list = []
    tool_definitions: list = []       # Fertige OpenAI Tool-Definitionen von Laravel
    images: list = []                 # Base64-Bilder für Vision-Analyse [{"data": "base64...", "media_type": "image/png"}]
    callback_url: str = ""
    mode: str = "async"  # "async" (default) oder "sync"
    # RAG-Konfiguration
    notebook_ids: list = []           # z.B. ["nb_1", "nb_3"]
    doc_processor_url: str = ""       # z.B. "https://trainer-doc-processor.ai-guide.at"
    doc_processor_secret: str = ""    # Bearer-Token für doc-processor
    # Trainer-Branding (Farben, Logo als Base64)
    trainer_branding: dict = None     # {"company_name", "primary_color", "logo_base64", ...}


# ── Skill-Labels (technischer Name → menschenlesbar) ────────────────

SKILL_LABELS = {
    "crm_search": {"label": "CRM durchsuchen", "icon": "🔍"},
    "crm_create_contact": {"label": "Kontakt anlegen", "icon": "👤"},
    "crm_update_deal": {"label": "Deal aktualisieren", "icon": "📊"},
    "crm_create_followup": {"label": "Wiedervorlage erstellen", "icon": "📅"},
    "content_generate": {"label": "Inhalt generieren", "icon": "✍️"},
    "image_generate": {"label": "Bild generieren", "icon": "🎨"},
    "pdf_generate": {"label": "PDF erstellen", "icon": "📄"},
    "email_generate": {"label": "E-Mail verfassen", "icon": "📧"},
    "campaign_analyze": {"label": "Kampagne analysieren", "icon": "📈"},
    "ai_background_task": {"label": "Hintergrund-Aufgabe starten", "icon": "⚙️"},
    "delegate_to_assistant": {"label": "An Assistenten delegieren", "icon": "🔄"},
}

def get_skill_label(skill_name: str) -> dict:
    """Gibt Label und Icon für einen Skill zurück."""
    default = {"label": skill_name.replace("_", " ").title(), "icon": "🔧"}
    return SKILL_LABELS.get(skill_name, default)


# ── Artifact-System (Universal HTML Visualizer) ──────────────────────

# ── Artifact Design System CSS ───────────────────────────────────────
# Wird automatisch in jedes Artifact injiziert. GPT schreibt nur semantisches HTML
# mit diesen Klassen — das CSS macht den Rest. Branding-Variablen (--primary etc.)
# werden dynamisch vom Gateway mit den Trainer-Farben befüllt.
ARTIFACT_BASE_CSS = """
/* ─── Artifact Design System v2.0 — Premium ─── */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --primary:#6366f1;--primary-light:#818cf8;--primary-dark:#4f46e5;
  --secondary:#8b5cf6;--accent:#f59e0b;
  --success:#10b981;--warning:#f59e0b;--danger:#ef4444;--info:#3b82f6;
  --bg:transparent;--bg-card:rgba(255,255,255,0.72);--bg-subtle:rgba(0,0,0,0.025);
  --bg-glass:rgba(255,255,255,0.55);
  --text:#1e293b;--text-secondary:#64748b;--text-muted:#94a3b8;
  --border:rgba(0,0,0,0.06);--border-strong:rgba(0,0,0,0.12);
  --radius:14px;--radius-sm:10px;--radius-xs:6px;
  --shadow:0 1px 2px rgba(0,0,0,0.04),0 4px 16px rgba(0,0,0,0.06);
  --shadow-md:0 2px 4px rgba(0,0,0,0.04),0 8px 24px rgba(0,0,0,0.1);
  --shadow-lg:0 4px 6px rgba(0,0,0,0.03),0 12px 40px rgba(0,0,0,0.12);
  --shadow-glow:0 0 20px rgba(99,102,241,0.15);
  --font:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
  --font-mono:'SF Mono',SFMono-Regular,Menlo,Consolas,monospace;
  --transition:0.2s cubic-bezier(.4,0,.2,1);
  --gradient:linear-gradient(135deg,var(--primary),var(--secondary));
}
@media(prefers-color-scheme:dark){
  :root{
    --bg:transparent;--bg-card:rgba(30,30,50,0.65);--bg-subtle:rgba(255,255,255,0.04);
    --bg-glass:rgba(30,30,50,0.5);
    --text:#e2e8f0;--text-secondary:#94a3b8;--text-muted:#64748b;
    --border:rgba(255,255,255,0.06);--border-strong:rgba(255,255,255,0.12);
    --shadow:0 1px 2px rgba(0,0,0,0.2),0 4px 16px rgba(0,0,0,0.3);
    --shadow-md:0 2px 4px rgba(0,0,0,0.2),0 8px 24px rgba(0,0,0,0.35);
    --shadow-lg:0 4px 6px rgba(0,0,0,0.2),0 12px 40px rgba(0,0,0,0.4);
    --shadow-glow:0 0 24px rgba(99,102,241,0.2);
  }
}
body{background:var(--bg);color:var(--text);font-family:var(--font);font-size:14px;line-height:1.6;padding:0;margin:0;-webkit-font-smoothing:antialiased}
a{color:var(--primary);text-decoration:none;transition:var(--transition)}a:hover{opacity:0.85}

/* ─── Layout ─── */
.grid{display:grid;gap:1rem}.cols-2{grid-template-columns:repeat(2,1fr)}.cols-3{grid-template-columns:repeat(3,1fr)}.cols-4{grid-template-columns:repeat(4,1fr)}
.flex{display:flex}.flex-col{flex-direction:column}.flex-wrap{flex-wrap:wrap}.items-center{align-items:center}.items-start{align-items:flex-start}.justify-between{justify-content:space-between}.justify-center{justify-content:center}
.gap-1{gap:0.25rem}.gap-2{gap:0.5rem}.gap-3{gap:0.75rem}.gap-4{gap:1rem}.gap-5{gap:1.25rem}.gap-6{gap:1.5rem}
.p-2{padding:0.5rem}.p-3{padding:0.75rem}.p-4{padding:1rem}.p-5{padding:1.25rem}.p-6{padding:1.5rem}.p-8{padding:2rem}
.px-4{padding-left:1rem;padding-right:1rem}.py-2{padding-top:0.5rem;padding-bottom:0.5rem}
.mt-1{margin-top:0.25rem}.mt-2{margin-top:0.5rem}.mt-3{margin-top:0.75rem}.mt-4{margin-top:1rem}.mt-6{margin-top:1.5rem}.mb-1{margin-bottom:0.25rem}.mb-2{margin-bottom:0.5rem}.mb-3{margin-bottom:0.75rem}.mb-4{margin-bottom:1rem}.mb-6{margin-bottom:1.5rem}
.w-full{width:100%}.text-center{text-align:center}.text-right{text-align:right}.text-left{text-align:left}
.hidden{display:none}.block{display:block}.inline{display:inline}.inline-block{display:inline-block}
.overflow-auto{overflow:auto}.overflow-hidden{overflow:hidden}
.relative{position:relative}.absolute{position:absolute}
@media(max-width:640px){.cols-2,.cols-3,.cols-4{grid-template-columns:1fr}.p-6,.p-8{padding:1rem}}

/* ─── Typography ─── */
h1,h2,h3,h4{font-weight:700;color:var(--text);line-height:1.25;letter-spacing:-0.01em}
h1{font-size:1.5rem;margin-bottom:0.5rem}h2{font-size:1.25rem;margin-bottom:0.5rem}h3{font-size:1.1rem;margin-bottom:0.375rem}h4{font-size:1rem}
.section-title{font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-muted);margin-bottom:0.75rem;display:flex;align-items:center;gap:0.5rem}
.section-title::after{content:'';flex:1;height:1px;background:var(--border)}
.text-xs{font-size:0.75rem}.text-sm{font-size:0.875rem}.text-base{font-size:1rem}.text-lg{font-size:1.125rem}.text-xl{font-size:1.25rem}.text-2xl{font-size:1.5rem}.text-3xl{font-size:1.875rem}.text-4xl{font-size:2.25rem}
.font-bold{font-weight:700}.font-semibold{font-weight:600}.font-medium{font-weight:500}.font-normal{font-weight:400}
.text-primary{color:var(--primary)}.text-success{color:var(--success)}.text-warning{color:var(--warning)}.text-danger{color:var(--danger)}.text-muted{color:var(--text-muted)}.text-secondary{color:var(--text-secondary)}
.text-gradient{background:var(--gradient);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.mono{font-family:var(--font-mono)}
.truncate{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.leading-tight{line-height:1.25}.leading-relaxed{line-height:1.75}

/* ─── Cards — Glassmorphism ─── */
.card{background:var(--bg-card);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);border:1px solid var(--border);border-radius:var(--radius);padding:1.5rem;box-shadow:var(--shadow);transition:var(--transition)}
.card:hover{box-shadow:var(--shadow-md);transform:translateY(-1px)}
.card-flat{background:var(--bg-subtle);border:1px solid var(--border);border-radius:var(--radius);padding:1.25rem}
.card-accent{border-left:3px solid var(--primary);background:var(--bg-card);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);border-radius:var(--radius);padding:1.25rem;box-shadow:var(--shadow)}
.card-glass{background:var(--bg-glass);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);border:1px solid var(--border);border-radius:var(--radius);padding:1.5rem;box-shadow:var(--shadow)}
.card-gradient{background:var(--gradient);border-radius:var(--radius);padding:1.5rem;color:#fff;box-shadow:var(--shadow-md)}
.card-gradient .text-muted,.card-gradient .text-secondary,.card-gradient .kpi-label{color:rgba(255,255,255,0.75)}
.card-gradient .kpi-value,.card-gradient h1,.card-gradient h2,.card-gradient h3{color:#fff}

/* ─── KPI / Metric Cards — Premium ─── */
.kpi{background:var(--bg-card);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);border:1px solid var(--border);border-radius:var(--radius);padding:1.25rem 1.5rem;position:relative;overflow:hidden;box-shadow:var(--shadow);transition:var(--transition)}
.kpi:hover{box-shadow:var(--shadow-md);transform:translateY(-2px)}
.kpi::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:var(--gradient);opacity:0.8}
.kpi-icon{width:2.5rem;height:2.5rem;border-radius:var(--radius-sm);background:linear-gradient(135deg,rgba(99,102,241,0.12),rgba(139,92,246,0.12));display:flex;align-items:center;justify-content:center;font-size:1.15rem;margin-bottom:0.75rem}
.kpi-value{font-size:1.85rem;font-weight:800;background:var(--gradient);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1.1;letter-spacing:-0.02em}
.kpi-label{font-size:0.72rem;color:var(--text-muted);margin-top:0.25rem;text-transform:uppercase;letter-spacing:0.06em;font-weight:600}
.kpi-trend{font-size:0.75rem;margin-top:0.5rem;font-weight:600;display:inline-flex;align-items:center;gap:0.25rem;padding:0.15rem 0.5rem;border-radius:999px}
.kpi-trend.up{color:var(--success);background:rgba(16,185,129,0.1)}.kpi-trend.down{color:var(--danger);background:rgba(239,68,68,0.1)}.kpi-trend.neutral{color:var(--text-muted);background:var(--bg-subtle)}
.kpi-row{display:flex;align-items:flex-end;justify-content:space-between;gap:0.75rem}

/* ─── Tables — Polished ─── */
.table-wrap{overflow-x:auto;border-radius:var(--radius);border:1px solid var(--border);box-shadow:var(--shadow);background:var(--bg-card);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px)}
table{width:100%;border-collapse:collapse;font-size:0.85rem}
th{background:var(--bg-subtle);font-weight:600;text-align:left;padding:0.75rem 1rem;border-bottom:1px solid var(--border-strong);color:var(--text-secondary);font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;position:sticky;top:0;z-index:1}
td{padding:0.7rem 1rem;border-bottom:1px solid var(--border);vertical-align:middle}
tr:hover td{background:rgba(99,102,241,0.03)}
tr:last-child td{border-bottom:none}
td .flex{gap:0.5rem}

/* ─── Badges / Pills ─── */
.badge{display:inline-flex;align-items:center;gap:0.25rem;padding:0.2rem 0.65rem;border-radius:999px;font-size:0.72rem;font-weight:600;line-height:1.4;letter-spacing:0.01em}
.badge-primary{background:rgba(99,102,241,0.1);color:var(--primary)}
.badge-success{background:rgba(16,185,129,0.1);color:var(--success)}
.badge-warning{background:rgba(245,158,11,0.1);color:var(--warning)}
.badge-danger{background:rgba(239,68,68,0.1);color:var(--danger)}
.badge-info{background:rgba(59,130,246,0.1);color:var(--info)}
.badge-neutral{background:var(--bg-subtle);color:var(--text-secondary)}
.badge-dot::before{content:'';width:6px;height:6px;border-radius:50%;background:currentColor}

/* ─── Progress Bars ─── */
.progress{height:6px;background:var(--bg-subtle);border-radius:999px;overflow:hidden}
.progress-bar{height:100%;border-radius:999px;background:var(--gradient);transition:width 0.8s cubic-bezier(.4,0,.2,1)}
.progress-bar.success{background:var(--success)}.progress-bar.warning{background:var(--warning)}.progress-bar.danger{background:var(--danger)}
.progress-lg{height:10px}
.progress-label{display:flex;justify-content:space-between;font-size:0.78rem;color:var(--text-secondary);margin-bottom:0.3rem;font-weight:500}

/* ─── Lists ─── */
.list{list-style:none;padding:0;margin:0}
.list-item{display:flex;align-items:center;gap:0.75rem;padding:0.65rem 0;border-bottom:1px solid var(--border)}
.list-item:last-child{border-bottom:none}
.list-icon{width:2.25rem;height:2.25rem;border-radius:var(--radius-sm);background:linear-gradient(135deg,rgba(99,102,241,0.1),rgba(139,92,246,0.08));display:flex;align-items:center;justify-content:center;font-size:1rem;flex-shrink:0}
.list-content{flex:1;min-width:0}

/* ─── Tabs ─── */
.tabs{display:flex;gap:0;border-bottom:2px solid var(--border);margin-bottom:1rem}
.tab{padding:0.6rem 1rem;font-size:0.85rem;font-weight:500;color:var(--text-secondary);cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-2px;transition:var(--transition);user-select:none}
.tab:hover{color:var(--text)}.tab.active{color:var(--primary);border-bottom-color:var(--primary);font-weight:600}

/* ─── Alerts ─── */
.alert{padding:0.875rem 1rem;border-radius:var(--radius-sm);font-size:0.85rem;display:flex;align-items:flex-start;gap:0.5rem;backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px)}
.alert-info{background:rgba(59,130,246,0.07);color:var(--info);border:1px solid rgba(59,130,246,0.12)}
.alert-success{background:rgba(16,185,129,0.07);color:var(--success);border:1px solid rgba(16,185,129,0.12)}
.alert-warning{background:rgba(245,158,11,0.07);color:var(--warning);border:1px solid rgba(245,158,11,0.12)}
.alert-danger{background:rgba(239,68,68,0.07);color:var(--danger);border:1px solid rgba(239,68,68,0.12)}

/* ─── Divider ─── */
.divider{border:none;border-top:1px solid var(--border);margin:1.25rem 0}

/* ─── Tags/Chips ─── */
.tag{display:inline-flex;align-items:center;gap:0.25rem;padding:0.2rem 0.5rem;background:var(--bg-subtle);border:1px solid var(--border);border-radius:var(--radius-xs);font-size:0.72rem;color:var(--text-secondary);font-weight:500}

/* ─── Tooltips ─── */
[data-tip]{position:relative;cursor:help}
[data-tip]::after{content:attr(data-tip);position:absolute;bottom:calc(100% + 6px);left:50%;transform:translateX(-50%);background:#1e293b;color:#fff;padding:0.3rem 0.6rem;border-radius:var(--radius-xs);font-size:0.7rem;white-space:nowrap;opacity:0;pointer-events:none;transition:opacity 0.15s}
[data-tip]:hover::after{opacity:1}

/* ─── Charts (CSS-only Bars) — Enhanced ─── */
.bar-chart{display:flex;align-items:flex-end;gap:0.5rem;height:180px;padding:1.5rem 0 1.5rem 0;border-bottom:1px solid var(--border)}
.bar{flex:1;border-radius:var(--radius-xs) var(--radius-xs) 0 0;min-width:20px;position:relative;transition:filter 0.3s,box-shadow 0.3s;background:var(--gradient)}
.bar:hover{filter:brightness(1.1);box-shadow:0 -4px 12px rgba(99,102,241,0.2)}
.bar-label{position:absolute;bottom:-1.5rem;left:50%;transform:translateX(-50%);font-size:0.65rem;color:var(--text-muted);white-space:nowrap;font-weight:500}
.bar-value{position:absolute;top:-1.4rem;left:50%;transform:translateX(-50%);font-size:0.7rem;font-weight:700;color:var(--text)}
.bar.secondary{background:var(--secondary)}.bar.accent{background:var(--accent)}.bar.success{background:var(--success)}.bar.info{background:var(--info)}.bar.warning{background:var(--warning)}.bar.danger{background:var(--danger)}

/* ─── Donut Chart (SVG helper) ─── */
.donut-chart{position:relative;display:inline-flex;align-items:center;justify-content:center}
.donut-label{position:absolute;text-align:center}
.donut-label strong{display:block;font-size:1.5rem;font-weight:800;background:var(--gradient);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.donut-label span{font-size:0.7rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em}

/* ─── Timeline — Refined ─── */
.timeline{position:relative;padding-left:2rem}
.timeline::before{content:'';position:absolute;left:0.55rem;top:0;bottom:0;width:2px;background:linear-gradient(to bottom,var(--primary),var(--secondary),var(--border));border-radius:1px}
.timeline-item{position:relative;padding-bottom:1.25rem}
.timeline-item::before{content:'';position:absolute;left:-1.45rem;top:0.35rem;width:10px;height:10px;border-radius:50%;background:var(--primary);box-shadow:0 0 0 3px var(--bg-card),var(--shadow);z-index:1}
.timeline-item.done::before{background:var(--success)}.timeline-item.pending::before{background:var(--text-muted)}
.timeline-date{font-size:0.72rem;color:var(--text-muted);margin-bottom:0.125rem;font-weight:500}

/* ─── Score / Rating ─── */
.score-ring{width:64px;height:64px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:1.25rem;font-weight:800;color:#fff;background:var(--gradient);box-shadow:var(--shadow-glow)}
.score-ring.lg{width:80px;height:80px;font-size:1.5rem}
.score-ring.accent{background:linear-gradient(135deg,var(--accent),var(--warning))}
.stars{color:var(--accent);letter-spacing:2px;font-size:1.1rem}

/* ─── Accent Highlights ─── */
.accent-bar{height:3px;background:linear-gradient(90deg,var(--accent),var(--primary));border-radius:2px;margin:0.75rem 0}
.badge-accent{background:rgba(245,158,11,0.12);color:var(--accent);font-weight:600}
.card-accent-top{position:relative;overflow:hidden}.card-accent-top::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,var(--accent),var(--primary))}
.kpi.accent::before{background:linear-gradient(90deg,var(--accent),var(--primary))}
.text-accent{color:var(--accent)}
.highlight{background:linear-gradient(120deg,rgba(245,158,11,0.1) 0%,rgba(245,158,11,0.05) 100%);padding:0.75rem 1rem;border-radius:var(--radius-sm);border-left:3px solid var(--accent)}

/* ─── Stat Highlight Row ─── */
.stat-row{display:flex;align-items:center;justify-content:space-between;padding:0.6rem 0;border-bottom:1px solid var(--border)}
.stat-row:last-child{border-bottom:none}
.stat-row .label{font-size:0.85rem;color:var(--text-secondary)}
.stat-row .value{font-size:0.95rem;font-weight:700;color:var(--text)}

/* ─── Animations — Premium ─── */
@keyframes fadeIn{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
@keyframes slideIn{from{opacity:0;transform:translateX(-12px)}to{opacity:1;transform:translateX(0)}}
@keyframes slideUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
@keyframes scaleIn{from{opacity:0;transform:scale(0.92)}to{opacity:1;transform:scale(1)}}
@keyframes shimmer{0%{background-position:-200% 0}100%{background-position:200% 0}}
@keyframes countUp{from{opacity:0;transform:translateY(8px) scale(0.95)}to{opacity:1;transform:translateY(0) scale(1)}}
@keyframes glowPulse{0%,100%{box-shadow:var(--shadow-glow)}50%{box-shadow:0 0 28px rgba(99,102,241,0.25)}}
@keyframes barGrow{from{transform:scaleY(0)}to{transform:scaleY(1)}}
@keyframes progressFill{from{width:0}}
.animate-in{animation:fadeIn 0.55s cubic-bezier(.22,1,.36,1) both}
.animate-in-1{animation-delay:0.08s}.animate-in-2{animation-delay:0.16s}.animate-in-3{animation-delay:0.24s}.animate-in-4{animation-delay:0.32s}.animate-in-5{animation-delay:0.4s}.animate-in-6{animation-delay:0.48s}.animate-in-7{animation-delay:0.56s}
.animate-slide{animation:slideIn 0.5s cubic-bezier(.22,1,.36,1) both}
.animate-slide-up{animation:slideUp 0.6s cubic-bezier(.22,1,.36,1) both}
.animate-scale{animation:scaleIn 0.45s cubic-bezier(.22,1,.36,1) both}
.animate-glow{animation:glowPulse 2s ease-in-out 0.5s 2}
.kpi-value{animation:countUp 0.6s cubic-bezier(.22,1,.36,1) both}
.bar{transform-origin:bottom;animation:barGrow 0.7s cubic-bezier(.22,1,.36,1) both}
.bar:nth-child(1){animation-delay:0.1s}.bar:nth-child(2){animation-delay:0.2s}.bar:nth-child(3){animation-delay:0.3s}.bar:nth-child(4){animation-delay:0.4s}.bar:nth-child(5){animation-delay:0.5s}.bar:nth-child(6){animation-delay:0.6s}
.progress-bar{animation:progressFill 1s cubic-bezier(.22,1,.36,1) both;animation-delay:0.3s}
.score-ring{animation:scaleIn 0.5s cubic-bezier(.22,1,.36,1) both, glowPulse 2s ease-in-out 0.8s 2}

/* ─── Utility Decorators ─── */
.rounded{border-radius:var(--radius)}.rounded-sm{border-radius:var(--radius-sm)}.rounded-full{border-radius:999px}
.shadow{box-shadow:var(--shadow)}.shadow-md{box-shadow:var(--shadow-md)}.shadow-lg{box-shadow:var(--shadow-lg)}
.border{border:1px solid var(--border)}.border-b{border-bottom:1px solid var(--border)}
.opacity-75{opacity:0.75}.opacity-50{opacity:0.5}

/* ─── Interactive Controls ─── */
input[type="range"]{-webkit-appearance:none;appearance:none;width:100%;height:6px;border-radius:3px;background:var(--bg-subtle);outline:none;transition:var(--transition)}
input[type="range"]::-webkit-slider-thumb{-webkit-appearance:none;width:18px;height:18px;border-radius:50%;background:var(--gradient);cursor:pointer;box-shadow:var(--shadow)}
input[type="range"]::-moz-range-thumb{width:18px;height:18px;border-radius:50%;background:var(--gradient);cursor:pointer;border:none;box-shadow:var(--shadow)}
input[type="number"],input[type="text"],select{padding:0.5rem 0.75rem;border:1px solid var(--border-strong);border-radius:var(--radius-sm);font-size:0.9rem;background:var(--bg-card);color:var(--text);outline:none;transition:var(--transition);font-family:var(--font)}
input[type="number"]:focus,input[type="text"]:focus,select:focus{border-color:var(--primary);box-shadow:0 0 0 3px rgba(99,102,241,0.1)}
.btn{display:inline-flex;align-items:center;gap:0.4rem;padding:0.5rem 1rem;border-radius:var(--radius-sm);font-size:0.85rem;font-weight:600;cursor:pointer;transition:var(--transition);border:none;font-family:var(--font)}
.btn-primary{background:var(--gradient);color:#fff;box-shadow:var(--shadow)}.btn-primary:hover{opacity:0.9;box-shadow:var(--shadow-md)}
.btn-secondary{background:var(--bg-subtle);color:var(--text);border:1px solid var(--border-strong)}.btn-secondary:hover{background:var(--border);border-color:var(--border-strong)}
.btn-accent{background:var(--accent);color:#fff;box-shadow:var(--shadow)}.btn-accent:hover{opacity:0.9}
.btn-sm{padding:0.3rem 0.7rem;font-size:0.78rem}.btn-lg{padding:0.7rem 1.5rem;font-size:1rem}
.btn-group{display:flex;gap:0.5rem;flex-wrap:wrap}
.btn-toggle{background:var(--bg-subtle);color:var(--text-secondary);border:1px solid var(--border)}.btn-toggle.active{background:var(--gradient);color:#fff;border-color:transparent;box-shadow:var(--shadow)}

/* ─── Panels / Content Sections ─── */
.panel{background:var(--bg-card);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden}
.panel-header{padding:0.75rem 1.25rem;background:var(--bg-subtle);border-bottom:1px solid var(--border);font-weight:600;font-size:0.85rem;display:flex;align-items:center;justify-content:space-between}
.panel-body{padding:1.25rem}
.tab-content{display:none}.tab-content.active{display:block}
.side-by-side{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem}
@media(max-width:640px){.side-by-side{grid-template-columns:1fr}}

/* ─── Text Highlighting ─── */
.text-highlight{padding:2px 6px;border-radius:4px;transition:all 0.3s}
.text-highlight-primary{background:rgba(99,102,241,0.1);border-bottom:2px solid var(--primary)}
.text-highlight-success{background:rgba(16,185,129,0.1);border-bottom:2px solid var(--success)}
.text-highlight-danger{background:rgba(239,68,68,0.1);border-bottom:2px solid var(--danger)}
.text-highlight-warning{background:rgba(245,158,11,0.1);border-bottom:2px solid var(--warning)}
.text-highlight-accent{background:rgba(245,158,11,0.08);border-bottom:2px solid var(--accent)}

/* ─── Comparison / Pro-Contra ─── */
.pro-contra{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
.pro-list,.contra-list{padding:1rem;border-radius:var(--radius-sm)}
.pro-list{background:rgba(16,185,129,0.05);border:1px solid rgba(16,185,129,0.15)}
.contra-list{background:rgba(239,68,68,0.05);border:1px solid rgba(239,68,68,0.15)}
.pro-list li::before{content:'✓ ';color:var(--success);font-weight:700}
.contra-list li::before{content:'✗ ';color:var(--danger);font-weight:700}
.pro-list li,.contra-list li{list-style:none;padding:0.35rem 0;font-size:0.88rem}
@media(max-width:640px){.pro-contra{grid-template-columns:1fr}}

/* ─── Scrollbar ─── */
::-webkit-scrollbar{width:5px;height:5px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:var(--border-strong);border-radius:3px}::-webkit-scrollbar-thumb:hover{background:var(--text-muted)}
"""

# Resize-Script das in jedes Artifact injiziert wird (postMessage für Auto-Höhe)
ARTIFACT_RESIZE_SCRIPT = """<script>
(function(){
  function h(){var b=document.body,d=document.documentElement;
    var height=Math.max(b.scrollHeight,b.offsetHeight,d.scrollHeight,d.offsetHeight);
    parent.postMessage({type:'artifact-resize',height:height},'*');
  }
  h();
  new MutationObserver(h).observe(document.body,{childList:true,subtree:true,attributes:true});
  window.addEventListener('resize',h);
  document.addEventListener('click',function(e){var a=e.target.closest('a');if(a&&a.href&&!a.target)a.target='_top';});
})();
</script>"""


def _hex_to_rgb(hex_color: str) -> tuple:
    """Konvertiert #RRGGBB zu (r, g, b)."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join(c * 2 for c in hex_color)
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def _lighten_hex(hex_color: str, factor: float = 0.3) -> str:
    """Hellt eine Hex-Farbe auf."""
    try:
        r, g, b = _hex_to_rgb(hex_color)
        r = min(255, int(r + (255 - r) * factor))
        g = min(255, int(g + (255 - g) * factor))
        b = min(255, int(b + (255 - b) * factor))
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return hex_color


def _darken_hex(hex_color: str, factor: float = 0.2) -> str:
    """Dunkelt eine Hex-Farbe ab."""
    try:
        r, g, b = _hex_to_rgb(hex_color)
        r = max(0, int(r * (1 - factor)))
        g = max(0, int(g * (1 - factor)))
        b = max(0, int(b * (1 - factor)))
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return hex_color


def inject_artifact_css(html: str, branding: dict = None) -> str:
    """
    Injiziert das Design-System CSS und Branding-Variablen in Artifact-HTML.
    Wird automatisch bei jedem render_artifact Call aufgerufen.
    """
    # Branding CSS Custom Property Overrides
    brand_overrides = ""
    if branding:
        overrides = []
        primary = branding.get("primary_color")
        if primary:
            overrides.append(f"--primary:{primary}")
            overrides.append(f"--primary-light:{_lighten_hex(primary, 0.3)}")
            overrides.append(f"--primary-dark:{_darken_hex(primary, 0.2)}")
        if branding.get("secondary_color"):
            overrides.append(f"--secondary:{branding['secondary_color']}")
        if branding.get("accent_color"):
            overrides.append(f"--accent:{branding['accent_color']}")
        if branding.get("font_family"):
            overrides.append(f"--font:{branding['font_family']},system-ui,sans-serif")
        if overrides:
            brand_overrides = f"\n:root{{{';'.join(overrides)}}}"
            log.info("Artifact branding: %s", '; '.join(overrides))

    css_block = f"<style>{ARTIFACT_BASE_CSS}{brand_overrides}</style>"

    # Inject: Wenn <head> vorhanden, dort rein. Sonst ans Ende vom <body> oder einfach davor.
    if "<head>" in html.lower():
        html = html.replace("<head>", f"<head>{css_block}", 1).replace("<HEAD>", f"<HEAD>{css_block}", 1)
    elif "<body>" in html.lower():
        html = html.replace("<body>", f"<body>{css_block}", 1).replace("<BODY>", f"<BODY>{css_block}", 1)
    else:
        html = css_block + html

    # Resize-Script injizieren (vor </body> oder am Ende)
    if "</body>" in html.lower():
        html = html.replace("</body>", f"{ARTIFACT_RESIZE_SCRIPT}</body>", 1).replace("</BODY>", f"{ARTIFACT_RESIZE_SCRIPT}</BODY>", 1)
    else:
        html = html + ARTIFACT_RESIZE_SCRIPT

    return html


RENDER_ARTIFACT_TOOL = {
    "type": "function",
    "function": {
        "name": "render_artifact",
        "description": (
            "Rendere eine kreative, interaktive HTML-Visualisierung inline im Chat. "
            "Rufe dieses Tool pro Antwort MAXIMAL EINMAL auf. "
            "Frage dich: Wie kann ich dieses Thema visuell am besten darstellen? "
            "Ein CSS Design-System wird automatisch injiziert. "
            "Schreibe eigenes CSS und JavaScript für einzigartige, kreative Erlebnisse. Sei kreativ!"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Interner Titel (wird NICHT sichtbar angezeigt, nur für Logging).",
                },
                "html": {
                    "type": "string",
                    "description": (
                        "Vollständiges HTML mit eigenem CSS und JavaScript. "
                        "Basis-Klassen verfügbar (.card, .grid, .kpi, .btn, .animate-in etc.) — "
                        "aber schreibe EIGENES <style> und <script> für kreative, einzigartige Erlebnisse. "
                        "Denke wie ein Designer: Wie stellt man dieses Thema visuell am besten dar?"
                    ),
                },
            },
            "required": ["title", "html"],
        },
    },
}


ARTIFACT_SYSTEM_PROMPT = """

## Artifact-System — Kreative interaktive HTML-Visualisierungen

Du hast das Tool `render_artifact` mit dem du **kreative, interaktive HTML/CSS/JS-Visualisierungen** direkt im Chat rendern kannst. Ein CSS Design-System wird automatisch injiziert — aber du hast **volle kreative Freiheit** eigenes CSS und JavaScript zu schreiben.

### Deine Aufgabe:
**Erstelle eine gut durchdachte, kreative HTML-Seite die die Aufgabe des Users visuell löst.** Frage dich: "Wie kann ich das am besten visuell und interaktiv darstellen?" — und baue etwas das den User überrascht und begeistert. Du bist ein kreativer Designer, kein Template-Ausfüller!

### ⚠️ REGELN:
1. **EIN Artifact pro Antwort** — Rufe `render_artifact` NIEMALS mehrmals auf.
2. **JEDES interaktive Element MUSS funktionieren** — Kein Dummy-UI. Jeder Button, Slider, Klick braucht funktionales `<script>`.
3. **Hintergrund TRANSPARENT** — fügt sich nahtlos in den Chat ein. Dark-Mode automatisch.
4. **Branding-Farben** `var(--primary)`, `var(--secondary)`, `var(--accent)` werden automatisch injiziert — nutze sie!
5. **Erstelle ein Artifact** wenn etwas visuell/interaktiv besser wirkt als Text. NICHT für simple Kurzantworten.

### 🎨 Kreative Freiheit — SO sollst du arbeiten:
- Schreibe **eigenes `<style>`** für einzigartige, zum Thema passende Designs
- Schreibe **eigenes `<script>`** für Interaktivität — Klicks, Slider, Step-by-Step Flows, Live-Berechnungen, Animationen, alles was Sinn macht
- Die Basis-Klassen (siehe unten) sind ein Startpunkt — du SOLLST darüber hinaus kreativ werden
- **Jedes Artifact soll visuell einzigartig sein** und perfekt zum jeweiligen Thema passen
- Frage dich immer: "Wie würde ein Top-Designer das für dieses spezifische Thema lösen?"

### 💡 Inspirationen (kopiere NICHT — erfinde eigene kreative Varianten!):
- **Persönlichkeitstest:** Fragen Step-by-Step nacheinander anzeigen, Progress-Bar ("Frage 2 von 5"), am Ende Ergebnis-Reveal mit Charakter/Typ-Beschreibung und passendem Emoji
- **Lern-Explorer:** Text mit farbig hervorgehobenen Stellen, Slider/Buttons steuern Gewichtung, Live-Score der sich verändert
- **Rechner/Simulator:** Eingabefelder und Slider → Live-Ergebnis mit animierten Zahlen, Gradient-Result-Card
- **Daten-Dashboard:** KPI-Cards mit Trend-Pfeilen, animierte Bar-Charts, filterbare Tabellen, Score-Rings
- **Quiz:** Fragen nacheinander mit Ja/Nein oder Multiple-Choice, Auflösung mit ✓/✗, Endergebnis-Screen
- **Vergleich:** Pro/Contra nebeneinander, Toggle zwischen Optionen, Bewertungs-Slider
- **Timeline/Story:** Interaktive Schritte mit Fortschrittsanzeige, klickbare Meilensteine

### Verfügbare Basis-Klassen (erweitere frei mit eigenem CSS!):
**Layout:** `.grid .cols-2/3/4`, `.flex .items-center .justify-between`, `.gap-1` bis `-6`, `.p-2` bis `-8`, `.mb-1` bis `-6`, `.w-full`
**Cards:** `.card` (Glassmorphism), `.card-glass` (extra Blur), `.card-gradient` (Gradient-BG + weiß), `.card-flat`, `.card-accent`
**Text:** `.text-xs/sm/base/lg/xl/2xl/3xl`, `.font-bold`, `.text-gradient`, `.text-primary/success/warning/danger/muted/accent`, `.section-title`
**Daten:** `.kpi > .kpi-icon + .kpi-value + .kpi-label + .kpi-trend.up/.down`, `.score-ring`, `.progress + .progress-bar`, `.bar-chart > .bar`, `.stat-row`, `.table-wrap > table`
**Interaktiv:** `.btn .btn-primary/.btn-accent/.btn-secondary`, `.btn-toggle.active`, `.btn-group`, `.tabs > .tab`, Inputs (`range/number/text`) automatisch gestylt
**Badges:** `.badge .badge-primary/success/warning/danger`, `.badge-dot`
**Animation:** `.animate-in .animate-in-1` bis `-7` (gestaffelte Einblend-Effekte — IMMER verwenden!), `.animate-slide-up`, `.animate-scale`
**Mehr:** `.timeline > .timeline-item`, `.alert`, `.side-by-side`, `.pro-contra > .pro-list + .contra-list`, `.text-highlight-primary/success/danger/accent`, `.highlight`, `.panel > .panel-header + .panel-body`

### JavaScript-Pattern für Interaktivität:
```javascript
(function(){
  const state = { step: 0 /* deine Werte */ };
  function update() { /* DOM basierend auf state aktualisieren, Elemente ein-/ausblenden, Werte berechnen */ }
  document.querySelectorAll('[data-action]').forEach(el => el.addEventListener('click', e => { /* state ändern */ update(); }));
  document.querySelectorAll('input').forEach(el => el.addEventListener('input', e => { /* state ändern */ update(); }));
  update(); // Initial-Render
})();
```

### Technische Hinweise:
- Starte HTML mit `<meta charset="utf-8">`
- CRM-Links: `<a href="/crm/contacts/{id}" target="_top">`
- Nutze `var(--primary)`, `var(--accent)`, `var(--gradient)` für Branding-konsistente Farben
- `.animate-in` mit `.animate-in-1` bis `-7` auf sichtbaren Elementen für elegante Lade-Animationen
"""


# ── Health ────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "provider": AI_PROVIDER,
    }


# ── Chat Endpoint ────────────────────────────────────────────────────

@app.post("/chat")
async def chat(request: Request, bg: BackgroundTasks):
    verify_auth(request)
    try:
        raw = await request.body()
        text = raw.decode("utf-8")
        body = json.loads(text)
    except UnicodeDecodeError:
        raise HTTPException(400, "Request body must be UTF-8 encoded")
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON in request body")
    data = ChatRequest(**body)

    # Sync-Modus: LLM-Call direkt ausführen und Ergebnis zurückgeben
    if data.mode == "sync":
        log.info("Chat %d: Sync-Modus", data.chat_id)
        result = chat_pipeline_sync(data)
        return result

    # Async-Modus (default): Sofort 202 Accepted, Verarbeitung im Background
    bg.add_task(chat_pipeline, data)

    return {"status": "accepted", "chat_id": data.chat_id}


# ── SSE Streaming ────────────────────────────────────────────────────

@app.post("/chat/stream/init")
async def stream_init(request: Request):
    """
    Laravel ruft diesen Endpoint auf, um einen Stream vorzubereiten.
    Gibt einen kurzlebigen Token zurück, den der Browser für EventSource nutzt.
    """
    verify_auth(request)
    try:
        raw = await request.body()
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise HTTPException(400, "Invalid request")

    data = ChatRequest(**body)
    token = str(uuid.uuid4())

    # Abgelaufene Tokens aufräumen
    now = time.time()
    expired = [k for k, v in stream_tokens.items() if now - v["created_at"] > STREAM_TOKEN_TTL]
    for k in expired:
        del stream_tokens[k]

    stream_tokens[token] = {"data": data, "created_at": now}
    log.info("Stream init: chat_id=%d token=%s", data.chat_id, token[:8])

    return {"stream_token": token, "chat_id": data.chat_id}


@app.get("/chat/stream")
async def stream_chat(token: str):
    """
    SSE-Endpoint: Browser öffnet EventSource mit ?token=xxx.
    Streamt LLM-Tokens als Server-Sent Events.
    """
    entry = stream_tokens.pop(token, None)
    if not entry:
        raise HTTPException(401, "Invalid or expired stream token")

    if time.time() - entry["created_at"] > STREAM_TOKEN_TTL:
        raise HTTPException(401, "Stream token expired")

    data: ChatRequest = entry["data"]
    log.info("Stream start: chat_id=%d", data.chat_id)

    return StreamingResponse(
        stream_chat_generator(data),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


async def stream_chat_generator(data: ChatRequest):
    """Generator der SSE-Events yieldet. Führt Tool-Calls synchron aus,
    streamt nur die finale Text-Antwort."""
    start_time = time.time()
    full_content = ""
    usage = {"input_tokens": 0, "output_tokens": 0}
    model = get_model_name(data.model)
    all_tool_calls = []
    all_tool_results = []
    artifacts = []  # Gesammelte render_artifact Ergebnisse

    try:
        # RAG
        rag_context = []
        if data.notebook_ids and data.doc_processor_url:
            rag_context = search_documents(
                query=data.message,
                notebook_ids=data.notebook_ids,
                doc_processor_url=data.doc_processor_url,
                doc_processor_secret=data.doc_processor_secret,
            )

        messages = build_messages(data, rag_context=rag_context)
        tools = build_tools(data.tool_definitions, data.skills)
        max_iterations = 5

        # Tool-Call-Loop (nicht-gestreamt — nur finale Antwort wird gestreamt)
        for iteration in range(max_iterations):
            call_params = {
                "model": model,
                "messages": messages,
                "temperature": data.temperature,
                "max_tokens": data.max_tokens,
            }

            is_last_iteration = iteration >= max_iterations - 1
            has_tools = tools and not is_last_iteration

            if has_tools:
                call_params["tools"] = tools
                call_params["tool_choice"] = "auto"

            # Prüfen ob wir Tool-Calls erwarten — wenn ja, nicht streamen
            if has_tools:
                response = oai_client.chat.completions.create(**call_params)
                choice = response.choices[0]

                if choice.message.tool_calls:
                    # Tool-Calls verarbeiten
                    messages.append(choice.message)
                    for tc in choice.message.tool_calls:
                        tool_name = tc.function.name
                        tool_args = json.loads(tc.function.arguments)
                        log.info("Stream chat %d: Tool-Call → %s", data.chat_id, tool_name)
                        skill_info = get_skill_label(tool_name)

                        # render_artifact: Gateway-intercepted (nicht an Laravel senden)
                        if tool_name == "render_artifact":
                            artifact_title = tool_args.get("title", "Visualisierung")
                            artifact_html = inject_artifact_css(tool_args.get("html", ""), data.trainer_branding)
                            artifact_data = {"title": artifact_title, "html": artifact_html}
                            artifacts.append(artifact_data)
                            log.info("Stream chat %d: Artifact '%s' (%d bytes HTML)", data.chat_id, artifact_title, len(artifact_html))

                            # SSE artifact Event an Frontend senden
                            yield f"event: artifact\ndata: {json.dumps(artifact_data, ensure_ascii=False)}\n\n"

                            # Erfolg an GPT zurückmelden (damit GPT weitermachen kann)
                            all_tool_calls.append({"id": tc.id, "name": tool_name, "arguments": tool_args})
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": json.dumps({"success": True, "rendered": True, "title": artifact_title}, ensure_ascii=False),
                            })
                            continue

                        # Delegation: eigene Start/End Events
                        is_delegation = tool_name == "delegate_to_assistant"
                        target_name = tool_args.get("target_assistant", "") if is_delegation else ""

                        if is_delegation:
                            yield f"event: delegation_start\ndata: {json.dumps({'tool': tool_name, 'label': skill_info['label'], 'icon': skill_info['icon'], 'target': target_name})}\n\n"
                        else:
                            yield f"event: tool_start\ndata: {json.dumps({'tool': tool_name, 'label': skill_info['label'], 'icon': skill_info['icon']})}\n\n"

                        all_tool_calls.append({"id": tc.id, "name": tool_name, "arguments": tool_args})
                        result = execute_skill(tool_name, tool_args, data.user_id, data.chat_id)
                        all_tool_results.append({"tool_call_id": tc.id, "name": tool_name, "result": result})
                        skill_success = "error" not in result

                        if is_delegation:
                            delegate_name = result.get("assistant_name", target_name)
                            yield f"event: delegation_end\ndata: {json.dumps({'tool': tool_name, 'label': delegate_name, 'success': skill_success})}\n\n"
                        else:
                            yield f"event: tool_end\ndata: {json.dumps({'tool': tool_name, 'label': skill_info['label'], 'icon': skill_info['icon'], 'success': skill_success})}\n\n"

                        # Base64-Daten entfernen bevor sie an GPT gehen (zu groß)
                        result.pop("_image_b64", None)
                        result.pop("_pdf_b64", None)

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(result, ensure_ascii=False),
                        })

                    # Usage akkumulieren
                    if response.usage:
                        usage["input_tokens"] += response.usage.prompt_tokens
                        usage["output_tokens"] += response.usage.completion_tokens
                    continue

                # Kein Tool-Call aber auch nicht gestreamt — Content extrahieren
                full_content = choice.message.content or ""
                if response.usage:
                    usage["input_tokens"] += response.usage.prompt_tokens
                    usage["output_tokens"] += response.usage.completion_tokens

                # Content auf einmal als Tokens senden
                for i in range(0, len(full_content), 4):
                    chunk = full_content[i:i+4]
                    yield f"data: {json.dumps({'token': chunk})}\n\n"
                break

            # Letzte Iteration oder keine Tools → Streaming!
            call_params["stream"] = True
            call_params["stream_options"] = {"include_usage": True}
            stream_response = oai_client.chat.completions.create(**call_params)

            for chunk in stream_response:
                if not chunk.choices:
                    # Usage kommt im letzten Chunk
                    if chunk.usage:
                        usage["input_tokens"] += chunk.usage.prompt_tokens
                        usage["output_tokens"] += chunk.usage.completion_tokens
                    continue

                delta = chunk.choices[0].delta
                if delta and delta.content:
                    full_content += delta.content
                    yield f"data: {json.dumps({'token': delta.content})}\n\n"

            break  # Nach Streaming sind wir fertig

        duration_ms = int((time.time() - start_time) * 1000)

        # Done-Event mit Metadaten
        done_data = {'usage': usage, 'model': model, 'duration_ms': duration_ms, 'tool_calls': all_tool_calls or None, 'tool_results': all_tool_results or None}
        if artifacts:
            done_data['artifacts'] = artifacts
        yield f"event: done\ndata: {json.dumps(done_data)}\n\n"

        log.info("Stream chat %d: Fertig. %d tokens, %d ms, %d artifacts", data.chat_id, usage["output_tokens"], duration_ms, len(artifacts))

        # Callback an Laravel (Antwort persistieren + Battery-Tracking)
        callback_url = data.callback_url or LARAVEL_CALLBACK_URL
        if callback_url:
            callback_data = {
                "chat_id": data.chat_id,
                "content": full_content,
                "usage": usage,
                "model": model,
                "duration_ms": duration_ms,
                "tool_calls": all_tool_calls or None,
                "tool_results": all_tool_results or None,
            }
            if artifacts:
                callback_data["artifacts"] = artifacts
            send_callback(callback_url, callback_data)

    except Exception as e:
        log.error("Stream chat %d: Fehler — %s", data.chat_id, str(e))
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

        # Error-Callback
        callback_url = data.callback_url or LARAVEL_CALLBACK_URL
        if callback_url:
            send_callback(callback_url, {
                "chat_id": data.chat_id,
                "content": f"Fehler bei der Verarbeitung: {str(e)}",
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "model": data.model,
                "duration_ms": int((time.time() - start_time) * 1000),
            })


# ── Chat Pipeline (Sync) ─────────────────────────────────────────────

def chat_pipeline_sync(data: ChatRequest) -> dict:
    """
    Synchrone Version: Führt LLM-Call aus und gibt Ergebnis direkt zurück.
    Kein Callback, kein Background-Task.
    """
    start_time = time.time()
    try:
        # RAG: Dokumente suchen wenn Notebooks verknüpft sind
        rag_context = []
        if data.notebook_ids and data.doc_processor_url:
            log.info("Chat %d [sync]: RAG-Suche in %d Notebooks", data.chat_id, len(data.notebook_ids))
            rag_context = search_documents(
                query=data.message,
                notebook_ids=data.notebook_ids,
                doc_processor_url=data.doc_processor_url,
                doc_processor_secret=data.doc_processor_secret,
            )
            log.info("Chat %d [sync]: %d RAG-Chunks gefunden", data.chat_id, len(rag_context))

        messages = build_messages(data, rag_context=rag_context)
        tools = build_tools(data.tool_definitions, data.skills)
        model = get_model_name(data.model)
        all_tool_calls = []
        all_tool_results = []
        pending_images = []
        pending_pdfs = []
        artifacts = []  # Gesammelte render_artifact Ergebnisse
        max_iterations = 5

        for iteration in range(max_iterations):
            call_params = {
                "model": model,
                "messages": messages,
                "temperature": data.temperature,
                "max_tokens": data.max_tokens,
            }
            if tools and iteration < max_iterations - 1:
                call_params["tools"] = tools
                call_params["tool_choice"] = "auto"

            log.info("Chat %d [sync]: LLM-Call iteration %d (model=%s)", data.chat_id, iteration, model)
            response = oai_client.chat.completions.create(**call_params)
            choice = response.choices[0]

            if not choice.message.tool_calls:
                break

            messages.append(choice.message)
            for tc in choice.message.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments)
                log.info("Chat %d [sync]: Tool-Call → %s(%s)", data.chat_id, tool_name, tool_args)

                all_tool_calls.append({"id": tc.id, "name": tool_name, "arguments": tool_args})

                # render_artifact: Gateway-intercepted
                if tool_name == "render_artifact":
                    artifact_title = tool_args.get("title", "Visualisierung")
                    artifact_html = inject_artifact_css(tool_args.get("html", ""), data.trainer_branding)
                    artifacts.append({"title": artifact_title, "html": artifact_html})
                    log.info("Chat %d [sync]: Artifact '%s' (%d bytes HTML)", data.chat_id, artifact_title, len(artifact_html))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"success": True, "rendered": True, "title": artifact_title}, ensure_ascii=False),
                    })
                    continue

                result = execute_skill(tool_name, tool_args, data.user_id, data.chat_id)
                all_tool_results.append({"tool_call_id": tc.id, "name": tool_name, "result": result})

                # Base64-Bilddaten extrahieren und separat sammeln
                # (wird NICHT an GPT gesendet — zu groß für den Context)
                image_b64 = result.pop("_image_b64", None)
                if image_b64:
                    pending_images.append({
                        "image_b64": image_b64,
                        "prompt": result.get("prompt", ""),
                        "size": result.get("size", "1024x1024"),
                        "quality": result.get("quality", "medium"),
                        "model": result.get("model", "gpt-image-1"),
                    })

                # Base64-PDF-Daten extrahieren und separat sammeln
                pdf_b64 = result.pop("_pdf_b64", None)
                if pdf_b64:
                    pending_pdfs.append({
                        "pdf_b64": pdf_b64,
                        "title": result.get("title", "Dokument"),
                        "size_bytes": result.get("size_bytes", 0),
                    })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

        content = choice.message.content or ""
        usage = {
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
        }
        duration_ms = int((time.time() - start_time) * 1000)

        log.info("Chat %d [sync]: Fertig. %d input, %d output tokens, %d ms",
                 data.chat_id, usage["input_tokens"], usage["output_tokens"], duration_ms)

        resp = {
            "status": "completed",
            "chat_id": data.chat_id,
            "content": content,
            "usage": usage,
            "model": model,
            "duration_ms": duration_ms,
            "tool_calls": all_tool_calls or None,
            "tool_results": all_tool_results or None,
        }

        if pending_images:
            resp["pending_images"] = pending_images
            log.info("Chat %d [sync]: %d Bilder in Response (pending_images)",
                     data.chat_id, len(pending_images))

        if pending_pdfs:
            resp["pending_pdfs"] = pending_pdfs
            log.info("Chat %d [sync]: %d PDFs in Response (pending_pdfs)",
                     data.chat_id, len(pending_pdfs))

        if artifacts:
            resp["artifacts"] = artifacts
            log.info("Chat %d [sync]: %d Artifacts in Response",
                     data.chat_id, len(artifacts))

        return resp

    except Exception as e:
        log.error("Chat %d [sync]: Fehler — %s", data.chat_id, str(e))
        duration_ms = int((time.time() - start_time) * 1000)
        return {
            "status": "error",
            "chat_id": data.chat_id,
            "content": f"Fehler bei der Verarbeitung: {str(e)}",
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "model": data.model,
            "duration_ms": duration_ms,
        }


# ── Chat Pipeline (Background/Async) ────────────────────────────────

def chat_pipeline(data: ChatRequest):
    """
    Orchestriert den kompletten Chat-Flow:
    1. System-Prompt mit Memories anreichern
    2. Tool-Definitionen bauen (falls Skills vorhanden)
    3. LLM-Call (mit Tool-Call-Loop)
    4. Callback an Laravel
    """
    start_time = time.time()
    try:
        # 0. RAG: Dokumente suchen wenn Notebooks verknüpft sind
        rag_context = []
        if data.notebook_ids and data.doc_processor_url:
            log.info("Chat %d: RAG-Suche in %d Notebooks", data.chat_id, len(data.notebook_ids))
            rag_context = search_documents(
                query=data.message,
                notebook_ids=data.notebook_ids,
                doc_processor_url=data.doc_processor_url,
                doc_processor_secret=data.doc_processor_secret,
            )
            log.info("Chat %d: %d RAG-Chunks gefunden", data.chat_id, len(rag_context))

        # 1. Messages aufbauen (mit RAG-Kontext)
        messages = build_messages(data, rag_context=rag_context)

        # 2. Tool-Definitionen (OpenAI Function Calling Format)
        tools = build_tools(data.tool_definitions, data.skills)

        # 3. LLM-Call (mit Tool-Call-Loop)
        model = get_model_name(data.model)
        all_tool_calls = []
        all_tool_results = []
        artifacts = []  # Gesammelte render_artifact Ergebnisse
        max_iterations = 5  # Schutz vor Endlos-Loops

        for iteration in range(max_iterations):
            call_params = {
                "model": model,
                "messages": messages,
                "temperature": data.temperature,
                "max_tokens": data.max_tokens,
            }
            if tools and iteration < max_iterations - 1:
                call_params["tools"] = tools
                call_params["tool_choice"] = "auto"

            log.info("Chat %d: LLM-Call iteration %d (model=%s)", data.chat_id, iteration, model)
            response = oai_client.chat.completions.create(**call_params)
            choice = response.choices[0]

            # Kein Tool-Call → fertig
            if not choice.message.tool_calls:
                break

            # Tool-Calls verarbeiten
            messages.append(choice.message)
            for tc in choice.message.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments)
                log.info("Chat %d: Tool-Call → %s(%s)", data.chat_id, tool_name, tool_args)

                all_tool_calls.append({
                    "id": tc.id,
                    "name": tool_name,
                    "arguments": tool_args,
                })

                # render_artifact: Gateway-intercepted
                if tool_name == "render_artifact":
                    artifact_title = tool_args.get("title", "Visualisierung")
                    artifact_html = inject_artifact_css(tool_args.get("html", ""), data.trainer_branding)
                    artifacts.append({"title": artifact_title, "html": artifact_html})
                    log.info("Chat %d: Artifact '%s' (%d bytes HTML)", data.chat_id, artifact_title, len(artifact_html))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"success": True, "rendered": True, "title": artifact_title}, ensure_ascii=False),
                    })
                    continue

                # Skill bei Laravel ausführen
                result = execute_skill(tool_name, tool_args, data.user_id, data.chat_id)
                all_tool_results.append({
                    "tool_call_id": tc.id,
                    "name": tool_name,
                    "result": result,
                })

                # Base64-Daten entfernen bevor sie an GPT gehen (zu groß)
                result.pop("_image_b64", None)
                result.pop("_pdf_b64", None)

                # Tool-Result als Message für nächste Iteration
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

        # 4. Ergebnis extrahieren
        content = choice.message.content or ""
        usage = {
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
        }
        duration_ms = int((time.time() - start_time) * 1000)

        log.info(
            "Chat %d: Fertig. %d input, %d output tokens, %d ms",
            data.chat_id, usage["input_tokens"], usage["output_tokens"], duration_ms
        )

        # 5. Callback an Laravel
        callback_url = data.callback_url or LARAVEL_CALLBACK_URL
        if callback_url:
            callback_data = {
                "chat_id": data.chat_id,
                "content": content,
                "usage": usage,
                "model": model,
                "duration_ms": duration_ms,
                "tool_calls": all_tool_calls or None,
                "tool_results": all_tool_results or None,
            }
            if artifacts:
                callback_data["artifacts"] = artifacts
            send_callback(callback_url, callback_data)

    except Exception as e:
        log.error("Chat %d: Fehler — %s", data.chat_id, str(e))
        duration_ms = int((time.time() - start_time) * 1000)
        callback_url = data.callback_url or LARAVEL_CALLBACK_URL
        if callback_url:
            send_callback(callback_url, {
                "chat_id": data.chat_id,
                "content": f"Fehler bei der Verarbeitung: {str(e)}",
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "model": data.model,
                "duration_ms": duration_ms,
            })


# ── Helper: RAG — Dokumente suchen ───────────────────────────────────

def search_documents(query: str, notebook_ids: list, doc_processor_url: str, doc_processor_secret: str, top_k: int = 5) -> list:
    """
    Sucht in allen verknüpften Notebooks nach relevanten Dokumenten-Chunks.
    Gibt eine Liste von {text, filename, score} zurück.
    """
    if not notebook_ids or not doc_processor_url:
        return []

    all_results = []
    try:
        with httpx.Client(timeout=15) as client:
            for nb_id in notebook_ids:
                resp = client.post(
                    f"{doc_processor_url}/search",
                    data={
                        "query": query,
                        "notebook_id": nb_id,
                        "top_k": top_k,
                    },
                    headers={
                        "Authorization": f"Bearer {doc_processor_secret}",
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                )
                if resp.status_code == 200:
                    results = resp.json().get("results", [])
                    all_results.extend(results)
                else:
                    log.warning("RAG search for %s failed: %d", nb_id, resp.status_code)
    except Exception as e:
        log.error("RAG search error: %s", str(e))

    # Nach Score sortieren und Top-K zurückgeben
    all_results.sort(key=lambda r: r.get("score", 0), reverse=True)
    return all_results[:top_k]


# ── Helper: Messages bauen ───────────────────────────────────────────

def build_messages(data: ChatRequest, rag_context: list = None) -> list:
    """System-Prompt mit Memories + RAG-Kontext anreichern + Conversation History."""
    system_parts = []

    # System-Prompt des Assistenten
    if data.system_prompt:
        system_parts.append(data.system_prompt)

    # Trainer Memories injizieren
    if data.trainer_memories:
        memory_text = "\n".join(
            f"- [{m['category']}] {m['content']}" for m in data.trainer_memories
        )
        system_parts.append(
            f"\n\n## Bekannte Informationen über den Trainer:\n{memory_text}"
        )

    # RAG-Kontext injizieren
    if rag_context:
        context_parts = []
        for i, chunk in enumerate(rag_context, 1):
            source = chunk.get("filename", "Unbekannt")
            text = chunk.get("text", "")
            score = chunk.get("score", 0)
            context_parts.append(f"[Quelle {i}: {source} (Relevanz: {score:.2f})]\n{text}")
        context_block = "\n\n".join(context_parts)
        system_parts.append(
            f"\n\n## Relevanter Kontext aus Dokumenten:\n"
            f"Nutze die folgenden Informationen aus den Wissensdatenbanken des Trainers, "
            f"um die Frage zu beantworten. Verweise auf die Quellen wenn möglich.\n\n"
            f"{context_block}"
        )

    # Trainer-Branding injizieren
    if data.trainer_branding:
        b = data.trainer_branding
        branding_lines = []
        if b.get("company_name"):
            branding_lines.append(f"- Firmenname: {b['company_name']}")
        if b.get("primary_color"):
            branding_lines.append(f"- Primärfarbe: {b['primary_color']}")
        if b.get("secondary_color"):
            branding_lines.append(f"- Sekundärfarbe: {b['secondary_color']}")
        if b.get("accent_color"):
            branding_lines.append(f"- Akzentfarbe: {b['accent_color']}")
        if b.get("font_family"):
            branding_lines.append(f"- Schriftart: {b['font_family']}")
        if b.get("logo_base64") and b.get("logo_mime_type"):
            branding_lines.append(f"- Logo: data:{b['logo_mime_type']};base64,{b['logo_base64']}")
            branding_lines.append("  (Verwende diesen Data-URI direkt als <img src=\"...\"> in HTML/PDF-Artefakten)")
        if branding_lines:
            system_parts.append(
                "\n\n## Trainer-Branding\n"
                "Verwende diese Branding-Informationen wenn du PDFs, HTML-Artefakte oder Dokumente erstellst:\n"
                + "\n".join(branding_lines)
            )

    # Artifact-System Anweisungen immer anhängen
    system_parts.append(ARTIFACT_SYSTEM_PROMPT)

    messages = []
    if system_parts:
        messages.append({"role": "system", "content": "\n".join(system_parts)})

    # Conversation History (bereits als role/content dicts)
    messages.extend(data.conversation_history)

    # Aktuelle Nachricht (mit Vision-Bildern wenn vorhanden)
    if data.images:
        # OpenAI Vision Format: content ist ein Array aus text + image_url Objekten
        content_parts = [{"type": "text", "text": data.message}]
        for img in data.images:
            media_type = img.get("media_type", "image/png")
            b64_data = img.get("data", "")
            if b64_data:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{b64_data}",
                        "detail": "high",
                    }
                })
        messages.append({"role": "user", "content": content_parts})
        log.info("Chat %d: Vision-Nachricht mit %d Bildern", data.chat_id, len(data.images))
    else:
        messages.append({"role": "user", "content": data.message})

    return messages


# ── Helper: Tool-Definitionen ────────────────────────────────────────

def build_tools(tool_definitions: list = None, skill_slugs: list = None, include_artifact: bool = True) -> list:
    """
    Baut OpenAI Function Calling Tool-Definitionen.
    Bevorzugt die vom Laravel-Payload mitgelieferten Definitionen (tool_definitions).
    Fallback: Holt sie per HTTP von LARAVEL_SKILL_URL (für Rückwärtskompatibilität).
    render_artifact wird immer automatisch angehängt.
    """
    tools = []

    # Option A: Tool-Definitionen direkt aus dem Payload (bevorzugt)
    if tool_definitions:
        log.info("build_tools: %d Tool-Definitionen aus Payload (skills: %s)",
                 len(tool_definitions),
                 [t.get("function", {}).get("name") for t in tool_definitions])
        tools = list(tool_definitions)
    else:
        # Fallback: Von Laravel per HTTP holen (nur wenn LARAVEL_SKILL_URL gesetzt)
        log.info("build_tools: Keine tool_definitions im Payload, versuche LARAVEL_SKILL_URL")
        try:
            skill_url = LARAVEL_SKILL_URL
            if not skill_url:
                log.info("build_tools: LARAVEL_SKILL_URL ist leer — nur render_artifact verfügbar")
            else:
                slugs = skill_slugs or []
                log.info("build_tools: Lade Skill-Definitionen von %s (skills: %s)", skill_url, slugs)
                with httpx.Client(timeout=10) as client:
                    resp = client.get(
                        skill_url,
                        headers={"Authorization": f"Bearer {AI_GATEWAY_SECRET}"},
                        params={"skills": ",".join(slugs)},
                    )
                    if resp.status_code != 200:
                        log.warning("Skill-Definitionen laden fehlgeschlagen: %d — %s", resp.status_code, resp.text[:500])
                    else:
                        skills_data = resp.json()
                        for skill in skills_data.get("skills", []):
                            tools.append({
                                "type": "function",
                                "function": {
                                    "name": skill["name"],
                                    "description": skill["description"],
                                    "parameters": skill.get("parameters", {"type": "object", "properties": {}}),
                                },
                            })
                        log.info("build_tools: %d Tools via HTTP geladen", len(tools))

        except Exception as e:
            log.error("Skill-Definitionen laden Fehler: %s", str(e))

    # render_artifact immer anhängen (universeller HTML-Visualizer)
    if include_artifact:
        has_artifact = any(
            t.get("function", {}).get("name") == "render_artifact" for t in tools
        )
        if not has_artifact:
            tools.append(RENDER_ARTIFACT_TOOL)
            log.info("build_tools: render_artifact Tool angehängt")

    return tools


# ── Helper: Skill bei Laravel ausführen ──────────────────────────────

def execute_skill(skill_name: str, params: dict, user_id: int, chat_id: int = 0) -> dict:
    """
    Sendet Tool-Call an Laravel /api/ai/skills.
    Einige Skills (z.B. image_generate) werden direkt im Gateway ausgeführt.
    """
    # Gateway-intercepted Skills
    if skill_name == "image_generate":
        return generate_image(params, user_id, chat_id)
    if skill_name == "pdf_generate":
        return generate_pdf(params, user_id, chat_id)
    if skill_name == "delegate_to_assistant":
        return _delegate_to_assistant(params, user_id, chat_id)

    try:
        skill_url = LARAVEL_SKILL_URL
        if not skill_url:
            return {"error": "Keine LARAVEL_SKILL_URL konfiguriert"}

        with httpx.Client(timeout=30) as client:
            resp = client.post(
                skill_url,
                json={
                    "skill": skill_name,
                    "params": params,
                    "user_id": user_id,
                },
                headers={"Authorization": f"Bearer {AI_GATEWAY_SECRET}"},
            )
            if resp.status_code == 200:
                result = resp.json()
                # Artifact-Hints für strukturierte Skill-Ergebnisse
                result = _inject_artifact_hint(skill_name, result)
                return result
            else:
                log.warning("Skill %s fehlgeschlagen: %d — %s", skill_name, resp.status_code, resp.text)
                return {"error": f"Skill-Fehler: HTTP {resp.status_code}"}

    except Exception as e:
        log.error("Skill %s Fehler: %s", skill_name, str(e))
        return {"error": str(e)}


# Artifact-Hints für Skill-Ergebnisse — GPT bekommt Hinweise, wann ein Artifact sinnvoll ist
SKILL_ARTIFACT_HINTS = {
    "crm_search": "Stelle diese Ergebnisse als schöne interaktive HTML-Tabelle oder Karten dar. Verwende render_artifact. Links zu Kontakten: /crm/contacts/{id}, zu Deals: /crm/deals/{id}. Links sollen target=\"_top\" haben.",
    "crm_update_deal": "Zeige die Deal-Änderungen als übersichtliche Karte mit render_artifact. Link zum Deal: /crm/deals/{deal_id}.",
    "campaign_analyze": "Visualisiere die Kampagnen-Statistiken als KPI-Dashboard mit render_artifact (Balken, Prozent-Kreise o.ä.).",
}


def _inject_artifact_hint(skill_name: str, result: dict) -> dict:
    """Fügt _artifact_hint in Skill-Ergebnisse ein, damit GPT weiß, dass ein Artifact sinnvoll wäre."""
    hint = SKILL_ARTIFACT_HINTS.get(skill_name)
    if hint and "error" not in result:
        result["_artifact_hint"] = hint
    return result


# ── Helper: Bildgenerierung (Gateway-intercepted) ────────────────────

def generate_image(params: dict, user_id: int, chat_id: int) -> dict:
    """
    Generiert ein Bild über OpenAI gpt-image-1.
    Gibt die Base64-Bilddaten im Result zurück (unter '_image_b64').
    Die sync/stream pipeline sammelt diese und gibt sie als 'pending_images'
    in der Response an Laravel zurück, damit Laravel das Bild lokal speichert.
    """
    prompt = params.get("prompt", "")
    size = params.get("size", "1024x1024")
    quality = params.get("quality", "medium")
    background = params.get("background", "auto")

    if not prompt:
        return {"error": "Kein Prompt angegeben"}

    try:
        log.info("Bildgenerierung: prompt='%s', size=%s, quality=%s", prompt[:80], size, quality)

        # OpenAI Images API aufrufen (gpt-image-1 liefert immer Base64)
        image_response = oai_client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size=size,
            quality=quality,
            background=background,
            output_format="png",
            n=1,
        )

        # Base64-Daten aus der Response extrahieren
        image_b64 = image_response.data[0].b64_json

        if not image_b64:
            return {"error": "Keine Bilddaten von OpenAI erhalten"}

        log.info("Bild generiert: %d bytes Base64-Daten", len(image_b64))

        # Base64 in _image_b64 zurückgeben (wird von der Pipeline extrahiert,
        # NICHT an GPT gesendet — GPT bekommt nur die Metadaten)
        return {
            "status": "success",
            "_image_b64": image_b64,
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "model": "gpt-image-1",
            "message": "Bild wurde erfolgreich generiert.",
        }

    except openai.BadRequestError as e:
        log.warning("Bildgenerierung abgelehnt: %s", str(e))
        return {"error": f"Bildgenerierung wurde abgelehnt (Content Policy): {str(e)}"}
    except Exception as e:
        log.error("Bildgenerierung Fehler: %s", str(e))
        return {"error": f"Bildgenerierung fehlgeschlagen: {str(e)}"}


# ── Helper: PDF-Generierung (Gateway-intercepted) ───────────────────

def generate_pdf(params: dict, user_id: int, chat_id: int) -> dict:
    """
    Generiert ein PDF aus HTML via WeasyPrint.
    Gibt die Base64-PDF-Daten im Result zurück (unter '_pdf_b64').
    Die sync/stream pipeline sammelt diese und gibt sie als 'pending_pdfs'
    in der Response an Laravel zurück, damit Laravel das PDF lokal speichert.
    """
    title = params.get("title", "Dokument")
    html_content = params.get("html_content", "")

    if not html_content:
        return {"error": "Kein HTML-Content angegeben"}

    try:
        import weasyprint

        log.info("PDF-Generierung: title='%s', html_length=%d", title[:80], len(html_content))

        # WeasyPrint: HTML → PDF (A4 ist Standard)
        html_doc = weasyprint.HTML(string=html_content)
        pdf_bytes = html_doc.write_pdf()

        if not pdf_bytes:
            return {"error": "PDF-Generierung fehlgeschlagen — keine Daten"}

        # PDF → Base64
        pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")

        log.info("PDF generiert: %d bytes (%d bytes Base64)", len(pdf_bytes), len(pdf_b64))

        return {
            "status": "success",
            "_pdf_b64": pdf_b64,
            "title": title,
            "size_bytes": len(pdf_bytes),
            "message": f"PDF '{title}' wurde erfolgreich generiert.",
        }

    except Exception as e:
        log.error("PDF-Generierung Fehler: %s", str(e))
        return {"error": f"PDF-Generierung fehlgeschlagen: {str(e)}"}


# ── Helper: Callback an Laravel ──────────────────────────────────────

def send_callback(url: str, payload: dict):
    """Sendet Ergebnis per POST an Laravel AiCallbackController."""
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {AI_GATEWAY_SECRET}"},
            )
            if resp.status_code != 200:
                log.warning("Callback fehlgeschlagen: %d — %s", resp.status_code, resp.text)
            else:
                log.info("Callback gesendet: chat_id=%d", payload.get("chat_id", 0))
    except Exception as e:
        log.error("Callback Fehler: %s", str(e))


# ── Background Task Endpoint ─────────────────────────────────────────

class TaskExecuteRequest(BaseModel):
    job_id: int
    user_id: int
    chat_id: int
    assistant_id: int
    task_description: str
    context: list = []              # Bisheriger Kontext / Schritt-History
    system_prompt: str = ""
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096
    max_runtime_minutes: int = 30
    confirmation_skills: list = []  # ["crm_update_deal", "crm_create_contact"]
    skills: list = []               # Alle aktiven Skill-Slugs
    tool_definitions: list = []     # OpenAI Tool-Definitionen
    trainer_memories: list = []
    notebook_ids: list = []
    doc_processor_url: str = ""
    doc_processor_secret: str = ""
    callback_url: str = ""          # Laravel callback URL
    execution_seconds: int = 0      # Bisherige kumulierte Laufzeit (bei Resume)
    laravel_base_url: str = ""      # z.B. https://trainer.example.com/api/ai
    trainer_branding: dict = None   # Trainer-Branding (Farben, Logo als Base64)


@app.post("/task/execute")
async def execute_task(request: Request, bg: BackgroundTasks):
    """Startet autonome Background-Task-Verarbeitung."""
    verify_auth(request)
    body = await request.json()
    data = TaskExecuteRequest(**body)
    bg.add_task(task_pipeline, data)
    return {"status": "accepted", "job_id": data.job_id}


def task_pipeline(data: TaskExecuteRequest):
    """
    Autonome Schleife für Background Tasks:
    1. Battery-Check vor jedem Schritt
    2. GPT plant nächsten Schritt (LLM-Call mit Tools)
    3. Falls Tool-Call: Prüfe ob Confirmation nötig → Pause oder Ausführen
    4. Ausführungszeit prüfen
    5. Wiederholen oder fertig
    """
    start_time = time.time()
    execution_seconds = data.execution_seconds  # Resume-Wert
    step_count = 0
    max_steps = 50  # Sicherheitslimit
    max_runtime = data.max_runtime_minutes * 60
    step_history = list(data.context)  # Kopie
    all_tool_calls = []
    all_tool_results = []
    total_usage = {"input_tokens": 0, "output_tokens": 0}

    model = get_model_name(data.model)
    base_url = data.laravel_base_url or LARAVEL_CALLBACK_URL.rsplit("/", 1)[0] if LARAVEL_CALLBACK_URL else ""

    log.info("Task %d: Starte autonome Schleife (max %d min, %d bisherige Sek.)",
             data.job_id, data.max_runtime_minutes, execution_seconds)

    try:
        # RAG-Kontext laden (einmalig am Anfang)
        rag_context = []
        if data.notebook_ids and data.doc_processor_url:
            rag_context = search_documents(
                query=data.task_description,
                notebook_ids=data.notebook_ids,
                doc_processor_url=data.doc_processor_url,
                doc_processor_secret=data.doc_processor_secret,
            )
            log.info("Task %d: %d RAG-Chunks geladen", data.job_id, len(rag_context))

        # System-Prompt für autonomen Modus erweitern
        system_prompt = _build_task_system_prompt(data, rag_context)

        # Tools vorbereiten
        tools = build_tools(data.tool_definitions, data.skills)

        # Messages für die Schleife
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Aufgabe: {data.task_description}"},
        ]
        # Bisherige Schritte als Kontext einfügen
        for ctx in step_history:
            if isinstance(ctx, dict) and "role" in ctx:
                messages.append(ctx)
            else:
                messages.append({"role": "assistant", "content": str(ctx)})

        while step_count < max_steps:
            step_start = time.time()
            step_count += 1
            log.info("Task %d: Schritt %d (%.0f Sek. bisherig)", data.job_id, step_count, execution_seconds)

            # ① Battery-Check
            battery_ok = _check_battery(data.user_id, base_url)
            if not battery_ok:
                _task_callback_failed(data, execution_seconds, "Monatliches AI-Limit erreicht", base_url)
                return

            # ② GPT: Nächster Schritt
            response = oai_client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                temperature=data.temperature,
                max_tokens=data.max_tokens,
            )

            choice = response.choices[0]
            usage = response.usage
            if usage:
                total_usage["input_tokens"] += usage.prompt_tokens
                total_usage["output_tokens"] += usage.completion_tokens

            # ③ Kein Tool-Call → GPT ist fertig
            if not choice.message.tool_calls:
                content = choice.message.content or ""
                step_duration = time.time() - step_start
                execution_seconds += step_duration

                log.info("Task %d: Fertig nach %d Schritten (%.0f Sek.)", data.job_id, step_count, execution_seconds)
                _task_callback_complete(data, content, execution_seconds, total_usage, all_tool_calls, all_tool_results, base_url)
                return

            # ④ Tool-Calls verarbeiten
            messages.append(choice.message)

            for tc in choice.message.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments)
                log.info("Task %d: Tool-Call → %s", data.job_id, tool_name)

                # render_artifact: Gateway-intercepted (kein Laravel-Call)
                if tool_name == "render_artifact":
                    artifact_title = tool_args.get("title", "Visualisierung")
                    artifact_html = inject_artifact_css(tool_args.get("html", ""), data.trainer_branding)
                    log.info("Task %d: Artifact '%s' (%d bytes HTML)", data.job_id, artifact_title, len(artifact_html))
                    all_tool_calls.append({"id": tc.id, "name": tool_name, "arguments": tool_args})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"success": True, "rendered": True, "title": artifact_title}, ensure_ascii=False),
                    })
                    continue

                # Confirmation nötig?
                if tool_name in data.confirmation_skills:
                    # Timer pausieren: bisherige Laufzeit speichern
                    step_duration = time.time() - step_start
                    execution_seconds += step_duration

                    description = _build_confirmation_description(tool_name, tool_args)
                    log.info("Task %d: Confirmation nötig für %s — pausiert", data.job_id, tool_name)

                    _task_request_confirmation(data, tool_name, tool_args, description, execution_seconds, base_url)
                    return  # Pausiert — wird nach Bestätigung mit POST /task/execute neu gestartet

                # Skill ausführen
                all_tool_calls.append({"id": tc.id, "name": tool_name, "arguments": tool_args})
                result = execute_skill(tool_name, tool_args, data.user_id, data.chat_id)
                all_tool_results.append({"tool_call_id": tc.id, "name": tool_name, "result": result})

                # Ergebnis für GPT aufbereiten (ohne große Binärdaten)
                result_for_gpt = {k: v for k, v in result.items() if k not in ("_image_b64", "_pdf_b64")}
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result_for_gpt, ensure_ascii=False),
                })

            # ⑤ Ausführungszeit prüfen
            step_duration = time.time() - step_start
            execution_seconds += step_duration

            if execution_seconds >= max_runtime:
                log.warning("Task %d: Max Ausführungszeit erreicht (%.0f Sek.)", data.job_id, execution_seconds)
                _task_callback_failed(data, execution_seconds, "Maximale Ausführungszeit erreicht", base_url)
                return

            # ⑥ Execution time an Laravel melden (für Tracking)
            _task_update_execution(data.job_id, execution_seconds, base_url)

        # Sicherheitslimit erreicht
        log.warning("Task %d: Max Schritte (%d) erreicht", data.job_id, max_steps)
        _task_callback_failed(data, execution_seconds, f"Maximale Schrittanzahl ({max_steps}) erreicht", base_url)

    except Exception as e:
        log.error("Task %d: Fehler — %s", data.job_id, str(e))
        elapsed = time.time() - start_time
        _task_callback_failed(data, execution_seconds + elapsed, f"Fehler: {str(e)}", base_url)



# ── Delegation / Orchestrierung (Phase 8.4) ──────────────────────────

DELEGATION_TIMEOUT = 60  # Sekunden

def _delegate_to_assistant(params: dict, user_id: int, chat_id: int) -> dict:
    """
    Nested Completion: Delegiert eine Teilaufgabe an einen anderen Assistenten.
    1. Access-Check bei Laravel
    2. Config des Ziel-Assistenten laden (prompt, skills, notebooks, memories)
    3. Eigener LLM-Call mit Ziel-Assistent Profil
    4. Ergebnis zurück an aufrufenden Assistenten
    """
    target_slug = params.get("target_assistant", "")
    task = params.get("task", "")
    context_data = params.get("context_data", {})

    if not target_slug or not task:
        return {"error": "target_assistant und task sind Pflichtfelder."}

    base_url = LARAVEL_CALLBACK_URL.rsplit("/", 1)[0] if LARAVEL_CALLBACK_URL else ""
    if not base_url:
        return {"error": "Keine Laravel-URL konfiguriert für Delegation."}

    log.info("Delegation: %s → %s (User %d, Chat %d)", "source", target_slug, user_id, chat_id)

    # ① Access-Check
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.post(
                f"{base_url}/check-assistant-access",
                json={"user_id": user_id, "assistant_slug": target_slug},
                headers={"Authorization": f"Bearer {AI_GATEWAY_SECRET}"},
            )
            if resp.status_code != 200:
                return {"error": f"Access-Check fehlgeschlagen: HTTP {resp.status_code}"}
            access = resp.json()
            if not access.get("has_access"):
                reason = access.get("reason", "Kein Zugriff")
                return {"error": reason, "assistant_slug": target_slug}
    except Exception as e:
        log.error("Delegation Access-Check Fehler: %s", str(e))
        return {"error": f"Access-Check Fehler: {str(e)}"}

    log.info("Delegation: Access OK für %s (ID %d)", target_slug, access.get("assistant_id", 0))

    # ② Config laden
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.post(
                f"{base_url}/delegate-config",
                json={"user_id": user_id, "assistant_slug": target_slug},
                headers={"Authorization": f"Bearer {AI_GATEWAY_SECRET}"},
            )
            if resp.status_code != 200:
                return {"error": f"Config-Laden fehlgeschlagen: HTTP {resp.status_code}"}
            config = resp.json()
    except Exception as e:
        log.error("Delegation Config Fehler: %s", str(e))
        return {"error": f"Config-Laden Fehler: {str(e)}"}

    log.info("Delegation: Config geladen für %s — %d Skills, %d Notebooks",
             target_slug, len(config.get("skills", [])), len(config.get("notebook_ids", [])))

    # ③ System-Prompt für Ziel-Assistent bauen
    system_parts = []
    if config.get("system_prompt"):
        system_parts.append(config["system_prompt"])

    system_parts.append("""
## Delegierte Aufgabe
Du wurdest von einem anderen Assistenten beauftragt, eine spezifische Teilaufgabe zu erledigen.
Konzentriere dich NUR auf die beschriebene Aufgabe.
Gib ein klares, strukturiertes Ergebnis zurück.
Falls du Dateien erstellst (PDFs, Bilder), gib die URLs im Ergebnis an.
""")

    # Trainer-Memories einfügen
    memories = config.get("trainer_memories", [])
    if memories:
        mem_text = "\n".join(f"- [{m['category']}] {m['content']}" for m in memories)
        system_parts.append(f"\n## Bekannte Informationen über den Trainer:\n{mem_text}")

    # RAG: Dokumente suchen für den Ziel-Assistenten
    rag_context = []
    notebook_ids = config.get("notebook_ids", [])
    doc_processor_url = config.get("doc_processor_url", "")
    doc_processor_secret = config.get("doc_processor_secret", "")

    if notebook_ids and doc_processor_url:
        try:
            rag_context = search_documents(
                query=task,
                notebook_ids=notebook_ids,
                doc_processor_url=doc_processor_url,
                doc_processor_secret=doc_processor_secret,
            )
            log.info("Delegation: %d RAG-Chunks für %s", len(rag_context), target_slug)
        except Exception as e:
            log.warning("Delegation RAG-Suche fehlgeschlagen: %s", str(e))

    if rag_context:
        rag_text = "\n\n".join(
            f"[Quelle: {c.get('source', 'Unbekannt')}]\n{c.get('content', '')}"
            for c in rag_context
        )
        system_parts.append(f"\n## Relevantes Wissen:\n{rag_text}")

    system_prompt = "\n\n".join(system_parts)

    # ④ Kontext-Nachricht für den Ziel-Assistenten
    user_message = f"Aufgabe: {task}"
    if context_data:
        context_str = json.dumps(context_data, ensure_ascii=False, indent=2)
        user_message += f"\n\nKontextdaten:\n```json\n{context_str}\n```"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    # Tools für den Ziel-Assistenten (ohne delegate_to_assistant — Max 1 Tiefe!)
    tools = config.get("tool_definitions", [])
    model = get_model_name(config.get("model", "gpt-4o"))

    # ⑤ LLM-Call mit Timeout und Tool-Loop
    try:
        max_iterations = 5
        all_tool_calls = []

        for iteration in range(max_iterations):
            call_params = {
                "model": model,
                "messages": messages,
                "temperature": config.get("temperature", 0.7),
                "max_tokens": config.get("max_tokens", 4096),
                "timeout": DELEGATION_TIMEOUT,
            }
            if tools and iteration < max_iterations - 1:
                call_params["tools"] = tools
                call_params["tool_choice"] = "auto"

            log.info("Delegation %s: LLM-Call Iteration %d", target_slug, iteration)
            response = oai_client.chat.completions.create(**call_params)
            choice = response.choices[0]

            # Kein Tool-Call → Ziel-Assistent ist fertig
            if not choice.message.tool_calls:
                break

            # Tool-Calls verarbeiten
            messages.append(choice.message)
            for tc in choice.message.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments)
                log.info("Delegation %s: Tool-Call → %s", target_slug, tool_name)

                # Sicherheit: delegate_to_assistant darf NICHT rekursiv aufgerufen werden
                if tool_name == "delegate_to_assistant":
                    result = {"error": "Delegation darf nicht rekursiv aufgerufen werden (max. 1 Tiefe)."}
                else:
                    result = execute_skill(tool_name, tool_args, user_id, chat_id)

                all_tool_calls.append({"name": tool_name, "arguments": tool_args})

                result_for_gpt = {k: v for k, v in result.items() if k not in ("_image_b64", "_pdf_b64")}
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result_for_gpt, ensure_ascii=False),
                })

        content = choice.message.content or ""
        log.info("Delegation %s: Fertig — %d Zeichen Ergebnis, %d Tool-Calls",
                 target_slug, len(content), len(all_tool_calls))

        return {
            "status": "success",
            "result": content,
            "delegated_to": target_slug,
            "assistant_name": config.get("assistant_name", target_slug),
            "tool_calls_count": len(all_tool_calls),
        }

    except Exception as e:
        log.error("Delegation %s LLM-Fehler: %s", target_slug, str(e))
        return {
            "error": f"Delegation an {target_slug} fehlgeschlagen: {str(e)}",
            "delegated_to": target_slug,
        }


# ── Background Task Helper ──────────────────────────────────────────

def _build_task_system_prompt(data: TaskExecuteRequest, rag_context: list) -> str:
    """System-Prompt für autonomen Modus zusammenbauen."""
    parts = []

    if data.system_prompt:
        parts.append(data.system_prompt)

    parts.append("""
## Autonomer Modus
Du arbeitest im Hintergrund an einer Aufgabe. Plane Schritt für Schritt:
1. Überlege was als nächstes getan werden muss
2. Nutze die verfügbaren Tools um die Aufgabe zu erledigen
3. Wenn du fertig bist, gib eine Zusammenfassung des Ergebnisses
4. Sei effizient — minimiere die Anzahl der Schritte
""")

    # Trainer Memories
    if data.trainer_memories:
        memory_text = "\n".join(
            f"- [{m['category']}] {m['content']}" for m in data.trainer_memories
        )
        parts.append(f"\n## Bekannte Informationen über den Trainer:\n{memory_text}")

    # RAG-Kontext
    if rag_context:
        rag_text = "\n\n".join(
            f"[Quelle: {c.get('source', 'Unbekannt')}]\n{c.get('content', '')}"
            for c in rag_context
        )
        parts.append(f"\n## Relevantes Wissen:\n{rag_text}")

    # Trainer-Branding
    if data.trainer_branding:
        b = data.trainer_branding
        branding_lines = []
        if b.get("company_name"):
            branding_lines.append(f"- Firmenname: {b['company_name']}")
        if b.get("primary_color"):
            branding_lines.append(f"- Primärfarbe: {b['primary_color']}")
        if b.get("secondary_color"):
            branding_lines.append(f"- Sekundärfarbe: {b['secondary_color']}")
        if b.get("accent_color"):
            branding_lines.append(f"- Akzentfarbe: {b['accent_color']}")
        if b.get("font_family"):
            branding_lines.append(f"- Schriftart: {b['font_family']}")
        if b.get("logo_base64") and b.get("logo_mime_type"):
            branding_lines.append(f"- Logo: data:{b['logo_mime_type']};base64,{b['logo_base64']}")
            branding_lines.append("  (Verwende diesen Data-URI direkt als <img src=\"...\"> in HTML/PDF-Artefakten)")
        if branding_lines:
            parts.append(
                "\n## Trainer-Branding\n"
                "Verwende diese Branding-Informationen wenn du PDFs, HTML-Artefakte oder Dokumente erstellst:\n"
                + "\n".join(branding_lines)
            )

    return "\n\n".join(parts)


def _check_battery(user_id: int, base_url: str) -> bool:
    """Battery-Check bei Laravel."""
    try:
        url = f"{base_url}/battery-check"
        with httpx.Client(timeout=10) as client:
            resp = client.post(
                url,
                json={"user_id": user_id},
                headers={"Authorization": f"Bearer {AI_GATEWAY_SECRET}"},
            )
            if resp.status_code == 200:
                result = resp.json()
                return result.get("has_capacity", False)
            log.warning("Battery-Check fehlgeschlagen: %d", resp.status_code)
            return False
    except Exception as e:
        log.error("Battery-Check Fehler: %s", str(e))
        return False  # Bei Fehler sicherheitshalber stoppen


def _build_confirmation_description(tool_name: str, tool_args: dict) -> str:
    """Menschenlesbare Beschreibung einer Skill-Aktion."""
    descriptions = {
        "crm_create_contact": "Neuen Kontakt im CRM anlegen",
        "crm_update_deal": "Deal im CRM aktualisieren",
        "crm_create_followup": "Follow-Up im CRM erstellen",
    }
    base = descriptions.get(tool_name, f"Skill '{tool_name}' ausführen")
    details = ", ".join(f"{k}={v}" for k, v in tool_args.items() if not str(v).startswith("base64"))
    return f"{base}: {details}" if details else base


def _task_request_confirmation(data: TaskExecuteRequest, skill: str, params: dict, description: str, execution_seconds: float, base_url: str):
    """Sendet Confirmation-Request an Laravel → Job wird pausiert."""
    try:
        url = f"{base_url}/task/confirm-request"
        with httpx.Client(timeout=15) as client:
            resp = client.post(
                url,
                json={
                    "job_id": data.job_id,
                    "skill": skill,
                    "params": params,
                    "description": description,
                    "execution_seconds": int(execution_seconds),
                },
                headers={"Authorization": f"Bearer {AI_GATEWAY_SECRET}"},
            )
            if resp.status_code != 200:
                log.warning("Confirm-Request fehlgeschlagen: %d — %s", resp.status_code, resp.text[:500])
    except Exception as e:
        log.error("Confirm-Request Fehler: %s", str(e))


def _task_update_execution(job_id: int, execution_seconds: float, base_url: str):
    """Execution-Time Update an Laravel."""
    try:
        url = f"{base_url}/task/update-execution"
        with httpx.Client(timeout=10) as client:
            client.post(
                url,
                json={"job_id": job_id, "execution_seconds": int(execution_seconds)},
                headers={"Authorization": f"Bearer {AI_GATEWAY_SECRET}"},
            )
    except Exception as e:
        log.error("Execution-Update Fehler: %s", str(e))


def _task_callback_complete(data: TaskExecuteRequest, content: str, execution_seconds: float, usage: dict, tool_calls: list, tool_results: list, base_url: str):
    """Task erfolgreich abgeschlossen → Callback an Laravel."""
    try:
        url = f"{base_url}/task/complete"
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                url,
                json={
                    "job_id": data.job_id,
                    "chat_id": data.chat_id,
                    "content": content,
                    "execution_seconds": int(execution_seconds),
                    "usage": usage,
                    "tool_calls": tool_calls or None,
                    "tool_results": tool_results or None,
                },
                headers={"Authorization": f"Bearer {AI_GATEWAY_SECRET}"},
            )
            if resp.status_code != 200:
                log.warning("Task-Complete Callback fehlgeschlagen: %d — %s", resp.status_code, resp.text[:500])
            else:
                log.info("Task %d: Complete-Callback gesendet", data.job_id)
    except Exception as e:
        log.error("Task-Complete Callback Fehler: %s", str(e))


def _task_callback_failed(data: TaskExecuteRequest, execution_seconds: float, reason: str, base_url: str):
    """Task fehlgeschlagen → Callback an Laravel."""
    try:
        url = f"{base_url}/task/failed"
        with httpx.Client(timeout=15) as client:
            resp = client.post(
                url,
                json={
                    "job_id": data.job_id,
                    "chat_id": data.chat_id,
                    "reason": reason,
                    "execution_seconds": int(execution_seconds),
                },
                headers={"Authorization": f"Bearer {AI_GATEWAY_SECRET}"},
            )
            if resp.status_code != 200:
                log.warning("Task-Failed Callback fehlgeschlagen: %d — %s", resp.status_code, resp.text[:500])
            else:
                log.info("Task %d: Failed-Callback gesendet (%s)", data.job_id, reason)
    except Exception as e:
        log.error("Task-Failed Callback Fehler: %s", str(e))


# ── Memory-Extraktion Endpoint ───────────────────────────────────────

class MemoryExtractionRequest(BaseModel):
    user_id: int
    chat_id: int
    messages: list = []
    existing_memories: list = []
    callback_url: str = ""


@app.post("/extract-memories")
async def extract_memories(request: Request, bg: BackgroundTasks):
    verify_auth(request)
    body = await request.json()
    data = MemoryExtractionRequest(**body)
    bg.add_task(memory_extraction_pipeline, data)
    return {"status": "accepted", "chat_id": data.chat_id}


def memory_extraction_pipeline(data: MemoryExtractionRequest):
    """Extrahiert lernbare Fakten aus Chat-Verlauf via LLM."""
    try:
        model = get_model_name("gpt-4o-mini")

        existing_str = "\n".join(f"- {m}" for m in data.existing_memories) if data.existing_memories else "Keine"

        conversation = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in data.messages
        )

        messages = [
            {"role": "system", "content": (
                "Du bist ein Memory-Extraktions-Assistent. Analysiere den Chat-Verlauf "
                "und extrahiere wichtige Fakten über den Trainer, die für zukünftige "
                "Gespräche nützlich sein könnten.\n\n"
                "Bereits bekannte Fakten (nicht duplizieren):\n"
                f"{existing_str}\n\n"
                "Antworte als JSON-Array mit Objekten: "
                '[{"category": "...", "content": "...", "confidence": 0.0-1.0}]\n'
                "Kategorien:\n"
                "- style: Kommunikationsstil, Tonalität (z.B. 'duzt Kunden', 'formell bei Erstansprache')\n"
                "- preference: Vorlieben und Präferenzen (z.B. 'Bevorzugt kurze E-Mails')\n"
                "- fact: Allgemeine Fakten über den Trainer (z.B. 'Spezialisiert auf Outdoor-Training')\n"
                "- business: Geschäftliche Infos (z.B. 'Firma: TrainerPro GmbH, 5 Mitarbeiter')\n"
                "- audience: Zielgruppen-Infos (z.B. 'Zielgruppe: Führungskräfte 35-55')\n\n"
                "Nur Fakten mit confidence >= 0.6 extrahieren. "
                "Leeres Array [] wenn keine neuen Fakten."
            )},
            {"role": "user", "content": f"Chat-Verlauf:\n{conversation}"},
        ]

        response = oai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=1000,
        )

        result_text = response.choices[0].message.content.strip()

        # JSON parsen
        try:
            memories = json.loads(result_text)
        except json.JSONDecodeError:
            # Versuche JSON aus Markdown-Block zu extrahieren
            if "```json" in result_text:
                json_str = result_text.split("```json")[1].split("```")[0].strip()
                memories = json.loads(json_str)
            elif "```" in result_text:
                json_str = result_text.split("```")[1].split("```")[0].strip()
                memories = json.loads(json_str)
            else:
                memories = []

        log.info("Memory-Extraktion Chat %d: %d neue Memories", data.chat_id, len(memories))

        # Callback an Laravel
        callback_url = data.callback_url
        if callback_url:
            send_callback(callback_url, {
                "chat_id": data.chat_id,
                "user_id": data.user_id,
                "memories": memories,
            })

    except Exception as e:
        log.error("Memory-Extraktion Chat %d Fehler: %s", data.chat_id, str(e))


# ── Prompt Builder ──────────────────────────────────────────────────

BUILDER_SYSTEM_PROMPT = """Du bist ein Experte für Prompt-Engineering. Der Benutzer verbessert den System-Prompt eines KI-Assistenten durch natürliche Sprache.

DEINE AUFGABE:
1. Verstehe was der Benutzer möchte (Verhalten, Tonalität, Fähigkeiten, Einschränkungen etc.)
2. Schlage einen verbesserten System-Prompt vor
3. Erkläre kurz was du geändert hast

REGELN:
- Antworte IMMER auf Deutsch
- Gib den vorgeschlagenen Prompt IMMER innerhalb von <proposed_prompt>...</proposed_prompt> Tags aus
- Gib eine kurze Zusammenfassung der Änderung in <change_summary>...</change_summary> Tags aus
- Der vorgeschlagene Prompt soll professionell, klar strukturiert und effektiv sein
- Wenn der aktuelle Prompt leer ist, erstelle einen komplett neuen basierend auf der Beschreibung
- Behalte bestehende gute Aspekte bei und verbessere/erweitere nur was nötig ist
- Schreibe den Prompt in der Du-Form an die KI gerichtet ("Du bist...", "Du hilfst...")

BEISPIEL-ANTWORT:
Ich habe den Prompt angepasst und folgende Änderungen vorgenommen:
- Tonalität auf formeller gestellt
- Zielgruppe konkretisiert

<change_summary>Tonalität formeller, Zielgruppe konkretisiert</change_summary>

<proposed_prompt>
Du bist ein professioneller Berater für Führungskräfte...
</proposed_prompt>"""


class BuilderChatRequest(BaseModel):
    assistant_id: int
    current_system_prompt: str = ""
    assistant_name: str = ""
    assistant_description: str = ""
    message: str
    conversation_history: list = []  # Bisheriger Builder-Chat-Verlauf
    model: str = "gpt-4o"


@app.post("/builder/chat")
async def builder_chat(request: Request):
    """
    Prompt-Builder: Meta-Assistent der System-Prompts iterativ verbessert.
    Synchroner Call — gibt sofort die Antwort zurück.
    """
    verify_auth(request)
    try:
        body = await request.json()
    except (ValueError, json.JSONDecodeError):
        raise HTTPException(400, "Invalid JSON")

    data = BuilderChatRequest(**body)
    log.info("Builder chat: assistant_id=%d", data.assistant_id)

    # Kontext für den Meta-Assistenten aufbauen
    context_parts = []
    if data.assistant_name:
        context_parts.append(f"Assistenten-Name: {data.assistant_name}")
    if data.assistant_description:
        context_parts.append(f"Beschreibung: {data.assistant_description}")
    context_parts.append(f"Aktueller System-Prompt:\n```\n{data.current_system_prompt or '(leer — noch kein Prompt definiert)'}\n```")
    context = "\n".join(context_parts)

    messages = [
        {"role": "system", "content": BUILDER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Kontext zum Assistenten:\n{context}"},
    ]

    # Bisherigen Builder-Chat-Verlauf hinzufügen
    for msg in data.conversation_history:
        messages.append({
            "role": msg.get("role", "user"),
            "content": msg.get("content", ""),
        })

    # Aktuelle Nachricht
    messages.append({"role": "user", "content": data.message})

    try:
        model = get_model_name(data.model)
        response = oai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=4000,
        )

        reply = response.choices[0].message.content or ""

        # Proposed prompt und change summary extrahieren
        proposed_prompt = None
        change_summary = None

        if "<proposed_prompt>" in reply and "</proposed_prompt>" in reply:
            proposed_prompt = reply.split("<proposed_prompt>")[1].split("</proposed_prompt>")[0].strip()

        if "<change_summary>" in reply and "</change_summary>" in reply:
            change_summary = reply.split("<change_summary>")[1].split("</change_summary>")[0].strip()

        # Tags aus der sichtbaren Antwort entfernen für saubere Chat-Darstellung
        clean_reply = reply
        if "<proposed_prompt>" in clean_reply:
            clean_reply = clean_reply.split("<proposed_prompt>")[0] + clean_reply.split("</proposed_prompt>")[-1]
        if "<change_summary>" in clean_reply:
            clean_reply = clean_reply.split("<change_summary>")[0] + clean_reply.split("</change_summary>")[-1]
        clean_reply = clean_reply.strip()

        return {
            "reply": clean_reply,
            "proposed_prompt": proposed_prompt,
            "change_summary": change_summary,
            "usage": {
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            },
            "model": model,
        }

    except Exception as e:
        log.error("Builder chat error: %s", str(e))
        raise HTTPException(500, f"Builder error: {str(e)}")

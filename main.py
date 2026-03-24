"""
AI-Gateway — FastAPI Service
Empfängt Chat-Requests von Laravel, orchestriert LLM-Calls,
und sendet Ergebnisse per Callback zurück.
"""

import os
import time
import json
import uuid
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
    callback_url: str = ""
    mode: str = "async"  # "async" (default) oder "sync"
    # RAG-Konfiguration
    notebook_ids: list = []           # z.B. ["nb_1", "nb_3"]
    doc_processor_url: str = ""       # z.B. "https://trainer-doc-processor.ai-guide.at"
    doc_processor_secret: str = ""    # Bearer-Token für doc-processor


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
        tools = build_tools(data.skills) if data.skills else None
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

                        # Status-Event an Browser senden
                        yield f"event: status\ndata: {json.dumps({'tool': tool_name})}\n\n"

                        all_tool_calls.append({"id": tc.id, "name": tool_name, "arguments": tool_args})
                        result = execute_skill(tool_name, tool_args, data.user_id)
                        all_tool_results.append({"tool_call_id": tc.id, "name": tool_name, "result": result})

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
        yield f"event: done\ndata: {json.dumps({'usage': usage, 'model': model, 'duration_ms': duration_ms, 'tool_calls': all_tool_calls or None, 'tool_results': all_tool_results or None})}\n\n"

        log.info("Stream chat %d: Fertig. %d tokens, %d ms", data.chat_id, usage["output_tokens"], duration_ms)

        # Callback an Laravel (Antwort persistieren + Battery-Tracking)
        callback_url = data.callback_url or LARAVEL_CALLBACK_URL
        if callback_url:
            send_callback(callback_url, {
                "chat_id": data.chat_id,
                "content": full_content,
                "usage": usage,
                "model": model,
                "duration_ms": duration_ms,
                "tool_calls": all_tool_calls or None,
                "tool_results": all_tool_results or None,
            })

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
        tools = build_tools(data.skills) if data.skills else None
        model = get_model_name(data.model)
        all_tool_calls = []
        all_tool_results = []
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
                result = execute_skill(tool_name, tool_args, data.user_id)
                all_tool_results.append({"tool_call_id": tc.id, "name": tool_name, "result": result})

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

        return {
            "status": "completed",
            "chat_id": data.chat_id,
            "content": content,
            "usage": usage,
            "model": model,
            "duration_ms": duration_ms,
            "tool_calls": all_tool_calls or None,
            "tool_results": all_tool_results or None,
        }

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
        tools = build_tools(data.skills) if data.skills else None

        # 3. LLM-Call (mit Tool-Call-Loop)
        model = get_model_name(data.model)
        all_tool_calls = []
        all_tool_results = []
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

                # Skill bei Laravel ausführen
                result = execute_skill(tool_name, tool_args, data.user_id)
                all_tool_results.append({
                    "tool_call_id": tc.id,
                    "name": tool_name,
                    "result": result,
                })

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
            send_callback(callback_url, {
                "chat_id": data.chat_id,
                "content": content,
                "usage": usage,
                "model": model,
                "duration_ms": duration_ms,
                "tool_calls": all_tool_calls or None,
                "tool_results": all_tool_results or None,
            })

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

    messages = []
    if system_parts:
        messages.append({"role": "system", "content": "\n".join(system_parts)})

    # Conversation History (bereits als role/content dicts)
    messages.extend(data.conversation_history)

    # Aktuelle Nachricht
    messages.append({"role": "user", "content": data.message})

    return messages


# ── Helper: Tool-Definitionen ────────────────────────────────────────

def build_tools(skill_slugs: list) -> list:
    """
    Baut OpenAI Function Calling Tool-Definitionen.
    Holt die Skill-Definitionen von Laravel.
    """
    try:
        skill_url = LARAVEL_SKILL_URL
        if not skill_url:
            return []

        with httpx.Client(timeout=10) as client:
            resp = client.get(
                skill_url,
                headers={"Authorization": f"Bearer {AI_GATEWAY_SECRET}"},
                params={"skills": ",".join(skill_slugs)},
            )
            if resp.status_code != 200:
                log.warning("Skill-Definitionen laden fehlgeschlagen: %d", resp.status_code)
                return []

            skills_data = resp.json()
            tools = []
            for skill in skills_data.get("skills", []):
                tools.append({
                    "type": "function",
                    "function": {
                        "name": skill["name"],
                        "description": skill["description"],
                        "parameters": skill.get("parameters", {"type": "object", "properties": {}}),
                    },
                })
            return tools

    except Exception as e:
        log.error("Skill-Definitionen laden Fehler: %s", str(e))
        return []


# ── Helper: Skill bei Laravel ausführen ──────────────────────────────

def execute_skill(skill_name: str, params: dict, user_id: int) -> dict:
    """Sendet Tool-Call an Laravel /api/ai/skills."""
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
                return resp.json()
            else:
                log.warning("Skill %s fehlgeschlagen: %d — %s", skill_name, resp.status_code, resp.text)
                return {"error": f"Skill-Fehler: HTTP {resp.status_code}"}

    except Exception as e:
        log.error("Skill %s Fehler: %s", skill_name, str(e))
        return {"error": str(e)}


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


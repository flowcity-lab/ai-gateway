"""
Hilfsfunktionen für das SSE-Streaming im Gateway.

Bewusst dependency-frei (nur Python-Stdlib), damit die Tests isoliert laufen
können, ohne FastAPI, OpenAI oder andere Gateway-Abhängigkeiten zu ziehen.
"""


def accumulate_tool_call_delta(buffer: dict, tc_delta) -> None:
    """Mergt ein einzelnes OpenAI tool_call-Delta in den Buffer.

    OpenAI schickt Tool-Call-Informationen fragmentiert über viele Chunks:
    - id und name kommen jeweils nur einmal
    - arguments werden inkrementell pro Chunk aufgebaut

    Der Buffer ist ein dict indexed by ``tc_delta.index``, damit auch mehrere
    parallele Tool-Calls korrekt getrennt zusammengesetzt werden können.

    Args:
        buffer: Mapping ``index -> {"id": str, "name": str, "arguments": str}``.
            Wird in-place mutiert.
        tc_delta: Objekt mit ``.index``, ``.id``, ``.function.name``,
            ``.function.arguments`` (OpenAI ChoiceDeltaToolCall oder Äquivalent).
    """
    idx = tc_delta.index
    buf = buffer.setdefault(idx, {"id": "", "name": "", "arguments": ""})

    if tc_delta.id:
        buf["id"] = tc_delta.id

    fn = getattr(tc_delta, "function", None)
    if fn is not None:
        if fn.name:
            buf["name"] = fn.name
        if fn.arguments:
            buf["arguments"] += fn.arguments


def build_assistant_tool_calls(buffer: dict) -> list:
    """Baut aus einem befüllten Buffer die Liste der Tool-Calls für die
    Message-History zusammen.

    Die Reihenfolge richtet sich nach dem ``index`` aus dem Buffer (stabil,
    aufsteigend), sodass parallele Tool-Calls deterministisch verarbeitet
    werden.

    Args:
        buffer: Ergebnis-Dict von ``accumulate_tool_call_delta``.

    Returns:
        Liste von Tool-Call-Dicts im OpenAI-Message-Format, bereit für
        ``messages.append({"role": "assistant", "tool_calls": ...})``.
    """
    tool_calls = []
    for idx in sorted(buffer.keys()):
        buf = buffer[idx]
        tool_calls.append({
            "id": buf["id"],
            "type": "function",
            "function": {
                "name": buf["name"],
                "arguments": buf["arguments"] or "{}",
            },
        })
    return tool_calls

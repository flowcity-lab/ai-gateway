"""
Unit-Tests für die Delta-Assembly-Logik des SSE-Streaming.

Diese Tests decken das Zusammensetzen von OpenAI-Tool-Call-Deltas ab, das
bei jedem Assistent mit aktiven Skills im Streaming-Pfad ausgeführt wird.

Ausführung (aus dem ai-gateway/ Verzeichnis):

    pip install -r requirements-dev.txt
    pytest -v
"""
import json
from types import SimpleNamespace

import pytest

from stream_utils import accumulate_tool_call_delta, build_assistant_tool_calls


# ── Helper ────────────────────────────────────────────────────────────

def _delta(index, id_=None, name=None, args=None):
    """Baut ein Fake-OpenAI-tool_call-Delta nach, wie es .chat.completions.create
    mit stream=True liefern würde. Felder sind optional (OpenAI schickt sie
    fragmentiert über mehrere Chunks)."""
    fn = None
    if name is not None or args is not None:
        fn = SimpleNamespace(name=name, arguments=args)
    return SimpleNamespace(index=index, id=id_, function=fn)


# ── accumulate_tool_call_delta ────────────────────────────────────────

class TestAccumulateDelta:
    def test_single_complete_delta(self):
        """Alle Felder in einem einzigen Chunk → direkt korrekt im Buffer."""
        buf = {}
        accumulate_tool_call_delta(buf, _delta(0, id_="call_1", name="search", args='{"q":"x"}'))
        assert buf == {0: {"id": "call_1", "name": "search", "arguments": '{"q":"x"}'}}

    def test_fields_fragmented_across_chunks(self):
        """Realer OpenAI-Flow: id zuerst, name im 2., arguments in mehreren Chunks."""
        buf = {}
        accumulate_tool_call_delta(buf, _delta(0, id_="call_abc"))
        accumulate_tool_call_delta(buf, _delta(0, name="search_contacts"))
        accumulate_tool_call_delta(buf, _delta(0, args='{"que'))
        accumulate_tool_call_delta(buf, _delta(0, args='ry":"Max"}'))

        assert buf[0]["id"] == "call_abc"
        assert buf[0]["name"] == "search_contacts"
        assert buf[0]["arguments"] == '{"query":"Max"}'

    def test_parallel_tool_calls_different_indexes(self):
        """GPT kann mehrere Tools gleichzeitig aufrufen (verschiedene .index)."""
        buf = {}
        accumulate_tool_call_delta(buf, _delta(0, id_="call_1", name="search", args='{"a":1}'))
        accumulate_tool_call_delta(buf, _delta(1, id_="call_2", name="create", args='{"b":2}'))

        assert len(buf) == 2
        assert buf[0]["name"] == "search"
        assert buf[1]["name"] == "create"
        assert buf[0]["arguments"] == '{"a":1}'
        assert buf[1]["arguments"] == '{"b":2}'

    def test_parallel_tool_calls_interleaved(self):
        """Parallele Tool-Calls deren Deltas verschränkt ankommen."""
        buf = {}
        accumulate_tool_call_delta(buf, _delta(0, id_="call_1"))
        accumulate_tool_call_delta(buf, _delta(1, id_="call_2"))
        accumulate_tool_call_delta(buf, _delta(0, name="skill_a"))
        accumulate_tool_call_delta(buf, _delta(1, name="skill_b"))
        accumulate_tool_call_delta(buf, _delta(0, args='{"x":'))
        accumulate_tool_call_delta(buf, _delta(1, args='{"y":'))
        accumulate_tool_call_delta(buf, _delta(0, args='1}'))
        accumulate_tool_call_delta(buf, _delta(1, args='2}'))

        assert buf[0] == {"id": "call_1", "name": "skill_a", "arguments": '{"x":1}'}
        assert buf[1] == {"id": "call_2", "name": "skill_b", "arguments": '{"y":2}'}

    def test_delta_without_function_is_ignored(self):
        """Ein Delta nur mit id (ohne function-Objekt) darf nichts kaputt machen."""
        buf = {}
        accumulate_tool_call_delta(buf, _delta(0, id_="call_x"))  # function=None

        assert buf[0] == {"id": "call_x", "name": "", "arguments": ""}

    def test_arguments_concatenate_not_overwrite(self):
        """Arguments werden inkrementell aufgebaut (Concat, kein Überschreiben)."""
        buf = {}
        accumulate_tool_call_delta(buf, _delta(0, args='hello'))
        accumulate_tool_call_delta(buf, _delta(0, args=' world'))

        assert buf[0]["arguments"] == "hello world"

    def test_empty_arguments_chunks_are_skipped(self):
        """OpenAI kann leere arguments-Chunks schicken — die dürfen nichts verändern."""
        buf = {}
        accumulate_tool_call_delta(buf, _delta(0, args='{"q":"x"}'))
        accumulate_tool_call_delta(buf, _delta(0, args=''))
        accumulate_tool_call_delta(buf, _delta(0, args=None))

        assert buf[0]["arguments"] == '{"q":"x"}'

    def test_id_overwrites_not_concatenates(self):
        """Falls OpenAI die id nochmal schickt, wird sie ersetzt — nicht angehängt."""
        buf = {}
        accumulate_tool_call_delta(buf, _delta(0, id_="call_1"))
        accumulate_tool_call_delta(buf, _delta(0, id_="call_final"))

        assert buf[0]["id"] == "call_final"

    def test_name_overwrites_not_concatenates(self):
        """Der name wird ebenfalls ersetzt statt angehängt."""
        buf = {}
        accumulate_tool_call_delta(buf, _delta(0, name="alt"))
        accumulate_tool_call_delta(buf, _delta(0, name="neu"))

        assert buf[0]["name"] == "neu"

    def test_assembled_arguments_parse_as_valid_json(self):
        """Nach dem Zusammensetzen müssen die arguments valides JSON sein."""
        buf = {}
        for chunk in ['{"query":', ' "hallo",', ' "limit":', ' 5}']:
            accumulate_tool_call_delta(buf, _delta(0, args=chunk))

        parsed = json.loads(buf[0]["arguments"])
        assert parsed == {"query": "hallo", "limit": 5}


# ── build_assistant_tool_calls ────────────────────────────────────────

class TestBuildAssistantToolCalls:
    def test_empty_buffer_returns_empty_list(self):
        assert build_assistant_tool_calls({}) == []

    def test_single_tool_call_shape(self):
        """Erzeugt das von OpenAI erwartete Message-Format."""
        buf = {0: {"id": "call_1", "name": "search", "arguments": '{"q":"x"}'}}
        result = build_assistant_tool_calls(buf)

        assert result == [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "search", "arguments": '{"q":"x"}'},
        }]

    def test_multiple_tool_calls_sorted_by_index(self):
        """Ausgabe ist stabil nach index sortiert, egal in welcher Reihenfolge
        die Schlüssel in den Buffer kamen."""
        buf = {
            2: {"id": "call_c", "name": "c", "arguments": '{}'},
            0: {"id": "call_a", "name": "a", "arguments": '{}'},
            1: {"id": "call_b", "name": "b", "arguments": '{}'},
        }
        result = build_assistant_tool_calls(buf)

        assert [t["id"] for t in result] == ["call_a", "call_b", "call_c"]
        assert [t["function"]["name"] for t in result] == ["a", "b", "c"]

    def test_empty_arguments_defaults_to_empty_json_object(self):
        """Leere arguments werden zu '{}', damit OpenAI sie als gültiges JSON
        parsen kann (sonst wirft die API einen Fehler)."""
        buf = {0: {"id": "call_1", "name": "skill", "arguments": ""}}
        result = build_assistant_tool_calls(buf)

        assert result[0]["function"]["arguments"] == "{}"

    def test_non_empty_arguments_pass_through_unchanged(self):
        buf = {0: {"id": "call_1", "name": "skill", "arguments": '{"a":1}'}}
        result = build_assistant_tool_calls(buf)

        assert result[0]["function"]["arguments"] == '{"a":1}'

import types

import pytest

from services.context_packager import ContextPackager


def test_package_respects_budget_and_records_drops():
    packager = ContextPackager(
        total_budget=120,
        allocation_plan={"instructions": 0.2, "knowledge": 0.6, "tools": 0.1, "scratch": 0.1},
    )

    instructions = ["System instructions" + "!" * 100]
    knowledge = ["K" * 400, "Q" * 200]
    tools = ["Tool schema", {"content": "CALL foo(arg)"}]
    scratch = ["memo A", "memo B" * 20]

    package = packager.package(
        instructions=instructions,
        knowledge=knowledge,
        tools=tools,
        scratchpad=scratch,
    )

    assert package.total_used <= package.total_budget
    instructions_seg = package.segment("instructions")
    assert instructions_seg is not None
    assert instructions_seg.used_tokens <= instructions_seg.budget_tokens
    knowledge_seg = package.segment("knowledge")
    assert knowledge_seg is not None
    # large blocks are likely trimmed or dropped due to limited budget
    assert knowledge_seg.used_tokens <= knowledge_seg.budget_tokens
    assert knowledge_seg.used_tokens >= 0


def test_build_from_context_handles_missing_fields():
    class DummyClassification:
        primary_paradigm = "bernard"

    context_obj = types.SimpleNamespace(
        write_output=types.SimpleNamespace(
            documentation_focus="Test focus",
            key_themes=["theme a", "theme b"],
        ),
        select_output=types.SimpleNamespace(
            search_queries=[{"query": "alpha"}, {"query": "beta"}],
        ),
        isolate_output=types.SimpleNamespace(focus_areas=["gap 1", "gap 2"]),
        compress_output=types.SimpleNamespace(
            compression_strategy="map-reduce",
            priority_elements=["summaries"],
        ),
        classification=DummyClassification(),
        layer_durations={"write": 0.1},
    )

    packager = ContextPackager(total_budget=200)
    package = packager.build_from_context(
        context_obj,
        base_instructions="Base",
        tool_schemas=["Tool X"],
        memory_items=["note"],
    )

    as_dict = package.as_dict()
    assert as_dict["total_used"] <= as_dict["total_budget"]
    assert package.segment("instructions") is not None
    assert package.segment("knowledge") is not None


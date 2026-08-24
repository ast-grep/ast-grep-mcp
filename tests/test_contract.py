from __future__ import annotations

from pathlib import Path

import pytest

from ast_soleaux.contracts import (
    contract_profile_tools,
    load_contract,
    profile_fingerprint,
    tool_fingerprint,
)
from ast_soleaux.server import SUPPORTED_AST_GREP_VERSION, ResolvedExecutable, RuntimeServices, create_mcp

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "tests" / "fixtures" / "contracts" / "ast-soleaux-0.5.0.json"
PROFILE_ORDER = (
    "baseline",
    "oxc",
    "semantic",
    "configured",
    "typescript_compiler",
    "postgresql",
    "all",
)


def object_value(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    assert all(isinstance(key, str) for key in value)
    return {key: item for key, item in value.items() if isinstance(key, str)}


def list_value(value: object) -> list[object]:
    assert isinstance(value, list)
    return value


def integer_value(value: object) -> int:
    assert isinstance(value, int) and not isinstance(value, bool)
    return value


def string_dict(value: object) -> dict[str, str]:
    mapping = object_value(value)
    assert all(isinstance(item, str) for item in mapping.values())
    return {key: item for key, item in mapping.items() if isinstance(item, str)}


def runtime(profile: str) -> RuntimeServices:
    index = PROFILE_ORDER.index(profile)
    executable = ResolvedExecutable(path=ROOT / "ast_soleaux" / "server.py", command_prefix=("placeholder",))
    return RuntimeServices(
        working_directory=ROOT,
        executable=executable,
        ast_grep_version=SUPPORTED_AST_GREP_VERSION,
        config_path=None,
        allowed_roots=(ROOT,),
        command_timeout_seconds=30,
        default_max_results=50,
        max_results_cap=500,
        forbid_regex_rules=False,
        oxc_helper=executable if index >= PROFILE_ORDER.index("oxc") else None,
        oxc_versions=({"helper": "0.1.0", "parser": "0.147.0", "resolver": "11.24.2"} if index >= PROFILE_ORDER.index("oxc") else None),
        analysis_helper=executable if index >= PROFILE_ORDER.index("semantic") else None,
        analysis_versions=({"worker": "0.1.0", "oxc": "0.75.0"} if index >= PROFILE_ORDER.index("semantic") else None),
        typescript_project_helper=executable if index >= PROFILE_ORDER.index("typescript_compiler") else None,
        typescript_versions=({"typescript": "6.0.2"} if index >= PROFILE_ORDER.index("typescript_compiler") else None),
        postgres_helper=executable if index >= PROFILE_ORDER.index("postgresql") else None,
        postgres_versions=({"parser": "18.0.0", "deparser": "18.3.6"} if index >= PROFILE_ORDER.index("postgresql") else None),
        typescript_execution_helper=executable if profile == "all" else None,
        typescript_execution_versions={"helper": "0.1.0"} if profile == "all" else None,
    )


def test_contract_catalog_is_unique_and_profile_counts_are_consistent() -> None:
    contract = load_contract(CONTRACT)
    tools = list_value(contract["tools"])
    names = [object_value(tool)["name"] for tool in tools]
    assert all(isinstance(name, str) for name in names)
    assert len(names) == len(set(names)) == 22
    assert {"oxc_format", "oxc_format_files"}.isdisjoint(names)
    profiles = object_value(contract["profiles"])
    assert integer_value(object_value(profiles["baseline"])["count"]) == 6
    assert integer_value(object_value(profiles["oxc"])["count"]) == 11
    assert integer_value(object_value(profiles["all"])["count"]) == 22
    assert {name: len(tool_names) for name, tool_names in contract_profile_tools(contract).items()} == {
        name: integer_value(object_value(profiles[name])["count"]) for name in PROFILE_ORDER
    }


@pytest.mark.asyncio
async def test_every_capability_profile_matches_generated_catalog_fingerprints() -> None:
    contract = load_contract(CONTRACT)
    profiles = contract_profile_tools(contract)
    generated = object_value(contract["generated"])
    expected_tools = string_dict(generated["tool_fingerprints"])
    expected_profiles = string_dict(generated["profile_fingerprints"])

    for profile in PROFILE_ORDER:
        server = create_mcp(runtime(profile))
        if PROFILE_ORDER.index(profile) >= PROFILE_ORDER.index("configured"):
            server.enable(names={"scan_project_rules", "test_project_rules"})
        first = await server.list_tools(run_middleware=False)
        second = await server.list_tools(run_middleware=False)
        first_fingerprints = {tool.name: tool_fingerprint(tool.to_mcp_tool(), key=str(tool.key), version=tool.version) for tool in first}
        second_fingerprints = {tool.name: tool_fingerprint(tool.to_mcp_tool(), key=str(tool.key), version=tool.version) for tool in second}
        assert frozenset(first_fingerprints) == profiles[profile]
        assert first_fingerprints == second_fingerprints
        assert first_fingerprints == {name: expected_tools[name] for name in profiles[profile]}
        assert profile_fingerprint(first_fingerprints, profiles[profile]) == expected_profiles[profile]

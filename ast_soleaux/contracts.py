from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mcp.types import Tool
from pydantic import TypeAdapter

type JsonValue = bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None
type JsonObject = dict[str, JsonValue]

JSON_OBJECT_ADAPTER: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)


def tool_fingerprint(tool: Tool, *, key: str | None = None, version: str | None = None) -> str:
    metadata = JSON_OBJECT_ADAPTER.validate_python(tool.meta or {}, strict=True)
    fastmcp_value = metadata.get("fastmcp")
    fastmcp_metadata = fastmcp_value if isinstance(fastmcp_value, dict) else {}
    tags_value = fastmcp_metadata.get("tags")
    tags = tags_value if isinstance(tags_value, list) else []
    payload = {
        "key": key or f"tool:{tool.name}@",
        "name": tool.name,
        "version": version,
        "title": tool.title,
        "description": tool.description,
        "input_schema": tool.input_schema,
        "output_schema": tool.output_schema,
        "annotations": tool.annotations.model_dump(mode="json") if tool.annotations is not None else None,
        "tags": sorted(tag for tag in tags if isinstance(tag, str)),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def contract_profile_tools(contract: JsonObject) -> dict[str, frozenset[str]]:
    profiles_value = contract.get("profiles")
    if not isinstance(profiles_value, dict):
        raise ValueError("contract profiles must be an object")
    resolved: dict[str, frozenset[str]] = {}
    current: set[str] = set()
    for profile_name, profile_value in profiles_value.items():
        if not isinstance(profile_value, dict):
            raise ValueError(f"contract profile {profile_name} must be an object")
        tools_value = profile_value.get("tools")
        if isinstance(tools_value, list):
            current = {tool for tool in tools_value if isinstance(tool, str)}
        adds_value = profile_value.get("adds")
        if isinstance(adds_value, list):
            current.update(tool for tool in adds_value if isinstance(tool, str))
        count_value = profile_value.get("count")
        if not isinstance(count_value, int) or count_value != len(current):
            raise ValueError(f"contract profile {profile_name} count does not match its tools")
        resolved[profile_name] = frozenset(current)
    return resolved


def profile_fingerprint(tool_fingerprints: dict[str, str], tool_names: frozenset[str]) -> str:
    payload = {name: tool_fingerprints[name] for name in sorted(tool_names)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_contract(path: Path) -> JsonObject:
    return JSON_OBJECT_ADAPTER.validate_json(path.read_bytes(), strict=True)

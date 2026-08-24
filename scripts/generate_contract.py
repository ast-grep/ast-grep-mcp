from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from ast_soleaux.contracts import (
    JSON_OBJECT_ADAPTER,
    JsonObject,
    contract_profile_tools,
    load_contract,
    profile_fingerprint,
    tool_fingerprint,
)
from ast_soleaux.server import SUPPORTED_AST_GREP_VERSION, ResolvedExecutable, RuntimeServices, create_mcp

DEFAULT_CONTRACT = Path("tests/fixtures/contracts/ast-soleaux-0.5.0.json")


async def generated_contract_metadata(contract: JsonObject) -> JsonObject:
    executable = ResolvedExecutable(path=Path("/nonexistent/ast-soleaux-contract-helper"), command_prefix=("helper",))
    runtime = RuntimeServices(
        working_directory=Path.cwd(),
        executable=executable,
        ast_grep_version=SUPPORTED_AST_GREP_VERSION,
        config_path=None,
        allowed_roots=(Path.cwd(),),
        command_timeout_seconds=30,
        default_max_results=50,
        max_results_cap=500,
        forbid_regex_rules=False,
        oxc_helper=executable,
        oxc_versions={"helper": "0.1.0", "parser": "0.147.0", "resolver": "11.24.2"},
        analysis_helper=executable,
        analysis_versions={"worker": "0.1.0", "oxc": "0.75.0"},
        typescript_project_helper=executable,
        typescript_versions={"typescript": "6.0.2"},
        postgres_helper=executable,
        postgres_versions={"parser": "18.0.0", "deparser": "18.3.6"},
        typescript_execution_helper=executable,
        typescript_execution_versions={"helper": "0.1.0"},
    )
    server = create_mcp(runtime)
    server.enable(names={"scan_project_rules", "test_project_rules"})
    tools = await server.list_tools(run_middleware=False)
    fingerprints = {tool.name: tool_fingerprint(tool.to_mcp_tool(), key=str(tool.key), version=tool.version) for tool in tools}
    profiles = contract_profile_tools(contract)
    expected_all = profiles["all"]
    if frozenset(fingerprints) != expected_all:
        missing = sorted(expected_all - fingerprints.keys())
        extra = sorted(fingerprints.keys() - expected_all)
        raise RuntimeError(f"runtime catalog differs from contract: missing={missing}, extra={extra}")
    return JSON_OBJECT_ADAPTER.validate_python(
        {
            "tool_fingerprints": dict(sorted(fingerprints.items())),
            "profile_fingerprints": {name: profile_fingerprint(fingerprints, names) for name, names in profiles.items()},
        },
        strict=True,
    )


async def run(path: Path, *, check: bool) -> int:
    contract = load_contract(path)
    generated = await generated_contract_metadata(contract)
    if check:
        if contract.get("generated") != generated:
            print(f"{path}: generated contract metadata is stale")
            return 1
        print(f"{path}: generated contract metadata is current")
        return 0
    contract["generated"] = generated
    payload = json.dumps(contract, indent=2, ensure_ascii=False) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
    print(f"updated {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate ast-soleaux contract fingerprints")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    return asyncio.run(run(arguments.contract, check=arguments.check))


if __name__ == "__main__":
    raise SystemExit(main())

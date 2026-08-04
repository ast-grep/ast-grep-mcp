from __future__ import annotations

import hashlib
import os
import shutil
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import yaml

import config_snapshot as snapshot_module
from config_snapshot import create_config_snapshot, load_strict_yaml_documents
from main import validate_rule_yaml


def write_project_config(root: Path, config: str) -> Path:
    path = root / "sgconfig.yml"
    path.write_text(config, encoding="utf-8", newline="")
    return path


def create_snapshot(root: Path, config: str = "ruleDirs: []\n", **kwargs: Any) -> snapshot_module.ConfigSnapshot:
    write_project_config(root, config)
    return create_config_snapshot(
        config_path="sgconfig.yml",
        working_directory=root,
        allowed_roots=(root.resolve(),),
        **kwargs,
    )


@pytest.mark.parametrize(
    ("source", "message"),
    [
        pytest.param("value: &shared text\n", "anchor", id="anchor"),
        pytest.param("value: *missing\n", "alias", id="alias"),
        pytest.param("value: !!str text\n", "explicit YAML tag", id="tag"),
        pytest.param("<<: text\n", "merge key", id="merge"),
        pytest.param("value: one\nvalue: two\n", "duplicate YAML key", id="duplicate-key"),
        pytest.param("1: value\n", "non-string YAML mapping key", id="non-string-key"),
    ],
)
def test_strict_yaml_rejects_graph_and_ambiguous_mapping_features(source: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_strict_yaml_documents(source, label="test YAML")


def test_strict_yaml_rejects_document_node_and_depth_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="2-document"):
        load_strict_yaml_documents("---\na: 1\n---\nb: 2\n---\nc: 3\n", label="test YAML", max_documents=2)

    monkeypatch.setattr(snapshot_module, "MAX_YAML_NODES", 4)
    with pytest.raises(ValueError, match="4-node"):
        load_strict_yaml_documents("values: [one, two, three, four]\n", label="test YAML")

    monkeypatch.setattr(snapshot_module, "MAX_YAML_NODES", 10_000)
    monkeypatch.setattr(snapshot_module, "MAX_YAML_DEPTH", 3)
    with pytest.raises(ValueError, match="3-level"):
        load_strict_yaml_documents("value: [[[[item]]]]\n", label="test YAML")


def test_snapshot_freezes_separate_least_privilege_configs_and_cleans_up(tmp_path: Path) -> None:
    rules = tmp_path / "rules"
    utils = tmp_path / "utils"
    tests = tmp_path / "rule-tests"
    snapshots = tests / "baselines"
    rules.mkdir()
    utils.mkdir()
    snapshots.mkdir(parents=True)
    (rules / "configured.yml").write_text(
        "id: configured.rule\nlanguage: Python\nmessage: configured\nseverity: info\nrule:\n  matches: shared-call\n",
        encoding="utf-8",
    )
    (utils / "shared.yml").write_text(
        "id: shared-call\nlanguage: Python\nrule:\n  pattern: print($A)\n",
        encoding="utf-8",
    )
    (tests / "configured-test.yml").write_text(
        "id: configured.rule\nvalid: [value]\ninvalid: [print(1)]\n",
        encoding="utf-8",
    )
    (snapshots / "configured-snapshot.yml").write_text(
        "id: configured.rule\nsnapshots: {}\n",
        encoding="utf-8",
    )
    config_path = write_project_config(
        tmp_path,
        "ruleDirs: [rules]\nutilDirs: [utils]\ntestConfigs:\n  - testDir: rule-tests\n    snapshotDir: baselines\n",
    )

    snapshot = create_config_snapshot(
        config_path=str(config_path),
        working_directory=tmp_path,
        allowed_roots=(tmp_path.resolve(),),
    )
    bundle = snapshot.bundle_root
    assert snapshot.configured_rule_ids == ("configured.rule",)
    assert snapshot.capabilities == {
        "inline_search": True,
        "outline": True,
        "configured_scan": True,
        "configured_tests": True,
        "custom_languages": False,
    }
    assert snapshot.provenance["source_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert snapshot.provenance["resource_files"] == 5
    assert snapshot.provenance["resource_bytes"] > 0
    assert len(snapshot.digest) == 64
    if os.name == "posix":
        assert stat.S_IMODE(bundle.stat().st_mode) == 0o500
        assert all(stat.S_IMODE(path.stat().st_mode) == 0o400 for path in bundle.rglob("*") if path.is_file())

    inline = yaml.safe_load(snapshot.inline_config_path.read_text(encoding="utf-8"))
    project = yaml.safe_load(snapshot.project_config_path.read_text(encoding="utf-8"))
    test = yaml.safe_load(snapshot.test_config_path.read_text(encoding="utf-8"))
    assert inline == {"ruleDirs": []}
    assert set(project) == {"ruleDirs", "utilDirs"}
    assert set(test) == {"ruleDirs", "utilDirs", "testConfigs"}
    assert "testConfigs" not in project
    assert project.get("ruleDirs")
    assert (bundle / project["ruleDirs"][0] / "configured.yml").read_text(encoding="utf-8").startswith("id: configured.rule")
    assert (bundle / test["testConfigs"][0]["testDir"] / "baselines" / "configured-snapshot.yml").is_file()

    frozen_rule = (bundle / project["ruleDirs"][0] / "configured.yml").read_bytes()
    original_digest = snapshot.digest
    config_path.write_text("ruleDirs: []\n", encoding="utf-8")
    (rules / "configured.yml").write_text("id: replacement\n", encoding="utf-8")
    assert (bundle / project["ruleDirs"][0] / "configured.yml").read_bytes() == frozen_rule
    assert snapshot.digest == original_digest

    snapshot.close()
    snapshot.close()
    assert not bundle.exists()


@pytest.mark.parametrize(
    ("config", "message"),
    [
        pytest.param("ruleDirs: []\nunknownKey: true\n", "unknown ast-grep 0.45 keys", id="top-level"),
        pytest.param(
            "ruleDirs: []\ntestConfigs:\n  - testDir: tests\n    unknown: true\n",
            "testConfigs\\[0\\].*unknown",
            id="test-config",
        ),
        pytest.param(
            "ruleDirs: []\ncustomLanguages:\n  Demo:\n    libraryPath: parser.bin\n    extensions: [demo]\n    unknown: true\n",
            "customLanguages.Demo.*unknown",
            id="custom-language",
        ),
        pytest.param(
            "ruleDirs: []\nlanguageInjections:\n  - hostLanguage: Html\n    injected: JavaScript\n    unknown: true\n",
            "languageInjections\\[0\\].*unknown",
            id="language-injection",
        ),
    ],
)
def test_snapshot_rejects_unknown_ast_grep_045_configuration_keys(
    tmp_path: Path,
    config: str,
    message: str,
) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "parser.bin").write_bytes(b"parser")
    with pytest.raises(ValueError, match=message):
        create_snapshot(tmp_path, config)


def test_snapshot_accepts_ast_grep_045_defaults_and_injection_rule_core(tmp_path: Path) -> None:
    snapshot = create_snapshot(
        tmp_path,
        "languageInjections:\n"
        "  - hostLanguage: Html\n"
        "    injected: JavaScript\n"
        "    rule: {kind: script_element}\n"
        "    constraints: {NODE: {kind: identifier}}\n"
        "    utils: {shared: {kind: identifier}}\n"
        "    transform: {TEXT: {substring: {source: $NODE, startChar: 0}}}\n",
    )
    try:
        private_config = yaml.safe_load(snapshot.inline_config_path.read_text(encoding="utf-8"))
        assert private_config["ruleDirs"] == []
        assert set(private_config["languageInjections"][0]) == {
            "hostLanguage",
            "injected",
            "rule",
            "constraints",
            "utils",
            "transform",
        }
    finally:
        snapshot.close()


@pytest.mark.parametrize(
    ("config", "message"),
    [
        pytest.param(
            'ruleDirs: []\nlanguageGlobs:\n  "": ["*.demo"]\n',
            "invalid language name",
            id="empty-language-glob-name",
        ),
        pytest.param(
            "ruleDirs: []\nlanguageInjections:\n  - hostLanguage: Html\n    injected: JavaScript\n",
            r"languageInjections\[0\]\.rule",
            id="missing-injection-rule",
        ),
        pytest.param(
            "ruleDirs: []\nlanguageInjections:\n  - hostLanguage: Html\n    injected: []\n    rule: {kind: script_element}\n",
            "injected must be a non-empty string or list",
            id="empty-injected-language-list",
        ),
    ],
)
def test_snapshot_rejects_invalid_project_language_configuration(tmp_path: Path, config: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        create_snapshot(tmp_path, config)


def test_snapshot_rejects_config_resource_and_native_symlinks(tmp_path: Path) -> None:
    real_config = write_project_config(tmp_path, "ruleDirs: []\n")
    config_link = tmp_path / "linked-config.yml"
    config_link.symlink_to(real_config.name)
    with pytest.raises(ValueError, match="symlinks"):
        create_config_snapshot(
            config_path=config_link.name,
            working_directory=tmp_path,
            allowed_roots=(tmp_path.resolve(),),
        )

    rules = tmp_path / "rules"
    rules.mkdir()
    outside = tmp_path / "outside.yml"
    outside.write_text("id: outside\n", encoding="utf-8")
    (rules / "linked.yml").symlink_to(outside)
    with pytest.raises(ValueError, match="cannot contain symlinks"):
        create_snapshot(tmp_path, "ruleDirs: [rules]\n")


def test_windows_reparse_metadata_is_treated_as_a_link() -> None:
    metadata = cast(
        os.stat_result,
        SimpleNamespace(
            st_mode=stat.S_IFDIR,
            st_file_attributes=getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024),
        ),
    )
    assert snapshot_module._is_link_like(metadata) is True


def test_snapshot_fallback_copy_rejects_links_and_retains_regular_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(snapshot_module.os, "supports_dir_fd", set())
    rules = tmp_path / "rules"
    rules.mkdir()
    rule = rules / "rule.yml"
    rule.write_text("id: safe\nlanguage: Python\nrule: {kind: identifier}\n", encoding="utf-8", newline="")

    snapshot = create_snapshot(tmp_path, "ruleDirs: [rules]\n")
    try:
        private_config = yaml.safe_load(snapshot.project_config_path.read_text(encoding="utf-8"))
        retained = snapshot.bundle_root / private_config["ruleDirs"][0] / rule.name
        assert retained.read_bytes() == rule.read_bytes()
    finally:
        snapshot.close()

    (rules / "linked.yml").symlink_to(rule)
    with pytest.raises(ValueError, match="symlinks or reparse points"):
        create_snapshot(tmp_path, "ruleDirs: [rules]\n")


def test_snapshot_rejects_a_resource_directory_swapped_to_a_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.open not in os.supports_dir_fd:
        pytest.skip("descriptor-relative opening is unavailable")
    rules = tmp_path / "rules"
    outside = tmp_path / "outside"
    rules.mkdir()
    outside.mkdir()
    (rules / "rule.yml").write_text("id: safe\nlanguage: Python\nrule: {kind: identifier}\n", encoding="utf-8")
    (outside / "rule.yml").write_text("id: outside\nlanguage: Python\nrule: {kind: identifier}\n", encoding="utf-8")
    original = snapshot_module._open_directory_chain
    swapped = False

    def swap_before_open(path: Path, *, boundary: Path, label: str) -> int:
        nonlocal swapped
        if path == rules and not swapped:
            swapped = True
            rules.rename(tmp_path / "original-rules")
            rules.symlink_to(outside, target_is_directory=True)
        return original(path, boundary=boundary, label=label)

    monkeypatch.setattr(snapshot_module, "_open_directory_chain", swap_before_open)
    with pytest.raises(ValueError, match="securely open configuration resource directory"):
        create_snapshot(tmp_path, "ruleDirs: [rules]\n")


@pytest.mark.parametrize("entry", ["../outside", "/outside"])
def test_snapshot_rejects_resource_path_escapes(tmp_path: Path, entry: str) -> None:
    with pytest.raises(ValueError, match=r"contained relative path|outside the allowed roots|escapes its permitted directory"):
        create_snapshot(tmp_path, f"ruleDirs: [{entry!r}]\n")


def test_snapshot_rejects_config_outside_allowed_roots(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    config = write_project_config(outside, "ruleDirs: []\n")
    with pytest.raises(ValueError, match="outside the allowed roots"):
        create_config_snapshot(
            config_path=str(config),
            working_directory=allowed,
            allowed_roots=(allowed.resolve(),),
        )


def test_snapshot_rejects_invalid_config_bytes_and_path_kinds(tmp_path: Path) -> None:
    allowed_roots = (tmp_path.resolve(),)
    config = tmp_path / "sgconfig.yml"
    config.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="valid UTF-8"):
        create_config_snapshot(
            config_path=config.name,
            working_directory=tmp_path,
            allowed_roots=allowed_roots,
        )

    config.write_text("ruleDirs: [rules]\n", encoding="utf-8")
    (tmp_path / "rules").write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
        create_config_snapshot(
            config_path=config.name,
            working_directory=tmp_path,
            allowed_roots=allowed_roots,
        )

    config_directory = tmp_path / "config-directory"
    config_directory.mkdir()
    with pytest.raises(ValueError, match="not a regular file"):
        create_config_snapshot(
            config_path=config_directory.name,
            working_directory=tmp_path,
            allowed_roots=allowed_roots,
        )

    with pytest.raises(ValueError, match="does not exist"):
        create_config_snapshot(
            config_path="missing.yml",
            working_directory=tmp_path,
            allowed_roots=allowed_roots,
        )

    with pytest.raises(ValueError, match="NUL"):
        create_config_snapshot(
            config_path="invalid\0.yml",
            working_directory=tmp_path,
            allowed_roots=allowed_roots,
        )


def test_snapshot_native_library_requires_exact_hash_and_is_copied(tmp_path: Path) -> None:
    parser = tmp_path / "parser.dylib"
    parser.write_bytes(b"trusted parser bytes")
    config = (
        "ruleDirs: []\n"
        "customLanguages:\n"
        "  Demo:\n"
        "    libraryPath: parser.dylib\n"
        "    extensions: [demo]\n"
        "    languageSymbol: tree_sitter_demo\n"
        "    metaVarChar: '$'\n"
        "    expandoChar: _\n"
    )
    write_project_config(tmp_path, config)
    digest = hashlib.sha256(parser.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="requires --trusted-native-library"):
        create_snapshot(tmp_path, config)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        create_config_snapshot(
            config_path="sgconfig.yml",
            working_directory=tmp_path,
            allowed_roots=(tmp_path.resolve(),),
            trusted_native_libraries=((str(parser), "0" * 64),),
        )

    snapshot = create_config_snapshot(
        config_path="sgconfig.yml",
        working_directory=tmp_path,
        allowed_roots=(tmp_path.resolve(),),
        trusted_native_libraries=((str(parser), digest.upper()),),
    )
    try:
        assert snapshot.capabilities["custom_languages"] is True
        assert snapshot.native_library_hashes == {"Demo": digest}
        private_config = yaml.safe_load(snapshot.inline_config_path.read_text(encoding="utf-8"))
        assert private_config["customLanguages"]["Demo"]["metaVarChar"] == "$"
        private_library = snapshot.bundle_root / private_config["customLanguages"]["Demo"]["libraryPath"]
        assert private_library.read_bytes() == b"trusted parser bytes"
        assert private_library != parser
    finally:
        snapshot.close()


def test_snapshot_rejects_unreferenced_or_malformed_native_trust(tmp_path: Path) -> None:
    parser = tmp_path / "parser.dylib"
    parser.write_bytes(b"parser")
    write_project_config(tmp_path, "ruleDirs: []\n")
    with pytest.raises(ValueError, match="64-character"):
        create_config_snapshot(
            config_path="sgconfig.yml",
            working_directory=tmp_path,
            allowed_roots=(tmp_path.resolve(),),
            trusted_native_libraries=((str(parser), "not-a-digest"),),
        )
    with pytest.raises(ValueError, match="not referenced"):
        create_config_snapshot(
            config_path="sgconfig.yml",
            working_directory=tmp_path,
            allowed_roots=(tmp_path.resolve(),),
            trusted_native_libraries=((str(parser), hashlib.sha256(parser.read_bytes()).hexdigest()),),
        )


def test_snapshot_rejects_resource_limits_and_removes_partial_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rules = tmp_path / "rules"
    rules.mkdir()
    (rules / "one.yml").write_text("id: one\nlanguage: Python\nrule: {kind: identifier}\n", encoding="utf-8")
    (rules / "two.yml").write_text("id: two\nlanguage: Python\nrule: {kind: identifier}\n", encoding="utf-8")
    write_project_config(tmp_path, "ruleDirs: [rules]\n")
    bundle = tmp_path / "partial-bundle"

    def make_bundle(*, prefix: str, dir: str | os.PathLike[str]) -> str:
        assert prefix == "config-"
        assert Path(dir).resolve() == (tmp_path / snapshot_module.RUNTIME_DIRECTORY_NAME).resolve()
        bundle.mkdir()
        return os.fspath(bundle)

    monkeypatch.setattr(snapshot_module, "mkdtemp", make_bundle)
    monkeypatch.setattr(snapshot_module, "MAX_CONFIG_RESOURCE_FILES", 1)
    with pytest.raises(ValueError, match="1-file resource limit"):
        create_config_snapshot(
            config_path="sgconfig.yml",
            working_directory=tmp_path,
            allowed_roots=(tmp_path.resolve(),),
        )
    assert not bundle.exists()


def test_snapshot_runtime_bundle_stays_inside_the_repository_boundary(tmp_path: Path) -> None:
    snapshot = create_snapshot(tmp_path)
    bundle = snapshot.bundle_root
    runtime_root = snapshot.runtime_root
    assert bundle.is_relative_to(tmp_path)
    assert runtime_root == tmp_path / snapshot_module.RUNTIME_DIRECTORY_NAME
    snapshot.close()
    assert not bundle.exists()
    assert not runtime_root.exists()


def test_runtime_bundle_does_not_require_the_working_directory_to_be_inspectable(tmp_path: Path) -> None:
    server = tmp_path / "server"
    project = tmp_path / "project"
    server.mkdir()
    project.mkdir()

    runtime_root = snapshot_module.private_runtime_root(server, (project.resolve(),))

    assert runtime_root == server.resolve() / snapshot_module.RUNTIME_DIRECTORY_NAME
    assert stat.S_IMODE(runtime_root.lstat().st_mode) == 0o700
    runtime_root.rmdir()
    runtime_root.symlink_to(project, target_is_directory=True)
    with pytest.raises(ValueError, match="must be a real directory"):
        snapshot_module.private_runtime_root(server, (project.resolve(),))


def test_snapshot_bounds_traversed_entries_that_are_never_copied(tmp_path: Path) -> None:
    rules = tmp_path / "rules"
    rules.mkdir()
    for index in range(snapshot_module.MAX_CONFIG_RESOURCE_FILES + 2):
        (rules / f"ignored{index}.txt").write_text("x", encoding="utf-8", newline="")

    with pytest.raises(ValueError, match="file resource limit"):
        create_snapshot(tmp_path, "ruleDirs: [rules]\n")


def test_snapshot_fallback_rejects_a_directory_swapped_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(snapshot_module.os, "supports_dir_fd", set())
    rules = tmp_path / "rules"
    nested = rules / "nested"
    outside = tmp_path / "outside"
    nested.mkdir(parents=True)
    outside.mkdir()
    (nested / "contained.yml").write_text("id: contained\n", encoding="utf-8", newline="")
    (outside / "escaped.yml").write_text("id: escaped\n", encoding="utf-8", newline="")

    real_scandir = snapshot_module.os.scandir

    def swap_before_reopening(target: Any) -> Any:
        if isinstance(target, (str, Path)) and Path(target) == nested and not nested.is_symlink():
            shutil.rmtree(nested)
            nested.symlink_to(outside, target_is_directory=True)
        return real_scandir(target)

    monkeypatch.setattr(snapshot_module.os, "scandir", swap_before_reopening)
    with pytest.raises(ValueError, match="changed during traversal"):
        create_snapshot(tmp_path, "ruleDirs: [rules]\n")


def test_snapshot_cleanup_can_be_retried_after_a_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = create_snapshot(tmp_path)
    original = snapshot_module.shutil.rmtree
    attempts = 0

    def fail_once(path: str | os.PathLike[str]) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("cleanup failed")
        original(path)

    monkeypatch.setattr(snapshot_module.shutil, "rmtree", fail_once)
    with pytest.raises(OSError, match="cleanup failed"):
        snapshot.close()
    assert snapshot.bundle_root.exists()
    snapshot.close()
    assert not snapshot.bundle_root.exists()


def test_snapshot_rejects_zero_progress_cycles_but_preserves_relational_recursion(tmp_path: Path) -> None:
    utils = tmp_path / "utils"
    utils.mkdir()
    (utils / "cycle.yml").write_text(
        "id: first\nlanguage: Python\nrule: {matches: second}\n---\nid: second\nlanguage: Python\nrule: {matches: first}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="zero-progress utility-rule cycle"):
        create_snapshot(tmp_path, "ruleDirs: []\nutilDirs: [utils]\n")

    (utils / "cycle.yml").write_text(
        "id: recursive\nlanguage: Python\nrule:\n  has:\n    matches: recursive\n",
        encoding="utf-8",
    )
    snapshot = create_snapshot(tmp_path, "ruleDirs: []\nutilDirs: [utils]\n")
    snapshot.close()


def test_inline_rule_cycle_validation_rejects_composites_and_allows_relations() -> None:
    with pytest.raises(ValueError, match="zero-progress utility-rule cycle"):
        validate_rule_yaml(
            "id: cycle\nlanguage: Python\nrule: {matches: first}\nutils:\n  first: {matches: second}\n  second: {matches: first}\n",
            forbid_regex_rules=False,
        )

    validate_rule_yaml(
        "id: recursive\nlanguage: Python\nrule: {matches: recursive}\nutils:\n  recursive:\n    has: {matches: recursive}\n",
        forbid_regex_rules=False,
    )


def test_zero_progress_cycle_validation_handles_long_acyclic_graphs() -> None:
    definitions: dict[str, object] = {f"utility-{index}": {"rule": {"matches": f"utility-{index + 1}"}} for index in range(1_500)}
    definitions["utility-1500"] = {"rule": {"kind": "identifier"}}
    snapshot_module._reject_zero_progress_cycles(definitions, label="utility graph")


def test_zero_progress_cycle_validation_respects_parameter_shadowing() -> None:
    snapshot_module._reject_zero_progress_cycles(
        {"X": {"arguments": ["X"], "rule": {"matches": "X"}}},
        label="utility graph",
    )
    with pytest.raises(ValueError, match="zero-progress utility-rule cycle"):
        snapshot_module._reject_zero_progress_cycles(
            {
                "first": {"arguments": ["VALUE"], "rule": {"matches": {"second": {"ARG": {"matches": "VALUE"}}}}},
                "second": {"arguments": ["ARG"], "rule": {"matches": {"first": {"VALUE": {"matches": "ARG"}}}}},
            },
            label="utility graph",
        )

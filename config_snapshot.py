from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import stat
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Final, cast

import yaml
from yaml.events import (
    AliasEvent,
    CollectionEndEvent,
    CollectionStartEvent,
    DocumentStartEvent,
    Event,
    NodeEvent,
    ScalarEvent,
)
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

MAX_CONFIG_FILE_BYTES: Final = 1024 * 1024
MAX_CONFIG_RESOURCE_BYTES: Final = 16 * 1024 * 1024
MAX_CONFIG_RESOURCE_FILES: Final = 1024
MAX_NATIVE_LIBRARY_BYTES: Final = 16 * 1024 * 1024
MAX_YAML_DOCUMENTS: Final = 64
MAX_YAML_NODES: Final = 10_000
MAX_YAML_DEPTH: Final = 64
MAX_SNAPSHOT_YAML_NODES: Final = 200_000
RUNTIME_DIRECTORY_NAME: Final = ".project-sast-runtime"

_CONFIG_KEYS: Final = frozenset({"ruleDirs", "testConfigs", "utilDirs", "customLanguages", "languageGlobs", "languageInjections"})
_TEST_CONFIG_KEYS: Final = frozenset({"testDir", "snapshotDir"})
_CUSTOM_LANGUAGE_KEYS: Final = frozenset({"libraryPath", "languageSymbol", "metaVarChar", "expandoChar", "extensions", "outlineRules"})
_INJECTION_KEYS: Final = frozenset({"hostLanguage", "injected", "rule", "constraints", "utils", "transform"})
_SHA256_RE: Final = re.compile(r"[0-9a-fA-F]{64}\Z")


def is_within(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)


def _containing_root(path: Path, roots: Sequence[Path]) -> Path:
    matches = [root for root in roots if is_within(path, root)]
    if not matches:
        raise ValueError(f"Path resolves outside the allowed roots: {path}")
    return max(matches, key=lambda root: len(root.parts))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _directory_identity(directory: Path) -> tuple[int, bool]:
    """Report the file type and link nature of a path without following it.

    The descriptor walker holds an O_NOFOLLOW handle, so the directory it reads
    is the directory it validated. The fallback reopens by name, leaving a window
    where the path may be replaced with a link to another tree between validation
    and traversal. Device and inode cannot settle that: deleting a directory frees
    its inode for immediate reuse, so a replacement link can carry the same number,
    and Windows does not report a stable inode across calls. A junction reports
    the directory type it replaced, so the reparse attribute is carried alongside.
    """
    try:
        metadata = os.stat(directory, follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"Could not inspect configuration resource directory: {directory}") from error
    return stat.S_IFMT(metadata.st_mode), _is_link_like(metadata)


def _count_visited_entry(writer: _BundleWriter, display_path: Path) -> None:
    """Charge every traversed entry against the resource budget.

    Only copied files reach the writer, so a tree of filtered or empty entries
    would otherwise permit unbounded walking beneath the advertised limit.
    """
    writer.visited += 1
    if writer.visited > MAX_CONFIG_RESOURCE_FILES:
        raise ValueError(f"Configuration exceeds the {MAX_CONFIG_RESOURCE_FILES}-file resource limit: {display_path}")


def _is_link_like(metadata: os.stat_result) -> bool:
    if stat.S_ISLNK(metadata.st_mode):
        return True
    reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(reparse_point and getattr(metadata, "st_file_attributes", 0) & reparse_point)


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    raw = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in raw):
        raise ValueError(f"{label} must use string keys")
    return cast(dict[str, Any], value)


def _list(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return cast(list[Any], value)


def _string_list(value: object, *, label: str, allow_empty: bool = True) -> list[str]:
    values = _list(value, label=label)
    if any(not isinstance(item, str) or "\0" in item or (not allow_empty and not item) for item in values):
        raise ValueError(f"{label} must contain valid strings")
    return cast(list[str], values)


def _reject_unknown_keys(value: Mapping[str, Any], allowed: frozenset[str], *, label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{label} contains unknown ast-grep 0.45 keys: {', '.join(unknown)}")


def _inspect_yaml_node(node: Node, *, label: str, depth: int, count: list[int]) -> None:
    count[0] += 1
    if count[0] > MAX_YAML_NODES:
        raise ValueError(f"{label} exceeds the {MAX_YAML_NODES}-node YAML limit")
    if depth > MAX_YAML_DEPTH:
        raise ValueError(f"{label} exceeds the {MAX_YAML_DEPTH}-level YAML depth limit")
    if isinstance(node, MappingNode):
        seen: set[str] = set()
        for key, item in node.value:
            if isinstance(key, ScalarNode) and key.value == "<<":
                raise ValueError(f"{label} contains a forbidden YAML merge key")
            if not isinstance(key, ScalarNode) or key.tag != "tag:yaml.org,2002:str":
                raise ValueError(f"{label} contains a non-string YAML mapping key")
            if key.value in seen:
                raise ValueError(f"{label} contains duplicate YAML key {key.value!r}")
            seen.add(key.value)
            _inspect_yaml_node(key, label=label, depth=depth + 1, count=count)
            _inspect_yaml_node(item, label=label, depth=depth + 1, count=count)
    elif isinstance(node, SequenceNode):
        for item in node.value:
            _inspect_yaml_node(item, label=label, depth=depth + 1, count=count)
    elif not isinstance(node, ScalarNode):
        raise ValueError(f"{label} contains an unsupported YAML node")


def load_strict_yaml_documents(
    source: str,
    *,
    label: str,
    max_documents: int = MAX_YAML_DOCUMENTS,
) -> list[object]:
    try:
        yaml_api = cast(Any, yaml)
        parse_yaml = cast(Callable[..., Iterable[Event]], yaml_api.parse)
        events = parse_yaml(source, Loader=yaml.SafeLoader)
        depth = 0
        documents = 0
        event_nodes = 0
        for event in events:
            if isinstance(event, DocumentStartEvent):
                documents += 1
                if documents > max_documents:
                    raise ValueError(f"{label} exceeds the {max_documents}-document YAML limit")
            if isinstance(event, AliasEvent):
                raise ValueError(f"{label} contains a forbidden YAML alias")
            if isinstance(event, NodeEvent):
                event_nodes += 1
                if event_nodes > MAX_YAML_NODES:
                    raise ValueError(f"{label} exceeds the {MAX_YAML_NODES}-node YAML limit")
                if event.anchor is not None:
                    raise ValueError(f"{label} contains a forbidden YAML anchor")
                if isinstance(event, (ScalarEvent, CollectionStartEvent)) and event.tag is not None:
                    raise ValueError(f"{label} contains a forbidden explicit YAML tag")
            if isinstance(event, CollectionStartEvent):
                depth += 1
                if depth > MAX_YAML_DEPTH:
                    raise ValueError(f"{label} exceeds the {MAX_YAML_DEPTH}-level YAML depth limit")
            elif isinstance(event, CollectionEndEvent):
                depth -= 1
        compose_yaml = cast(
            Callable[..., Iterable[Node | None]],
            yaml_api.compose_all,
        )
        composed = list(compose_yaml(source, Loader=yaml.SafeLoader))
        if len(composed) > max_documents:
            raise ValueError(f"{label} exceeds the {max_documents}-document YAML limit")
        count = [0]
        for node in composed:
            if node is not None:
                _inspect_yaml_node(node, label=label, depth=1, count=count)
        loaded: list[object] = list(yaml.safe_load_all(source))
    except ValueError:
        raise
    except (yaml.YAMLError, RecursionError) as error:
        raise ValueError(f"Invalid {label}: {error}") from error
    return loaded


def _absolute_without_resolution(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _assert_no_symlink_components(path: Path, *, boundary: Path, label: str) -> None:
    lexical_path = _absolute_without_resolution(path)
    lexical_boundary = _absolute_without_resolution(boundary)
    if not is_within(lexical_path, lexical_boundary):
        raise ValueError(f"{label} escapes its permitted directory: {path}")
    current = lexical_boundary
    for part in lexical_path.relative_to(lexical_boundary).parts:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError as error:
            raise ValueError(f"{label} does not exist: {path}") from error
        if _is_link_like(metadata):
            raise ValueError(f"{label} cannot contain symlinks or reparse points: {current}")


def _secure_existing_path(
    raw_path: str | Path,
    *,
    base: Path,
    allowed_roots: Sequence[Path],
    kind: str,
    label: str,
    require_relative: bool = False,
    boundary: Path | None = None,
) -> Path:
    raw = Path(raw_path)
    if "\0" in os.fspath(raw):
        raise ValueError(f"{label} contains a NUL character")
    if require_relative and (raw.is_absolute() or ".." in raw.parts):
        raise ValueError(f"{label} must be a contained relative path: {raw_path}")
    candidate = raw if raw.is_absolute() else base / raw
    lexical = _absolute_without_resolution(candidate)
    permitted = boundary.resolve(strict=True) if boundary is not None else None
    if permitted is not None and not is_within(lexical, permitted):
        raise ValueError(f"{label} escapes its permitted directory: {raw_path}")
    matching_roots = [root for root in allowed_roots if is_within(lexical, root)]
    if not matching_roots:
        raise ValueError(f"{label} resolves outside the allowed roots: {raw_path}")
    symlink_boundary = permitted or max(matching_roots, key=lambda root: len(root.parts))
    _assert_no_symlink_components(lexical, boundary=symlink_boundary, label=label)
    try:
        resolved = lexical.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"{label} does not exist: {raw_path}") from error
    if not any(is_within(resolved, root) for root in allowed_roots):
        raise ValueError(f"{label} resolves outside the allowed roots: {raw_path}")
    if permitted is not None and not is_within(resolved, permitted):
        raise ValueError(f"{label} escapes its permitted directory: {raw_path}")
    if kind == "file" and not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {raw_path}")
    if kind == "directory" and not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {raw_path}")
    return resolved


def _read_open_file(descriptor: int, path: Path, *, byte_limit: int, label: str) -> bytes:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} is not a regular file: {path}")
    if metadata.st_size > byte_limit:
        raise ValueError(f"{label} exceeds the {byte_limit}-byte limit: {path}")
    chunks: list[bytes] = []
    consumed = 0
    while True:
        chunk = os.read(descriptor, min(64 * 1024, byte_limit + 1 - consumed))
        if not chunk:
            break
        chunks.append(chunk)
        consumed += len(chunk)
        if consumed > byte_limit:
            raise ValueError(f"{label} exceeds the {byte_limit}-byte limit: {path}")
    final_metadata = os.fstat(descriptor)
    if (
        final_metadata.st_dev != metadata.st_dev
        or final_metadata.st_ino != metadata.st_ino
        or final_metadata.st_size != metadata.st_size
        or final_metadata.st_mtime_ns != metadata.st_mtime_ns
    ):
        raise ValueError(f"{label} changed while it was being copied: {path}")
    return b"".join(chunks)


def _open_directory_chain(path: Path, *, boundary: Path, label: str) -> int:
    lexical_path = _absolute_without_resolution(path)
    lexical_boundary = _absolute_without_resolution(boundary)
    if not is_within(lexical_path, lexical_boundary):
        raise ValueError(f"{label} escapes its permitted directory: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lexical_boundary, flags)
    try:
        for part in lexical_path.relative_to(lexical_boundary).parts:
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _assert_opened_name_is_not_a_link(path: Path, *, label: str) -> None:
    """Confirm a name opened without O_NOFOLLOW still denotes a regular file.

    Windows has neither O_NOFOLLOW nor dir_fd, so the open follows the name. A link
    restored before this recheck still passes.
    """
    try:
        named = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"Could not revalidate {label}: {path}") from error
    if _is_link_like(named):
        raise ValueError(f"{label} was replaced with a link while it was being opened: {path}")
    if not stat.S_ISREG(named.st_mode):
        raise ValueError(f"{label} is not a regular file: {path}")


def _read_file(path: Path, *, byte_limit: int, label: str, boundary: Path | None = None) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory_descriptor: int | None = None
    opened_by_name = False
    try:
        if boundary is not None and os.open in os.supports_dir_fd:
            directory_descriptor = _open_directory_chain(path.parent, boundary=boundary, label=label)
            descriptor = os.open(path.name, flags, dir_fd=directory_descriptor)
        else:
            opened_by_name = True
            descriptor = os.open(path, flags)
    except OSError as error:
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        raise ValueError(f"Could not securely open {label}: {path}") from error
    try:
        if opened_by_name:
            _assert_opened_name_is_not_a_link(path, label=label)
        return _read_open_file(descriptor, path, byte_limit=byte_limit, label=label)
    finally:
        os.close(descriptor)
        if directory_descriptor is not None:
            os.close(directory_descriptor)


def _decode_yaml(payload: bytes, *, label: str, budget: _NodeBudget | None = None) -> tuple[str, list[object]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must be valid UTF-8") from error
    documents = load_strict_yaml_documents(text, label=label)
    if budget is not None:
        budget.charge(_count_yaml_nodes(documents), label=label)
    return text, documents


def _count_yaml_nodes(value: object) -> int:
    if isinstance(value, dict):
        return 1 + sum(_count_yaml_nodes(item) for item in cast(dict[object, object], value).values())
    if isinstance(value, list):
        return 1 + sum(_count_yaml_nodes(item) for item in cast(list[object], value))
    return 1


@dataclass
class _NodeBudget:
    """Carry one YAML node allowance across every resource in a snapshot.

    The per-file limits bound a single document, so a configuration spread over
    the permitted file count could retain that allowance once per file and still
    expand into far more objects than any individual limit suggests.
    """

    limit: int = field(default_factory=lambda: MAX_SNAPSHOT_YAML_NODES)
    used: int = 0

    def charge(self, nodes: int, *, label: str) -> None:
        self.used += nodes
        if self.used > self.limit:
            raise ValueError(f"Configuration exceeds the {self.limit}-node aggregate YAML limit: {label}")


class _BundleWriter:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.files = 0
        self.visited = 0
        self.bytes = 0
        self._digest = hashlib.sha256()

    def reserve(self, relative: Path) -> Path:
        """Create a copied directory even when the source retains no entries.

        A configured test directory that is valid but empty would otherwise leave
        the private configuration pointing at a path that was never created, and
        ast-grep would be handed a missing directory instead of an empty one.
        """
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid private bundle path: {relative}")
        destination = self.root / relative
        destination.mkdir(parents=True, exist_ok=True, mode=0o700)
        return destination

    def add(self, relative: Path, payload: bytes, *, resource: bool = True) -> Path:
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid private bundle path: {relative}")
        if resource:
            if self.files + 1 > MAX_CONFIG_RESOURCE_FILES:
                raise ValueError(f"Configuration exceeds the {MAX_CONFIG_RESOURCE_FILES}-file resource limit")
            if self.bytes + len(payload) > MAX_CONFIG_RESOURCE_BYTES:
                raise ValueError(f"Configuration exceeds the {MAX_CONFIG_RESOURCE_BYTES // (1024 * 1024)} MiB resource limit")
            self.files += 1
            self.bytes += len(payload)
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with destination.open("xb") as output:
            output.write(payload)
        destination.chmod(0o400)
        self._digest.update(relative.as_posix().encode("utf-8"))
        self._digest.update(b"\0")
        self._digest.update(len(payload).to_bytes(8, "big"))
        self._digest.update(payload)
        return destination

    def digest(self) -> str:
        return self._digest.hexdigest()


def _yaml_bytes(value: Mapping[str, Any]) -> bytes:
    return yaml.safe_dump(dict(value), sort_keys=False, allow_unicode=True).encode("utf-8")


def _iter_yaml_documents(documents: Sequence[object], *, label: str) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for document in documents:
        if document is None:
            continue
        if isinstance(document, list):
            values = cast(list[object], document)
        else:
            values = [document]
        for value in values:
            flattened.append(_mapping(value, label=label))
    return flattened


def _matches_dependencies(rule: object, *, shadowed: frozenset[str] = frozenset()) -> set[str]:
    if not isinstance(rule, dict):
        return set()
    mapping = cast(dict[str, Any], rule)
    dependencies: set[str] = set()
    matches = mapping.get("matches")
    if isinstance(matches, str) and matches not in shadowed:
        dependencies.add(matches)
    elif isinstance(matches, dict):
        call = cast(dict[object, object], matches)
        for callee, arguments in call.items():
            if isinstance(callee, str) and callee not in shadowed:
                dependencies.add(callee)
            if isinstance(arguments, dict):
                for argument in cast(dict[object, object], arguments).values():
                    dependencies.update(_matches_dependencies(argument, shadowed=shadowed))
    for key in ("all", "any"):
        values = mapping.get(key)
        if isinstance(values, list):
            for value in cast(list[object], values):
                dependencies.update(_matches_dependencies(value, shadowed=shadowed))
    dependencies.update(_matches_dependencies(mapping.get("not"), shadowed=shadowed))
    return dependencies


def _reject_zero_progress_cycles(definitions: Mapping[str, object], *, label: str) -> None:
    dependencies: dict[str, set[str]] = {}
    for name, definition in definitions.items():
        if not isinstance(definition, dict):
            continue
        mapping = cast(dict[str, Any], definition)
        arguments = _string_list(mapping.get("arguments", []), label=f"{label} {name!r} arguments", allow_empty=False)
        if len(arguments) != len(set(arguments)):
            raise ValueError(f"{label} {name!r} arguments must be unique")
        shadowed = frozenset(arguments)
        refs = _matches_dependencies(mapping.get("rule"), shadowed=shadowed)
        for section in ("constraints", "utils"):
            values = mapping.get(section)
            if isinstance(values, dict):
                for value in cast(dict[object, object], values).values():
                    refs.update(_matches_dependencies(value, shadowed=shadowed))
        dependencies[name] = {ref for ref in refs if ref in definitions}

    states: dict[str, int] = {}
    for start in definitions:
        if states.get(start) == 2:
            continue
        states[start] = 1
        pending: list[tuple[str, Iterator[str]]] = [(start, iter(dependencies.get(start, set())))]
        while pending:
            name, iterator = pending[-1]
            try:
                dependency = next(iterator)
            except StopIteration:
                states[name] = 2
                pending.pop()
                continue
            state = states.get(dependency, 0)
            if state == 1:
                raise ValueError(f"{label} contains a zero-progress utility-rule cycle at {dependency!r}")
            if state == 0:
                states[dependency] = 1
                pending.append((dependency, iter(dependencies.get(dependency, set()))))


def _as_cycle_definition(value: object) -> dict[str, Any]:
    """Present a utility entry in the shape the cycle detector reads.

    A local utility value is the rule object itself, so it is wrapped under
    ``rule``. A parameterized entry already carries ``arguments`` alongside its
    own ``rule``; wrapping that again would bury both the parameters and the
    calls inside, letting mutually recursive definitions pass unnoticed.
    """
    if isinstance(value, dict) and "arguments" in value and "rule" in value:
        return cast(dict[str, Any], value)
    return {"rule": value}


def _validate_local_utility_cycles(rule: Mapping[str, Any], *, label: str) -> None:
    utils = rule.get("utils")
    if isinstance(utils, dict):
        definitions = {
            name: _as_cycle_definition(value) for name, value in cast(dict[object, object], utils).items() if isinstance(name, str)
        }
        _reject_zero_progress_cycles(definitions, label=label)


def validate_rule_utility_cycles(documents: Sequence[object], *, label: str) -> None:
    for index, document in enumerate(documents, start=1):
        if isinstance(document, dict):
            mapping = _mapping(cast(object, document), label=f"{label} document {index}")
            _validate_local_utility_cycles(mapping, label=f"{label} document {index}")


def _safe_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    return normalized or "resource"


def _copy_directory(
    source: Path,
    *,
    destination: Path,
    writer: _BundleWriter,
    yaml_only: bool,
    yaml_documents: list[tuple[str, list[dict[str, Any]]]],
    boundary: Path,
    budget: _NodeBudget,
) -> None:
    writer.reserve(destination)
    if os.open in os.supports_dir_fd:
        try:
            descriptor = _open_directory_chain(source, boundary=boundary, label="configuration resource directory")
        except OSError as error:
            raise ValueError(f"Could not securely open configuration resource directory: {source}") from error
        try:
            _copy_directory_descriptor(
                descriptor,
                source=source,
                relative=Path(),
                destination=destination,
                writer=writer,
                yaml_only=yaml_only,
                yaml_documents=yaml_documents,
                depth=0,
                budget=budget,
            )
        finally:
            os.close(descriptor)
        return
    pending = [(source, _directory_identity(source))]
    while pending:
        directory, expected_identity = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as error:
            raise ValueError(f"Could not read configuration resource directory: {directory}") from error
        if _directory_identity(directory) != expected_identity:
            raise ValueError(f"Configuration resource directory changed during traversal: {directory}")
        for entry in entries:
            entry_path = Path(entry.path)
            _count_visited_entry(writer, entry_path)
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise ValueError(f"Could not inspect configuration resource: {entry_path}") from error
            if _is_link_like(metadata):
                raise ValueError(f"Configuration resources cannot contain symlinks or reparse points: {entry_path}")
            if stat.S_ISDIR(metadata.st_mode):
                pending.append((entry_path, (stat.S_IFMT(metadata.st_mode), _is_link_like(metadata))))
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"Configuration resources must be regular files: {entry_path}")
            if yaml_only and entry_path.suffix.lower() not in {".yml", ".yaml"}:
                continue
            payload = _read_file(entry_path, byte_limit=MAX_CONFIG_FILE_BYTES, label="configuration resource")
            relative = entry_path.relative_to(source)
            writer.add(destination / relative, payload)
            if entry_path.suffix.lower() in {".yml", ".yaml"}:
                _, documents = _decode_yaml(payload, label=f"configuration resource {entry_path}", budget=budget)
                yaml_documents.append(
                    (entry_path.as_posix(), _iter_yaml_documents(documents, label=f"configuration resource {entry_path}"))
                )


def _copy_directory_descriptor(
    descriptor: int,
    *,
    source: Path,
    relative: Path,
    destination: Path,
    writer: _BundleWriter,
    yaml_only: bool,
    yaml_documents: list[tuple[str, list[dict[str, Any]]]],
    depth: int,
    budget: _NodeBudget,
) -> None:
    if depth > MAX_YAML_DEPTH:
        raise ValueError(f"Configuration resource directory exceeds the {MAX_YAML_DEPTH}-level depth limit")
    try:
        with os.scandir(descriptor) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name)
    except OSError as error:
        raise ValueError(f"Could not read configuration resource directory: {source / relative}") from error
    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    for entry in entries:
        display_path = source / relative / entry.name
        _count_visited_entry(writer, display_path)
        try:
            metadata = entry.stat(follow_symlinks=False)
        except OSError as error:
            raise ValueError(f"Could not inspect configuration resource: {display_path}") from error
        if _is_link_like(metadata):
            raise ValueError(f"Configuration resources cannot contain symlinks or reparse points: {display_path}")
        child_relative = relative / entry.name
        if stat.S_ISDIR(metadata.st_mode):
            try:
                child_descriptor = os.open(entry.name, directory_flags, dir_fd=descriptor)
            except OSError as error:
                raise ValueError(f"Could not securely open configuration resource directory: {display_path}") from error
            try:
                _copy_directory_descriptor(
                    child_descriptor,
                    source=source,
                    relative=child_relative,
                    destination=destination,
                    writer=writer,
                    yaml_only=yaml_only,
                    yaml_documents=yaml_documents,
                    depth=depth + 1,
                    budget=budget,
                )
            finally:
                os.close(child_descriptor)
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"Configuration resources must be regular files: {display_path}")
        suffix = Path(entry.name).suffix.lower()
        if yaml_only and suffix not in {".yml", ".yaml"}:
            continue
        try:
            file_descriptor = os.open(entry.name, file_flags, dir_fd=descriptor)
        except OSError as error:
            raise ValueError(f"Could not securely open configuration resource: {display_path}") from error
        try:
            payload = _read_open_file(
                file_descriptor,
                display_path,
                byte_limit=MAX_CONFIG_FILE_BYTES,
                label="configuration resource",
            )
        finally:
            os.close(file_descriptor)
        writer.add(destination / child_relative, payload)
        if suffix in {".yml", ".yaml"}:
            _, documents = _decode_yaml(payload, label=f"configuration resource {display_path}", budget=budget)
            yaml_documents.append(
                (display_path.as_posix(), _iter_yaml_documents(documents, label=f"configuration resource {display_path}"))
            )


def _target_triples() -> tuple[str, ...]:
    """List the platform keys a libraryPath mapping may use, most specific first.

    libc_ver names the C library, so a truthy name is not evidence of glibc.
    """
    system = platform.system().lower()
    machine = platform.machine().lower()
    architecture = {
        "arm64": "aarch64",
        "aarch64": "aarch64",
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "i386": "i686",
        "i686": "i686",
        "x86": "i686",
    }.get(machine)
    if architecture is None:
        raise ValueError(f"Unsupported native-parser architecture: {machine}")
    if system == "darwin":
        return (f"{architecture}-apple-darwin",)
    if system == "linux":
        library, _ = platform.libc_ver()
        if library == "glibc":
            return (f"{architecture}-unknown-linux-gnu",)
        if library == "musl":
            return (f"{architecture}-unknown-linux-musl",)
        return (f"{architecture}-unknown-linux-musl", f"{architecture}-unknown-linux-gnu")
    if system == "windows":
        return (f"{architecture}-pc-windows-msvc",)
    raise ValueError(f"Unsupported native-parser platform: {system}")


@dataclass
class ConfigSnapshot:
    source_path: Path | None
    bundle_root: Path
    inline_config_path: Path
    project_config_path: Path
    test_config_path: Path
    digest: str
    provenance: dict[str, Any]
    configured_rule_ids: tuple[str, ...]
    capabilities: dict[str, bool]
    native_library_hashes: dict[str, str]
    runtime_root: Path
    _closed: bool = field(default=False, init=False, repr=False)

    def close(self) -> None:
        if self._closed:
            return
        if not self.bundle_root.exists():
            self._closed = True
            return
        _relax_bundle_permissions(self.bundle_root)
        self.bundle_root.chmod(0o700)
        shutil.rmtree(self.bundle_root)
        self._closed = True
        try:
            self.runtime_root.rmdir()
        except OSError:
            pass

    def close_best_effort(self) -> None:
        try:
            self.close()
        except OSError:
            pass


def _relax_bundle_permissions(bundle_root: Path) -> None:
    for root, directories, files in os.walk(bundle_root, topdown=False):
        for name in files:
            try:
                (Path(root) / name).chmod(0o600)
            except OSError:
                pass
        for name in directories:
            try:
                (Path(root) / name).chmod(0o700)
            except OSError:
                pass


def private_runtime_root(working_directory: Path, allowed_roots: Sequence[Path]) -> Path:
    """Create the private bundle directory beneath the working directory.

    The allowed roots state which project trees callers may inspect, which is a
    separate question from where the server keeps its own runtime state. Requiring
    the working directory to appear in that list breaks the documented invocation,
    where the launcher runs from the server checkout while the inspected project
    lives elsewhere, and pushes operators to expose the server's own source as an
    inspection root. The bundle stays confined to one private directory here, and
    the reparse, symlink, and permission checks below are what protect it.
    """
    del allowed_roots
    resolved_working_directory = working_directory.resolve(strict=True)
    runtime_root = resolved_working_directory / RUNTIME_DIRECTORY_NAME
    try:
        runtime_root.mkdir(mode=0o700, exist_ok=True)
        metadata = runtime_root.lstat()
    except OSError as error:
        raise ValueError(f"Could not create private runtime directory: {runtime_root}") from error
    if _is_link_like(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"Private runtime path must be a real directory: {runtime_root}")
    runtime_root.chmod(0o700)
    return runtime_root


def create_config_snapshot(
    *,
    config_path: str | None,
    working_directory: Path,
    allowed_roots: Sequence[Path],
    trusted_native_libraries: Sequence[tuple[str, str]] = (),
) -> ConfigSnapshot:
    runtime_root = private_runtime_root(working_directory, allowed_roots)
    bundle_root = Path(mkdtemp(prefix="config-", dir=runtime_root))
    bundle_root.chmod(0o700)
    writer = _BundleWriter(bundle_root)
    source_path: Path | None = None
    source_digest: str | None = None
    config: dict[str, Any] = {"ruleDirs": []}
    config_directory = working_directory

    try:
        if config_path is not None:
            source_path = _secure_existing_path(
                config_path,
                base=working_directory,
                allowed_roots=allowed_roots,
                kind="file",
                label="Config path",
            )
            config_directory = source_path.parent
            source_payload = _read_file(
                source_path,
                byte_limit=MAX_CONFIG_FILE_BYTES,
                label="sgconfig.yml",
                boundary=_containing_root(source_path, allowed_roots),
            )
            source_digest = _sha256_bytes(source_payload)
            _, documents = _decode_yaml(source_payload, label="sgconfig.yml")
            if len(documents) != 1 or documents[0] is None:
                raise ValueError("sgconfig.yml must contain exactly one non-empty YAML document")
            config = _mapping(documents[0], label="sgconfig.yml")
            _reject_unknown_keys(config, _CONFIG_KEYS, label="sgconfig.yml")
            writer.add(Path("provenance/source-sgconfig.yml"), source_payload)

        trusted: dict[Path, str] = {}
        for raw_path, raw_digest in trusted_native_libraries:
            if not _SHA256_RE.fullmatch(raw_digest):
                raise ValueError(f"Trusted native library digest must be a 64-character SHA-256 value: {raw_digest!r}")
            path = _secure_existing_path(
                raw_path,
                base=working_directory,
                allowed_roots=allowed_roots,
                kind="file",
                label="Trusted native library",
            )
            digest = raw_digest.lower()
            if path in trusted and trusted[path] != digest:
                raise ValueError(f"Trusted native library has conflicting SHA-256 values: {path}")
            trusted[path] = digest

        node_budget = _NodeBudget()
        rule_dirs = _string_list(config.get("ruleDirs", []), label="sgconfig.yml ruleDirs", allow_empty=False)
        util_dirs = _string_list(config.get("utilDirs", []), label="sgconfig.yml utilDirs", allow_empty=False)
        copied_rule_dirs: list[str] = []
        copied_util_dirs: list[str] = []
        rule_documents: list[tuple[str, list[dict[str, Any]]]] = []
        util_documents: list[tuple[str, list[dict[str, Any]]]] = []

        for index, raw_dir in enumerate(rule_dirs):
            source = _secure_existing_path(
                raw_dir,
                base=config_directory,
                allowed_roots=allowed_roots,
                kind="directory",
                label="ruleDirs entry",
                require_relative=True,
                boundary=config_directory,
            )
            destination = Path(f"resources/rules/{index}")
            _copy_directory(
                source,
                destination=destination,
                writer=writer,
                yaml_only=True,
                yaml_documents=rule_documents,
                boundary=config_directory,
                budget=node_budget,
            )
            copied_rule_dirs.append(destination.as_posix())

        for index, raw_dir in enumerate(util_dirs):
            source = _secure_existing_path(
                raw_dir,
                base=config_directory,
                allowed_roots=allowed_roots,
                kind="directory",
                label="utilDirs entry",
                require_relative=True,
                boundary=config_directory,
            )
            destination = Path(f"resources/utils/{index}")
            _copy_directory(
                source,
                destination=destination,
                writer=writer,
                yaml_only=True,
                yaml_documents=util_documents,
                boundary=config_directory,
                budget=node_budget,
            )
            copied_util_dirs.append(destination.as_posix())

        configured_ids: list[str] = []
        for source_name, rule_docs in rule_documents:
            multiple = len(rule_docs) > 1
            for index, rule in enumerate(rule_docs):
                _validate_local_utility_cycles(rule, label=f"rule {source_name}")
                rule_id = rule.get("id")
                if rule_id is None or rule_id == "":
                    stem = Path(source_name).stem
                    rule_id = f"{stem}-{index}" if multiple else stem
                if not isinstance(rule_id, str) or not rule_id or "\0" in rule_id:
                    raise ValueError(f"Configured rule in {source_name} has an invalid id")
                if rule_id in configured_ids:
                    raise ValueError(f"Configured rules contain duplicate id {rule_id!r}")
                configured_ids.append(rule_id)

        global_utils: dict[str, object] = {}
        for source_name, utility_docs in util_documents:
            for utility in utility_docs:
                utility_id = utility.get("id")
                if not isinstance(utility_id, str) or not utility_id:
                    raise ValueError(f"Global utility rule in {source_name} has an invalid id")
                if utility_id in global_utils:
                    raise ValueError(f"Global utility rules contain duplicate id {utility_id!r}")
                global_utils[utility_id] = utility
                _validate_local_utility_cycles(utility, label=f"global utility {utility_id}")
        _reject_zero_progress_cycles(global_utils, label="Global utility rules")

        test_configs_value = config.get("testConfigs", [])
        test_configs = _list(test_configs_value, label="sgconfig.yml testConfigs")
        copied_test_configs: list[dict[str, Any]] = []
        for index, value in enumerate(test_configs):
            test = _mapping(value, label=f"sgconfig.yml testConfigs[{index}]")
            _reject_unknown_keys(test, _TEST_CONFIG_KEYS, label=f"sgconfig.yml testConfigs[{index}]")
            raw_test_dir = test.get("testDir")
            if not isinstance(raw_test_dir, str) or not raw_test_dir:
                raise ValueError(f"sgconfig.yml testConfigs[{index}].testDir must be a non-empty string")
            source = _secure_existing_path(
                raw_test_dir,
                base=config_directory,
                allowed_roots=allowed_roots,
                kind="directory",
                label="testConfigs testDir",
                require_relative=True,
                boundary=config_directory,
            )
            destination = Path(f"resources/tests/{index}")
            ignored_documents: list[tuple[str, list[dict[str, Any]]]] = []
            _copy_directory(
                source,
                destination=destination,
                writer=writer,
                yaml_only=False,
                yaml_documents=ignored_documents,
                boundary=config_directory,
                budget=node_budget,
            )
            copied_test: dict[str, Any] = {"testDir": destination.as_posix()}
            snapshot_dir = test.get("snapshotDir")
            if snapshot_dir is not None:
                if (
                    not isinstance(snapshot_dir, str)
                    or not snapshot_dir
                    or Path(snapshot_dir).is_absolute()
                    or ".." in Path(snapshot_dir).parts
                ):
                    raise ValueError(f"sgconfig.yml testConfigs[{index}].snapshotDir must stay within testDir")
                snapshot_source = source / snapshot_dir
                if snapshot_source.exists():
                    _assert_no_symlink_components(snapshot_source, boundary=source, label="snapshotDir")
                    if not snapshot_source.is_dir():
                        raise ValueError(f"sgconfig.yml testConfigs[{index}].snapshotDir is not a directory")
                copied_test["snapshotDir"] = snapshot_dir
            copied_test_configs.append(copied_test)

        language_globs: dict[str, Any] | None = None
        if "languageGlobs" in config:
            language_globs = _mapping(config["languageGlobs"], label="sgconfig.yml languageGlobs")
            for language, globs in language_globs.items():
                if not language or "\0" in language:
                    raise ValueError("sgconfig.yml languageGlobs contains an invalid language name")
                _string_list(globs, label=f"sgconfig.yml languageGlobs.{language}", allow_empty=False)

        injections: list[Any] = []
        if "languageInjections" in config:
            injections = _list(config["languageInjections"], label="sgconfig.yml languageInjections")
            for index, value in enumerate(injections):
                injection = _mapping(value, label=f"sgconfig.yml languageInjections[{index}]")
                _reject_unknown_keys(
                    injection,
                    _INJECTION_KEYS,
                    label=f"sgconfig.yml languageInjections[{index}]",
                )
                host_language = injection.get("hostLanguage")
                if not isinstance(host_language, str) or not host_language or "\0" in host_language:
                    raise ValueError(f"sgconfig.yml languageInjections[{index}].hostLanguage must be a non-empty string")
                _mapping(injection.get("rule"), label=f"sgconfig.yml languageInjections[{index}].rule")
                for section in ("constraints", "utils", "transform"):
                    if section in injection:
                        _mapping(injection[section], label=f"sgconfig.yml languageInjections[{index}].{section}")
                _validate_local_utility_cycles(injection, label=f"sgconfig.yml languageInjections[{index}]")
                injected = injection.get("injected")
                if isinstance(injected, str):
                    if not injected or "\0" in injected:
                        raise ValueError(f"sgconfig.yml languageInjections[{index}].injected must be a non-empty string or list")
                else:
                    injected_languages = _string_list(
                        injected,
                        label=f"sgconfig.yml languageInjections[{index}].injected",
                        allow_empty=False,
                    )
                    if not injected_languages:
                        raise ValueError(f"sgconfig.yml languageInjections[{index}].injected must be a non-empty string or list")

        native_hashes: dict[str, str] = {}
        used_trusted: set[Path] = set()
        copied_custom_languages: dict[str, Any] = {}
        normalized_language_names: dict[str, str] = {}
        if "customLanguages" in config:
            custom_languages = _mapping(config["customLanguages"], label="sgconfig.yml customLanguages")
            targets = _target_triples()
            for name, value in custom_languages.items():
                if not name or "\0" in name:
                    raise ValueError("sgconfig.yml customLanguages contains an invalid language name")
                component = _safe_component(name)
                collision = normalized_language_names.setdefault(component, name)
                if collision != name:
                    raise ValueError(f"Custom languages {collision!r} and {name!r} both normalize to {component!r}; rename one")
                custom = _mapping(value, label=f"sgconfig.yml customLanguages.{name}")
                _reject_unknown_keys(custom, _CUSTOM_LANGUAGE_KEYS, label=f"sgconfig.yml customLanguages.{name}")
                extensions = _string_list(
                    custom.get("extensions"),
                    label=f"sgconfig.yml customLanguages.{name}.extensions",
                    allow_empty=False,
                )
                raw_library = custom.get("libraryPath")
                if isinstance(raw_library, str):
                    selected_library = raw_library
                elif isinstance(raw_library, dict):
                    platforms = _mapping(
                        cast(object, raw_library),
                        label=f"sgconfig.yml customLanguages.{name}.libraryPath",
                    )
                    selected = next((platforms[triple] for triple in targets if isinstance(platforms.get(triple), str)), None)
                    if not isinstance(selected, str):
                        raise ValueError(f"Custom language {name!r} has no native library for {' or '.join(targets)}")
                    selected_library = selected
                else:
                    raise ValueError(f"Custom language {name!r} must configure libraryPath")
                library_path = _secure_existing_path(
                    selected_library,
                    base=config_directory,
                    allowed_roots=allowed_roots,
                    kind="file",
                    label=f"Custom language {name!r} libraryPath",
                    require_relative=True,
                    boundary=config_directory,
                )
                expected_digest = trusted.get(library_path)
                if expected_digest is None:
                    raise ValueError(f"Custom language {name!r} requires --trusted-native-library {library_path} SHA256")
                library_payload = _read_file(
                    library_path,
                    byte_limit=MAX_NATIVE_LIBRARY_BYTES,
                    label=f"Custom language {name!r} native library",
                    boundary=config_directory,
                )
                actual_digest = _sha256_bytes(library_payload)
                if actual_digest != expected_digest:
                    raise ValueError(
                        f"Custom language {name!r} native library SHA-256 mismatch: expected {expected_digest}, got {actual_digest}"
                    )
                used_trusted.add(library_path)
                suffix = "".join(library_path.suffixes)
                library_destination = Path(f"resources/native/{_safe_component(name)}{suffix}")
                writer.add(library_destination, library_payload)
                native_hashes[name] = actual_digest

                copied_custom: dict[str, Any] = {
                    "libraryPath": library_destination.as_posix(),
                    "extensions": extensions,
                }
                for optional in ("languageSymbol", "metaVarChar", "expandoChar"):
                    if optional in custom:
                        optional_value = custom[optional]
                        if not isinstance(optional_value, str) or (optional in {"metaVarChar", "expandoChar"} and len(optional_value) != 1):
                            raise ValueError(f"sgconfig.yml customLanguages.{name}.{optional} is invalid")
                        copied_custom[optional] = optional_value
                if "outlineRules" in custom:
                    raw_outline = custom["outlineRules"]
                    if not isinstance(raw_outline, str) or not raw_outline:
                        raise ValueError(f"sgconfig.yml customLanguages.{name}.outlineRules is invalid")
                    outline_path = _secure_existing_path(
                        raw_outline,
                        base=config_directory,
                        allowed_roots=allowed_roots,
                        kind="file",
                        label=f"Custom language {name!r} outlineRules",
                        require_relative=True,
                        boundary=config_directory,
                    )
                    outline_payload = _read_file(
                        outline_path,
                        byte_limit=MAX_CONFIG_FILE_BYTES,
                        label=f"Custom language {name!r} outline rules",
                        boundary=config_directory,
                    )
                    _decode_yaml(outline_payload, label=f"Custom language {name!r} outline rules")
                    outline_destination = Path(f"resources/outlines/{_safe_component(name)}.yml")
                    writer.add(outline_destination, outline_payload)
                    copied_custom["outlineRules"] = outline_destination.as_posix()
                copied_custom_languages[name] = copied_custom

        unused_trusted = sorted(str(path) for path in set(trusted) - used_trusted)
        if unused_trusted:
            raise ValueError(f"Trusted native libraries are not referenced by sgconfig.yml: {', '.join(unused_trusted)}")

        language_configuration: dict[str, Any] = {}
        if copied_custom_languages:
            language_configuration["customLanguages"] = copied_custom_languages
        if language_globs is not None:
            language_configuration["languageGlobs"] = language_globs
        if injections:
            language_configuration["languageInjections"] = injections

        inline_config: dict[str, Any] = {"ruleDirs": [], **language_configuration}
        project_config: dict[str, Any] = {
            "ruleDirs": copied_rule_dirs,
            **({"utilDirs": copied_util_dirs} if copied_util_dirs else {}),
            **language_configuration,
        }
        test_config: dict[str, Any] = {
            "ruleDirs": copied_rule_dirs,
            **({"utilDirs": copied_util_dirs} if copied_util_dirs else {}),
            **({"testConfigs": copied_test_configs} if copied_test_configs else {}),
            **language_configuration,
        }
        inline_path = writer.add(Path("inline-sgconfig.yml"), _yaml_bytes(inline_config), resource=False)
        project_path = writer.add(Path("project-sgconfig.yml"), _yaml_bytes(project_config), resource=False)
        test_path = writer.add(Path("test-sgconfig.yml"), _yaml_bytes(test_config), resource=False)

        for root, directories, files in os.walk(bundle_root, topdown=False):
            for name in files:
                (Path(root) / name).chmod(0o400)
            for name in directories:
                (Path(root) / name).chmod(0o500)
        bundle_root.chmod(0o500)

        capabilities = {
            "inline_search": True,
            "outline": True,
            "configured_scan": bool(configured_ids),
            "configured_tests": bool(copied_test_configs),
            "custom_languages": bool(copied_custom_languages),
        }
        provenance: dict[str, Any] = {
            "source": str(source_path) if source_path is not None else None,
            "source_sha256": source_digest,
            "snapshot": "private-read-only",
            "resource_files": writer.files,
            "resource_bytes": writer.bytes,
            "configured_rule_ids": sorted(configured_ids),
            "native_library_sha256": dict(sorted(native_hashes.items())),
        }
        return ConfigSnapshot(
            source_path=source_path,
            bundle_root=bundle_root,
            inline_config_path=inline_path,
            project_config_path=project_path,
            test_config_path=test_path,
            digest=writer.digest(),
            provenance=provenance,
            configured_rule_ids=tuple(sorted(configured_ids)),
            capabilities=capabilities,
            native_library_hashes=native_hashes,
            runtime_root=runtime_root,
        )
    except BaseException:
        _relax_bundle_permissions(bundle_root)
        try:
            bundle_root.chmod(0o700)
            shutil.rmtree(bundle_root)
        except OSError:
            pass
        try:
            runtime_root.rmdir()
        except OSError:
            pass
        raise

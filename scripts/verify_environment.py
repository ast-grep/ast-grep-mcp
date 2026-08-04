from __future__ import annotations

import ast
import importlib
import inspect
import os
import sys
import textwrap
from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any, Final, cast

ROOT: Final = Path(__file__).resolve().parents[1]
EXPECTED_ENVIRONMENT: Final = (ROOT / ".venv").resolve()
EXPECTED_PYTHON: Final = (3, 14, 6)
SHELL_FENCE_LANGUAGES: Final = frozenset({"bash", "sh", "shell", "zsh"})
PATH_ENVIRONMENT_VARIABLES: Final = frozenset({"TMPDIR", "TEMP", "TMP", "UV_CACHE_DIR"})
PROCESS_CALLS: Final = frozenset(
    {
        "asyncio.create_subprocess_exec",
        "asyncio.create_subprocess_shell",
        "os.execv",
        "os.execve",
        "os.execvp",
        "os.execvpe",
        "os.popen",
        "os.spawnl",
        "os.spawnle",
        "os.spawnlp",
        "os.spawnlpe",
        "os.spawnv",
        "os.spawnve",
        "os.spawnvp",
        "os.spawnvpe",
        "os.system",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "subprocess.Popen",
        "subprocess.run",
        "main.run_ndjson_process",
        "main.run_text_process",
    }
)
SHELL_PROCESS_CALLS: Final = frozenset(
    {
        "asyncio.create_subprocess_shell",
        "os.popen",
        "os.system",
    }
)
TEMPORARY_APIS: Final = frozenset({"tempfile.mkdtemp", "tempfile.TemporaryDirectory"})
ARGV_PARAMETERS: Final = frozenset({"argv", "args", "command", "cmd", "program"})
REQUIRED_SYNC_FLAGS: Final = frozenset({"--locked", "--all-extras", "--no-python-downloads"})


def resolve_from_root(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _line_number(source: str, offset: int, line_offset: int) -> int:
    return line_offset + source.count("\n", 0, offset)


def _node_children(node: Any) -> Iterator[Any]:
    attributes = cast(dict[str, Any], vars(node))
    for value in attributes.values():
        if hasattr(value, "kind"):
            yield value
        elif isinstance(value, list):
            for item in cast(list[Any], value):
                if hasattr(item, "kind"):
                    yield item


def _walk_shell_nodes(node: Any) -> Iterator[Any]:
    yield node
    for child in _node_children(node):
        yield from _walk_shell_nodes(child)


def _split_assignment(word: str) -> tuple[str, str] | None:
    name, separator, value = word.partition("=")
    if not separator or not name:
        return None
    if not (name[0].isalpha() or name[0] == "_"):
        return None
    if not all(character.isalnum() or character == "_" for character in name):
        return None
    return name, value


def _path_is_external(value: str) -> bool:
    if not value or "$" in value or "%" in value:
        return True
    try:
        return not resolve_from_root(value).is_relative_to(ROOT)
    except OSError, RuntimeError, ValueError:
        return True


def _word_uses_external_temporary_path(word: str) -> bool:
    normalized = word.replace("\\", "/").lower()
    if normalized == "/tmp" or normalized.startswith("/tmp/"):
        return True
    if normalized == "/private/tmp" or normalized.startswith("/private/tmp/"):
        return True
    upper = word.upper()
    return any(marker in upper for marker in ("$TMPDIR", "${TMPDIR}", "$TEMP", "${TEMP}", "$TMP", "${TMP}", "%TEMP%", "%TMP%"))


def _word_uses_uv_cache_environment(word: str) -> bool:
    components = [component.lower() for component in word.replace("\\", "/").split("/") if component]
    for component in components:
        suffix = component.removeprefix("archive-v")
        if suffix != component and suffix.isdigit():
            return True
    for index, component in enumerate(components):
        if component != "uv":
            continue
        remainder = components[index + 1 : index + 3]
        if remainder[:1] == ["tools"] or remainder == ["cache", "tools"]:
            return True
    return False


def _assignment_policy_failures(name: str, value: str, *, location: str) -> list[str]:
    failures: list[str] = []
    if name == "VIRTUAL_ENV":
        failures.append(f"{location}: virtual environment override")
    if name == "UV_ISOLATED":
        failures.append(f"{location}: isolated run environment")
    if name == "UV_PROJECT_ENVIRONMENT" and value != ".venv":
        failures.append(f"{location}: alternate UV_PROJECT_ENVIRONMENT")
    if name in PATH_ENVIRONMENT_VARIABLES and _path_is_external(value):
        failures.append(f"{location}: external temporary path")
    if name == "UV_PROJECT_ENVIRONMENT" and _path_is_external(value):
        failures.append(f"{location}: external temporary path")
    return failures


def _executable_basename(value: str) -> str:
    return value.replace("\\", "/").rsplit("/", 1)[-1].lower()


def _is_python_executable(value: str) -> bool:
    executable = _executable_basename(value)
    if executable in {"python", "python.exe"}:
        return True
    suffix = executable.removeprefix("python")
    if suffix.endswith(".exe"):
        suffix = suffix[:-4]
    return bool(suffix) and all(character.isdigit() or character == "." for character in suffix)


def _command_argv(words: Sequence[str]) -> list[str]:
    """Strip the wrappers that stand between a shell word list and the real executable.

    Leading assignments, ``env`` with its own options, and the ``command`` and
    ``exec`` builtins all delay the executable. ``env -u NAME`` and ``env -C DIR``
    take an operand, so skipping only the flag would mistake that operand for the
    command being run.
    """
    argv = list(words)
    while argv:
        index = 0
        while index < len(argv) and _split_assignment(argv[index]) is not None:
            index += 1
        argv = argv[index:]
        if not argv:
            return argv
        executable = _executable_basename(argv[0])
        if executable == "env":
            argv = _env_operand(argv)
            continue
        if executable in {"command", "exec"}:
            index = 1
            while index < len(argv) and argv[index].startswith("-"):
                index += 1
            argv = argv[index:]
            continue
        return argv
    return argv


def _env_operand(argv: Sequence[str]) -> list[str]:
    operand_options = {"-u", "--unset", "-C", "--chdir", "-S", "--split-string"}
    index = 1
    while index < len(argv):
        word = argv[index]
        if _split_assignment(word) is not None:
            index += 1
            continue
        if word in operand_options:
            index += 2
            continue
        if word.startswith("-"):
            index += 1
            continue
        break
    return list(argv[index:])


def _uv_subcommand(argv: Sequence[str]) -> tuple[str, int] | None:
    """Identify the uv subcommand, skipping the operands that global options consume.

    ``uv --directory run sync`` names a directory called ``run``, so scanning for the
    first word that looks like a subcommand would read the operand instead.
    """
    subcommands = {"python", "run", "sync", "tool", "venv"}
    global_operand_options = {"--directory", "--project", "--config-file", "--cache-dir", "--python", "--color", "--managed-python"}
    index = 1
    while index < len(argv):
        word = argv[index]
        if word in global_operand_options:
            index += 2
            continue
        if word.startswith("-"):
            index += 1
            continue
        if word in subcommands:
            return word, index
        index += 1
    return None


def _argv_policy_failures(argv: Sequence[str], *, location: str) -> list[str]:
    failures: list[str] = []
    for word in argv:
        assignment = _split_assignment(word)
        if assignment is not None:
            failures.extend(_assignment_policy_failures(*assignment, location=location))
        if _word_uses_external_temporary_path(word):
            failures.append(f"{location}: external temporary path")
        if _word_uses_uv_cache_environment(word):
            failures.append(f"{location}: uv cache environment execution")
    command = _command_argv(argv)
    if not command:
        return list(dict.fromkeys(failures))
    executable = _executable_basename(command[0])
    if executable == "uvx":
        failures.append(f"{location}: cache-isolated tool execution")
    if executable == "virtualenv" or executable == "mktemp":
        label = "alternate environment creation" if executable == "virtualenv" else "temporary workspace creation"
        failures.append(f"{location}: {label}")
    if _is_python_executable(executable) and any(
        command[index] == "-m" and command[index + 1] in {"venv", "virtualenv"} for index in range(len(command) - 1)
    ):
        failures.append(f"{location}: alternate environment creation")
    if executable == "uv":
        located = _uv_subcommand(command)
        if located is not None:
            subcommand, index = located
            uv_arguments = command[1:]
            if subcommand == "venv":
                failures.append(f"{location}: alternate environment creation")
            elif subcommand == "tool":
                failures.append(f"{location}: cache-isolated tool execution")
            elif subcommand == "python" and index + 1 < len(command) and command[index + 1] == "install":
                failures.append(f"{location}: managed Python installation")
            elif subcommand == "run":
                forbidden = {"--active", "--isolated", "--with", "--with-requirements", "-w"}
                if any(
                    argument in forbidden or argument.startswith("--with=") or argument.startswith("--with-requirements=")
                    for argument in uv_arguments
                ):
                    label = "active environment override" if "--active" in uv_arguments else "isolated run environment"
                    failures.append(f"{location}: {label}")
            elif subcommand == "sync":
                if "--active" in uv_arguments:
                    failures.append(f"{location}: active environment override")
                if "--no-build-isolation" in uv_arguments:
                    failures.append(f"{location}: disabled build isolation")
                if "--check" not in uv_arguments and not REQUIRED_SYNC_FLAGS.issubset(uv_arguments):
                    failures.append(f"{location}: incomplete synchronization")
            if subcommand == "run" and "--no-sync" not in uv_arguments:
                failures.append(f"{location}: implicit synchronization")
    return list(dict.fromkeys(failures))


def _shell_command_words(node: Any) -> list[str]:
    words: list[str] = []
    for part in cast(list[Any], getattr(node, "parts", [])):
        if getattr(part, "kind", None) in {"assignment", "word"}:
            words.append(cast(str, part.word))
    return words


def shell_policy_failures(source: str, *, label: str, line_offset: int = 1) -> list[str]:
    try:
        bashlex = cast(Any, importlib.import_module("bashlex"))
        roots = cast(list[Any], bashlex.parse(source))
    except Exception as error:
        return [f"{label}:{line_offset}: invalid shell syntax: {error}"]
    failures: list[str] = []
    for root in roots:
        for node in _walk_shell_nodes(root):
            if getattr(node, "kind", None) != "command":
                continue
            position = cast(tuple[int, int], node.pos)
            location = f"{label}:{_line_number(source, position[0], line_offset)}"
            words = _shell_command_words(node)
            failures.extend(_argv_policy_failures(words, location=location))
            command = _command_argv(words)
            if len(command) >= 3 and _executable_basename(command[0]) in {"bash", "sh", "zsh"} and command[1] in {"-c", "-lc"}:
                failures.extend(shell_policy_failures(command[2], label=label, line_offset=_line_number(source, position[0], line_offset)))
            if len(command) >= 3 and _is_python_executable(command[0]) and command[1] == "-c":
                failures.extend(
                    python_source_policy_failures(command[2], label=label, line_offset=_line_number(source, position[0], line_offset))
                )
    return list(dict.fromkeys(failures))


def _dotted_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        if parent is not None:
            return f"{parent}.{node.attr}"
    return None


def _import_aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name.split(".", 1)[0]] = item.name
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            for item in node.names:
                aliases[item.asname or item.name] = f"{node.module}.{item.name}"
    return aliases


def _resolve_dotted_name(node: ast.expr, aliases: Mapping[str, str]) -> str | None:
    name = _dotted_name(node)
    if name is None:
        return None
    head, separator, tail = name.partition(".")
    resolved = aliases.get(head, head)
    return f"{resolved}.{tail}" if separator else resolved


def _argv_argument(node: ast.Call, name: str) -> ast.expr | None:
    """Return the expression carrying argv, located through the callee's real signature."""
    located = _argv_parameter(name)
    if located is None:
        return None
    parameter, index, variadic = located
    if variadic and len(node.args) > index:
        return ast.List(elts=list(node.args[index:]), ctx=ast.Load())
    if len(node.args) > index:
        return node.args[index]
    for keyword in node.keywords:
        if keyword.arg == parameter:
            return keyword.value
    return None


@cache
def _argv_parameter(name: str) -> tuple[str, int, bool] | None:
    """Name and position the parameter carrying argv, read from the callee's own signature.

    Argv is not always the first argument: ``os.execv`` takes a path before it and
    ``os.spawnv`` a mode and a file, so binding position zero inspects the wrong
    expression. Functions that collect argv through ``*popenargs`` expose no such
    parameter and forward it to a wrapped callable, which is resolved through that
    wrapper. ``asyncio.create_subprocess_exec`` spreads argv across a leading
    program and a variadic remainder, which is gathered back into one list.
    """
    module_name, _, attribute = name.rpartition(".")
    if not module_name:
        return None
    try:
        target = cast(Callable[..., Any], getattr(importlib.import_module(module_name), attribute))
        parameters = list(inspect.signature(target).parameters.values())
    except ImportError, AttributeError, TypeError, ValueError:
        return None
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.VAR_POSITIONAL}
    ]
    for index, parameter in enumerate(positional):
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL and index == 0:
            delegate = _forwarded_callee(target, parameter.name)
            return _argv_parameter(f"{module_name}.{delegate}") if delegate else None
        if parameter.name in ARGV_PARAMETERS:
            return parameter.name, index, parameter.kind is inspect.Parameter.VAR_POSITIONAL
    return None


def _forwarded_callee(target: Callable[..., Any], collected: str) -> str | None:
    """Name the callable that receives ``*collected``, read from the callee's own source."""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(target)))
    except OSError, TypeError, SyntaxError, ValueError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for argument in node.args:
            if isinstance(argument, ast.Starred) and _dotted_name(argument.value) == collected:
                return _dotted_name(node.func)
    return None


def _literal_argv(node: ast.expr) -> list[str] | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    values: list[str] = []
    for item in node.elts:
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            values.append(item.value)
        else:
            values.append("")
    return values


def python_source_policy_failures(source: str, *, label: str, line_offset: int = 1) -> list[str]:
    try:
        tree = ast.parse(source, filename=label)
    except SyntaxError as error:
        line = line_offset + (error.lineno or 1) - 1
        return [f"{label}:{line}: invalid Python syntax: {error.msg}"]
    aliases = _import_aliases(tree)
    failures: list[str] = []
    for node in ast.walk(tree):
        location = f"{label}:{line_offset + getattr(node, 'lineno', 1) - 1}"
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _word_uses_external_temporary_path(node.value):
                failures.append(f"{location}: external temporary path")
            if _word_uses_uv_cache_environment(node.value):
                failures.append(f"{location}: uv cache environment execution")
        if not isinstance(node, ast.Call):
            continue
        name = _resolve_dotted_name(node.func, aliases)
        if name in TEMPORARY_APIS:
            failures.append(f"{location}: temporary workspace creation")
            continue
        if name not in PROCESS_CALLS:
            continue
        argument = _argv_argument(node, name)
        if argument is None:
            continue
        argv = _literal_argv(argument)
        if argv is None:
            continue
        if name in SHELL_PROCESS_CALLS and len(argv) == 1:
            failures.extend(shell_policy_failures(argv[0], label=label, line_offset=line_offset + node.lineno - 1))
        else:
            failures.extend(_argv_policy_failures(argv, location=location))
    return list(dict.fromkeys(failures))


def _markdown_policy_failures(path: Path) -> list[str]:
    from markdown_it import MarkdownIt

    source = path.read_text(encoding="utf-8")
    failures: list[str] = []
    for token in MarkdownIt("commonmark").parse(source):
        info = token.info.strip().split(maxsplit=1)
        if token.type != "fence" or not info or info[0].lower() not in SHELL_FENCE_LANGUAGES:
            continue
        start = token.map[0] + 2 if token.map is not None else 1
        failures.extend(shell_policy_failures(token.content, label=str(path.relative_to(ROOT)), line_offset=start))
    return failures


def _mapping_environment_failures(value: object, *, label: str) -> list[str]:
    failures: list[str] = []
    if isinstance(value, dict):
        mapping = cast(dict[object, object], value)
        environment = mapping.get("env")
        if isinstance(environment, dict):
            for key, item in cast(dict[object, object], environment).items():
                if isinstance(key, str):
                    failures.extend(_assignment_policy_failures(key, str(item), location=label))
        for item in mapping.values():
            failures.extend(_mapping_environment_failures(item, label=label))
    elif isinstance(value, list):
        for item in cast(list[object], value):
            failures.extend(_mapping_environment_failures(item, label=label))
    return failures


def _yaml_run_nodes(node: Any) -> Iterator[Any]:
    import yaml

    if isinstance(node, yaml.MappingNode):
        for key, value in node.value:
            if isinstance(key, yaml.ScalarNode) and key.value == "run" and isinstance(value, yaml.ScalarNode):
                yield value
            yield from _yaml_run_nodes(value)
    elif isinstance(node, yaml.SequenceNode):
        for item in node.value:
            yield from _yaml_run_nodes(item)


def _workflow_policy_failures(path: Path) -> list[str]:
    import yaml

    from config_snapshot import load_strict_yaml_documents

    source = path.read_text(encoding="utf-8")
    label = str(path.relative_to(ROOT))
    try:
        documents = load_strict_yaml_documents(source, label=label, max_documents=1)
        yaml_api = cast(Any, yaml)
        composed = yaml_api.compose(source, Loader=yaml.SafeLoader)
    except ValueError as error:
        return [f"{label}:1: invalid workflow YAML: {error}"]
    failures = _mapping_environment_failures(documents[0] if documents else None, label=label)
    if composed is not None:
        for node in _yaml_run_nodes(composed):
            failures.extend(shell_policy_failures(cast(str, node.value), label=label, line_offset=node.start_mark.line + 1))
    return list(dict.fromkeys(failures))


def static_policy_failures() -> list[str]:
    failures: list[str] = []
    for path in sorted(ROOT.glob("*.md")):
        failures.extend(_markdown_policy_failures(path))
    for path in sorted((ROOT / ".github" / "workflows").glob("*.y*ml")):
        failures.extend(_workflow_policy_failures(path))
    for path in sorted((ROOT / "scripts").rglob("*")):
        if not path.is_file() or path == Path(__file__).resolve() or "__pycache__" in path.parts:
            continue
        relative = str(path.relative_to(ROOT))
        if path.suffix == ".py":
            failures.extend(python_source_policy_failures(path.read_text(encoding="utf-8"), label=relative))
        elif path.suffix in {"", ".sh"}:
            failures.extend(shell_policy_failures(path.read_text(encoding="utf-8"), label=relative))
        elif path.suffix == ".ps1":
            failures.append(f"{relative}:1: unsupported executable verification surface")
    return list(dict.fromkeys(failures))


def repository_environment_failures() -> list[str]:
    failures: list[str] = []
    environment_entry = ROOT / ".venv"
    if Path.cwd().resolve() != ROOT:
        failures.append(f"working directory must be {ROOT}")
    if environment_entry.is_symlink():
        failures.append(f"Python environment must not be a symlink: {environment_entry}")
    if not EXPECTED_ENVIRONMENT.is_relative_to(ROOT):
        failures.append(f"Python environment must remain inside {ROOT}")
    if Path(sys.prefix).resolve() != EXPECTED_ENVIRONMENT:
        failures.append(f"Python environment must be {EXPECTED_ENVIRONMENT}")
    if sys.version_info[:3] != EXPECTED_PYTHON:
        failures.append(f"Python version must be {'.'.join(map(str, EXPECTED_PYTHON))}")
    virtual_environment = os.environ.get("VIRTUAL_ENV")
    if virtual_environment is None or resolve_from_root(virtual_environment) != EXPECTED_ENVIRONMENT:
        failures.append(f"VIRTUAL_ENV must be {EXPECTED_ENVIRONMENT}")
    project_environment = os.environ.get("UV_PROJECT_ENVIRONMENT")
    if project_environment is not None and resolve_from_root(project_environment) != EXPECTED_ENVIRONMENT:
        failures.append(f"UV_PROJECT_ENVIRONMENT must be {EXPECTED_ENVIRONMENT}")
    if failures:
        return failures
    return static_policy_failures()


def main() -> None:
    failures = repository_environment_failures()
    if failures:
        raise SystemExit("repository environment verification failed:\n" + "\n".join(failures))
    print(f"verified repository environment: {EXPECTED_ENVIRONMENT}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
import shutil
import stat
from collections.abc import Callable
from pathlib import Path

from pytest import Config


def _clear_readonly_and_retry(function: Callable[[str], object], path: str, _: BaseException) -> None:
    """Clear the read-only bit and reattempt removal, per the shutil.rmtree docs.

    Directories additionally need the traversal bit restored on POSIX, so grant
    owner write plus execute rather than the docs' Windows-only S_IWRITE.
    """
    os.chmod(path, stat.S_IWRITE | stat.S_IREAD | (stat.S_IEXEC if os.path.isdir(path) else 0))
    function(path)


def pytest_unconfigure(config: Config) -> None:
    root = Path(__file__).resolve().parents[1]
    target = root / "test-runtime"
    try:
        metadata = target.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"Refusing to remove unexpected pytest runtime path: {target}")
    for directory, directories, files in os.walk(target, topdown=False):
        for name in files:
            _chmod_best_effort(Path(directory) / name, 0o600)
        for name in directories:
            path = Path(directory) / name
            if not path.is_symlink():
                _chmod_best_effort(path, 0o700)
    _chmod_best_effort(target, 0o700)
    shutil.rmtree(target, onexc=_clear_readonly_and_retry)


def _chmod_best_effort(path: Path, mode: int) -> None:
    try:
        path.chmod(mode)
    except OSError:
        pass

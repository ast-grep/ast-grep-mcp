from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

from pytest import Config


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
            (Path(directory) / name).chmod(0o600)
        for name in directories:
            path = Path(directory) / name
            if not path.is_symlink():
                path.chmod(0o700)
    target.chmod(0o700)
    shutil.rmtree(target)

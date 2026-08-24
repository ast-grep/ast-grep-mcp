from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

from ast_soleaux.worker import JsonLineWorker


def worker(command: str, cwd: Path) -> JsonLineWorker:
    return JsonLineWorker(
        command=(sys.executable, "-c", command),
        cwd=str(cwd),
        environment=dict(os.environ),
    )


def test_worker_close_falls_back_and_reaps_when_group_signal_is_denied(tmp_path: Path) -> None:
    instance = worker("import time; print('ready', flush=True); time.sleep(30)", tmp_path)
    instance.start()
    process = instance._process
    assert process is not None
    response = instance._responses.get(timeout=2)
    assert isinstance(response, str) and response.strip() == "ready"

    try:
        with patch(
            "ast_soleaux.worker._signal_process_group",
            side_effect=PermissionError(1, "Operation not permitted"),
        ) as signal_group:
            instance.close()
        signal_group.assert_called_once()
        assert process.poll() is not None
        assert instance._process is None
        assert instance._reader is None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)


def test_worker_close_allows_clean_exit_after_stdin_eof(tmp_path: Path) -> None:
    instance = worker("import sys; print('ready', flush=True); sys.stdin.read()", tmp_path)
    instance.start()
    process = instance._process
    assert process is not None
    response = instance._responses.get(timeout=2)
    assert isinstance(response, str) and response.strip() == "ready"

    try:
        with patch("ast_soleaux.worker._signal_process_group") as signal_group:
            instance.close()
        signal_group.assert_not_called()
        assert process.poll() == 0
        assert instance._process is None
        assert instance._reader is None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)

from __future__ import annotations

import difflib
import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ConflictPolicy = Literal["error", "overwrite", "skip"]


@dataclass(frozen=True)
class PlannedWrite:
    source: Path | None
    target: Path
    content: str


@dataclass(frozen=True)
class AppliedWrite:
    target: str
    source_digest: str | None
    output_digest: str
    changed: bool
    overwritten: bool
    diff: str | None


@dataclass(frozen=True)
class MutationBatchResult:
    applied: tuple[AppliedWrite, ...]
    skipped: tuple[str, ...]


class MutationService:
    def __init__(self, *, max_file_bytes: int, max_total_bytes: int) -> None:
        self.max_file_bytes = max_file_bytes
        self.max_total_bytes = max_total_bytes

    @staticmethod
    def _digest(payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _contained(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def apply(
        self,
        *,
        project: Path,
        writes: list[PlannedWrite],
        conflict_policy: ConflictPolicy,
        allow_source_overwrite: bool,
    ) -> MutationBatchResult:
        project = project.resolve(strict=True)
        if conflict_policy not in {"error", "overwrite", "skip"}:
            raise ValueError(f"Unsupported conflict policy: {conflict_policy}")
        normalized: list[tuple[PlannedWrite, Path, bytes | None, bytes]] = []
        skipped: list[str] = []
        total = 0
        seen: set[Path] = set()
        for write in writes:
            target = write.target.resolve(strict=False)
            if not self._contained(target, project):
                raise ValueError(f"Output path resolves outside project: {write.target}")
            if target in seen:
                raise ValueError(f"Duplicate output path: {target}")
            seen.add(target)
            source_is_target = False
            if write.source is not None:
                source = write.source.resolve(strict=True)
                if not self._contained(source, project):
                    raise ValueError(f"Source path resolves outside project: {write.source}")
                source_is_target = source == target
                if source_is_target and not allow_source_overwrite:
                    raise ValueError(f"Source overwrite is not enabled: {target}")
            output_bytes = write.content.encode("utf-8")
            if len(output_bytes) > self.max_file_bytes:
                raise ValueError(f"Output exceeds the per-file byte limit: {target}")
            total += len(output_bytes)
            if total > self.max_total_bytes:
                raise ValueError("Outputs exceed the aggregate byte limit")
            previous = target.read_bytes() if target.exists() else None
            if previous is not None and not source_is_target and conflict_policy == "error":
                raise FileExistsError(f"Output already exists: {target}")
            if previous is not None and not source_is_target and conflict_policy == "skip":
                skipped.append(target.relative_to(project).as_posix())
                continue
            normalized.append((write, target, previous, output_bytes))

        staged: list[tuple[Path, Path]] = []
        backups: list[tuple[Path, Path]] = []
        applied: list[AppliedWrite] = []
        try:
            for _, target, _, output_bytes in normalized:
                target.parent.mkdir(parents=True, exist_ok=True)
                descriptor, raw_staged = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
                staged_path = Path(raw_staged)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(output_bytes)
                    stream.flush()
                    os.fsync(stream.fileno())
                staged.append((target, staged_path))
            for _, target, previous, output_bytes in normalized:
                staged_path = next(path for staged_target, path in staged if staged_target == target)
                backup: Path | None = None
                if target.exists():
                    descriptor, raw_backup = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".bak", dir=target.parent)
                    os.close(descriptor)
                    backup = Path(raw_backup)
                    shutil.copy2(target, backup)
                    backups.append((target, backup))
                os.replace(staged_path, target)
                if backup is not None:
                    shutil.copymode(backup, target)
                previous_text = previous.decode("utf-8") if previous is not None else ""
                output_text = output_bytes.decode("utf-8")
                diff = None
                if previous != output_bytes:
                    diff = "\n".join(
                        difflib.unified_diff(
                            previous_text.splitlines(),
                            output_text.splitlines(),
                            fromfile=f"a/{target.relative_to(project).as_posix()}",
                            tofile=f"b/{target.relative_to(project).as_posix()}",
                            lineterm="",
                        )
                    )
                applied.append(
                    AppliedWrite(
                        target=target.relative_to(project).as_posix(),
                        source_digest=self._digest(previous) if previous is not None else None,
                        output_digest=self._digest(output_bytes),
                        changed=previous != output_bytes,
                        overwritten=previous is not None,
                        diff=diff,
                    )
                )
        except BaseException:
            for target, backup in reversed(backups):
                if backup.exists():
                    os.replace(backup, target)
            raise
        finally:
            for _, path in staged:
                path.unlink(missing_ok=True)
            for _, path in backups:
                path.unlink(missing_ok=True)
        return MutationBatchResult(applied=tuple(applied), skipped=tuple(skipped))

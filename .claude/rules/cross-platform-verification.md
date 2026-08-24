---
paths:
  - "**/*.py"
  - "pyproject.toml"
  - ".python-version"
  - "uv.lock"
  - ".github/workflows/*.{yml,yaml}"
---

# Cross-platform verification

`uv run --no-sync` reuses the populated `.venv`, and default type-check and test runs describe only the host platform. Verification must also cover the following:

- Run `uv run --no-sync mypy --platform win32 ast_soleaux execution-sidecar scripts` and `uv run --no-sync pyright --pythonplatform Windows` because platform-specific stubs change which branches resolve. Repeat with `linux`/`Linux` and `darwin`/`Darwin`.
- Treat byte offsets, digests, and file contents as line-ending dependent. Fixtures write through `newline=""`, and assertions derive offsets from on-disk bytes rather than LF-only constants.
- Confirm a clean environment can build the project. `uv run --no-sync` alone does not exercise build dependencies.
- Treat `platform.libc_ver` as the C-library name, not a glibc predicate. Alpine reports `("musl", "1")`, and a generic scan may report `("libc", ...)`; test the name and measure on both `python:3.14-alpine` and `python:3.14-slim` when that behavior changes.

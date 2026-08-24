---
paths:
  - "ast_soleaux/**/*.py"
  - "execution-sidecar/**/*.py"
  - "tests/**/*.py"
---

# Linux testing

Use a container to reproduce platform-dependent behavior locally. The container is a separate machine using this checkout, not an alternate repository environment.

```bash
docker run --rm -v "$PWD":/w -w /w \
  -e UV_PROJECT_ENVIRONMENT=/tmp/ast-soleaux-venv \
  -e PYTHONDONTWRITEBYTECODE=1 \
  python:3.14-slim sh -c '
pip install -q uv
uv sync --locked --all-extras --dev --no-python-downloads
uv run --no-sync pytest \
  tests/test_config_snapshot.py tests/test_mutation.py tests/test_execution_sandbox.py \
  -k "not real_docker" -p no:cacheprovider -o addopts="" -q'
```

`-o addopts=""` drops coverage flags that the container does not install, and `-p no:cacheprovider` prevents a cache write into the working tree. Apply the defect-fixing rule's negative-control proof to security checks.

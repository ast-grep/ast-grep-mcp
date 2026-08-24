---
paths:
  - "tests/**/*.py"
  - "pyproject.toml"
---

# Pytest runtime cleanup

`tests/conftest.py` clears `test-runtime/` after a run. Snapshot bundles are deliberately read-only, so a manual `rm -rf test-runtime` can fail partway and poison later collection.

When manual cleanup is necessary, use:

```bash
chmod -R u+rwX test-runtime && rm -rf test-runtime
```

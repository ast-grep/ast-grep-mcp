---
paths:
  - "**/*.py"
  - "pyproject.toml"
---

# Python code

- Preserve the repository-owned Python toolchain and its configured checks.
- Bind subprocess arguments from the real signature: argv is the first argument to `subprocess.run`, the second to `os.execv`, and the third to `os.spawnv`.

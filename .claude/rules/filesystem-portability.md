---
paths:
  - "ast_soleaux/**/*.py"
  - "execution-sidecar/**/*.py"
  - "tests/test_unit.py"
  - "tests/test_config_snapshot.py"
---

# Filesystem portability

Type checking cannot prove these runtime behaviors. When a change touches path traversal, permissions, or process teardown, execute the affected test on Linux using the repository's Linux-testing rule.

- Inode numbers do not identify a path across a change. Linux may immediately reuse a removed directory's inode, and Windows does not report a stable inode across calls. Compare file type, which a link cannot match.
- `chmod` sets only the read-only flag on Windows; guard other mode assertions with `os.name == "posix"`.
- `os.scandir` receives a `Path` for source trees and a `str` or descriptor elsewhere, and entries arrive in filesystem order. Patches must accept every form and tests must not assume entry order.
- Resolving a path replaced with a symlink follows the target, so match the entry name rather than the pre-swap resolved path.
- Windows defines neither `O_NOFOLLOW` nor `dir_fd`; validate again after holding the descriptor, and document that a link restored before the recheck still passes.

# Fixing a reported defect

- Reproduce the defect before fixing it.
- Confirm the reproduction fails when the fix is removed. For a security fix, run that negative control on the same platform used for the passing proof.
- Fix the defect class rather than one observed instance. A guard keyed on name plus suffix still collides when suffixes differ, and a symlink-only rejection still admits a junction.
- Check adjacent guards; prior incomplete fixes left an existing predicate uncalled.
- When dismissing a review finding, reply on its thread with the evidence.

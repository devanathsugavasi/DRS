# Dead duplicates (archived 2026-06-16)

These files were **superseded duplicates** not used by the live pipeline. Moved
here (instead of deleted) so history and any reference value are preserved.
Removing them from `core/` and `tests/` removes false-confidence test runs and
import ambiguity.

| Archived file | Superseded by (LIVE) | Used by live code? |
|---|---|---|
| `lbw.py` (`LBWDecisionEngine`) | `core/lbw_engine.py` | No — `core/drs_decision.py` imports `lbw_engine` |
| `tracker.py` | `core/ball_tracker.py` | No — `core/integration.py` imports `ball_tracker` |
| `sync.py` | `core/synchronization.py` | No — `integration.py` + `api_server.py` import `synchronization` |
| `test_tracker.py` | — | Only tested the dead `tracker.py` |
| `test_sync.py` | — | Only tested the dead `sync.py` |

`test_lbw_engine.py` was kept in `tests/` (it tests the live engine).

## Follow-up (not done here)
The live `ball_tracker.py` and `synchronization.py` have **no tests** — the only
tracker/sync tests were the dead ones archived here. Worth adding real tests for
the live modules.

## To restore
`git mv archive/dead_duplicates/<file> <original/path>`

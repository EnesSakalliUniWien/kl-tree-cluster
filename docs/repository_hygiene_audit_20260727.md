# Repository Hygiene and Completion Audit — 2026-07-27

## Executive Conclusion

This checkout is the most advanced branch in the main local development
lineage, but it is not a finished or canonical release.

- `codex/tree-consensus-validation` at `a7f0f1c9` contains both local `main`
  and `dev` and is respectively 291 and 27 commits ahead of them.
- The branch has no upstream and does not exist on `origin`; the canonical
  remote default is still `main` at `c87a6f13`.
- At least one remote development branch has commits not contained here, so
  the repository does not yet have a single consolidated final line.
- There are no release tags and no GitHub Actions workflows.
- The maintained mathematical record still labels selected-hierarchy
  calibration, categorical transfer, and method-constant promotion as open.

If “finished one” means the newest coherent local implementation, this is the
best candidate found. If it means the published, merged, tagged, or
publication-ready repository, the answer is no.

## Git and Release State

| Check | Result |
| --- | --- |
| Worktree before this cleanup | Clean |
| Current branch | `codex/tree-consensus-validation` |
| Current commit | `a7f0f1c9` (2026-07-14) |
| Remote branch with the same name | Absent |
| `main` ancestry | Contained; current branch is 291 commits ahead |
| `dev` ancestry | Contained; current branch is 27 commits ahead |
| Release tags | None |
| Package version | `0.1.0` |
| Latest changelog section | 2026-02-17; stale relative to July work |

The current branch should not be called final until its divergent work is
reviewed, pushed, integrated into an agreed canonical branch, and tagged.
Those actions change shared Git state and were deliberately not performed by
this audit.

### Development merge decision

The reviewed organization candidate is technically fast-forwardable to `dev`:
local `dev` and `origin/dev` both resolve to `8aaa2615`, that commit is the
candidate's exact merge base, an isolated candidate index passes source diff
checks, and the complete checkout passes the health gate. There is no merge
conflict to resolve.

Do not merge the current `HEAD` as-is, because the 189-file organization pass
is still in the worktree (109 modified files, 50 deletions, and 19 untracked
path groups). It must be reviewed as one commit or a deliberate series first.
The resulting `dev` payload would span 2,439 files relative to current `dev`,
including 865 already-committed top-level `results/` files. That evidence scope
is intentional-looking and test-safe, but its retention policy needs explicit
owner acceptance before publication; it is not a source-code conflict.

## Executable Health

The implementation is healthy as a development checkout:

- `uv lock --check` resolves the locked environment.
- Package build succeeds for both wheel and source distribution.
- `quick_start.py` completes with six clusters and ARI `0.9530`.
- Pytest collects 1,252 tests after the organization pass added direct
  space-separation coverage.
- The purpose-ordered suite passes all 1,252 tests after adding the previously
  omitted `tests/tree/` and `tests/wiki/` targets.
- Ruff passes across project-owned Python code; the BranchArchitect submodule
  is excluded because it has its own quality policy.
- Wiki lint passes across the complete durable-memory layer.

The passing suite emits non-fatal warnings for empty-slice diagnostics,
disconnected spectral graphs, and duplicate sample distances in graphtools.
These are follow-up hygiene items rather than current test failures. The
deprecated Scanpy version access found during the first pass was repaired.

## Cleanup Applied

- Consolidated Pytest configuration in `pyproject.toml` and removed the
  duplicate `pytest.ini`.
- Added `make check`, `make lint`, and `make test` as a small repository-health
  interface.
- Corrected the ordered test runner and test documentation so every collected
  test directory is included.
- Added the root MIT license already promised by `README.md` and connected it
  to package metadata.
- Added an explicit research-software status section to `README.md`.
- Excluded the pinned `vendor/BranchArchitect` submodule from root Ruff checks.
- Applied Ruff's safe import ordering to 34 project-owned benchmark and
  quick-start files.
- Replaced deprecated `scanpy.__version__` access with package metadata.
- Grouped endotype/GO, scRNA, and MNIST commands under `applications/`, with
  scRNA follow-up analysis and plot composition separated by responsibility.
- Promoted reusable cosine, invariant/equivariant, and diffusion separation to
  `tree_break_selection/space_separation/` and generic plot engines to
  `tree_break_selection/plot/`.
- Reduced root `scripts/` to maintenance/external-request helpers; moved the
  discrete replay to benchmark diagnostics and explanatory paper figures to
  `manuscript/tools/figures/`.

## Remaining Hygiene Risks

### High priority

1. Agree the canonical integration branch, reconcile divergent remote work,
   publish it, and make the release lineage explicit.
2. Define statistical acceptance criteria for selected-hierarchy calibration,
   categorical transfer, and promotion of method constants.
3. Decide which retained evidence belongs in Git, Git LFS, or release storage
   before attempting any destructive history cleanup.

### Before a public release

4. Add CI for the lean test/lint gate and a documented optional full-extras
   job.
5. Refresh `CHANGELOG.md`, choose a release version, and create a signed or
   annotated tag after integration.
6. Resolve or suppress the known test warnings with evidence, especially the
   empty-slice diagnostics.
7. Decide whether the branch's 865 tracked top-level `results/` files (about
   215 MB of Git blob content) belong in Git, Git LFS, or release storage, then
   make the documented path policy match that decision. The checkout also has
   3,465 ignored local result files, bringing the on-disk directory to 1.7 GB;
   those ignored files are not part of the merge. `dev` currently tracks no
   top-level `results/` files.

## Commands Used

```bash
git status --porcelain=v2 --branch
git ls-remote --heads --tags origin
git rev-list --left-right --count main...HEAD
git rev-list --left-right --count dev...HEAD
git submodule status
uv lock --check
uv build
uv run --no-sync python quick_start.py
uv run --no-sync pytest --collect-only -q
make check
git count-objects -vH
make wiki-lint
uv run --no-sync deptry applications tree_break_selection benchmarks
uv run --no-sync vulture applications benchmarks scripts tree_break_selection --min-confidence 90
```

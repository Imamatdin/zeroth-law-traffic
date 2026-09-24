# Repository instructions

Before any change, read `docs/task_spec.md`, `docs/ARCHITECTURE.md`, and `private/PLAN.md` when it exists. The organizer spec in `docs/task_spec.md` is the source of truth; where it conflicts with the plan or any other document, the spec wins. Implement only the plan step you are asked for and nothing past its gate.

`private/` is local-only (excluded in `.git/info/exclude`). Plans, handoffs, decision logs and working notes go there and are never committed.

`run_submission.py` and `evaluate.py` must stay byte-identical to the starter kit. Never edit them; call them.

Keep `main` runnable. Read the official starter kit before changing submission-facing signatures, output formats, or scoring assumptions. Commit the starter kit unchanged first, then make adaptations in later commits.

`RiskEstimator.step` (Part B) must never read the video file, frames after time t, or any Part A output. Part A may read Part B output.

Teammate work lives in `contrib/<name>/` on `team/<name>` branches (see `docs/TEAM_BRIEF.md`). Do not edit it; pull it into the pipeline through the interfaces it defines.

Keep rich internal evidence separate from the official output adapter. Store videos, caches, weights, and generated outputs outside Git. Do not invent labels, geometry, scores, deadlines, or evaluator results. Preserve unrelated work and report checks actually run.

# Spec: Core-computation placement for the analysis framework

Status: **proposed** · Scope: `gigaseal/analysis/`, `gigaseal/QC.py`, legacy core modules
Owner action: author will handle actual code placement/moves.

## Context

- The modular framework (`gigaseal/analysis/`) is the long-term source of truth.
  The built-in analysis modules live directly under `gigaseal/analysis/`
  (e.g. `spike.py`, `qc.py`) and currently delegate to the frozen
  legacy modules (`featureExtractor.py`, `patch_subthres.py`). Framework
  internals (`AnalysisBase`, registry, result, runner) live in
  `gigaseal/analysis/core/`.
- `gigaseal/_legacy/readme.md` records the intent to move `featureExtractor.py`
  and `patch_subthres.py` into `_legacy/` at 1.0b.
- **Consumers:** end users run the **GUI** — internal module paths do not affect
  them provided the GUI is re-wired. The only *programmatic* (import-level)
  consumer is the maintainer. There is therefore **no internal-API-stability
  contract** to preserve; organization can be optimized purely for long-term
  readability.

## Decision

**Option A — colocate computation with its analysis.**

- **Single-analysis helpers live in the analysis's own module**, as module-level
  functions alongside the `AnalysisBase` subclass. Example: QC's
  `find_zero`, `find_baseline`, `compute_rms`, `compute_vm_drift`, `run_qc`
  live in `analysis/qc.py` next to `QcAnalysis`.
- **Shared helpers** (used by >1 analysis) live in a neutral location, **not**
  duplicated:
  - cross-cutting sweep utilities → `gigaseal/patch_utils.py`
    (e.g. `crop_spikes`, `build_running_bin`);
  - shared computation that is clearly "analysis core" → the
    `gigaseal/analysis/core/` subpackage, which already houses the framework
    internals (`base`, `registry`, `result`, `runner`).
- The `AnalysisBase.analyze()` body stays a **thin** call into these functions;
  do not inline computation into the class.

### Rationale

- Reads top-to-bottom as "here is everything for analysis X" — best readability
  for the code that survives 1.0.
- The layering objection (core → framework) is temporary: the current core
  consumer (`featureExtractor.py`) is itself scheduled for `_legacy/`.
- No compatibility cost, since the GUI is re-wired and the maintainer is the
  sole programmatic consumer.

## Constraints that still bind (mechanical, not stylistic)

1. **No import-time circular imports.** `analysis/spike.py` imports
   `featureExtractor` **lazily inside** `analyze()`; keep any legacy imports lazy
   so relocation never creates a cycle. Pure-numpy helpers (QC) are safe.
2. **Pickling / multiprocessing.** `run_batch(n_jobs>1)` uses
   `ProcessPoolExecutor`; every built-in module under `analysis/` must be
   importable at module level (no closures) and picklable.
3. **IPFX import hygiene.** Never import IPFX (or other optional deps) at the top
   of `analysis/**`; import inside `analyze()` or guard it, so the framework
   imports without optional deps installed.

## QC pilot (reference implementation for the other analyses)

1. Move `find_zero`, `find_baseline`, `compute_rms`, `compute_vm_drift`,
   `run_qc` from `gigaseal/QC.py` into `gigaseal/analysis/qc.py`
   (module-level, above `QcAnalysis`).
2. Fill `QcAnalysis.analyze()` to apply the optional `filter`, call
   `run_qc(realY=y, realC=c)`, and map the returned
   `[mean_rms, max_rms, mean_drift, max_drift]` onto the output keys
   (`mean_rms`, `max_rms`, `mean_vm_drift`, `max_vm_drift`).
3. Re-wire importers:
   - `gigaseal/featureExtractor.py` — `from .QC import run_qc` and call sites at
     lines ~376 and ~769.
   - `gigaseal/bin/run_CM_CALC.py` — `from gigaseal.QC import *`.
4. Delete `gigaseal/QC.py` once no importer references it.
5. Verify (see below).

### Sequencing options

- **Pilot now:** do the QC move immediately; accept that the still-live
  `featureExtractor.py` imports *upward* into the analysis package until it is
  quarantined. Mechanically fine (lazy imports); sets the pattern early.
- **Bundle later:** keep `QC.py` in place and perform the relocation atomically
  as part of the `featureExtractor.py`/`patch_subthres.py` → `_legacy/` move, so
  a live frozen module never imports from a just-relocated target.

## Opportunistic duplication cleanup

Local copies of the QC helpers exist and should be deleted in favor of the single
canonical location:

- `gigaseal/bin/run_QC.py` — local `find_zero` / `compute_rms` / `compute_vm_drift`.
- `gigaseal/dev/epsp_analysis.py` — local copies + `run_qc` usage.
- `gigaseal/dev/run_APisolation_ipfx_fv_ic1.py` — local copies.

## Target end-state (illustrative)

```
gigaseal/
  analysis/
    builtins/
      qc.py            # find_zero/…/run_qc + QcAnalysis   (Option A)
      rmp.py           # RMP helpers + RmpAnalysis
      membrane_fit.py  # MembraneAnalysis (wraps core fits)
      growth_factor.py # GrowthFactorAnalysis (experimental)
    core/              # shared computation ONLY (create on first need)
  patch_utils.py       # crop_spikes, build_running_bin (cross-cutting)
  _legacy/
    featureExtractor.py
    patch_subthres.py
```

## Verification checklist

- [ ] `python -c "import gigaseal.analysis as a; print(sorted(a.list_modules()))"`
- [ ] `pytest tests/test_analysis_framework.py -q`
- [ ] `pytest tests/test_feature_extractor.py -q` (spike/QC pipeline intact)
- [ ] `pytest tests/test_gui_smoke.py tests/test_gui_imports.py -q` (GUI re-wired)
- [ ] `grep -r "gigaseal.QC\|from .QC" gigaseal/` returns no stale importers

# Function-duplication inventory

Working checklist of duplicated function definitions across `gigaseal/`.
Tick items off as each duplication is consolidated. Entry-point `def main():`
launchers across `bin/` and `gui/` are intentionally excluded (expected, not
real duplication).

Legend: **canonical** = keep this one; delete/redirect the rest.

---

## 1. QC metrics — `find_zero`, `find_baseline`, `compute_vm_drift`, `compute_rms`, `run_qc`

Currently a **live triple** plus script copies.

- [ ] `gigaseal/QC.py` — original core module
- [ ] `gigaseal/_legacy/QC.py` — copy staged into `_legacy`
- [ ] `gigaseal/analysis/builtins/qc.py` — copy colocated with `QcAnalysis`
- [ ] `gigaseal/bin/run_QC.py` — local copies (all 5)
- [ ] `gigaseal/dev/epsp_analysis.py` — `find_zero`, `find_baseline` (+ others)
- [ ] `gigaseal/dev/run_APisolation_ipfx_fv_ic1.py` — all 5

**Action:** choose ONE canonical home (per `ANALYSIS_CORE_PLACEMENT.md`), delete
the other two live copies, and delete the dev/bin local copies (or import from
canonical).

---

## 2. Spike cropping — `crop_ap`

- [ ] `gigaseal/bin/run_QC.py` — `dv_cutoff=20`
- [ ] `gigaseal/bin/run_rmp.py` — `dv_cutoff=20, thresh_frac=0.2` variant

**Action:** consolidate into the stubbed `gigaseal/patch_utils.crop_spikes`.

---

## 3. RMP per-file loop — `rmp_abf`

- [ ] `gigaseal/bin/run_QC.py` — no running-bin
- [ ] `gigaseal/bin/run_rmp.py` — with running-bin

**Action:** fold into `RmpAnalysis.analyze()`.

---

## 4. Running-bin helpers — `running_bin` / `build_running_bin`

- [ ] `gigaseal/patch_utils.py` — `build_running_bin` (**canonical**)
- [ ] `gigaseal/bin/run_rmp.py` — local `running_bin`
- [ ] `gigaseal/dev/run_APisolation_ipfx_fv_ic1.py` — duplicate `build_running_bin`

---

## 5. Exponential growth/decay fitting

⚠️ Signature drift exists: `patch_subthres.exp_growth_factor(..., alpha, end_index=1)`
differs from the dev/bin `exp_growth_factor(..., end_index=300)`. **Not drop-in
identical — reconcile before consolidating.**

### `exp_grow`
- [ ] `gigaseal/patch_subthres.py`
- [ ] `gigaseal/bin/run_GROW_SPCA.py`
- [ ] `gigaseal/dev/grow.py`
- [ ] `gigaseal/dev/run_GROW_SPCA.py`
- [ ] `gigaseal/dev/run_APisolation_ipfx_fv_ic1.py`

### `exp_grow_2p`
- [ ] `gigaseal/bin/run_GROW_SPCA.py`
- [ ] `gigaseal/dev/grow.py`
- [ ] `gigaseal/dev/run_GROW_SPCA.py`
- [ ] `gigaseal/dev/run_APisolation_ipfx_fv_ic1.py`

### `exp_growth_factor`
- [ ] `gigaseal/patch_subthres.py` (α-param variant)
- [ ] `gigaseal/bin/run_GROW_SPCA.py`
- [ ] `gigaseal/dev/run_GROW_SPCA.py`
- [ ] `gigaseal/dev/run_APisolation_ipfx_fv_ic1.py`

### `exp_decay_factor`
- [ ] `gigaseal/patch_subthres.py`
- [ ] `gigaseal/dev/run_APisolation_ipfx_fv_ic1.py`

### `exp_decay_1p` / `exp_decay_2p`
- [ ] `gigaseal/patch_subthres.py`
- [ ] `gigaseal/dev/run_APisolation_ipfx_fv_ic1.py`

### `curvature`, `curvature_real`, `curvature_splines`, `derivative`
- [ ] `gigaseal/bin/run_GROW_SPCA.py`
- [ ] `gigaseal/dev/grow.py`
- [ ] `gigaseal/dev/run_GROW_SPCA.py`

**Action:** reconcile to `patch_subthres.py` as canonical; delete dev/bin copies.

---

## 6. Small utilities duplicated between `patch_subthres.py` and `patch_utils.py`

### `df_select_by_col`
- [ ] `gigaseal/patch_subthres.py`
- [ ] `gigaseal/patch_utils.py`

### `find_downward`
- [ ] `gigaseal/patch_subthres.py`
- [ ] `gigaseal/patch_utils.py`

### `rmp_mode`
- [ ] `gigaseal/patch_subthres.py`
- [ ] `gigaseal/dev/epsp_analysis.py` (adds `round_factor`)

**Action:** pick `patch_utils.py` as the home for generic helpers
(`df_select_by_col`, `find_downward`); have `patch_subthres.py` import them.

---

## 7. Whole-file near-duplicate

- [ ] `gigaseal/bin/run_GROW_SPCA.py` ↔ `gigaseal/dev/run_GROW_SPCA.py`
  — same script in two trees (`exp_grow*`, `curvature*`, `exp_growth_factor`,
  `derivative`). Decide which tree owns it (or fold into
  `builtins/growth_factor.py`) and delete the other.

---

## Suggested order (biggest win / least risk first)

1. Collapse the QC triple (§1) — pure numpy, low risk.
2. `crop_spikes` consolidation (§2).
3. `patch_subthres` ↔ `patch_utils` small utils (§6).
4. Growth/decay-fit family (§5) — **reconcile signatures first**.
5. `run_GROW_SPCA` whole-file dedupe (§7).

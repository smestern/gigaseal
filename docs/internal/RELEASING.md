# Releasing gigaseal

Version is defined once in [gigaseal/__init__.py](gigaseal/__init__.py) as `__version__`
and read by setuptools via `[tool.setuptools.dynamic]` in [pyproject.toml](pyproject.toml).
**Bump the version by editing that single line** — nothing else references it.

Versions follow [PEP 440](https://peps.python.org/pep-0440/):

```
0.9.1              ← previous release
1.0.0b1            ← current: first public beta (feedback window)
1.0.0b2, b3 …      ← bugfix betas as feedback lands
1.0.0rc1           ← optional release candidate once the API is frozen
1.0.0              ← the release
1.0.1, 1.0.2 …     ← patch releases
```

`pip install gigaseal` skips pre-releases automatically; a beta is only pulled by
`pip install --pre gigaseal` or an exact pin (`gigaseal==1.0.0b1`).

## Cutting a release

1. Update `__version__` in [gigaseal/__init__.py](gigaseal/__init__.py).
2. `python -m pip install -e . --no-deps` and confirm:
   - `python -c "import gigaseal; print(gigaseal.__version__)"`
   - `python -c "from importlib.metadata import version; print(version('gigaseal'))"`
   both report the new version.
3. `pytest tests/` — green.
4. `python -m build` (sdist + wheel).
5. Tag `vX.Y.Z[bN]`; mark GitHub release as **pre-release** for betas/rcs.
6. `twine upload dist/*` (or `dist/*bN*` for a targeted beta upload).

## 1.0.0 release gate

Do **not** ship the final `1.0.0` (drop the `bN`/`rcN` suffix) until every box is checked.
Until then, the beta line is where breaking changes are still allowed. From `1.0.0`
onward, breaking any item below requires a major bump (`2.0.0`).

### API stability
- [ ] `AnalysisBase` contract frozen: typed class-attribute parameters, `sweep_mode`
      semantics (`per_sweep` / `per_file`), `register()`, `run` / `run_batch`,
      `AnalysisResult` shape. No planned signature changes.
- [ ] CLI surface frozen: `gigaseal` subcommands and their flags are stable.
- [ ] Every builtin under `gigaseal/analysis/` has stable, documented parameters.

### Dual-API resolution
- [ ] Legacy `featureExtractor.py` / `patch_subthres.py` have a declared status
      (either officially "frozen but supported" with a documented deprecation
      timeline, or removed in favour of the modular framework).
- [ ] No code path silently depends on both APIs producing identical results without
      a test asserting it.

### GUI
- [ ] `gui/app.py` at feature parity with the modular framework
      (full param-form coverage, batch `progress_callback`, results export).
- [ ] `spikeFinder.py` deprecation decision made (keep for one more minor, or remove).

### Tests & CI
- [ ] `pytest tests/` green on 3.11 and 3.12.
- [ ] Each builtin has a `Test<Class>` block in
      [tests/test_analysis_framework.py](tests/test_analysis_framework.py):
      registered + synthetic sweep + demo-ABF (skipif).
- [ ] GUI end-to-end smoke test drives `AnalysisController` against a demo ABF for
      both `SpikeAnalysis` and `SubthresholdAnalysis`.
- [ ] Optional-dependency groups (`gui`, `web`, `server`, `ml`) install cleanly and
      core imports succeed without them (IPFX-missing path still works).

### Packaging & docs
- [ ] `python -m build` produces valid sdist + wheel; `twine check dist/*` passes.
- [ ] `README.md` install/usage reflects the modular framework as the primary path.
- [ ] Stale `build/` Dash code and `_legacy_*` files removed (see
      [COPILOT_PRIORITIES.md](COPILOT_PRIORITIES.md) Tier 3).

"""
Lightweight result container for analysis outputs.

AnalysisResult wraps whatever dict your analyze() method returns
and provides helpers for aggregation, DataFrame export, and serialization.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Reducer name -> callable used when collapsing many values to one.
# All are NaN-aware so partially-empty sweeps don't poison the aggregate.
_REDUCERS = {
    "mean": np.nanmean,
    "median": np.nanmedian,
    "sum": np.nansum,
    "max": np.nanmax,
    "min": np.nanmin,
    "std": np.nanstd,
    "first": lambda a: a[0] if len(a) else np.nan,
    "last": lambda a: a[-1] if len(a) else np.nan,
    "count": lambda a: int(np.size(a)),
}


@dataclass
class SummarySpec:
    """
    Declarative description of how to build a per-file summary from the
    per-sweep raw table.  Modules opt in by returning one of these from
    ``AnalysisBase._summary_spec()``; an empty spec reproduces the historic
    default behaviour (mean every numeric column, drop list columns).

    Attributes
    ----------
    list_reducer:
        Reducer name used to collapse list/array-valued cells (e.g. per-spike
        ``peak_v``) into a single per-sweep scalar *before* aggregation.
        ``None`` (default) drops such columns, matching legacy behaviour.
    aggregations:
        Per-column reducer overrides for the file-level aggregation, e.g.
        ``{"spike_count": "mean", "train_adapt": "mean"}``.  Columns not listed
        use ``default_reducer``.
    default_reducer:
        Reducer applied to numeric columns without an explicit override.
    per_sweep_columns:
        Columns to also emit pivoted one-column-per-sweep.  Either a list of
        column names (generic ``"Sweep {n:03d} {col}"`` labels) or a dict
        mapping ``col -> label_template`` where the template may reference
        ``{n}`` (sweep number) and ``{col}`` — e.g.
        ``{"spike_count": "Sweep {n:03d} spike count"}``.
    summary_exclude:
        Columns to drop entirely from the aggregated summary (e.g. spike-time
        columns that are meaningless to average).
    first_spike:
        Optional :class:`FirstSpikeSpec` describing "rheobase"-style columns
        taken from the *first* spike of the first triggering sweep.
    """

    list_reducer: Optional[str] = None
    aggregations: Dict[str, str] = field(default_factory=dict)
    default_reducer: str = "mean"
    per_sweep_columns: Union[List[str], Dict[str, str]] = field(default_factory=list)
    summary_exclude: List[str] = field(default_factory=list)
    first_spike: Optional["FirstSpikeSpec"] = None


@dataclass
class FirstSpikeSpec:
    """
    Describe "first spike of the first triggering sweep" (rheobase-style)
    columns.

    For every file the framework finds the first sweep whose *trigger* column
    is greater than zero, then extracts, from each requested column, the value
    at spike position *index* (element ``index`` of a list-valued cell, or the
    scalar itself for non-list columns like ``train_latency``).

    Attributes
    ----------
    columns:
        Either a list of source columns (output name = ``prefix + col``) or a
        dict mapping ``source_col -> output_name`` for custom labels such as
        ``{"threshold_v": "rheobase_threshold"}``.
    trigger:
        Column whose first ``> 0`` value marks the rheobase sweep.
    prefix:
        Output-name prefix used when *columns* is a plain list.
    index:
        Which spike within the sweep to take (0 = first spike).
    """

    columns: Union[List[str], Dict[str, str]] = field(default_factory=list)
    trigger: str = "spike_count"
    prefix: str = "rheobase_"
    index: int = 0


@dataclass
class AnalysisResult:
    """
    Container for the output of a single analysis run.

    Attributes:
        name:          Name of the analysis module that produced this result.
        file_path:     Path to the source file (or 'array_input' for raw data).
        success:       Whether the analysis completed without error.
        data:          The dict returned by analyze() (per-file mode) or an
                       aggregated dict built from sweep_results.
        sweep_results: List of per-sweep dicts (populated in per_sweep mode).
        errors:        Any error messages captured during the run.
        warnings:      Any warning messages captured during the run.
        metadata:      Extra info (sweep count, protocol, etc.).
    """

    name: str
    file_path: str = "unknown"
    success: bool = True

    # Main data -- the dict your analyze() returned
    data: Dict[str, Any] = field(default_factory=dict)

    # Per-sweep results (list of dicts, one per sweep)
    sweep_results: List[Dict[str, Any]] = field(default_factory=list)

    # Diagnostics
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Extra named output sheets (e.g. a module-specific summary).  These are
    # merged ahead of the generic "Raw" / "Summary" sheets by :meth:`to_sheets`.
    sheets: Dict[str, pd.DataFrame] = field(default_factory=dict)

    # Declarative summary configuration supplied by the producing module.
    # ``None`` -> historic default (mean numeric, drop lists).
    summary_spec: Optional[SummarySpec] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def add_error(self, msg: str) -> None:
        """Record an error and mark the result as failed."""
        self.errors.append(msg)
        self.success = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    @classmethod
    def concatenate(cls, results: List["AnalysisResult"]) -> "AnalysisResult":
        """
        Merge a list of AnalysisResult objects into one combined result.

        The combined ``data`` dict is empty; all information lives in the
        concatenated DataFrame accessible via ``to_dataframe()``.
        """
        if not results:
            return cls(name="empty", file_path="none", success=False,
                       errors=["No results to concatenate"])

        # Build a combined DataFrame from all individual results
        frames = [r.to_dataframe() for r in results if r.success]
        combined_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        all_errors = [e for r in results for e in r.errors]
        all_warnings = [w for r in results for w in r.warnings]
        all_files = [r.file_path for r in results]

        combined = cls(
            name=results[0].name,
            file_path=str(all_files),
            success=all(r.success for r in results),
            data={},
            sweep_results=[],
            errors=all_errors,
            warnings=all_warnings,
            metadata={"file_count": len(results), "files": all_files},
            summary_spec=results[0].summary_spec,
        )
        # Stash the pre-built DataFrame so to_dataframe() returns it
        combined._combined_df = combined_df
        return combined

    # ------------------------------------------------------------------
    # DataFrame export
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return the "raw" DataFrame -- one row per sweep (or per file).

        For a combined/batch result this returns the pre-built DataFrame
        stashed by :meth:`concatenate`; otherwise it is built on the fly:

        * If there are *sweep_results*, each sweep becomes a row and the
          file-level ``data`` dict is broadcast across all rows.
        * Otherwise a single-row DataFrame is built from ``data``.
        """
        if hasattr(self, "_combined_df"):
            return self._combined_df
        return self._build_dataframe()

    def _build_dataframe(self) -> pd.DataFrame:
        """Build a DataFrame from this single result."""
        if self.sweep_results:
            rows = []
            for i, sweep_dict in enumerate(self.sweep_results):
                row = {"file": self.file_path, "sweep": i}
                row.update(self.data)
                row.update(sweep_dict)
                rows.append(row)
            return pd.DataFrame(rows)
        else:
            row = {"file": self.file_path}
            row.update(self.data)
            return pd.DataFrame([row])

    # ------------------------------------------------------------------
    # Summary / multi-sheet export
    # ------------------------------------------------------------------

    def summary_dataframe(self) -> pd.DataFrame:
        """
        Build a per-file summary: one row per source file.

        Behaviour is driven by :attr:`summary_spec`.  With no spec (the
        default) the raw DataFrame (one row per sweep) is grouped by its
        ``file`` column and every *numeric scalar* column is reduced to its
        mean; list-valued columns (e.g. per-spike arrays like ``peak_t``) are
        dropped automatically because they are not numeric dtypes.

        When a :class:`SummarySpec` is present it may additionally:

        * collapse list/array cells to a per-sweep scalar (``list_reducer``)
          so per-spike features become averageable;
        * apply per-column reducers (``aggregations`` / ``default_reducer``);
        * drop columns (``summary_exclude``);
        * emit per-sweep pivot columns (``per_sweep_columns``), e.g. a
          ``Sweep 001 spike count`` column for every sweep.

        Two extra columns are always added:

        * ``n_sweeps`` -- number of raw rows that went into each file.
        * ``total_spike_count`` -- sum of ``spike_count`` per file, when a
          ``spike_count`` column is present.

        Returns an empty DataFrame if there is no data or no ``file`` column.
        """
        raw = self.to_dataframe()
        if raw is None or raw.empty or "file" not in raw.columns:
            return pd.DataFrame()

        spec = self.summary_spec or SummarySpec()
        raw = raw.copy()

        # 0) First-spike ("rheobase") extraction must run BEFORE list columns
        #    are collapsed, so it can still see the per-spike arrays.
        first_spike_df = None
        if spec.first_spike is not None:
            first_spike_df = self._build_first_spike_columns(raw, spec.first_spike)

        # 1) Collapse list/array-valued cells to a per-sweep scalar so that
        #    per-spike features can participate in the numeric aggregation.
        if spec.list_reducer is not None:
            raw = self._reduce_list_columns(raw, spec.list_reducer)

        grouped = raw.groupby("file", sort=False)

        # 2) Aggregate numeric columns with per-column reducers.
        numeric = raw.select_dtypes(include="number")
        # "sweep" is a row index, not a feature -- keep it out of the aggregate.
        exclude = self._expand_excludes(raw.columns, spec.summary_exclude)
        drop_cols = exclude | {"sweep"}
        numeric = numeric.drop(columns=[c for c in drop_cols if c in numeric.columns])

        if numeric.shape[1] > 0:
            summary = self._aggregate_numeric(numeric, raw["file"], spec)
        else:
            # No numeric features -- still emit one row per file.
            summary = pd.DataFrame(index=grouped.size().index)

        summary.insert(0, "n_sweeps", grouped.size())

        if "spike_count" in raw.columns:
            summary["total_spike_count"] = grouped["spike_count"].sum()

        # 3) First-spike (rheobase) columns, ahead of the per-sweep pivot.
        if first_spike_df is not None and not first_spike_df.empty:
            summary = summary.join(first_spike_df)

        # 4) Per-sweep pivot columns (e.g. "Sweep 001 spike count").
        if spec.per_sweep_columns:
            pivot = self._build_per_sweep_columns(raw, spec.per_sweep_columns)
            if not pivot.empty:
                summary = summary.join(pivot)

        return summary.reset_index()

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _expand_excludes(columns, patterns: List[str]) -> set:
        """
        Expand ``summary_exclude`` entries against the real column names.

        Entries may be exact column names or ``fnmatch``-style globs
        (e.g. ``"*_t"``, ``"*_index"``) so modules can drop whole families of
        columns without enumerating each ipfx feature by hand.
        """
        import fnmatch

        matched: set = set()
        cols = list(columns)
        for pat in patterns or []:
            if any(ch in pat for ch in "*?[]"):
                matched.update(fnmatch.filter(cols, pat))
            elif pat in cols:
                matched.add(pat)
        return matched

    @staticmethod
    def _reduce_list_columns(raw: pd.DataFrame, reducer: str) -> pd.DataFrame:
        """
        Collapse list/array-valued cells in object columns to a scalar.

        A column is treated as list-valued if any cell is a list/tuple/ndarray.
        Empty sequences reduce to NaN.  Columns whose lists hold non-numeric
        values (e.g. ipfx's string ``detour`` labels) are left untouched so
        they are simply dropped from the numeric aggregation downstream.
        """
        func = _REDUCERS.get(reducer, np.nanmean)

        def _collapse(val):
            if isinstance(val, (list, tuple, np.ndarray)):
                arr = np.asarray(val).ravel()
                if arr.size == 0:
                    return np.nan
                try:
                    arr = arr.astype("float64")
                except (ValueError, TypeError):
                    return np.nan
                if arr.size == 0 or np.all(np.isnan(arr)):
                    return np.nan
                with np.errstate(all="ignore"):
                    return float(func(arr))
            return val

        for col in raw.columns:
            if col in ("file", "sweep", "sweep_number"):
                continue
            if raw[col].dtype != object:
                continue
            cells = raw[col]
            is_seq = cells.map(lambda v: isinstance(v, (list, tuple, np.ndarray)))
            if not is_seq.any():
                continue
            # Skip columns whose lists are non-numeric (leave as object -> dropped).
            sample = next(
                (v for v in cells[is_seq] if np.asarray(v).ravel().size > 0), None
            )
            if sample is not None:
                try:
                    np.asarray(sample).astype("float64")
                except (ValueError, TypeError):
                    continue
            raw[col] = pd.to_numeric(cells.map(_collapse), errors="coerce")
        return raw

    @staticmethod
    def _aggregate_numeric(
        numeric: pd.DataFrame, files: pd.Series, spec: "SummarySpec"
    ) -> pd.DataFrame:
        """Group *numeric* by *files*, applying per-column reducers."""
        default = spec.default_reducer or "mean"
        agg_map = {}
        for col in numeric.columns:
            reducer = spec.aggregations.get(col, default)
            # pandas understands these string reducers directly.
            agg_map[col] = reducer if reducer in {
                "mean", "median", "sum", "max", "min", "std", "first", "last", "count"
            } else default
        return numeric.groupby(files, sort=False).agg(agg_map)

    @staticmethod
    def _build_first_spike_columns(raw: pd.DataFrame, fs: "FirstSpikeSpec") -> pd.DataFrame:
        """
        One row per file taken from the *first spike of the first triggering
        sweep* (rheobase-style features).

        For each file the first sweep whose ``fs.trigger`` value is ``> 0`` is
        located (rows are already ordered file-then-sweep).  From that sweep,
        each requested column contributes element ``fs.index`` of a list-valued
        cell, or the scalar itself for non-list columns.
        """
        if isinstance(fs.columns, dict):
            items = list(fs.columns.items())
        else:
            items = [(c, f"{fs.prefix}{c}") for c in fs.columns]

        if fs.trigger not in raw.columns or not items:
            return pd.DataFrame()

        rows: Dict[Any, Dict[str, Any]] = {}
        for file_key, grp in raw.groupby("file", sort=False):
            trig = pd.to_numeric(grp[fs.trigger], errors="coerce")
            hit = grp[trig > 0]
            if hit.empty:
                continue
            r = hit.iloc[0]
            out: Dict[str, Any] = {}
            for col, outname in items:
                if col not in raw.columns:
                    continue
                val = r[col]
                if isinstance(val, (list, tuple, np.ndarray)):
                    arr = np.asarray(val).ravel()
                    out[outname] = (
                        float(arr[fs.index]) if arr.size > fs.index else np.nan
                    )
                else:
                    out[outname] = val
            rows[file_key] = out

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index.name = "file"
        ordered = [outname for _, outname in items]
        return df.reindex(columns=[c for c in ordered if c in df.columns])

    @staticmethod
    def _build_per_sweep_columns(
        raw: pd.DataFrame,
        per_sweep_columns: Union[List[str], Dict[str, str]],
    ) -> pd.DataFrame:
        """
        Build a per-file frame with one column per (sweep, feature).

        *per_sweep_columns* is either a list of column names (generic
        ``"Sweep {n:03d} {col}"`` labels) or a dict mapping ``col`` to a label
        template that may reference ``{n}`` and ``{col}``.
        """
        if isinstance(per_sweep_columns, dict):
            items = list(per_sweep_columns.items())
        else:
            items = [(c, "Sweep {n:03d} {col}") for c in per_sweep_columns]

        # Sweep label uses the real sweep number when available.
        label_col = "sweep_number" if "sweep_number" in raw.columns else "sweep"

        rows: Dict[Any, Dict[str, Any]] = {}
        ordered_labels: List[str] = []
        seen = set()

        for col, template in items:
            if col not in raw.columns:
                continue
            for _, r in raw.iterrows():
                file_key = r["file"]
                try:
                    n = int(r[label_col])
                except (TypeError, ValueError, KeyError):
                    n = 0
                label = template.format(n=n, col=col)
                if label not in seen:
                    seen.add(label)
                    ordered_labels.append(label)
                rows.setdefault(file_key, {})[label] = r[col]

        if not rows:
            return pd.DataFrame()

        pivot = pd.DataFrame.from_dict(rows, orient="index")
        pivot.index.name = "file"
        # Preserve first-seen (sweep-then-feature) column order.
        pivot = pivot.reindex(columns=[c for c in ordered_labels if c in pivot.columns])
        return pivot

    def to_sheets(self) -> "OrderedDict[str, pd.DataFrame]":
        """
        Return the named sheets to display / export, in tab order.

        Always includes a generic ``"Summary"`` (one row per file) followed
        by ``"Raw"`` (one row per sweep).  Any module-supplied entries in
        :attr:`sheets` are inserted ahead of these so a module can override
        or augment the defaults.
        """
        out: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
        for key, frame in self.sheets.items():
            out[key] = frame
        out.setdefault("Summary", self.summary_dataframe())
        out.setdefault("Raw", self.to_dataframe())
        return out

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "file_path": self.file_path,
            "success": self.success,
            "data": self.data,
            "sweep_results": self.sweep_results,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        status = "OK" if self.success else f"FAILED ({len(self.errors)} errors)"
        n_sweeps = len(self.sweep_results)
        return (f"AnalysisResult(name='{self.name}', file='{self.file_path}', "
                f"{status}, sweeps={n_sweeps})")

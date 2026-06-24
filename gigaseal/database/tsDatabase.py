"""
tsDatabase - cell-level database for intracellular electrophysiology recordings.

Each row is one **cell**.  Columns fall into two categories:

* **Protocol columns** hold file paths (or ``;``-delimited lists of paths)
  linking recordings to the cell.  Within-cell conditions use the naming
  convention ``{protocol} - {condition}`` (e.g. ``IC1 - control``,
  ``IC1 + NE``).
* **Metadata columns** hold cell-level annotations such as ``condition``,
  ``group``, ``drug``, ``experimenter``, ``date``, ``notes``.

The database round-trips through Excel (``.xlsx``) and CSV so that
bench scientists can open and hand-edit the file.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Separator between protocol base name and condition in column headers
CONDITION_SEP = " - "

# Default metadata column names recognised on import
DEFAULT_METADATA_COLS = [
    "condition", "group", "drug", "experimenter", "date", "notes",
    "sex", "age", "animal_id", "cell_type", "well",
]

# Extra lab-CSV metadata names matched case-insensitively in addition to
# DEFAULT_METADATA_COLS. Anything matching here is auto-classified as
# metadata unless the caller overrides.
_EXTENDED_METADATA_NAMES = {
    "cell_id", "cell_num", "cell_number", "unique_id", "recording_id",
    "note", "exclude", "put. cell type", "putative cell type",
    "exclude?", "burst",
}

# Column names that, when present, suggest a per-recording row mode.
_UNIQUE_ID_CANDIDATES = ("unique_id", "recording_id", "rec_id")

_FILE_ID_RE = re.compile(r"^[A-Za-z0-9_\-./\\:]+$")


# ----------------------------------------------------------------------
# Value-type sniffing helpers (module-level, pure)
# ----------------------------------------------------------------------

def _clean_header_cell(value: Any) -> str:
    """Strip a header cell to its display text; treat NaN/blank/Unnamed as ''."""
    if value is None:
        return ""
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return ""
    if s.startswith("Unnamed:"):
        return ""
    return s


def _parses_as_float(value: Any) -> bool:
    """True iff *value* parses as a decimal number (and contains no ``_``).

    Python's :func:`float` accepts underscores between digits
    (``"240507_0000"`` → ``2.405070000e8``); for our purposes those are
    file-id tokens, not numbers.
    """
    s = str(value)
    if "_" in s:
        return False
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def _looks_like_file_id(value: Any) -> bool:
    """Return True if *value* looks like a recording-file identifier.

    Matches bare numeric IDs ≥4 chars (``26505004``), date-stamped tokens
    (``240507_0000``, ``2025_07_25_0000``), composite labels
    (``SLICE1_CELL1_240507_0000``), and full paths with extensions.
    Excludes pure floats, plain English words, Yes/No, and short
    integers.
    """
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return False
    if not _FILE_ID_RE.match(s):
        return False
    if not any(c.isdigit() for c in s):
        return False
    if "." in s and not ("/" in s or "\\" in s):
        # Numeric-with-decimal → metric, not file id.
        if _parses_as_float(s):
            return False
    # Pure-digit tokens must be long enough to look like an acquisition ID.
    if s.isdigit() and len(s) < 4:
        return False
    return True


def _classify_column_values(series: pd.Series) -> str:
    """Return ``'file_id'``, ``'flag'``, ``'metric'``, ``'empty'``, or ``'text'``."""
    non_empty = [
        str(v).strip() for v in series
        if pd.notna(v) and str(v).strip() not in ("", "nan", "None")
    ]
    if not non_empty:
        return "empty"
    lower = {v.lower() for v in non_empty}
    if lower <= {"yes", "no", "y", "n", "true", "false"}:
        return "flag"
    file_id_hits = sum(1 for v in non_empty if _looks_like_file_id(v))
    if file_id_hits / len(non_empty) >= 0.6:
        return "file_id"
    if all(_parses_as_float(v) for v in non_empty):
        return "metric"
    return "text"


# ----------------------------------------------------------------------
# CSV header / structure helpers (module-level, pure)
# ----------------------------------------------------------------------

def _detect_header_rows(path: str, max_levels: int = 3) -> int:
    """Probe top of *path* to detect how many rows form the header (1..max_levels).

    Strategy: read a few rows raw; the header section is the run of
    leading rows whose non-empty cells are mostly *labels* rather than
    bare numeric IDs, date-stamped tokens, or floats. Returns at least 1.
    """
    try:
        probe = pd.read_csv(
            path, nrows=max_levels + 1, header=None,
            dtype=str, keep_default_na=False,
        )
    except Exception:
        return 1
    if probe.empty:
        return 1
    n_rows = min(len(probe), max_levels + 1)

    _date_stamp_re = re.compile(r"^\d{4,}[_\-]\d{2,}")

    def _is_strong_data_token(v: str) -> bool:
        if not v:
            return False
        if _parses_as_float(v):  # plain float, no underscores
            return True
        if v.isdigit() and len(v) >= 5:  # long numeric acquisition ID
            return True
        if _date_stamp_re.match(v):  # 240507_0000 / 2025_07_25_0000
            return True
        if "/" in v or "\\" in v:  # path-like
            return True
        return False

    def _is_data_row(row: pd.Series) -> bool:
        vals = [_clean_header_cell(v) for v in row]
        vals = [v for v in vals if v]
        if not vals:
            return False
        # Any strong-data token (long ID, date-stamp, float, or path)
        # is enough — header rows shouldn't contain those.
        return any(_is_strong_data_token(v) for v in vals)

    for i in range(n_rows):
        if _is_data_row(probe.iloc[i]):
            return max(1, min(i, max_levels))
    return min(n_rows, max_levels)


def _flatten_grouped_columns(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Collapse a MultiIndex column header to flat names + group map.

    .. note::
        Prefer :func:`_read_grouped_csv` for fresh reads — it bypasses
        pandas' silent deduplication of duplicate column tuples (which
        loses the very repetition we need to detect pre/post-drug
        blocks). This function is kept for in-memory DataFrames whose
        columns are already a clean :class:`pd.MultiIndex`.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df, {c: "" for c in df.columns}

    n_levels = df.columns.nlevels
    upper_levels: List[List[str]] = []
    for i in range(n_levels - 1):
        raw = [_clean_header_cell(v) for v in df.columns.get_level_values(i)]
        upper_levels.append(raw)
    bottom = [_clean_header_cell(v) for v in df.columns.get_level_values(-1)]

    return _build_flat_from_raw(df.values, upper_levels, bottom, df.index)


def _read_grouped_csv(
    path: str, n_hdr: int,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Read a CSV whose top ``n_hdr`` rows form a (possibly grouped) header.

    Returns ``(df_with_flat_columns, group_map)``. Bypasses pandas'
    duplicate-column auto-suffixing by reading the header rows as raw
    strings and constructing flat column names ourselves.
    """
    raw = pd.read_csv(
        path, header=None, dtype=str, keep_default_na=False,
    )
    if raw.empty:
        return pd.DataFrame(), {}
    header_rows: List[List[str]] = []
    for i in range(min(n_hdr, len(raw))):
        header_rows.append([_clean_header_cell(v) for v in raw.iloc[i].tolist()])
    data = raw.iloc[n_hdr:].reset_index(drop=True)
    # Convert empty strings back to NaN so dtype inference & ``pd.isna`` work
    data = data.replace("", pd.NA)

    upper_levels = header_rows[:-1] if n_hdr >= 2 else []
    bottom = header_rows[-1]
    return _build_flat_from_raw(data.values, upper_levels, bottom, data.index)


def _build_flat_from_raw(
    data: Any,
    upper_levels: List[List[str]],
    bottom: List[str],
    index: Any,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Construct a flat-columned DataFrame from raw 2-D data + header rows."""
    # Forward-fill upper levels (treat blanks/Unnamed: as carry-down)
    filled_uppers: List[List[str]] = []
    for level in upper_levels:
        filled: List[str] = []
        last = ""
        for v in level:
            if not v:
                filled.append(last)
            else:
                last = v
                filled.append(v)
        filled_uppers.append(filled)

    counts: Dict[str, int] = {}
    for n in bottom:
        if n:
            counts[n] = counts.get(n, 0) + 1

    flat_names: List[Optional[str]] = []
    group_map: Dict[str, str] = {}
    seen: Dict[str, int] = {}
    emitted: set = set()
    for j, name in enumerate(bottom):
        # When the bottom-level name is blank but an upper-level group
        # exists, the upper label IS the column name (lab convention for
        # DRUG / NOTES blocks where the super-header carries the meaning).
        group_parts = [
            filled_uppers[i][j]
            for i in range(len(filled_uppers))
            if j < len(filled_uppers[i]) and filled_uppers[i][j]
        ]
        group = " / ".join(group_parts)
        if not name:
            if group_parts:
                name = group_parts[-1]
                group = " / ".join(group_parts[:-1])
            else:
                flat_names.append(None)
                continue
            # Recompute count contribution for promoted-name disambiguation
            counts[name] = counts.get(name, 0) + 1
        occ = seen.get(name, 0) + 1
        seen[name] = occ
        total = counts[name]
        if total == 1:
            flat = name
        elif total == 2:
            flat = f"{name}{CONDITION_SEP}control" if occ == 1 else f"{name}{CONDITION_SEP}drug"
        else:
            base_flat = f"{name}{CONDITION_SEP}{group}" if group else f"{name}_{occ}"
            flat = base_flat
            dedup = 2
            while flat in emitted:
                flat = f"{base_flat}_{dedup}"
                dedup += 1
        emitted.add(flat)
        flat_names.append(flat)
        group_map[flat] = group

    keep_idx = [j for j, n in enumerate(flat_names) if n is not None]
    kept_names = [flat_names[j] for j in keep_idx]
    # data is 2-D ndarray; slice columns by keep_idx
    if hasattr(data, "shape") and len(getattr(data, "shape", ())) == 2:
        sub = data[:, keep_idx] if len(keep_idx) else data[:, :0]
        df_out = pd.DataFrame(sub, index=index, columns=kept_names)
    else:
        df_out = pd.DataFrame(data, index=index)
    return df_out, group_map


def _pick_index_column(
    df: pd.DataFrame,
    cell_id_col: Optional[str],
    unique_id_col: Optional[str],
    row_mode: str,
) -> Tuple[str, str]:
    """Decide which column to use as the dataframe index and the effective row_mode.

    Returns ``(index_col_name, resolved_row_mode)``.
    """
    cols_lower = {c.lower(): c for c in df.columns}

    if unique_id_col and unique_id_col in df.columns:
        return unique_id_col, "per_recording"
    if cell_id_col and cell_id_col in df.columns:
        chosen = cell_id_col
    else:
        # Auto-detect cell-id column
        for candidate in ("cell_id", "cell", "cellid", "cellname"):
            if candidate in cols_lower:
                chosen = cols_lower[candidate]
                break
        else:
            chosen = df.columns[0]

    if row_mode == "per_recording":
        # Prefer an explicit unique-id column if one exists
        for candidate in _UNIQUE_ID_CANDIDATES:
            if candidate in cols_lower:
                return cols_lower[candidate], "per_recording"
        return chosen, "per_recording"

    if row_mode == "per_cell":
        return chosen, "per_cell"

    # row_mode == "auto"
    for candidate in _UNIQUE_ID_CANDIDATES:
        if candidate in cols_lower:
            return cols_lower[candidate], "per_recording"
    if chosen in df.columns and df[chosen].duplicated().any():
        return chosen, "per_recording"
    return chosen, "per_cell"


# ----------------------------------------------------------------------
# File-ID → path resolution
# ----------------------------------------------------------------------

def resolve_file_ids(
    db: "tsDatabase",
    folder: str,
    *,
    extensions: Tuple[str, ...] = (".abf", ".nwb"),
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Rewrite file-ID tokens inside *db*'s protocol cells to absolute paths.

    Walks *folder* recursively, building ``{stem: path}`` for files whose
    suffix is in *extensions*. For each protocol cell in ``db.cellindex``,
    splits the value on ``;`` and replaces tokens whose ``str(stem)``
    matches a known file. Unmatched tokens are preserved. Numeric tokens
    that match by zero-stripped equality are also accepted (``"4"`` ↔
    ``"240507_0004"`` does **not** match — only exact-stem and
    int-equality matches are honoured).

    Returns a summary dict ``{resolved, unresolved, samples_unresolved,
    collisions}`` for the caller to surface.
    """
    folder = os.path.abspath(folder)
    stem_map: Dict[str, str] = {}
    int_map: Dict[int, str] = {}
    collisions: List[str] = []

    if os.path.isdir(folder):
        for root, _dirs, files in os.walk(folder):
            for fname in files:
                stem, ext = os.path.splitext(fname)
                if ext.lower() not in extensions:
                    continue
                full = os.path.join(root, fname)
                if stem in stem_map and stem_map[stem] != full:
                    collisions.append(stem)
                else:
                    stem_map[stem] = full
                if stem.isdigit():
                    n = int(stem)
                    if n not in int_map:
                        int_map[n] = full

    resolved = 0
    unresolved = 0
    unresolved_samples: List[str] = []
    if db.cellindex.empty:
        return {
            "resolved": 0, "unresolved": 0,
            "samples_unresolved": [], "collisions": sorted(set(collisions)),
        }

    proto_cols = db.get_protocol_columns()
    for cell in db.cellindex.index:
        for col in proto_cols:
            try:
                val = db.cellindex.at[cell, col]
            except KeyError:
                continue
            if val is None or (isinstance(val, float) and pd.isna(val)):
                continue
            s = str(val).strip()
            if not s or s.lower() in {"nan", "none"}:
                continue
            tokens = [t.strip() for t in s.split(";") if t.strip()]
            new_tokens: List[str] = []
            changed = False
            for tok in tokens:
                if os.path.isabs(tok) or os.path.sep in tok or "/" in tok:
                    new_tokens.append(tok)
                    continue
                hit = stem_map.get(tok)
                if hit is None and tok.isdigit():
                    hit = int_map.get(int(tok))
                if hit:
                    new_tokens.append(hit)
                    resolved += 1
                    changed = True
                else:
                    new_tokens.append(tok)
                    unresolved += 1
                    if len(unresolved_samples) < 10:
                        unresolved_samples.append(tok)
            if changed and not dry_run:
                db.cellindex.at[cell, col] = ";".join(new_tokens)

    return {
        "resolved": resolved,
        "unresolved": unresolved,
        "samples_unresolved": unresolved_samples,
        "collisions": sorted(set(collisions)),
    }


# ======================================================================
# experimentalStructure - protocol & column-role registry
# ======================================================================

class experimentalStructure:
    """Track which columns are protocols vs metadata and manage conditions."""

    def __init__(self):
        # protocol name -> {altnames: [...], conditions: [...], ...}
        self._protocols: Dict[str, dict] = {}
        # set of column names that are metadata (not protocols)
        self._metadata_cols: set = set()
        self.primary: Optional[str] = None

    # -- protocols ---------------------------------------------------------

    def add_protocol(self, name: str, altnames: Optional[list] = None,
                     conditions: Optional[list] = None,
                     group: Optional[str] = None, **flags):
        """Register a protocol (idempotent).

        ``group`` records the original lab-CSV super-header (e.g. ``SIM1``,
        ``PRE / NETCLAMP``) so the grouped writer can restore it.
        """
        entry = self._protocols.setdefault(name, {
            "altnames": [],
            "conditions": [],
            "group": "",
        })
        if altnames:
            for a in altnames:
                if a not in entry["altnames"]:
                    entry["altnames"].append(a)
        if conditions:
            for c in conditions:
                if c not in entry["conditions"]:
                    entry["conditions"].append(c)
        if group is not None and group != "" and not entry.get("group"):
            entry["group"] = group
        entry.update(flags)

    def get_protocol(self, name: str) -> Optional[dict]:
        """Look up by name or altname; return the entry dict or *None*."""
        if name in self._protocols:
            return self._protocols[name]
        for pname, entry in self._protocols.items():
            if name in entry.get("altnames", []):
                return entry
        return None

    def remove_protocol(self, name: str):
        self._protocols.pop(name, None)

    def protocol_names(self) -> List[str]:
        return list(self._protocols.keys())

    def set_primary(self, name: str):
        self.primary = name

    # -- metadata columns --------------------------------------------------

    def mark_metadata(self, col: str):
        self._metadata_cols.add(col)

    def unmark_metadata(self, col: str):
        self._metadata_cols.discard(col)

    def is_metadata(self, col: str) -> bool:
        return col in self._metadata_cols

    def metadata_columns(self) -> List[str]:
        return sorted(self._metadata_cols)

    # -- serialisation helpers ---------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Serialise protocol registry to a DataFrame for saving."""
        rows = []
        for name, entry in self._protocols.items():
            rows.append({
                "name": name,
                "altnames": ";".join(entry.get("altnames", [])),
                "conditions": ";".join(entry.get("conditions", [])),
                "group": entry.get("group", "") or "",
            })
        if not rows:
            return pd.DataFrame(columns=["name", "altnames", "conditions", "group"])
        return pd.DataFrame(rows)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "experimentalStructure":
        """Reconstruct from a DataFrame (e.g. the *Protocols* sheet)."""
        exp = cls()
        if df is None or df.empty:
            return exp
        for _, row in df.iterrows():
            name = str(row.get("name", ""))
            if not name:
                continue
            raw_alt = row.get("altnames", "")
            altnames = [a for a in str(raw_alt).split(";") if a] if pd.notna(raw_alt) else []
            raw_cond = row.get("conditions", "")
            conditions = [c for c in str(raw_cond).split(";") if c] if pd.notna(raw_cond) else []
            raw_group = row.get("group", "")
            group = str(raw_group) if pd.notna(raw_group) and str(raw_group) else None
            exp.add_protocol(name, altnames=altnames, conditions=conditions,
                             group=group)
        return exp

    # -- backward-compat shims ---------------------------------------------

    @property
    def protocols(self) -> pd.DataFrame:
        """Legacy accessor - returns a DataFrame view."""
        return self.to_dataframe()

    @protocols.setter
    def protocols(self, df: pd.DataFrame):
        """Legacy setter - rebuild from DataFrame."""
        self._protocols.clear()
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                name = str(row.get("name", ""))
                if not name:
                    continue
                raw_alt = row.get("altnames", "")
                altnames = [a for a in str(raw_alt).split(";") if a] if pd.notna(raw_alt) else []
                self.add_protocol(name, altnames=altnames)

    def addProtocol(self, name, flags=None, **kw):
        """Legacy wrapper."""
        flags = flags or {}
        altnames = flags.get("altnames", kw.get("altnames"))
        if isinstance(altnames, np.ndarray):
            altnames = altnames.tolist()
        if isinstance(altnames, str):
            altnames = [altnames]
        self.add_protocol(name, altnames=altnames)

    def getProtocol(self, name):
        """Legacy wrapper - returns a one-row DataFrame or None."""
        entry = self.get_protocol(name)
        if entry is None:
            return None
        return pd.DataFrame([{"name": name, **entry}])

    def setPrimary(self, name):
        self.set_primary(name)


# ======================================================================
# tsDatabase
# ======================================================================

class tsDatabase:
    """Cell-level database mapping cells -> protocols -> recording files.

    The main data structure is ``cellindex``, a :class:`pandas.DataFrame`
    where each row is a cell and columns are either *protocol* columns
    (holding file paths) or *metadata* columns (holding annotations).

    Protocol columns may encode within-cell conditions using the naming
    convention ``{base_protocol} - {condition}`` (see ``CONDITION_SEP``).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, path: Optional[str] = None,
                 exp: Optional[experimentalStructure] = None):
        self.path: str = path or os.getcwd()
        self.exp: experimentalStructure = exp or experimentalStructure()
        self.cellindex: pd.DataFrame = pd.DataFrame()
        self.cellindex.index.name = "cell"
        self._save_path: Optional[str] = None  # last-used save location

    # ------------------------------------------------------------------
    # Cell CRUD
    # ------------------------------------------------------------------

    def add_cell(self, name: str, metadata: Optional[dict] = None):
        """Add a new cell row.  *metadata* sets cell-level columns."""
        if name in self.cellindex.index:
            logger.warning("Cell %r already exists", name)
            return
        row: Dict[str, Any] = {}
        if metadata:
            for k, v in metadata.items():
                row[k] = v
                self.exp.mark_metadata(k)
        new_row = pd.DataFrame([row], index=pd.Index([name], name="cell"))
        # align columns
        for col in self.cellindex.columns:
            if col not in new_row.columns:
                new_row[col] = None
        self.cellindex = pd.concat([self.cellindex, new_row])

    def remove_cell(self, name: str):
        """Remove a cell row."""
        self.cellindex = self.cellindex.drop(index=name, errors="ignore")

    def rename_cell(self, old: str, new: str):
        """Rename a cell (index label)."""
        if old not in self.cellindex.index:
            logger.warning("Cell %r not found", old)
            return
        self.cellindex = self.cellindex.rename(index={old: new})

    def cell_names(self) -> List[str]:
        return list(self.cellindex.index)

    def get_cell(self, name: str) -> dict:
        """Return a single cell as a flat dict."""
        if name not in self.cellindex.index:
            return {}
        return self.cellindex.loc[name].to_dict()

    def cell_count(self) -> int:
        return len(self.cellindex)

    # ------------------------------------------------------------------
    # Metadata CRUD
    # ------------------------------------------------------------------

    def set_cell_metadata(self, cell: str, key: str, value):
        """Set a metadata column value for a cell."""
        self.exp.mark_metadata(key)
        if key not in self.cellindex.columns:
            self.cellindex[key] = None
        if cell in self.cellindex.index:
            self.cellindex.loc[cell, key] = value
        else:
            logger.warning("Cell %r not found", cell)

    def get_metadata_columns(self) -> List[str]:
        """Return metadata column names present in cellindex."""
        return [c for c in self.cellindex.columns if self.exp.is_metadata(c)]

    # ------------------------------------------------------------------
    # Protocol / column CRUD
    # ------------------------------------------------------------------

    @staticmethod
    def _col_name(protocol: str, condition: Optional[str] = None) -> str:
        if condition:
            return f"{protocol}{CONDITION_SEP}{condition}"
        return protocol

    def add_protocol(self, name: str, condition: Optional[str] = None):
        """Add a protocol column (creates column if missing)."""
        col = self._col_name(name, condition)
        if col not in self.cellindex.columns:
            self.cellindex[col] = None
        # register in experimental structure
        conds = [condition] if condition else None
        self.exp.add_protocol(name, conditions=conds)

    def remove_protocol(self, name: str, condition: Optional[str] = None):
        """Drop a protocol column."""
        col = self._col_name(name, condition)
        if col in self.cellindex.columns:
            self.cellindex = self.cellindex.drop(columns=[col])
        if not condition:
            self.exp.remove_protocol(name)

    def get_protocol_columns(self) -> List[str]:
        """Return protocol column names (non-metadata) in cellindex."""
        return [c for c in self.cellindex.columns if not self.exp.is_metadata(c)]

    def protocol_base_name(self, col: str) -> str:
        """Extract the base protocol name (strip condition suffix)."""
        if CONDITION_SEP in col:
            return col.split(CONDITION_SEP, 1)[0]
        return col

    def protocol_condition(self, col: str) -> Optional[str]:
        """Extract the condition suffix, or None."""
        if CONDITION_SEP in col:
            return col.split(CONDITION_SEP, 1)[1]
        return None

    # ------------------------------------------------------------------
    # File assignment
    # ------------------------------------------------------------------

    def assign_file(self, cell: str, protocol: str, filepath: str,
                    condition: Optional[str] = None, *, append: bool = False):
        """Assign a recording file to *cell* under *protocol*.

        If *append* is True and the cell already has a value, append with
        ``;`` (multi-file protocol support).
        """
        col = self._col_name(protocol, condition)
        # ensure column exists
        if col not in self.cellindex.columns:
            self.add_protocol(protocol, condition)
        if cell not in self.cellindex.index:
            logger.warning("Cell %r not found - creating it", cell)
            self.add_cell(cell)

        existing = self.cellindex.loc[cell, col]
        if append and pd.notna(existing) and str(existing).strip():
            paths = str(existing).split(";")
            if filepath not in paths:
                paths.append(filepath)
            self.cellindex.loc[cell, col] = ";".join(paths)
        else:
            self.cellindex.loc[cell, col] = filepath

    def unassign_file(self, cell: str, protocol: str,
                      condition: Optional[str] = None):
        """Clear the file assignment for a cell+protocol."""
        col = self._col_name(protocol, condition)
        if col in self.cellindex.columns and cell in self.cellindex.index:
            self.cellindex.loc[cell, col] = None

    def get_file_list(self, cell: str, protocol: str,
                      condition: Optional[str] = None) -> List[str]:
        """Return the list of file paths for a cell+protocol."""
        col = self._col_name(protocol, condition)
        if col not in self.cellindex.columns or cell not in self.cellindex.index:
            return []
        val = self.cellindex.loc[cell, col]
        if pd.isna(val) or not str(val).strip():
            return []
        return [p.strip() for p in str(val).split(";") if p.strip()]

    # ------------------------------------------------------------------
    # Save / Load - Excel (.xlsx)
    # ------------------------------------------------------------------

    def save_xlsx(self, path: str) -> str:
        """Save to a multi-sheet Excel workbook.

        Sheets: *CellIndex*, *Protocols*, *_config*, *_metadata_cols*.
        """
        if not path.endswith(".xlsx"):
            path += ".xlsx"

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            # main table
            self.cellindex.to_excel(writer, sheet_name="CellIndex", index=True)

            # protocol registry
            self.exp.to_dataframe().to_excel(writer, sheet_name="Protocols", index=False)

            # config
            pd.DataFrame({
                "key": ["version", "created_by", "database_type", "path"],
                "value": ["2.0", "gigaseal", "tsDatabase", self.path],
            }).to_excel(writer, sheet_name="_config", index=False)

            # metadata column list (so we know which cols are metadata on load)
            pd.DataFrame({
                "column": sorted(self.exp._metadata_cols),
            }).to_excel(writer, sheet_name="_metadata_cols", index=False)

        self._save_path = path
        logger.info("Database saved to %s", path)
        return path

    def load_xlsx(self, path: str):
        """Load from a multi-sheet Excel workbook."""
        self.cellindex = pd.read_excel(path, sheet_name="CellIndex", index_col=0)
        self.cellindex.index.name = "cell"

        try:
            proto_df = pd.read_excel(path, sheet_name="Protocols")
            self.exp = experimentalStructure.from_dataframe(proto_df)
        except Exception:
            logger.warning("No Protocols sheet found - using empty registry")
            self.exp = experimentalStructure()

        try:
            meta_df = pd.read_excel(path, sheet_name="_metadata_cols")
            for col in meta_df["column"]:
                self.exp.mark_metadata(str(col))
        except Exception:
            # fall back: guess metadata cols by name
            for col in self.cellindex.columns:
                if col.lower() in {m.lower() for m in DEFAULT_METADATA_COLS}:
                    self.exp.mark_metadata(col)

        try:
            cfg = pd.read_excel(path, sheet_name="_config")
            cfg_dict = dict(zip(cfg["key"], cfg["value"]))
            self.path = cfg_dict.get("path", self.path)
        except Exception:
            pass

        self._save_path = path
        logger.info("Database loaded from %s (%d cells)", path, len(self.cellindex))

    # ------------------------------------------------------------------
    # Save / Load - CSV
    # ------------------------------------------------------------------

    def save_csv(self, path: str, *, grouped: bool = False,
                 drug_col: str = "drug") -> str:
        """Save ``cellindex`` as a CSV.

        With ``grouped=False`` (default) emit a flat single-row header.
        With ``grouped=True`` emit a two-row header where row 0 holds
        the lab group labels (recovered from ``exp._protocols[name]["group"]``)
        and row 1 holds the column names. Conditioned columns of the
        form ``base - control`` / ``base - drug`` are reduced back to
        ``base`` on row 1 so the spreadsheet matches the lab convention.
        """
        if not path.endswith(".csv"):
            path += ".csv"

        if not grouped:
            self.cellindex.to_csv(path, index=True)
            logger.info("Database exported to %s", path)
            return path

        # Build a two-level column header
        df = self.cellindex.copy()
        meta_cols = set(self.get_metadata_columns())
        group_row: List[str] = []
        name_row: List[str] = []
        for col in df.columns:
            if col in meta_cols:
                group_row.append("")
                name_row.append(str(col))
                continue
            base = self.protocol_base_name(col)
            cond = self.protocol_condition(col)
            entry = self.exp.get_protocol(base) or {}
            stored_group = (entry.get("group") or "").strip()
            if cond == "control":
                group_label = stored_group or "PRE"
            elif cond == "drug":
                group_label = stored_group or drug_col.upper()
            else:
                group_label = stored_group
            group_row.append(group_label)
            name_row.append(base)

        # Write manually so the index column slot sits inline with the
        # header rows (pandas' to_csv would emit the index *name* on its
        # own line when the columns are a MultiIndex, breaking
        # round-tripping).
        index_label = str(df.index.name) if df.index.name else "cell"
        import csv as _csv
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = _csv.writer(fh)
            w.writerow([""] + group_row)
            w.writerow([index_label] + name_row)
            for idx, row in zip(df.index, df.itertuples(index=False, name=None)):
                w.writerow(
                    [idx] + [
                        "" if pd.isna(v) else v for v in row
                    ]
                )
        logger.info("Database exported (grouped) to %s", path)
        return path

    def load_csv(self, path: str,
                 cell_id_col: Optional[str] = None,
                 protocol_cols: Optional[List[str]] = None,
                 metadata_cols: Optional[List[str]] = None,
                 *,
                 header_rows: Optional[int] = None,
                 header_mode: Literal["auto", "flat", "grouped"] = "auto",
                 row_mode: Literal["auto", "per_cell", "per_recording"] = "auto",
                 drug_col: str = "drug",
                 unique_id_col: Optional[str] = None):
        """Load from a CSV. Supports 1-, 2-, and 3-row headers.

        Parameters
        ----------
        path
            CSV file to load.
        cell_id_col
            Column name to use as the cell-id index. Auto-detected from
            ``CELL_ID``/``cell``/``cell_id`` when omitted.
        protocol_cols, metadata_cols
            Optional explicit classification. Anything not listed is
            auto-classified.
        header_rows
            Force a specific header depth (1, 2, or 3). ``None`` =
            auto-detect via :func:`_detect_header_rows`.
        header_mode
            ``"auto"`` (default) uses :func:`_detect_header_rows`;
            ``"flat"`` forces a single-row header; ``"grouped"`` forces
            multi-row reading (uses ``header_rows`` if set, else 2).
        row_mode
            ``"auto"`` chooses ``per_recording`` when a unique-id column
            is present or the cell-id column has duplicates; otherwise
            ``per_cell``.
        drug_col
            Name of the metadata column carrying drug labels (case-
            insensitive). Always marked as metadata when present.
        unique_id_col
            Explicit per-recording key column (forces ``per_recording``).
        """
        # 1. Decide header depth
        if header_mode == "flat":
            n_hdr = 1
        elif header_mode == "grouped":
            n_hdr = header_rows if header_rows else 2
        else:  # auto
            n_hdr = header_rows if header_rows else _detect_header_rows(path)
        n_hdr = max(1, min(n_hdr, 3))

        # 2. Read raw
        if n_hdr == 1:
            df = pd.read_csv(path)
            group_map: Dict[str, str] = {c: "" for c in df.columns}
        else:
            df, group_map = _read_grouped_csv(path, n_hdr)

        # 3. Pick the index column and resolve row mode
        idx_col, resolved_row_mode = _pick_index_column(
            df, cell_id_col, unique_id_col, row_mode,
        )
        if idx_col in df.columns:
            df = df.set_index(idx_col)
        # Drop rows with empty / sentinel index values
        df = df[df.index.notna()]
        df = df[~df.index.astype(str).str.strip().isin({"", "_", "nan", "None"})]
        df.index.name = "cell"

        # 4. Build experimentalStructure and classify columns
        self.exp = experimentalStructure()
        meta_lookup = {m.lower() for m in DEFAULT_METADATA_COLS} \
            | set(_EXTENDED_METADATA_NAMES) \
            | {drug_col.lower()}
        explicit_meta = {c for c in (metadata_cols or [])}
        explicit_proto = {c for c in (protocol_cols or [])}

        for col in df.columns:
            base = self.protocol_base_name(col)
            cond = self.protocol_condition(col)
            group_label = group_map.get(col, "")

            if col in explicit_meta:
                self.exp.mark_metadata(col)
                continue
            if col in explicit_proto:
                self.exp.add_protocol(
                    base, conditions=[cond] if cond else None,
                    group=group_label or None,
                )
                continue

            name_low = col.lower()
            base_low = base.lower()
            if name_low in meta_lookup or base_low in meta_lookup:
                self.exp.mark_metadata(col)
                continue

            # Auto-classify by value type
            value_kind = _classify_column_values(df[col])
            if value_kind == "file_id":
                self.exp.add_protocol(
                    base, conditions=[cond] if cond else None,
                    group=group_label or None,
                )
            else:
                # flag / metric / text / empty → safest as metadata
                self.exp.mark_metadata(col)

        self.cellindex = df
        self._save_path = path
        logger.info(
            "Database loaded from CSV %s (%d rows, header_rows=%d, mode=%s)",
            path, len(df), n_hdr, resolved_row_mode,
        )

    # ------------------------------------------------------------------
    # Convenience: new empty database
    # ------------------------------------------------------------------

    def clear(self):
        """Reset to an empty database."""
        self.cellindex = pd.DataFrame()
        self.cellindex.index.name = "cell"
        self.exp = experimentalStructure()
        self._save_path = None

    def next_cell_name(self) -> str:
        """Generate the next auto-incremented cell name."""
        n = self.cell_count() + 1
        while f"Cell_{n:03d}" in self.cellindex.index:
            n += 1
        return f"Cell_{n:03d}"

    # ------------------------------------------------------------------
    # Backward-compat shims (old API -> new API)
    # ------------------------------------------------------------------

    def addEntry(self, name: str, paths=None):
        """Legacy: add a cell, optionally with files."""
        self.add_cell(name)
        if paths is not None:
            if isinstance(paths, (str, os.PathLike)):
                paths = [paths]
            for p in paths:
                try:
                    from ..dataset import cellData
                    cd = cellData(str(p))
                    proto = getattr(cd, "protocol", "unknown")
                    self.assign_file(name, proto, str(p))
                except Exception as exc:
                    logger.warning("Could not parse %s: %s", p, exc)

    def addProtocol(self, cell, protocol, **kwargs):
        """Legacy: add protocol column and optionally assign a file path."""
        path = kwargs.pop("path", None)
        self.add_protocol(protocol)
        if path and cell in self.cellindex.index:
            self.assign_file(cell, protocol, path)

    def updateEntry(self, name, **kwargs):
        """Legacy: update cell-level values."""
        for key, val in kwargs.items():
            if key in self.get_protocol_columns():
                self.cellindex.loc[name, key] = val
            else:
                self.set_cell_metadata(name, key, val)

    def save(self, path):
        """Legacy: save to Excel."""
        return self.save_xlsx(path)

    def load_from_excel(self, path):
        """Legacy: load from Excel."""
        self.load_xlsx(path)

    def getEntries(self):
        return self.cellindex.to_dict(orient="records")

    def getCells(self):
        return self.cellindex.to_dict(orient="index")

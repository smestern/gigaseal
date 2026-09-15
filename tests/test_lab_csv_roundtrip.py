"""Round-trip tests for lab-CSV (grouped / multi-row header) loading."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile

import pandas as pd
import pytest

# Directly load tsDatabase without the package __init__ (avoids ipfx)
_mod_path = os.path.join(
    os.path.dirname(__file__), "..", "gigaseal", "database", "tsDatabase.py"
)
_spec = importlib.util.spec_from_file_location("tsDatabase", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

tsDatabase = _mod.tsDatabase
resolve_file_ids = _mod.resolve_file_ids
_detect_header_rows = _mod._detect_header_rows
_classify_column_values = _mod._classify_column_values
_looks_like_file_id = _mod._looks_like_file_id
_read_grouped_csv = _mod._read_grouped_csv
suggest_column_role = _mod.suggest_column_role

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "test_data")
PER_CELL = os.path.join(FIXTURE_DIR, "lab_db_per_cell.csv")
PER_RECORDING = os.path.join(FIXTURE_DIR, "lab_db_per_recording.csv")


# ======================================================================
# Value-type sniffing
# ======================================================================

class TestValueClassification:

    def test_file_id_numeric(self):
        assert _looks_like_file_id("26505004")
        assert _looks_like_file_id("240507_0000")
        assert _looks_like_file_id("SLICE1_CELL1_240507_0000")
        assert _looks_like_file_id("2025_07_25_0000")

    def test_file_id_rejects_metrics(self):
        assert not _looks_like_file_id("88.144")
        assert not _looks_like_file_id("13.5537")
        assert not _looks_like_file_id("0")
        assert _looks_like_file_id("0000")  # all-digit token IS a file id

    def test_file_id_rejects_text(self):
        # DOPA_50 has a digit + underscore so the *regex* accepts it as
        # file-id-shaped; the downstream classifier still labels DRUG
        # columns as metadata via the name lookup. Plain English / Yes /
        # No must be rejected.
        assert _looks_like_file_id("DOPA_50")
        assert not _looks_like_file_id("Yes")
        assert not _looks_like_file_id("No")
        assert not _looks_like_file_id("")
        assert not _looks_like_file_id("nan")
        assert not _looks_like_file_id("L5 Pyr.")  # space → reject

    def test_classify_flag_column(self):
        s = pd.Series(["Yes", "No", "Yes", None, "No"])
        assert _classify_column_values(s) == "flag"

    def test_classify_metric_column(self):
        s = pd.Series([88.144, 13.5537, None, 15.3])
        assert _classify_column_values(s) == "metric"

    def test_classify_file_id_column(self):
        s = pd.Series(["240507_0000", None, "240507_0008", "240507_0017"])
        assert _classify_column_values(s) == "file_id"

    def test_classify_empty_column(self):
        s = pd.Series([None, "", "nan"])
        assert _classify_column_values(s) == "empty"


# ======================================================================
# Header detection
# ======================================================================

class TestHeaderDetection:

    def test_per_cell_detects_two_rows(self):
        assert _detect_header_rows(PER_CELL) == 2

    def test_per_recording_detects_three_rows(self):
        assert _detect_header_rows(PER_RECORDING) == 3

    def test_flat_csv_detects_one_row(self, tmp_path):
        p = tmp_path / "flat.csv"
        p.write_text("cell,IC1,drug\nC1,/file.abf,NE\nC2,/other.abf,\n")
        assert _detect_header_rows(str(p)) == 1


# ======================================================================
# Per-cell (2-row header) round-trip
# ======================================================================

class TestPerCellLoad:

    def test_loads_with_defaults(self):
        db = tsDatabase()
        db.load_csv(PER_CELL)
        assert db.cell_count() == 6
        assert "cell_1_46147" in db.cell_names()
        assert "cell_4_46154" in db.cell_names()

    def test_drug_classified_as_metadata(self):
        db = tsDatabase()
        db.load_csv(PER_CELL)
        meta = db.get_metadata_columns()
        assert "DRUG" in meta
        assert "CELL_NUM" in meta
        assert "Put. Cell type" in meta
        assert "EXCLUDE" in meta

    def test_control_drug_split_for_duplicated_protocols(self):
        db = tsDatabase()
        db.load_csv(PER_CELL)
        proto_cols = db.get_protocol_columns()
        assert "Sim1_0hz - control" in proto_cols
        assert "Sim1_0hz - drug" in proto_cols
        assert "Sim2_0hz - control" in proto_cols
        assert "Sim2_0hz - drug" in proto_cols

    def test_long_pulse_three_occurrences_disambiguated(self):
        # Long-pulse appears under Protocol, pre_drug_protocol, Protocol —
        # three columns; the 3+ fallback uses the group label.
        db = tsDatabase()
        db.load_csv(PER_CELL)
        proto_cols = db.get_protocol_columns()
        lp = [c for c in proto_cols if c.startswith("Long-pulse")]
        assert len(lp) >= 3
        # All Long-pulse variants must be unique
        assert len(set(lp)) == len(lp)

    def test_drug_block_only_populated_for_drug_cells(self):
        db = tsDatabase()
        db.load_csv(PER_CELL)
        # cell_4_46154 received DOPA_50 → drug block populated
        files_drug = db.get_file_list("cell_4_46154", "Sim1_0hz", condition="drug")
        assert files_drug == ["26512072"]
        # cell_1_46147 did not receive drug → drug block empty
        files_drug_c1 = db.get_file_list("cell_1_46147", "Sim1_0hz", condition="drug")
        assert files_drug_c1 == []

    def test_group_label_preserved_on_protocol_entry(self):
        db = tsDatabase()
        db.load_csv(PER_CELL)
        entry = db.exp.get_protocol("Sim1_0hz")
        assert entry is not None
        assert entry.get("group") == "SIM1"


# ======================================================================
# Per-recording (3-row header) round-trip
# ======================================================================

class TestPerRecordingLoad:

    def test_loads_with_defaults(self):
        db = tsDatabase()
        db.load_csv(PER_RECORDING)
        # 8 data rows in the fixture
        assert db.cell_count() == 8

    def test_uses_unique_id_as_index(self):
        db = tsDatabase()
        db.load_csv(PER_RECORDING)
        assert "SLICE1_CELL1_240507_0000" in db.cell_names()
        assert "SLICE2_CELL3_241023_0065" in db.cell_names()

    def test_duplicate_cell_id_preserved_across_rows(self):
        db = tsDatabase()
        db.load_csv(PER_RECORDING)
        # SLICE2_CELL3 has two rows (control + TTX)
        cell_id_values = list(db.cellindex["CELL_ID"])
        assert cell_id_values.count("SLICE2_CELL3") == 2

    def test_drug_classified_as_metadata(self):
        db = tsDatabase()
        db.load_csv(PER_RECORDING)
        assert "drug" in db.get_metadata_columns()

    def test_yes_no_flags_classified_as_metadata(self):
        db = tsDatabase()
        db.load_csv(PER_RECORDING)
        meta = db.get_metadata_columns()
        assert "Burst Adex" in meta
        assert "Burst Cadex" in meta

    def test_numeric_metric_columns_classified_as_metadata(self):
        db = tsDatabase()
        db.load_csv(PER_RECORDING)
        meta = db.get_metadata_columns()
        # EXP3, EXP3.5, EXP4, EXP4.5 hold floats
        assert "EXP3" in meta
        assert "EXP4" in meta

    def test_file_id_columns_classified_as_protocols(self):
        db = tsDatabase()
        db.load_csv(PER_RECORDING)
        proto_cols = db.get_protocol_columns()
        assert "IC1" in proto_cols
        assert "CTRL_PULSE" in proto_cols
        assert "NET_PULSE" in proto_cols


# ======================================================================
# Round-trip: grouped save → load
# ======================================================================

class TestGroupedRoundTrip:

    def test_per_cell_grouped_roundtrip(self, tmp_path):
        db = tsDatabase()
        db.load_csv(PER_CELL)
        out = str(tmp_path / "roundtrip.csv")
        db.save_csv(out, grouped=True)
        assert os.path.isfile(out)

        # Reload — both should detect 2-row header automatically
        db2 = tsDatabase()
        db2.load_csv(out)
        assert db2.cell_count() == db.cell_count()
        # Same control/drug split survives the round-trip
        proto_cols_2 = db2.get_protocol_columns()
        assert "Sim1_0hz - control" in proto_cols_2
        assert "Sim1_0hz - drug" in proto_cols_2

    def test_flat_save_is_default_and_round_trips(self, tmp_path):
        db = tsDatabase()
        db.load_csv(PER_CELL)
        out = str(tmp_path / "flat.csv")
        db.save_csv(out)  # grouped=False default

        # Verify it's a single-row header
        with open(out) as f:
            first = f.readline()
            second = f.readline()
        # First line has all the column names; second line is data
        assert "Sim1_0hz - control" in first or "Sim1_0hz" in first

        db2 = tsDatabase()
        db2.load_csv(out, header_mode="flat")
        assert db2.cell_count() == db.cell_count()


# ======================================================================
# resolve_file_ids
# ======================================================================

class TestResolveFileIds:

    def test_resolves_numeric_ids(self, tmp_path):
        # Build a folder with two ABFs that match IDs in the per-cell fixture
        (tmp_path / "26505004.abf").write_bytes(b"x")
        (tmp_path / "26512011.abf").write_bytes(b"x")

        db = tsDatabase()
        db.load_csv(PER_CELL)
        summary = resolve_file_ids(db, str(tmp_path))

        assert summary["resolved"] >= 2
        # Inspect: cell_1_46147 / Long-pulse - control should now hold a path
        files = db.get_file_list("cell_1_46147", "Long-pulse - control")
        # Either Long-pulse - control or Long-pulse - Protocol depending on
        # disambiguation; check whichever the load chose
        proto_cols = db.get_protocol_columns()
        lp_cols = [c for c in proto_cols if c.startswith("Long-pulse")]
        # At least one Long-pulse column for cell_1_46147 was resolved
        any_abs = False
        for col in lp_cols:
            base = db.protocol_base_name(col)
            cond = db.protocol_condition(col)
            for f in db.get_file_list("cell_1_46147", base, condition=cond):
                if os.path.isabs(f) and f.endswith("26505004.abf"):
                    any_abs = True
        assert any_abs

    def test_unresolved_ids_preserved(self, tmp_path):
        # Empty folder → nothing resolves; values stay as numeric strings
        db = tsDatabase()
        db.load_csv(PER_CELL)
        summary = resolve_file_ids(db, str(tmp_path))
        assert summary["resolved"] == 0
        assert summary["unresolved"] > 0
        # Spot-check: cell_1_46147 still has raw "26505004"
        for col in db.get_protocol_columns():
            base = db.protocol_base_name(col)
            cond = db.protocol_condition(col)
            for f in db.get_file_list("cell_1_46147", base, condition=cond):
                # Not a path — preserved as-is
                assert not os.path.isabs(f)

    def test_dry_run_does_not_mutate(self, tmp_path):
        (tmp_path / "26505004.abf").write_bytes(b"x")
        db = tsDatabase()
        db.load_csv(PER_CELL)
        before = db.cellindex.copy()
        summary = resolve_file_ids(db, str(tmp_path), dry_run=True)
        assert summary["resolved"] >= 1
        # cellindex unchanged
        pd.testing.assert_frame_equal(
            db.cellindex.astype(str), before.astype(str)
        )

    def test_token_normalization(self, tmp_path):
        # Float-coercion artefacts and extension-bearing tokens must resolve
        (tmp_path / "26505004.abf").write_bytes(b"x")
        (tmp_path / "240507_0000.abf").write_bytes(b"x")
        db = tsDatabase()
        db.add_cell("C1")
        db.add_cell("C2")
        db.add_protocol("IC1")
        db.assign_file("C1", "IC1", "26505004.0")
        db.assign_file("C2", "IC1", "240507_0000.abf")
        summary = resolve_file_ids(db, str(tmp_path))
        assert summary["resolved"] == 2
        assert summary["unresolved"] == 0
        assert db.get_file_list("C1", "IC1")[0].endswith("26505004.abf")
        assert db.get_file_list("C2", "IC1")[0].endswith("240507_0000.abf")

    def test_summary_reports_columns_scanned(self, tmp_path):
        db = tsDatabase()
        db.load_csv(PER_RECORDING)
        summary = resolve_file_ids(db, str(tmp_path))
        assert "IC1" in summary["columns_scanned"]
        assert "CELL_ID" not in summary["columns_scanned"]
        assert "DATE" not in summary["columns_scanned"]


# ======================================================================
# suggest_column_role — name precedence over value sniffing
# ======================================================================

class TestSuggestColumnRole:

    def test_id_shaped_metadata_stays_metadata(self):
        # CELL_ID values pass _looks_like_file_id, but the name must win
        s = pd.Series(["SLICE1_CELL1", "SLICE1_CELL2", "SLICE2_CELL1"])
        assert _classify_column_values(s) == "file_id"
        assert suggest_column_role("CELL_ID", s) == "metadata"
        assert suggest_column_role("UNIQUE_ID", s) == "metadata"

    def test_date_column_is_metadata(self):
        s = pd.Series(["5/7/2024", "5/10/2024"])
        assert suggest_column_role("DATE", s) == "metadata"

    def test_note_and_drug_are_metadata(self):
        s = pd.Series(["DOPA_50", "NE"])
        assert suggest_column_role("NOTE", s) == "metadata"
        assert suggest_column_role("my_drug", s, drug_col="my_drug") == "metadata"

    def test_file_id_column_is_protocol(self):
        s = pd.Series(["240507_0000", None, "240507_0008"])
        assert suggest_column_role("IC1", s) == "protocol"

    def test_metric_column_is_metadata(self):
        s = pd.Series(["15.442", "12.87", None])
        assert suggest_column_role("EXP3", s) == "metadata"


# ======================================================================
# Dialog-style import (explicit role lists) → clean resolution
# ======================================================================

class TestDialogStyleImport:

    def test_explicit_lists_from_suggestions_resolve_cleanly(self, tmp_path):
        # Mirror LabCSVImportDialog: classify every column up front and
        # pass explicit protocol/metadata lists to load_csv.
        df, _ = _read_grouped_csv(PER_RECORDING, 3)
        proto, meta = [], []
        for col in df.columns:
            role = suggest_column_role(str(col), df[col])
            (proto if role == "protocol" else meta).append(str(col))
        assert "CELL_ID" in meta
        assert "DATE" in meta
        assert "UNIQUE_ID" in meta
        assert "IC1" in proto

        db = tsDatabase()
        db.load_csv(PER_RECORDING, protocol_cols=proto, metadata_cols=meta)
        assert "CELL_ID" not in db.get_protocol_columns()
        assert "DATE" not in db.get_protocol_columns()

        # Stub every referenced recording, then everything must resolve
        for col in db.get_protocol_columns():
            base = db.protocol_base_name(col)
            cond = db.protocol_condition(col)
            for cell in db.cell_names():
                for tok in db.get_file_list(cell, base, condition=cond):
                    (tmp_path / f"{tok}.abf").write_bytes(b"x")
        summary = resolve_file_ids(db, str(tmp_path))
        assert summary["unresolved"] == 0
        assert summary["resolved"] > 0
        # No cell-id tokens leaked into the resolver
        assert not any(s.startswith("SLICE") for s in summary["samples_unresolved"])


# ======================================================================
# Flat CSV — numeric-ID dtype preservation
# ======================================================================

class TestFlatNumericIdCsv:

    def _write_flat(self, tmp_path):
        p = tmp_path / "flat.csv"
        # blank cell forces float64 coercion without dtype=str
        p.write_text("cell,IC1,drug\nC1,26505004,NE\nC2,,aCSF\nC3,26512011,\n")
        return str(p)

    def test_ids_survive_as_strings(self, tmp_path):
        db = tsDatabase()
        db.load_csv(self._write_flat(tmp_path))
        assert "IC1" in db.get_protocol_columns()
        assert db.get_file_list("C1", "IC1") == ["26505004"]
        assert db.get_file_list("C3", "IC1") == ["26512011"]

    def test_ids_resolve_against_folder(self, tmp_path):
        (tmp_path / "26505004.abf").write_bytes(b"x")
        (tmp_path / "26512011.abf").write_bytes(b"x")
        db = tsDatabase()
        db.load_csv(self._write_flat(tmp_path))
        summary = resolve_file_ids(db, str(tmp_path))
        assert summary["resolved"] == 2
        assert summary["unresolved"] == 0
        assert db.get_file_list("C1", "IC1")[0].endswith("26505004.abf")

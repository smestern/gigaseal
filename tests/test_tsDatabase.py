"""Unit tests for the tsDatabase backend."""

import importlib.util
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# Directly load the tsDatabase module without triggering the main package __init__
# (which pulls in ipfx and other heavy deps that may not be installed)
_mod_path = os.path.join(
    os.path.dirname(__file__), "..", "gigaseal", "database", "tsDatabase.py"
)
_spec = importlib.util.spec_from_file_location("tsDatabase", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

CONDITION_SEP = _mod.CONDITION_SEP
DEFAULT_METADATA_COLS = _mod.DEFAULT_METADATA_COLS
experimentalStructure = _mod.experimentalStructure
tsDatabase = _mod.tsDatabase


# ======================================================================
# experimentalStructure
# ======================================================================

class TestExperimentalStructure:

    def test_add_and_get_protocol(self):
        exp = experimentalStructure()
        exp.add_protocol("IC1", altnames=["ic1", "IC_long_square"])
        assert "IC1" in exp.protocol_names()
        entry = exp.get_protocol("IC1")
        assert "ic1" in entry["altnames"]

    def test_get_by_altname(self):
        exp = experimentalStructure()
        exp.add_protocol("IC1", altnames=["ic1"])
        assert exp.get_protocol("ic1") is not None

    def test_conditions(self):
        exp = experimentalStructure()
        exp.add_protocol("IC1", conditions=["control", "NE"])
        entry = exp.get_protocol("IC1")
        assert "control" in entry["conditions"]
        assert "NE" in entry["conditions"]

    def test_metadata_tracking(self):
        exp = experimentalStructure()
        exp.mark_metadata("drug")
        assert exp.is_metadata("drug")
        exp.unmark_metadata("drug")
        assert not exp.is_metadata("drug")

    def test_round_trip_dataframe(self):
        exp = experimentalStructure()
        exp.add_protocol("IC1", altnames=["ic1"], conditions=["ctrl"])
        exp.add_protocol("Sag")
        df = exp.to_dataframe()
        exp2 = experimentalStructure.from_dataframe(df)
        assert set(exp2.protocol_names()) == {"IC1", "Sag"}
        assert "ic1" in exp2.get_protocol("IC1")["altnames"]
        assert "ctrl" in exp2.get_protocol("IC1")["conditions"]

    def test_remove_protocol(self):
        exp = experimentalStructure()
        exp.add_protocol("IC1")
        exp.remove_protocol("IC1")
        assert "IC1" not in exp.protocol_names()

    def test_group_field_round_trip(self):
        exp = experimentalStructure()
        exp.add_protocol("Sim1_0hz", group="SIM1")
        exp.add_protocol("IC1", group="PRE / NETCLAMP")
        exp.add_protocol("Sag")  # no group → empty
        df = exp.to_dataframe()
        assert "group" in df.columns
        exp2 = experimentalStructure.from_dataframe(df)
        assert exp2.get_protocol("Sim1_0hz")["group"] == "SIM1"
        assert exp2.get_protocol("IC1")["group"] == "PRE / NETCLAMP"
        assert exp2.get_protocol("Sag")["group"] == ""

    def test_group_first_write_wins(self):
        exp = experimentalStructure()
        exp.add_protocol("IC1", group="PRE")
        exp.add_protocol("IC1", group="POST")  # second call must not overwrite
        assert exp.get_protocol("IC1")["group"] == "PRE"


# ======================================================================
# tsDatabase — cell CRUD
# ======================================================================

class TestCellCRUD:

    def test_add_and_list(self):
        db = tsDatabase()
        db.add_cell("Cell_001")
        db.add_cell("Cell_002")
        assert db.cell_count() == 2
        assert db.cell_names() == ["Cell_001", "Cell_002"]

    def test_add_with_metadata(self):
        db = tsDatabase()
        db.add_cell("Cell_001", metadata={"drug": "NE", "group": "stress"})
        assert db.get_cell("Cell_001")["drug"] == "NE"
        assert "drug" in db.get_metadata_columns()

    def test_remove_cell(self):
        db = tsDatabase()
        db.add_cell("Cell_001")
        db.remove_cell("Cell_001")
        assert db.cell_count() == 0

    def test_rename_cell(self):
        db = tsDatabase()
        db.add_cell("Cell_001")
        db.rename_cell("Cell_001", "My_Cell")
        assert "My_Cell" in db.cell_names()
        assert "Cell_001" not in db.cell_names()

    def test_duplicate_cell_names(self):
        db = tsDatabase()
        db.add_cell("C1")
        db.add_cell("C1")  # should warn but not crash
        assert db.cell_count() == 1

    def test_next_cell_name(self):
        db = tsDatabase()
        assert db.next_cell_name() == "Cell_001"
        db.add_cell("Cell_001")
        assert db.next_cell_name() == "Cell_002"


# ======================================================================
# tsDatabase — protocol CRUD
# ======================================================================

class TestProtocolCRUD:

    def test_add_protocol(self):
        db = tsDatabase()
        db.add_cell("C1")
        db.add_protocol("IC1")
        assert "IC1" in db.get_protocol_columns()

    def test_add_protocol_with_condition(self):
        db = tsDatabase()
        db.add_cell("C1")
        db.add_protocol("IC1", condition="control")
        db.add_protocol("IC1", condition="NE")
        cols = db.get_protocol_columns()
        assert "IC1 - control" in cols
        assert "IC1 - NE" in cols

    def test_remove_protocol(self):
        db = tsDatabase()
        db.add_cell("C1")
        db.add_protocol("IC1")
        db.remove_protocol("IC1")
        assert "IC1" not in db.get_protocol_columns()

    def test_protocol_base_and_condition(self):
        db = tsDatabase()
        assert db.protocol_base_name("IC1 - control") == "IC1"
        assert db.protocol_condition("IC1 - control") == "control"
        assert db.protocol_base_name("IC1") == "IC1"
        assert db.protocol_condition("IC1") is None


# ======================================================================
# tsDatabase — file assignment
# ======================================================================

class TestFileAssignment:

    def test_assign_and_get(self):
        db = tsDatabase()
        db.add_cell("C1")
        db.add_protocol("IC1")
        db.assign_file("C1", "IC1", "/path/file.abf")
        files = db.get_file_list("C1", "IC1")
        assert files == ["/path/file.abf"]

    def test_assign_with_condition(self):
        db = tsDatabase()
        db.add_cell("C1")
        db.assign_file("C1", "IC1", "/ctrl.abf", condition="control")
        db.assign_file("C1", "IC1", "/ne.abf", condition="NE")
        assert db.get_file_list("C1", "IC1", condition="control") == ["/ctrl.abf"]
        assert db.get_file_list("C1", "IC1", condition="NE") == ["/ne.abf"]

    def test_append_multi_file(self):
        db = tsDatabase()
        db.add_cell("C1")
        db.add_protocol("IC1")
        db.assign_file("C1", "IC1", "/a.abf")
        db.assign_file("C1", "IC1", "/b.abf", append=True)
        files = db.get_file_list("C1", "IC1")
        assert files == ["/a.abf", "/b.abf"]

    def test_unassign(self):
        db = tsDatabase()
        db.add_cell("C1")
        db.add_protocol("IC1")
        db.assign_file("C1", "IC1", "/a.abf")
        db.unassign_file("C1", "IC1")
        assert db.get_file_list("C1", "IC1") == []

    def test_assign_creates_cell_if_missing(self):
        db = tsDatabase()
        db.add_protocol("IC1")
        db.assign_file("NewCell", "IC1", "/a.abf")
        assert "NewCell" in db.cell_names()

    def test_empty_get(self):
        db = tsDatabase()
        assert db.get_file_list("missing", "missing") == []


# ======================================================================
# tsDatabase — metadata
# ======================================================================

class TestMetadata:

    def test_set_and_get(self):
        db = tsDatabase()
        db.add_cell("C1")
        db.set_cell_metadata("C1", "drug", "NE")
        assert db.get_cell("C1")["drug"] == "NE"
        assert "drug" in db.get_metadata_columns()

    def test_metadata_not_in_protocol_cols(self):
        db = tsDatabase()
        db.add_cell("C1")
        db.add_protocol("IC1")
        db.set_cell_metadata("C1", "drug", "NE")
        assert "drug" not in db.get_protocol_columns()
        assert "IC1" not in db.get_metadata_columns()


# ======================================================================
# tsDatabase — XLSX round-trip
# ======================================================================

class TestXLSXRoundTrip:

    def test_save_and_load(self, tmp_path):
        db = tsDatabase()
        db.add_cell("C1", metadata={"drug": "NE", "group": "stress"})
        db.add_cell("C2", metadata={"drug": "aCSF", "group": "control"})
        db.add_protocol("IC1")
        db.add_protocol("Sag")
        db.assign_file("C1", "IC1", "/c1_ic1.abf")
        db.assign_file("C2", "Sag", "/c2_sag.abf")
        db.set_cell_metadata("C1", "notes", "good cell")

        path = str(tmp_path / "test.xlsx")
        db.save_xlsx(path)
        assert os.path.isfile(path)

        # Reload
        db2 = tsDatabase()
        db2.load_xlsx(path)
        assert db2.cell_count() == 2
        assert set(db2.cell_names()) == {"C1", "C2"}
        assert db2.get_file_list("C1", "IC1") == ["/c1_ic1.abf"]
        assert db2.get_cell("C1")["notes"] == "good cell"
        assert "drug" in db2.get_metadata_columns()
        assert "IC1" in db2.get_protocol_columns()

    def test_condition_columns_survive_roundtrip(self, tmp_path):
        db = tsDatabase()
        db.add_cell("C1")
        db.assign_file("C1", "IC1", "/ctrl.abf", condition="control")
        db.assign_file("C1", "IC1", "/ne.abf", condition="NE")

        path = str(tmp_path / "cond.xlsx")
        db.save_xlsx(path)

        db2 = tsDatabase()
        db2.load_xlsx(path)
        assert "IC1 - control" in db2.get_protocol_columns()
        assert "IC1 - NE" in db2.get_protocol_columns()
        assert db2.get_file_list("C1", "IC1", condition="control") == ["/ctrl.abf"]


# ======================================================================
# tsDatabase — CSV round-trip
# ======================================================================

class TestCSVRoundTrip:

    def test_save_and_load(self, tmp_path):
        db = tsDatabase()
        db.add_cell("C1", metadata={"group": "stress"})
        db.add_protocol("IC1")
        db.assign_file("C1", "IC1", "/file.abf")

        path = str(tmp_path / "test.csv")
        db.save_csv(path)
        assert os.path.isfile(path)

        db2 = tsDatabase()
        db2.load_csv(path)
        assert db2.cell_count() == 1
        assert "C1" in db2.cell_names()

    def test_default_save_is_flat_single_row_header(self, tmp_path):
        db = tsDatabase()
        db.add_cell("C1", metadata={"group": "stress"})
        db.add_protocol("IC1")
        db.assign_file("C1", "IC1", "/file.abf")

        path = str(tmp_path / "flat.csv")
        db.save_csv(path)  # default → grouped=False

        with open(path) as f:
            first_line = f.readline().rstrip("\n")
            second_line = f.readline().rstrip("\n")
        # Header row must contain column names directly, not group labels
        assert "IC1" in first_line
        # First data row carries the cell name
        assert second_line.startswith("C1,")


# ======================================================================
# tsDatabase — clear
# ======================================================================

class TestClear:

    def test_clear(self):
        db = tsDatabase()
        db.add_cell("C1")
        db.add_protocol("IC1")
        db.clear()
        assert db.cell_count() == 0
        assert db.get_protocol_columns() == []


# ======================================================================
# tsDatabase — from_dataframe (in-memory ingestion)
# ======================================================================

class TestFromDataFrame:

    def _demo_df(self):
        return pd.DataFrame({
            "CELL_ID": ["C1", "C2"],
            "IC1": ["240507_0000", "240507_0010"],
            "Sag": ["240507_0001", "240507_0011"],
            "drug": ["NE", "aCSF"],
            "NOTE": ["good", "ok"],
        })

    def test_basic_ingest(self):
        db = tsDatabase()
        ok = db.from_dataframe(
            self._demo_df(), cell_id_col="CELL_ID",
            metadata_cols=["drug", "NOTE"],
        )
        assert ok is True
        assert set(db.cell_names()) == {"C1", "C2"}
        assert "IC1" in db.get_protocol_columns()
        assert "Sag" in db.get_protocol_columns()
        assert "drug" in db.get_metadata_columns()
        assert "NOTE" in db.get_metadata_columns()

    def test_auto_classifies_file_id_columns(self):
        db = tsDatabase()
        db.from_dataframe(self._demo_df(), cell_id_col="CELL_ID")
        # date-stamped tokens should be classified as protocols
        assert "IC1" in db.get_protocol_columns()
        assert "Sag" in db.get_protocol_columns()

    def test_filename_cols_alias(self):
        db = tsDatabase()
        db.from_dataframe(
            self._demo_df(), cell_id_col="CELL_ID",
            filename_cols=["IC1"], metadata_cols=["drug", "NOTE"],
        )
        assert "IC1" in db.get_protocol_columns()

    def test_empty_dataframe_returns_false(self):
        db = tsDatabase()
        assert db.from_dataframe(pd.DataFrame()) is False

    def test_matches_load_csv(self, tmp_path):
        # from_dataframe on a flat CSV should equal load_csv on the same file
        df = self._demo_df()
        path = str(tmp_path / "flat.csv")
        df.to_csv(path, index=False)

        db_csv = tsDatabase()
        db_csv.load_csv(path, cell_id_col="CELL_ID",
                        metadata_cols=["drug", "NOTE"])
        db_df = tsDatabase()
        db_df.from_dataframe(df, cell_id_col="CELL_ID",
                             metadata_cols=["drug", "NOTE"])

        assert set(db_csv.get_protocol_columns()) == set(db_df.get_protocol_columns())
        assert set(db_csv.cell_names()) == set(db_df.cell_names())


# ======================================================================
# tsDatabase — feature import / store
# ======================================================================

class TestFeatureImport:

    def _build_db(self):
        db = tsDatabase()
        df = pd.DataFrame({
            "CELL_ID": ["C1", "C2"],
            "IC1": ["240507_0000", "240507_0010"],
            "drug": ["NE", "aCSF"],
        })
        db.from_dataframe(df, cell_id_col="CELL_ID", metadata_cols=["drug"])
        return db

    def _spike_csv(self, tmp_path, name="spikes.csv"):
        spike = pd.DataFrame({
            "foldername": ["/data", "/data"],
            "filename": ["240507_0000", "240507_0010"],
            "protocol": ["IC1", "IC1"],
            "mean_rate": [10.0, 20.0],
            "rheobase": [50.0, 60.0],
        })
        path = str(tmp_path / name)
        spike.to_csv(path, index=False)
        return path

    def test_import_matches_by_stem(self, tmp_path):
        db = self._build_db()
        res = db.import_spike_data(self._spike_csv(tmp_path))
        assert res["matched"] == 2
        assert res["unmatched"] == 0
        assert not db.features.empty
        assert "spike_mean_rate" in db.features.columns
        assert "spike_rheobase" in db.features.columns
        # link columns present
        for col in ("cell", "protocol", "source_file"):
            assert col in db.features.columns

    def test_features_not_added_to_cellindex(self, tmp_path):
        db = self._build_db()
        db.import_spike_data(self._spike_csv(tmp_path))
        assert not any(c.startswith("spike_") for c in db.cellindex.columns)

    def test_unmatched_rows_reported(self, tmp_path):
        db = self._build_db()
        spike = pd.DataFrame({
            "filename": ["999_nomatch"],
            "mean_rate": [1.0],
        })
        path = str(tmp_path / "nomatch.csv")
        spike.to_csv(path, index=False)
        res = db.import_spike_data(path)
        assert res["matched"] == 0
        assert res["unmatched"] == 1
        assert db.features.empty

    def test_create_missing(self, tmp_path):
        db = self._build_db()
        spike = pd.DataFrame({
            "filename": ["new_recording"],
            "protocol": ["IC1"],
            "mean_rate": [5.0],
        })
        path = str(tmp_path / "new.csv")
        spike.to_csv(path, index=False)
        res = db.import_spike_data(path, create_missing=True)
        assert res["matched"] == 1
        assert "new_recording" in db.cell_names()

    def test_get_features_merged(self, tmp_path):
        db = self._build_db()
        db.import_spike_data(self._spike_csv(tmp_path))
        merged = db.get_features(merge=True)
        assert "spike_mean_rate" in merged.columns
        assert merged.loc["C1", "spike_mean_rate"] == 10.0
        assert merged.loc["C2", "spike_mean_rate"] == 20.0

    def test_missing_filename_col_raises(self, tmp_path):
        db = self._build_db()
        bad = pd.DataFrame({"foo": [1]})
        path = str(tmp_path / "bad.csv")
        bad.to_csv(path, index=False)
        with pytest.raises(ValueError):
            db.import_spike_data(path)

    def test_clear_features(self, tmp_path):
        db = self._build_db()
        db.import_spike_data(self._spike_csv(tmp_path))
        db.clear_features()
        assert db.features.empty

    def test_xlsx_feature_sheet_roundtrip(self, tmp_path):
        db = self._build_db()
        db.import_spike_data(self._spike_csv(tmp_path))
        path = str(tmp_path / "db.xlsx")
        db.save_xlsx(path, feature_export="sheet")

        db2 = tsDatabase()
        db2.load_xlsx(path)
        assert not db2.features.empty
        assert "spike_mean_rate" in db2.features.columns
        # main sheet stays lean
        assert not any(c.startswith("spike_") for c in db2.cellindex.columns)

    def test_xlsx_feature_merged(self, tmp_path):
        db = self._build_db()
        db.import_spike_data(self._spike_csv(tmp_path))
        path = str(tmp_path / "merged.xlsx")
        db.save_xlsx(path, feature_export="merged")

        main = pd.read_excel(path, sheet_name="CellIndex", index_col=0)
        assert "spike_mean_rate" in main.columns

    def test_csv_feature_sidecar(self, tmp_path):
        db = self._build_db()
        db.import_spike_data(self._spike_csv(tmp_path))
        path = str(tmp_path / "db.csv")
        db.save_csv(path, feature_export="separate")
        side = path[:-4] + "_features.csv"
        assert os.path.isfile(side)
        sidecar = pd.read_csv(side)
        assert "spike_mean_rate" in sidecar.columns

    def test_to_anndata(self, tmp_path):
        pytest.importorskip("anndata")
        db = self._build_db()
        db.import_spike_data(self._spike_csv(tmp_path))
        adata = db.to_anndata()
        assert adata.n_obs == 2
        assert "spike_mean_rate" in list(adata.var_names)
        assert "cell" in adata.obs.columns

    def test_save_h5ad(self, tmp_path):
        pytest.importorskip("anndata")
        db = self._build_db()
        db.import_spike_data(self._spike_csv(tmp_path))
        path = str(tmp_path / "feats.h5ad")
        out = db.save_h5ad(path)
        assert os.path.isfile(out)


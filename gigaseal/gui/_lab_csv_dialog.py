"""Lab-CSV import dialog for the Database Builder.

Lets the user inspect and override how a multi-row-header lab CSV is
parsed before it is loaded into the active database.
"""

from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

import pandas as pd

from ..database.tsDatabase import (
    tsDatabase,
    _classify_column_values,
    _detect_header_rows,
    _read_grouped_csv,
    resolve_file_ids,
)


_ROLE_METADATA = "Metadata"
_ROLE_PROTOCOL = "Protocol"
_ROLE_SKIP = "Skip"


class LabCSVImportDialog(QDialog):
    """Preview + override dialog for grouped lab CSV imports.

    On *Import*, mutates the supplied :class:`tsDatabase` in place and
    sets :attr:`accepted_path` for the caller. Cancel leaves the DB
    untouched.
    """

    def __init__(self, path: str, db: tsDatabase, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Import lab CSV — {os.path.basename(path)}")
        self.resize(900, 600)
        self._path = path
        self._db = db
        self._preview_df: Optional[pd.DataFrame] = None
        self._group_map: dict = {}
        self.accepted_path: Optional[str] = None

        layout = QVBoxLayout(self)

        # -- top: read options ----------------------------------------
        opts = QGroupBox("Read options")
        form = QFormLayout(opts)
        detected = _detect_header_rows(path)
        self._header_spin = QSpinBox()
        self._header_spin.setRange(1, 3)
        self._header_spin.setValue(detected)
        self._header_spin.valueChanged.connect(self._reload_preview)
        form.addRow("Header rows:", self._header_spin)

        self._row_mode = QComboBox()
        self._row_mode.addItems(["auto", "per_cell", "per_recording"])
        form.addRow("Row mode:", self._row_mode)

        self._cell_id_combo = QComboBox()
        self._cell_id_combo.setEditable(True)
        form.addRow("Cell-ID column:", self._cell_id_combo)

        self._unique_id_combo = QComboBox()
        self._unique_id_combo.setEditable(True)
        form.addRow("Unique-ID column:", self._unique_id_combo)

        self._drug_edit = QLineEdit("drug")
        form.addRow("Drug column name:", self._drug_edit)

        layout.addWidget(opts)

        # -- middle: per-column table ---------------------------------
        layout.addWidget(QLabel("Columns (override role / condition as needed):"))
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["Column", "Group", "Detected", "Role", "Condition"]
        )
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        layout.addWidget(self._table, 1)

        # -- bottom: optional file-ID resolver ------------------------
        resolver_box = QGroupBox("Optional: resolve numeric file IDs against a folder")
        rl = QHBoxLayout(resolver_box)
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText("(none — values kept as-is)")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._pick_folder)
        rl.addWidget(self._folder_edit, 1)
        rl.addWidget(browse)
        layout.addWidget(resolver_box)

        # -- dialog buttons -------------------------------------------
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Import")
        bb.accepted.connect(self._on_import)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

        self._reload_preview()

    # ------------------------------------------------------------------

    def _reload_preview(self):
        """Re-parse the CSV with current header depth and repopulate UI."""
        n_hdr = self._header_spin.value()
        try:
            if n_hdr == 1:
                df = pd.read_csv(self._path)
                self._group_map = {c: "" for c in df.columns}
            else:
                df, self._group_map = _read_grouped_csv(self._path, n_hdr)
        except Exception as e:
            QMessageBox.warning(
                self, "Preview failed",
                f"Could not parse with {n_hdr} header row(s):\n{e}",
            )
            return
        self._preview_df = df

        # Refresh cell-id / unique-id dropdowns
        cols = list(df.columns)
        prev_cell = self._cell_id_combo.currentText()
        prev_uid = self._unique_id_combo.currentText()
        self._cell_id_combo.blockSignals(True)
        self._unique_id_combo.blockSignals(True)
        self._cell_id_combo.clear()
        self._unique_id_combo.clear()
        self._cell_id_combo.addItem("(auto)")
        self._unique_id_combo.addItem("(auto)")
        self._cell_id_combo.addItems([str(c) for c in cols])
        self._unique_id_combo.addItems([str(c) for c in cols])
        if prev_cell and prev_cell in cols:
            self._cell_id_combo.setCurrentText(prev_cell)
        if prev_uid and prev_uid in cols:
            self._unique_id_combo.setCurrentText(prev_uid)
        self._cell_id_combo.blockSignals(False)
        self._unique_id_combo.blockSignals(False)

        # Per-column table
        self._table.setRowCount(len(cols))
        for i, col in enumerate(cols):
            group = self._group_map.get(col, "")
            kind = _classify_column_values(df[col])
            role = _ROLE_PROTOCOL if kind == "file_id" else _ROLE_METADATA

            self._table.setItem(i, 0, _ro_item(str(col)))
            self._table.setItem(i, 1, _ro_item(group))
            self._table.setItem(i, 2, _ro_item(kind))

            role_combo = QComboBox()
            role_combo.addItems([_ROLE_METADATA, _ROLE_PROTOCOL, _ROLE_SKIP])
            role_combo.setCurrentText(role)
            self._table.setCellWidget(i, 3, role_combo)

            cond_item = QTableWidgetItem("")
            self._table.setItem(i, 4, cond_item)

    # ------------------------------------------------------------------

    def _pick_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Pick recording folder", os.path.dirname(self._path),
        )
        if folder:
            self._folder_edit.setText(folder)

    # ------------------------------------------------------------------

    def _on_import(self):
        df = self._preview_df
        if df is None:
            QMessageBox.warning(self, "Import", "Preview is not ready.")
            return

        # Collect role overrides
        explicit_meta: list = []
        explicit_proto: list = []
        skip_cols: list = []
        for i in range(self._table.rowCount()):
            col = self._table.item(i, 0).text()
            role_widget = self._table.cellWidget(i, 3)
            role = role_widget.currentText() if role_widget else _ROLE_METADATA
            if role == _ROLE_METADATA:
                explicit_meta.append(col)
            elif role == _ROLE_PROTOCOL:
                explicit_proto.append(col)
            else:
                skip_cols.append(col)

        cell_id = self._cell_id_combo.currentText()
        cell_id = None if cell_id in ("", "(auto)") else cell_id
        unique_id = self._unique_id_combo.currentText()
        unique_id = None if unique_id in ("", "(auto)") else unique_id

        try:
            self._db.load_csv(
                self._path,
                cell_id_col=cell_id,
                protocol_cols=explicit_proto or None,
                metadata_cols=explicit_meta or None,
                header_rows=self._header_spin.value(),
                header_mode="grouped" if self._header_spin.value() > 1 else "flat",
                row_mode=self._row_mode.currentText(),
                drug_col=self._drug_edit.text().strip() or "drug",
                unique_id_col=unique_id,
            )
        except Exception as e:
            QMessageBox.critical(self, "Import failed", str(e))
            return

        # Drop user-skipped columns after load
        if skip_cols:
            keep = [c for c in self._db.cellindex.columns if c not in skip_cols]
            self._db.cellindex = self._db.cellindex[keep]

        # Optional ID resolution
        folder = self._folder_edit.text().strip()
        if folder and os.path.isdir(folder):
            try:
                summary = resolve_file_ids(self._db, folder)
                QMessageBox.information(
                    self, "File-ID resolution",
                    f"Resolved {summary['resolved']} reference(s);\n"
                    f"unresolved: {summary['unresolved']}\n"
                    f"collisions: {len(summary['collisions'])}",
                )
            except Exception as e:
                QMessageBox.warning(self, "Resolution failed", str(e))

        self.accepted_path = self._path
        self.accept()


def _ro_item(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    return item

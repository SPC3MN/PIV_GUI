"""Qt table model backing run_panel's per-pair progress view."""

from PySide6.QtCore import QAbstractTableModel, Qt

# Short headers on purpose. "Rejected (range)" / "Rejected (std-dev)" were
# clipped to "ejected (rang" / "ected (std-d" at the panel's real width -- a
# header truncated mid-word is worse than a terse one, because the reader
# cannot tell which column they are looking at. The full meaning lives in the
# header tooltips (see run_panel).
COLUMNS = ["Pair", "Status", "Time (s)", "Valid", "Total", "Rej. range", "Rej. σ"]

#: Long-form column meanings, shown as header tooltips.
COLUMN_TOOLTIPS = [
    "Pair identifier, as listed in the Preview tab",
    "running / done / error",
    "Correlation time for this pair",
    "Vectors that survived validation",
    "Grid points in total",
    "Vectors rejected by the local median (universal outlier detection) filter",
    "Vectors rejected by the field-wide standard-deviation filter",
]


class JobModel(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows = []  # list of dicts: pair_id, status, elapsed, n_valid, n_total, ...

    def reset(self):
        self.beginResetModel()
        self._rows = []
        self.endResetModel()

    def start_pair(self, pair_id):
        self.beginInsertRows(self.index(0, 0).parent(), len(self._rows), len(self._rows))
        self._rows.append({"pair_id": pair_id, "status": "running"})
        self.endInsertRows()

    def _find_row(self, pair_id):
        for i, row in enumerate(self._rows):
            if row["pair_id"] == pair_id:
                return i
        return -1

    def finish_pair(self, pair_id, result):
        i = self._find_row(pair_id)
        if i == -1:
            return
        self._rows[i].update(status="done", **result)
        self.dataChanged.emit(self.index(i, 0), self.index(i, len(COLUMNS) - 1))

    def error_pair(self, pair_id, message):
        i = self._find_row(pair_id)
        if i == -1:
            return
        self._rows[i].update(status=f"error: {message}")
        self.dataChanged.emit(self.index(i, 0), self.index(i, len(COLUMNS) - 1))

    def rowCount(self, parent=None):
        return len(self._rows)

    def columnCount(self, parent=None):
        return len(COLUMNS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation != Qt.Horizontal:
            return None
        if role == Qt.DisplayRole:
            return COLUMNS[section]
        if role == Qt.ToolTipRole:
            return COLUMN_TOOLTIPS[section]
        return None

    def data(self, index, role=Qt.DisplayRole):
        if role != Qt.DisplayRole or not index.isValid():
            return None
        row = self._rows[index.row()]
        col = index.column()
        if col == 0:
            return row.get("pair_id", "")
        if col == 1:
            return row.get("status", "")
        if col == 2:
            elapsed = row.get("elapsed")
            return f"{elapsed:.3f}" if elapsed is not None else ""
        if col == 3:
            return row.get("n_valid", "")
        if col == 4:
            return row.get("n_total", "")
        if col == 5:
            return row.get("n_rejected_range_residual", "")
        if col == 6:
            return row.get("n_rejected_std_dev", "")
        return None

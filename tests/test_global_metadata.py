from __future__ import annotations

import ast
import csv
import importlib.util
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_global_metadata", PROJECT_ROOT / "data/build_global_metadata.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class GlobalMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.processed = self.root / "processed_data"
        self.processed.mkdir()
        self.output = self.root / "metadata.csv"

    def write_metadata(self, method_id, row):
        folder = self.processed / method_id
        folder.mkdir()
        path = folder / f"{method_id}_metadata.tsv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row), delimiter="\t")
            writer.writeheader()
            writer.writerow(row)
        return path

    def test_reordered_headers_are_matched_by_name(self):
        self.write_metadata("0002", {"column.temperature": "30", "id": "0002",
                                     "column.length": "100", "column.particle.size": "1.8"})
        self.write_metadata("0001", {"id": "0001", "column.particle.size": "1.7",
                                     "column.length": "150", "column.temperature": "40"})
        self.assertEqual(builder.build_global_metadata(self.processed, self.output), 2)
        with self.output.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["id"] for row in rows], ["0001", "0002"])
        self.assertEqual([row["column.temperature"] for row in rows], ["40", "30"])
        self.assertEqual([row["column.length"] for row in rows], ["150", "100"])
        self.assertEqual([row["column.particle.size"] for row in rows], ["1.7", "1.8"])

    def test_training_reader_can_use_first_column_as_method_index(self):
        self.write_metadata("0001", {"id": "0001", "column.temperature": "40"})
        builder.build_global_metadata(self.processed, self.output)
        frame = pd.read_csv(self.output, index_col=0)
        frame.index = frame.index.map(lambda value: str(value).zfill(4))
        self.assertEqual(frame.loc["0001", "column.temperature"], 40)
        self.assertEqual(frame.index.name, "id")

    def test_current_engine_reads_named_values_instead_of_wrong_positions(self):
        self.write_metadata("0001", {
            "id": "0001", "column.name": "C18", "column.usp.code": "L1",
            "column.length": "150", "column.id": "2.1",
            "column.particle.size": "1.8", "column.temperature": "40",
        })
        builder.build_global_metadata(self.processed, self.output)
        path = PROJECT_ROOT / "model/engine.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {"to_dataset_id", "_norm_text", "_case_insensitive_get",
                 "read_report_global_metadata", "get_meta_row",
                 "read_per_method_metadata_matrix", "get_matrix_value", "get_meta_or_matrix"}
        module = ast.Module(body=[node for node in tree.body
                                  if isinstance(node, ast.FunctionDef) and node.name in names],
                            type_ignores=[])
        namespace = dict(pd=pd, np=np, re=re, Path=Path, META_PATH=self.output,
                         PROCESSED_DIR=self.processed)
        exec(compile(module, str(path), "exec"), namespace)
        namespace["df_meta"] = namespace["read_report_global_metadata"](self.output)
        for field, legacy_position, expected in [
            ("column.length", 2, 150), ("column.id", 3, 2.1),
            ("column.particle.size", 4, 1.8), ("column.temperature", 5, 40),
        ]:
            actual = namespace["get_meta_or_matrix"]("0001", [field], matrix_col=legacy_position)
            self.assertEqual(float(actual), expected)

    def test_blank_values_and_additional_fields_are_preserved(self):
        self.write_metadata("0001", {"id": "0001", "column.temperature": "", "note": "A,B"})
        self.write_metadata("0002", {"id": "0002", "column.temperature": "0", "new.field": "abc"})
        builder.build_global_metadata(self.processed, self.output)
        with self.output.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["column.temperature"], "")
        self.assertEqual(rows[1]["column.temperature"], "0")
        self.assertEqual(rows[0]["note"], "A,B")
        self.assertEqual(rows[0]["new.field"], "")
        self.assertEqual(rows[1]["new.field"], "abc")

    def test_existing_output_is_not_overwritten_by_default(self):
        self.write_metadata("0001", {"id": "0001", "column.temperature": "40"})
        self.output.write_text("existing content", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            builder.build_global_metadata(self.processed, self.output)
        self.assertEqual(self.output.read_text(encoding="utf-8"), "existing content")

    def test_explicit_overwrite(self):
        self.write_metadata("0001", {"id": "0001", "column.temperature": "40"})
        self.output.write_text("old content", encoding="utf-8")
        builder.build_global_metadata(self.processed, self.output, overwrite=True)
        self.assertIn("0001,40", self.output.read_text(encoding="utf-8"))

    def test_mismatched_method_id_does_not_replace_existing_output(self):
        self.write_metadata("0001", {"id": "0002", "column.temperature": "40"})
        self.output.write_text("existing content", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differs"):
            builder.build_global_metadata(self.processed, self.output, overwrite=True)
        self.assertEqual(self.output.read_text(encoding="utf-8"), "existing content")

    def test_missing_metadata_is_not_silently_skipped(self):
        (self.processed / "0001").mkdir()
        with self.assertRaises(FileNotFoundError):
            builder.build_global_metadata(self.processed, self.output)
        self.assertFalse(self.output.exists())

    def test_multiple_rows_are_rejected(self):
        path = self.write_metadata("0001", {"id": "0001", "column.temperature": "40"})
        with path.open("a", encoding="utf-8") as handle:
            handle.write("0001\t50\n")
        with self.assertRaisesRegex(ValueError, "one metadata row"):
            builder.build_global_metadata(self.processed, self.output)
        self.assertFalse(self.output.exists())

    def test_duplicate_headers_are_rejected(self):
        path = self.write_metadata("0001", {"id": "0001"})
        path.write_text("id\tid\n0001\t0001\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            builder.build_global_metadata(self.processed, self.output)

    def test_empty_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "No four-digit"):
            builder.build_global_metadata(self.processed, self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()

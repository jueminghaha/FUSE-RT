from __future__ import annotations

import ast
import csv
from datetime import datetime
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METHODS = ['0391', '0390', '0437', '0420', '0419', '0411']


def extracted_helpers(relative, names, namespace):
    path = PROJECT_ROOT / relative
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    module = ast.Module(body=[
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ], type_ignores=[])
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace


def literal_assignment(relative, name):
    tree = ast.parse((PROJECT_ROOT / relative).read_text(encoding='utf-8'))
    return next(ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in node.targets))


class ExternalOODScopeTests(unittest.TestCase):
    def setUp(self):
        self.helpers = extracted_helpers('script/evaluate_external_ood.py', {
            'select_report_ood_rows', '_read_csv_flexible', 'safe_read_csv',
            'robust_read_csv', '_backup_path', 'normalize_existing_csv_to_schema',
        }, dict(pd=pd, np=np, Path=Path, datetime=datetime, shutil=shutil))

    def test_default_report_methods_are_unchanged(self):
        self.assertEqual(literal_assignment('script/evaluate_external_ood.py',
                                           'LOW_OVERLAP_METHOD_IDS'), DEFAULT_METHODS)
        self.assertEqual(literal_assignment('script/evaluate_unirt_external_ood.py',
                                           'TARGET_OOD_METHODS'), DEFAULT_METHODS)

    def test_scope_filter_preserves_report_methods_and_global_audits(self):
        frame = pd.DataFrame([
            dict(ood_tier='new_report_low_overlap_ood', ood_task_id='0411', mae_sec=12.345),
            dict(ood_tier='new_report_low_overlap_ood', ood_task_id='0186', mae_sec=23.456),
            dict(ood_tier='excluded_external_ood', ood_task_id='private_task', mae_sec=99.0),
            dict(ood_tier='new_report_low_overlap_ood', ood_task_id='private_task', mae_sec=88.0),
            dict(ood_tier='', ood_task_id='', mae_sec=0.0),
        ])
        original = frame.copy(deep=True)
        selected = self.helpers['select_report_ood_rows'](frame)
        pd.testing.assert_frame_equal(selected, frame.loc[[0, 1, 4]])
        pd.testing.assert_frame_equal(frame, original)

    def test_numeric_ids_keep_their_leading_zeroes_and_metrics(self):
        frame = pd.DataFrame({'ood_method_id': [186.0, 390.0, 411.0],
                              'mae_sec': [1.1, 2.2, 3.3]})
        selected = self.helpers['select_report_ood_rows'](frame)
        self.assertEqual(selected.ood_method_id.tolist(), ['0186', '0390', '0411'])
        self.assertEqual(selected.mae_sec.tolist(), [1.1, 2.2, 3.3])

    def test_summary_filter_without_task_ids(self):
        frame = pd.DataFrame({'ood_tier': ['new_report_low_overlap_ood', 'excluded_external_ood'],
                              'mae_sec_mean': [12.345, 99.0]})
        pd.testing.assert_frame_equal(self.helpers['select_report_ood_rows'](frame), frame.iloc[:1])

    def test_empty_and_task_independent_frames(self):
        for frame in [pd.DataFrame(columns=['ood_tier', 'ood_task_id']),
                      pd.DataFrame({'EXP_ID': ['E1'], 'seed': [2004]})]:
            pd.testing.assert_frame_equal(self.helpers['select_report_ood_rows'](frame), frame)

    def test_resume_and_aggregate_readers_filter_old_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metrics.csv'
            path.write_text(
                'ood_tier,ood_task_id,ood_method_id,mae_sec\n'
                'new_report_low_overlap_ood,0411,0411,12.345\n'
                'new_report_low_overlap_ood,0186,0186,23.456\n'
                'excluded_external_ood,private_task,private_task,99.0\n',
                encoding='utf-8',
            )
            for name in ['safe_read_csv', 'robust_read_csv']:
                selected = self.helpers[name](path)
                self.assertEqual(selected.ood_task_id.tolist(), ['0411', '0186'])
                self.assertEqual(selected.ood_method_id.tolist(), ['0411', '0186'])
                self.assertEqual(selected.mae_sec.tolist(), [12.345, 23.456])

    def test_normalization_removes_stale_rows_and_preserves_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metrics.csv'
            original = ('ood_tier,ood_task_id,mae_sec\n'
                        'new_report_low_overlap_ood,0411,12.345\n'
                        'excluded_external_ood,private_task,99.0\n')
            path.write_text(original, encoding='utf-8')
            self.helpers['normalize_existing_csv_to_schema'](
                path, ['ood_tier', 'ood_task_id', 'mae_sec'])
            selected = pd.read_csv(path, dtype={'ood_task_id': str})
            self.assertEqual(selected.ood_task_id.tolist(), ['0411'])
            self.assertEqual(selected.mae_sec.tolist(), [12.345])
            backups = list(Path(directory).glob('metrics.schema_backup.*.csv'))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(encoding='utf-8'), original)

    def test_flexible_csv_reader_filters_out_of_scope_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metrics.csv'
            path.write_text(
                'ood_tier,ood_task_id,mae_sec\n'
                'new_report_low_overlap_ood,0411,12.345\n'
                'excluded_external_ood,private_task,99.0,extra\n',
                encoding='utf-8',
            )
            result = self.helpers['safe_read_csv'](path)
            self.assertEqual(result.ood_task_id.tolist(), ['0411'])
            self.assertEqual(float(result.mae_sec.iloc[0]), 12.345)

    def test_published_tables_contain_only_report_records(self):
        for filename in ['per_model_seed.csv', 'summary.csv']:
            with (PROJECT_ROOT / 'result/external_ood' / filename).open(
                    encoding='utf-8', newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertEqual({row['ood_tier'] for row in rows}, {'new_report_low_overlap_ood'})
            if filename == 'per_model_seed.csv':
                self.assertIn('0186', {row['ood_task_id'] for row in rows})
                self.assertTrue(all(row['ood_task_id'].isdigit()
                                    and len(row['ood_task_id']) == 4 for row in rows))

    def test_unirt_prefers_the_report_only_export(self):
        path = PROJECT_ROOT / 'script/evaluate_unirt_external_ood.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        node = next(node for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'EXTERNAL_ROWS_CANDIDATES'
                            for t in node.targets))
        first = eval(compile(ast.Expression(node.value.elts[0]), str(path), 'eval'),
                     {'PROJECT_ROOT': PROJECT_ROOT})
        self.assertEqual(first, PROJECT_ROOT / 'result/_runs/external_ood/ood_rows_master_lowoverlap.csv')

    def test_migration_copies_curated_files_without_changing_source(self):
        namespace = extracted_helpers('script/migrate_notebooks.py', {'copy_curated_files'}, {
            '__file__': str(PROJECT_ROOT / 'script/migrate_notebooks.py'),
            'Path': Path, 'Sequence': list, 'shutil': shutil,
        })
        relative = 'script/evaluate_external_ood.py'
        before = (PROJECT_ROOT / relative).read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            namespace['copy_curated_files'](target, [relative])
            self.assertEqual((target / relative).read_bytes(), before)
        namespace['copy_curated_files'](PROJECT_ROOT, [relative])
        self.assertEqual((PROJECT_ROOT / relative).read_bytes(), before)


if __name__ == '__main__':
    unittest.main()

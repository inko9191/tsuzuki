"""CPU checks of fixed membership, preflight, durable single sends and pause."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import locomo_long_c8k as c


def fixture():
    ids = [f'g{i//3}/q{i}' for i in range(30)]
    items = {f'{c.LENGTH}|{c.ARM}|{q}': {'id': q, 'input_tokens': 1600} for q in ids}
    return ids, items


class C8k(unittest.TestCase):
    def test_all_frozen_thirty_fit(self):
        ids, items = fixture()
        c.preflight(items, ids)

    def test_exact_boundary_counts_output_and_margin(self):
        ids, items = fixture()
        first = next(iter(items.values()))
        first['input_tokens'] = 8192 - 384
        c.preflight(items, ids)
        first['input_tokens'] += 1
        with self.assertRaises(RuntimeError):
            c.preflight(items, ids)

    def test_extra_or_missing_inputs_not_allowed(self):
        ids, items = fixture()
        items.pop(next(iter(items)))
        with self.assertRaises(RuntimeError):
            c.preflight(items, ids)

    def test_duplicate_questions_rejected(self):
        ids, items = fixture()
        ids[-1] = ids[0]
        with self.assertRaises(RuntimeError):
            c.preflight(items, ids)

    def test_no_scoring_partial_responses(self):
        ids, items = fixture()
        with patch.object(c.base, 'main_row') as score:
            with self.assertRaises(RuntimeError):
                c.rows_after_complete(items, {}, ids)
        score.assert_not_called()

    def test_complete_rows_preserve_registered_order(self):
        ids, items = fixture()
        replies = {key: {} for key in reversed(list(items))}
        with patch.object(c.base, 'main_row', side_effect=lambda item, reply: item['id']):
            self.assertEqual(c.rows_after_complete(items, replies, ids), ids)

    def test_single_send_rejects_received_key_before_http(self):
        with patch.object(c.ae, 'gate'), patch.object(c.base.st, 'http_json') as http:
            with self.assertRaises(RuntimeError):
                c.base.send('saved', 'answer', {}, '', 0, {}, {'saved': {}}, None)
        http.assert_not_called()

    def test_bindings_restore_on_failure(self):
        saved = c.base.LOCAL
        with self.assertRaises(ValueError):
            with c.isolated():
                self.assertEqual(c.base.LOCAL, c.LOCAL)
                raise ValueError('synthetic')
        self.assertEqual(c.base.LOCAL, saved)

    def test_pause_persists_without_process_or_server_signal(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(c.ae, 'PAUSE', Path(tmp) / 'pause.json'), patch.object(c.os, 'kill') as kill:
            c.pause()
            value = json.loads(c.ae.PAUSE.read_text())
            self.assertEqual(value['reason'], 'human requested pause')
        kill.assert_not_called()

    def test_A_resume_worker_blocks_new_run(self):
        with patch.object(c.subprocess, 'check_output', return_value='999999 python scripts/eval/locomo_long_continue.py resume-a\n'):
            with self.assertRaises(RuntimeError):
                c.other_workers()


if __name__ == '__main__':
    unittest.main()

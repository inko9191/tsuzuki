"""CPU-only invariants for matched counts, cache isolation and peak inference."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import locomo_long_footprint as f


def startup(window=8192, off=False):
    return (f'initializing, n_slots = 1, n_ctx_slot = {window}, kv_unified = false\n'
            'context checkpoints enabled, max = 32, min spacing = 8192\n'+
            ('prompt cache is disabled\n' if off else 'prompt cache is enabled, size limit: 8192 MiB\n'))


def row(n, peak):
    return {'memory': {'cumulative_peak': {'physical_footprint_bytes': peak, 'rss_bytes': peak+100},
                       'after': {'physical_footprint_bytes': peak-1}},
            'timing': {'wall_seconds': n, 'finish_reason': 'stop'}, 'OS': {'swap_used_bytes': 3}, 'cache': {}}


class Footprint(unittest.TestCase):
    def test_balanced_reverse_order_two_repeats(self):
        jobs = f.schedule()
        self.assertEqual(len(jobs), 8)
        self.assertEqual([(a, c) for r, a, c in jobs[:4]], list(reversed([(a, c) for r, a, c in jobs[4:]])))
        for arm in ('default', 'host-cache-off'):
            for window in (8192, 131072):
                self.assertEqual(sum(a == arm and c == window for r, a, c in jobs), 2)

    def test_cache_setting_verified_from_effective_log(self):
        self.assertEqual(f.cache_settings(startup(), 'default', 8192)['host_cache_limit_bytes'], 8*f.GIB)
        self.assertFalse(f.cache_settings(startup(off=True), 'host-cache-off', 8192)['host_cache_enabled'])

    def test_mismatched_cache_or_slots_or_window_stop(self):
        for text, arm, window in ((startup(), 'host-cache-off', 8192),
                                  (startup().replace('slots = 1', 'slots = 2'), 'default', 8192),
                                  (startup(), 'default', 131072)):
            with self.assertRaises(RuntimeError):
                f.cache_settings(text, arm, window)

    def test_missing_context_checkpoint_setting_stops(self):
        with self.assertRaises(RuntimeError):
            f.cache_settings(startup().replace('max = 32', 'max = 0'), 'default', 8192)

    def test_cache_no_log_is_unknown_not_zero(self):
        self.assertIsNone(f.cache_observations('disabled')['largest_logged_bytes'])

    def test_cache_logged_size_and_current_size_differ(self):
        value = f.cache_observations('cache state: 4 prompts, 1024.000 MiB\ncache state: 1 prompts, 500.000 MiB\n')
        self.assertEqual(value['largest_logged_bytes'], f.GIB)
        self.assertEqual(value['last']['bytes'], 500*1024**2)

    def test_vmmap_rounded_swapped_not_compressed(self):
        value = f.vmmap_summary('TOTAL 400.0G 26.1G 8.9G 32K 0K 0K\nMalloc Large 8.0G 7.0G 7.0G 1.0G 0K\n')
        self.assertEqual(value['regions_bytes_rounded']['TOTAL']['swapped'], 32768)
        self.assertIsNone(value['per_process_compressed_bytes'])

    def test_only_registered_count_curves(self):
        with self.assertRaises(RuntimeError):
            f.curve([row(1, 100)]*2)

    def test_curve_uses_cumulative_peak_not_current_after(self):
        value = f.curve([row(1, 100), row(2, 120), row(3, 150)])
        self.assertEqual(value['peak']['physical_footprint_bytes'], 150)
        self.assertEqual(value['HTTP_seconds_per_question'], 2)
        self.assertTrue(value['provisional'])

    def test_comparison_needs_exactly_two_repeats(self):
        with self.assertRaises(RuntimeError):
            f.conservative_gap([100], [200, 210])

    def test_gap_must_exceed_repeat_range(self):
        self.assertTrue(f.conservative_gap([100, 105], [200, 210])['reduction_demonstrated'])
        self.assertFalse(f.conservative_gap([100, 150], [160, 200])['reduction_demonstrated'])
        self.assertFalse(f.conservative_gap([200, 210], [100, 105])['reduction_demonstrated'])

    def test_serial_pipeline_max_and_arithmetic_sum_are_distinct(self):
        a = {'30': f.curve([row(1, 100)]*30)}
        b = {'30': f.curve([row(1, 150)]*30)}
        value = f.pair_curves(a, b)['30']
        self.assertEqual(value['serial_pipeline_peak']['physical_footprint_bytes'], 150)
        self.assertEqual(value['arithmetic_sum_of_stage_peaks']['physical_footprint_bytes'], 250)

    def test_pair_curves_reject_unmatched_counts(self):
        with self.assertRaises(RuntimeError):
            f.pair_curves({'3': {}}, {'30': {}})

    def test_request_peak_resets_and_cumulative_peak_remains(self):
        fake = {'rss_bytes': 100, 'physical_footprint_bytes': 50, 'kernel_lifetime_peak_footprint_bytes': 50}
        system = {'available_bytes_estimate': 20*f.GIB, 'pressure_level': 1}
        with patch.object(f.base.resources, 'observe_os', return_value=system), patch.object(f.base.resources.audit, 'memory', return_value=fake):
            watch = f.SeriesWatch(123, lambda *a: None)
            watch.begin(1)
            watch.sample({k: v+100 for k, v in fake.items()})
            watch.begin(2)
            boundary = watch.boundary()
        self.assertEqual(boundary['request_peak']['rss_bytes'], 100)
        self.assertEqual(boundary['cumulative_peak']['rss_bytes'], 200)
        self.assertEqual(watch.series[-1]['question_number'], 2)

    def test_received_or_unknown_inflight_prevents_http(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(f.audit, 'LOCAL', Path(tmp)), patch.object(f.ae, 'gate'), patch.object(f.base.st, 'http_json') as http:
            for replies in ({'done': {}}, {}):
                if not replies:
                    (Path(tmp)/'inflight.json').write_text('{}')
                with self.assertRaises(RuntimeError):
                    f.audit.send('done', {}, {}, replies, None)
            http.assert_not_called()

    def test_runtime_binds_new_plan_and_restores_environment(self):
        saved = f.audit.PLAN
        with patch.dict(f.os.environ, {'LLAMA_ARG_CACHE_RAM': '123'}):
            with f.runtime('host-cache-off'):
                self.assertEqual(f.audit.PLAN, f.PLAN)
                self.assertEqual(f.os.environ['LLAMA_ARG_CACHE_RAM'], '0')
            self.assertEqual(f.os.environ['LLAMA_ARG_CACHE_RAM'], '123')
        self.assertEqual(f.audit.PLAN, saved)

    def test_pause_does_not_signal_qwen(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(f.ae, 'PAUSE', Path(tmp)/'pause.json'), patch.object(f.os, 'kill') as kill:
            f.pause()
            self.assertEqual(json.loads(f.ae.PAUSE.read_text())['reason'], 'human requested pause')
        kill.assert_not_called()

    def test_protected_journal_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(f.lm, 'ROOT', Path(tmp)):
            path = Path(tmp)/'received.jsonl'
            path.write_text('original')
            p = {'source_sha256': {}, 'private_source_sha256': {'received.jsonl': f.lm.file_hash(path)}}
            f.check_sources(p)
            path.write_text('changed')
            with self.assertRaises(RuntimeError):
                f.check_sources(p)

    def test_full_cells_report_answer_and_pipeline_separately(self):
        cells = {}
        for r, arm, window in f.schedule():
            peak = (100 if window == 8192 else 200)+(r-1)
            if arm == 'host-cache-off':
                peak -= 20
            answer = {'30': f.curve([row(1, peak)]*30)}
            rerank = {'30': f.curve([row(1, 250)]*30)}
            cells[f'r{r}-{arm}-c{window}'] = {'paired_curve': f.pair_curves(answer, rerank),
                                            'answer': {'curve': answer}, 'rerank': {'curve': rerank}}
        values = f.comparisons(cells)
        self.assertTrue(values['default|answer']['reduction_demonstrated'])
        self.assertFalse(values['default|serial_pipeline']['reduction_demonstrated'])
        self.assertTrue(values['host-cache-effect|8192|answer']['reduction_demonstrated'])
        self.assertFalse(values['host-cache-effect|8192|rerank']['reduction_demonstrated'])


if __name__ == '__main__':
    unittest.main()

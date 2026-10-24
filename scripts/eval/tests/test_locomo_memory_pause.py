"""Pause routing tests; never signal a real evaluation or model server."""
import json
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import locomo_memory_pause as pause


class Pause(unittest.TestCase):
    def test_only_this_checkout_python_runners_are_targets(self):
        target = pause.ROOT / 'scripts/eval/locomo_memory_stages.py'
        self.assertEqual(pause.script(f'python -u {target} generate rw', pause.ROOT), target)
        self.assertEqual(pause.script('python -u scripts/eval/locomo_memory_resume.py', pause.ROOT),
                         pause.ROOT / 'scripts/eval/locomo_memory_resume.py')
        for argv, cwd in [('llama-server -m Qwen.gguf', pause.ROOT),
                          ('zsh -c "python scripts/eval/locomo_memory_resume.py"', pause.ROOT),
                          ('python -c "locomo_memory_stages.py"', pause.ROOT),
                          ('python scripts/eval/locomo_memory_stages.py', Path('/another/project'))]:
            self.assertIsNone(pause.script(argv, cwd))

    def test_marker_precedes_one_signal_per_group(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(pause, 'MARKER', Path(tmp) / 'pause.json'), \
                patch.object(pause.os, 'getpgrp', return_value=90):
            def verify(group, signum):
                self.assertEqual(group, 10)
                self.assertEqual(signum, signal.SIGINT)
                self.assertEqual(json.loads(pause.MARKER.read_text())['reason'],
                                 'explicit user pause; do not automatically resume')
            with patch.object(pause.os, 'killpg', side_effect=verify) as kill:
                pause.pause([{'pid': 10, 'pgid': 10}, {'pid': 11, 'pgid': 10}])
                kill.assert_called_once()

    def test_dry_run_cannot_write_or_signal(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(pause, 'MARKER', Path(tmp) / 'pause.json'), \
                patch.object(pause.os, 'getpgrp', return_value=90), patch.object(pause.os, 'killpg') as kill:
            pause.pause([{'pid': 10, 'pgid': 10}], dry_run=True)
            self.assertFalse(pause.MARKER.exists())
            kill.assert_not_called()

    def test_caller_group_refused_before_marker(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(pause, 'MARKER', Path(tmp) / 'pause.json'), \
                patch.object(pause.os, 'getpgrp', return_value=10), patch.object(pause.os, 'killpg') as kill:
            with self.assertRaisesRegex(RuntimeError, 'caller process group'):
                pause.pause([{'pid': 10, 'pgid': 10}])
            self.assertFalse(pause.MARKER.exists())
            kill.assert_not_called()


if __name__ == '__main__':
    unittest.main()

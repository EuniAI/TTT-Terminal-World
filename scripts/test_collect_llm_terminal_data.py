"""Batch/output tests; model integration tests live in test_collect_mini.py."""
import io
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace
from unittest.mock import patch

from collect_llm_terminal_data import MODEL, collect_llm


class LLMCollectionTests(unittest.TestCase):
    def setUp(self):
        runtime = patch('collect_llm_terminal_data.check_runtime')
        runtime.start()
        self.addCleanup(runtime.stop)

    def test_direct_output_is_visible_between_trajectories(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(output=Path(tmp) / 'out.jsonl', overwrite=False,
                image='test', model=MODEL, seed=42, num_trajectories=2, workers=1)

            def worker(args, key, index, stop, states):
                self.assertEqual(list(Path(tmp).iterdir()), [args.output])
                rows = args.output.read_text().splitlines()
                self.assertEqual(len(rows), index)
                if rows:
                    self.assertEqual(json.loads(rows[0])['trajectory'][0]['output'], 'first')
                return [{'command': 'echo first', 'output': 'first'}], 'completed', None, Counter()

            with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'offline-test-key'}), \
                 patch('collect_llm_terminal_data.shutil.which', return_value='/fake/docker'), \
                 patch('collect_llm_terminal_data.subprocess.run', return_value=SimpleNamespace(returncode=0)), \
                 patch('collect_llm_terminal_data.collect_llm_trajectory', side_effect=worker), \
                 patch('sys.stderr', new_callable=io.StringIO):
                stats = collect_llm(args)
                before = args.output.read_bytes()
                with self.assertRaises(FileExistsError):
                    collect_llm(args)
                self.assertEqual(args.output.read_bytes(), before)
            self.assertEqual(stats['trajectories_written'], 2)

    def test_parallel_failure_stops_scheduling_and_drains_active_workers(self):
        barrier = Barrier(2, timeout=5)
        failed = Event()
        started = []

        def worker(args, key, index, stop, states):
            started.append(index)
            barrier.wait()
            if index == 0:
                stop.set()
                failed.set()
                return [{'command': 'first', 'output': 'valid prefix'}], 'error_prefix', 'test failure', Counter(prompt_tokens=3)
            self.assertTrue(failed.wait(5))
            return [{'command': 'second', 'output': 'finished'}], 'completed', None, Counter(prompt_tokens=4)

        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(output=Path(tmp) / 'out.jsonl', overwrite=False,
                image='test', model=MODEL, seed=42, num_trajectories=20, workers=2)
            with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'offline-test-key'}), \
                 patch('collect_llm_terminal_data.shutil.which', return_value='/fake/docker'), \
                 patch('collect_llm_terminal_data.subprocess.run', return_value=SimpleNamespace(returncode=0)), \
                 patch('collect_llm_terminal_data.collect_llm_trajectory', side_effect=worker), \
                 patch('sys.stderr', new_callable=io.StringIO):
                stats = collect_llm(args)
            self.assertEqual(sorted(started), [0, 1])
            self.assertEqual(stats['records_attempted'], 2)
            self.assertEqual(stats['stopped_on_error'], 1)
            self.assertEqual(stats['prompt_tokens'], 7)
            rows = [json.loads(line) for line in args.output.read_text().splitlines()]
            self.assertEqual({row['trajectory'][0]['output'] for row in rows}, {'valid prefix', 'finished'})


if __name__ == '__main__':
    unittest.main()

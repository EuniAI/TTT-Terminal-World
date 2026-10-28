import io
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

from collection_progress import CollectionProgress, TrajectoryStates


class ProgressTests(unittest.TestCase):
    def test_native_agent_state_is_only_observed(self):
        agent = SimpleNamespace(n_calls=1, messages=[{'role': 'user'}])
        states = TrajectoryStates()
        with patch('collection_progress.time.monotonic', return_value=100):
            states.set(9, 'task')
            states.set(2, 'container')
            states.set(9, 'model', agent=agent, max_steps=50)
            self.assertEqual(states.snapshot()[1][:4], (10, 'model', 1, 50))
        with patch('collection_progress.time.monotonic', return_value=120):
            self.assertEqual(states.snapshot()[1][-2:], (20, 20))
            agent.messages.append({'role': 'assistant'})
            self.assertEqual(states.snapshot()[1], (10, 'command', 1, 50, 0, 20))
        with patch('collection_progress.time.monotonic', return_value=125):
            agent.messages.append({'role': 'tool'})
            agent.n_calls = 2
            self.assertEqual(states.snapshot()[1][:5], (10, 'model', 2, 50, 0))
        self.assertEqual(agent.messages, [{'role': 'user'}, {'role': 'assistant'}, {'role': 'tool'}])
        states.remove(9)
        self.assertEqual(len(states.snapshot()), 1)

    def test_redirected_output_reports_state_changes(self):
        states = TrajectoryStates()
        states.set(0, 'task')
        with patch('sys.stderr', new_callable=io.StringIO) as stream:
            progress = CollectionProgress(10, Counter(), states)
            progress.update(1)
            count = len(stream.getvalue())
            progress.update(1)
            self.assertEqual(len(stream.getvalue()), count)
            states.set(0, 'container')
            progress.update(1)
            self.assertIn('#1 container', stream.getvalue())
            self.assertIn('phase=', stream.getvalue())
            self.assertIn('total=', stream.getvalue())

    def test_terminal_render_clears_previous_status_rows(self):
        class Terminal(io.StringIO):
            def isatty(self):
                return True
        stream = Terminal()
        states = TrajectoryStates()
        states.set(0, 'task')
        with patch('sys.stderr', stream):
            progress = CollectionProgress(1, Counter(), states)
            progress.update(1)
            progress.log('warning')
            self.assertIn('\x1b[2A\r\x1b[Jwarning', stream.getvalue())


if __name__ == '__main__':
    unittest.main()

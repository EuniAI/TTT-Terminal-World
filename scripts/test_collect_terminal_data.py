"""Quality checks for exporting native mini-swe-agent histories."""
import json
import unittest

from collect_llm_terminal_data import extract_trajectory


def action(command='cat file', call_id='call_1', finish='tool_calls'):
    return {'role': 'assistant', 'extra': {
        'actions': [{'command': command, 'tool_call_id': call_id}],
        'response': {'choices': [{'finish_reason': finish}]},
    }}


def observation(output='hello\n', call_id='call_1', code=0, error=''):
    return {'role': 'tool', 'tool_call_id': call_id, 'content': 'FORMATTED FEEDBACK',
            'extra': {'raw_output': output, 'returncode': code, 'exception_info': error}}


class ExportTests(unittest.TestCase):
    def test_protocol_pollution_truncates_at_first_bad_command(self):
        commands = [
            'ls</｜DSML｜  parameter>\n</｜DSML｜  invoke>',
            'ls</parameter>\n</｜DSML｜  invoke>',
            'ls</|DSML| parameter>',
            'ls</think>more reasoning',
        ]
        for command in commands:
            with self.subTest(command=command):
                steps, error = extract_trajectory([
                    action('first'), observation('valid'), action(command),
                    observation('syntax error', code=2),
                    action('later'), observation('must discard'),
                ])
                self.assertEqual(steps, [{'command': 'first', 'output': 'valid'}])
                self.assertIn('step 2', error)
                logged_command = error.split('Model bash command (JSON, untruncated): ', 1)[1]
                self.assertEqual(json.loads(logged_command), command)

    def test_protocol_pollution_in_first_command_produces_empty_prefix(self):
        steps, error = extract_trajectory([action('ls</think>'), observation('bad')])
        self.assertEqual(steps, [])
        self.assertTrue(error)

    def test_plain_words_and_real_output_are_not_filtered(self):
        steps, error = extract_trajectory([action('cat DSML.txt'), observation('<think>text</think>')])
        self.assertIsNone(error)
        self.assertEqual(steps, [{'command': 'cat DSML.txt', 'output': '<think>text</think>'}])

    def test_exact_raw_outputs_including_empty_and_shell_error(self):
        messages = [action('create'), observation(''), action('read'), observation('a\n\n'),
                    action('bad'), observation('error\n', code=1)]
        steps, error = extract_trajectory(messages)
        self.assertIsNone(error)
        self.assertEqual(steps, [{'command': 'create', 'output': ''},
                               {'command': 'read', 'output': 'a\n\n'},
                               {'command': 'bad', 'output': 'error\n'}])

    def test_timeout_discards_current_and_later_calls(self):
        messages = [action('first'), observation('valid'), action('slow'),
                    observation('partial', code=-1, error='timeout'), action('later'), observation('discard')]
        steps, error = extract_trajectory(messages)
        self.assertEqual(steps, [{'command': 'first', 'output': 'valid'}])
        self.assertTrue(error)

        details = json.loads(error.split('Execution details (JSON, untruncated): ', 1)[1])
        self.assertIn('step 2', error)
        self.assertEqual(details['command'], 'slow')
        self.assertEqual(details['exception_info'], 'timeout')
        self.assertEqual(details['returncode'], -1)
        self.assertEqual(details['raw_output'], 'partial')

    def test_missing_or_mismatched_observation(self):
        for tail in ([action()], [action(), action('later')],
                     [action(), observation(call_id='other')], [observation()]):
            with self.subTest(tail=tail):
                steps, error = extract_trajectory([action('first'), observation('valid'), *tail])
                self.assertEqual(len(steps), 1)
                self.assertTrue(error)

    def test_submission_is_not_a_training_step(self):
        steps, error = extract_trajectory([
            action('first'), observation('valid'), action('echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'),
            {'role': 'exit', 'extra': {'exit_status': 'Submitted'}},
        ])
        self.assertEqual(steps, [{'command': 'first', 'output': 'valid'}])
        self.assertIsNone(error)

    def test_missing_observation_on_exception_is_not_submission(self):
        steps, error = extract_trajectory([action(), {'role': 'exit', 'extra': {'exit_status': 'RuntimeError'}}])
        self.assertEqual(steps, [])
        self.assertTrue(error)

    def test_ambiguous_or_truncated_actions_are_rejected(self):
        multi = action()
        multi['extra']['actions'].append({'command': 'second', 'tool_call_id': 'call_2'})
        for bad in (multi, action(command=1), action(call_id=''), action(finish='length')):
            with self.subTest(bad=bad):
                steps, error = extract_trajectory([bad, observation()])
                self.assertEqual(steps, [])
                self.assertTrue(error)

    def test_format_error_stops_prefix(self):
        response = {'choices': [{'message': {'content': 'done', 'tool_calls': []}}]}
        steps, error = extract_trajectory([
            action('first'), observation('valid'),
            {'role': 'user', 'content': 'No tool calls found.',
             'extra': {'interrupt_type': 'FormatError', 'response': response}},
            {'role': 'exit', 'extra': {'exit_status': 'RepeatedFormatError'}},
        ])
        self.assertEqual(len(steps), 1)
        self.assertTrue(error)
        self.assertIn('step 2', error)
        self.assertIn('No tool calls found.', error)
        self.assertEqual(json.loads(error.split('Model response (JSON, untruncated): ', 1)[1]), response)

    def test_recovered_format_error_keeps_observed_steps(self):
        steps, error = extract_trajectory([
            action('first'), observation('valid'),
            {'role': 'user', 'extra': {'interrupt_type': 'FormatError'}},
            action('retry'), observation('recovered'),
        ])
        self.assertIsNone(error)
        self.assertEqual(steps, [{'command': 'first', 'output': 'valid'},
                                 {'command': 'retry', 'output': 'recovered'}])

    def test_never_falls_back_to_formatted_content(self):
        result = observation()
        del result['extra']['raw_output']
        steps, error = extract_trajectory([action(), result])
        self.assertEqual(steps, [])
        self.assertTrue(error)


if __name__ == '__main__':
    unittest.main()

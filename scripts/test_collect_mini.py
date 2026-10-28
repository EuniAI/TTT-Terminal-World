"""Real native mini components with mocked remote API and Docker subprocesses."""
import importlib.util
import json
import os
from collections import Counter
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from collect_llm_terminal_data import AGENT_PROMPT, MODEL, collect_llm_trajectory, extract_trajectory

HAS_MINI = importlib.util.find_spec('minisweagent') is not None and importlib.util.find_spec('litellm') is not None
if HAS_MINI:
    os.environ.setdefault('MSWEA_SILENT_STARTUP', '1')
    import litellm
    from tenacity import Retrying, stop_after_attempt, wait_none
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.docker import DockerEnvironment
    from minisweagent.models.litellm_model import LitellmModel
    from terminal_mini import FINISH_COMMAND, generate_task, make_model, make_environment, make_agent


def reply(command=None, *, text=None, finish='tool_calls'):
    calls = None if command is None else [{
        'id': 'call_test', 'type': 'function',
        'function': {'name': 'bash', 'arguments': json.dumps({'command': command})},
    }]
    return litellm.ModelResponse(model='test', choices=[{
        'finish_reason': finish, 'message': {'role': 'assistant', 'content': text, 'tool_calls': calls},
    }], usage={'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15})


@unittest.skipUnless(HAS_MINI, 'Install requirements-collection.txt to test real mini-swe-agent')
class MiniIntegrationTests(unittest.TestCase):
    def setUp(self):
        cost = patch('litellm.cost_calculator.completion_cost', return_value=0.01)
        cost.start()
        self.addCleanup(cost.stop)
        cleanup = patch.object(DockerEnvironment, 'cleanup')
        self.cleanup = cleanup.start()
        self.addCleanup(cleanup.stop)

    def test_native_classes_execution_and_submission(self):
        with patch('minisweagent.environments.docker.subprocess.run', side_effect=[
            SimpleNamespace(stdout='container-id\n'),
            SimpleNamespace(stdout='hello\n', returncode=0),
            SimpleNamespace(stdout='COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n', returncode=0),
        ]) as docker, patch('litellm.completion', side_effect=[reply('cat file'), reply(FINISH_COMMAND)]) as api:
            env = make_environment('ubuntu:24.04', 10)
            model = make_model(MODEL, 60)
            agent = make_agent(model, env, 5, AGENT_PROMPT)
            self.assertIs(type(env), DockerEnvironment)
            self.assertIsInstance(model, LitellmModel)
            self.assertIs(type(agent), DefaultAgent)
            result = agent.run('task', min_steps=1, max_steps=5)
        self.assertEqual(result['exit_status'], 'Submitted')
        self.assertEqual(extract_trajectory(agent.messages), ([{'command': 'cat file', 'output': 'hello\n'}], None))
        command = docker.call_args_list[1].args[0]
        self.assertEqual(command[-5:], ['bash', '--noprofile', '--norc', '-c', 'cat file'])
        self.assertEqual(command[0:4], ['docker', 'exec', '-w', '/workspace'])
        self.assertEqual(api.call_args_list[1].kwargs['messages'][-1]['role'], 'tool')
        sent = api.call_args_list[0].kwargs
        self.assertEqual(sent['model'], 'openrouter/' + MODEL)
        self.assertEqual(sent['temperature'], 0.3)
        self.assertNotIn('parallel_tool_calls', sent)
        self.assertEqual(sent['extra_body']['reasoning'], {'effort': 'none'})
        self.assertNotIn('max_tokens', sent)

    def test_format_error_retries_recover_or_stop_after_three_failures(self):
        for failures in (1, 3):
            with self.subTest(failures=failures):
                args = SimpleNamespace(model=MODEL, api_timeout=60, seed=42, difficulty='medium',
                                       task_temperature=0.3, image='test', timeout=1,
                                       min_steps=1, max_steps=10)
                stop = Event()
                invalid = reply(text='No valid tool call', finish='stop')
                replies = [reply('first')] + [invalid] * failures
                if failures == 1:
                    replies += [reply(FINISH_COMMAND)]
                with patch('terminal_mini.generate_task', return_value='Task.'), \
                     patch('minisweagent.environments.docker.subprocess.run', side_effect=[
                         SimpleNamespace(stdout='container-id\n'),
                         SimpleNamespace(stdout='valid', returncode=0),
                         SimpleNamespace(stdout='COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n', returncode=0),
                     ]) as docker, patch('litellm.completion', side_effect=replies) as api:
                    steps, ending, error, usage = collect_llm_trajectory(args, 'unused', 0, stop)
                self.assertEqual(steps, [{'command': 'first', 'output': 'valid'}])
                self.assertEqual(api.call_count, len(replies))
                self.assertEqual(usage['prompt_tokens'], 10 * len(replies))
                self.assertEqual(stop.is_set(), failures == 3)
                self.assertEqual(ending, 'completed' if failures == 1 else 'error_prefix')
                self.assertEqual(docker.call_count, 3 if failures == 1 else 2)
                if failures == 1:
                    self.assertIsNone(error)
                else:
                    self.assertIn('Invalid model tool call', error)

    def test_multiple_calls_retry_without_executing_rejected_commands(self):
        multi = reply('must not execute first')
        multi.choices[0].message.tool_calls.append(
            reply('must not execute second').choices[0].message.tool_calls[0])
        args = SimpleNamespace(model=MODEL, api_timeout=60, seed=42, difficulty='medium',
                               task_temperature=1.0, image='test', timeout=1,
                               min_steps=1, max_steps=5)
        stop = Event()
        with patch('terminal_mini.generate_task', return_value='Task.'), \
             patch('minisweagent.environments.docker.subprocess.run', side_effect=[
                 SimpleNamespace(stdout='container-id\n'),
                 SimpleNamespace(stdout='valid', returncode=0),
                 SimpleNamespace(stdout='COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n', returncode=0),
             ]) as docker, patch('litellm.completion', side_effect=[
                 multi, reply('recovered'), reply(FINISH_COMMAND),
             ]) as api:
            steps, ending, error, usage = collect_llm_trajectory(args, 'unused', 0, stop)
        self.assertEqual(ending, 'completed')
        self.assertIsNone(error)
        self.assertFalse(stop.is_set())
        self.assertEqual(steps, [{'command': 'recovered', 'output': 'valid'}])
        self.assertEqual(docker.call_count, 3)
        self.assertEqual(docker.call_args_list[1].args[0][-1], 'recovered')
        self.assertEqual(api.call_count, 3)
        self.assertEqual(usage['prompt_tokens'], 30)

    def test_early_submission_retries_until_minimum_observed_steps(self):
        with patch('minisweagent.environments.docker.subprocess.run', side_effect=[
            SimpleNamespace(stdout='container-id\n'),
            SimpleNamespace(stdout='first result', returncode=0),
            SimpleNamespace(stdout='second result', returncode=0),
            SimpleNamespace(stdout='COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n', returncode=0),
        ]) as docker, patch('litellm.completion', side_effect=[
            reply('first'), reply(FINISH_COMMAND), reply('second'), reply(FINISH_COMMAND),
        ]):
            model = make_model(MODEL, 60, min_steps=2)
            agent = make_agent(model, make_environment('test', 1), 6, AGENT_PROMPT)
            result = agent.run('task', min_steps=2, max_steps=6)
        self.assertEqual(result['exit_status'], 'Submitted')
        self.assertEqual(docker.call_count, 4)
        self.assertEqual(docker.call_args_list[2].args[0][-1], 'second')
        steps, error = extract_trajectory(agent.messages)
        self.assertIsNone(error)
        self.assertEqual([step['command'] for step in steps], ['first', 'second'])

    def test_worker_exports_prefix_of_native_timeout_and_cleans_up(self):
        args = SimpleNamespace(model=MODEL, api_timeout=60, seed=42, difficulty='easy',
                               task_temperature=1.2, image='test', timeout=1, min_steps=1, max_steps=3)
        stop = Event()
        with patch('terminal_mini.generate_task', return_value='Task.'), \
             patch('minisweagent.environments.docker.subprocess.run', side_effect=[
                 SimpleNamespace(stdout='container-id\n'),
                 SimpleNamespace(stdout='valid', returncode=0),
                 TimeoutError('timeout'),
                 SimpleNamespace(stdout='COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n', returncode=0),
             ]), patch('litellm.completion', side_effect=[reply('first'), reply('slow'), reply(FINISH_COMMAND)]):
            steps, ending, error, usage = collect_llm_trajectory(args, 'unused', 0, stop)
        self.assertEqual(steps, [{'command': 'first', 'output': 'valid'}])
        self.assertEqual(ending, 'error_prefix')
        self.assertTrue(error)
        self.assertTrue(stop.is_set())
        self.assertEqual(usage['prompt_tokens'], 30)
        self.assertTrue(self.cleanup.called)

    def test_task_generation_is_plain_text_with_no_cap(self):
        usage = Counter()
        task = 'A task.' * 2000
        with patch('litellm.completion', return_value=reply(text=task, finish='stop')) as api:
            self.assertEqual(generate_task(MODEL, [], 1.2, 60, usage), task)
        self.assertEqual(api.call_args.kwargs['temperature'], 1.2)
        self.assertEqual(api.call_args.kwargs['extra_body']['reasoning'], {'enabled': True})
        for key in ('tools', 'max_tokens', 'response_format'):
            self.assertNotIn(key, api.call_args.kwargs)
        self.assertEqual(usage['completion_tokens'], 5)

    def test_invalid_task_reply_retries_and_counts_all_usage(self):
        for invalid in (reply(text='', finish='stop'),
                        reply(text='Incomplete task', finish='length'),
                        reply('pwd', finish='stop')):
            with self.subTest(invalid=invalid.choices[0].finish_reason):
                usage = Counter()
                with patch('terminal_mini.retry', return_value=Retrying(
                        stop=stop_after_attempt(3), wait=wait_none(), reraise=True)), \
                     patch('litellm.completion', side_effect=[invalid, reply(text='A task.', finish='stop')]) as api:
                    self.assertEqual(generate_task(MODEL, [], 1.2, 60, usage), 'A task.')
                self.assertEqual(api.call_count, 2)
                self.assertEqual(usage['completion_tokens'], 10)

    def test_invalid_task_retries_are_bounded_and_report_reason(self):
        usage = Counter()
        with patch('terminal_mini.retry', return_value=Retrying(
                stop=stop_after_attempt(3), wait=wait_none(), reraise=True)), \
             patch('litellm.completion', return_value=reply(text='', finish='length')) as api:
            with self.assertRaisesRegex(RuntimeError, "finish_reason='length'; empty or non-text content"):
                generate_task(MODEL, [], 1.2, 60, usage)
        self.assertEqual(api.call_count, 3)
        self.assertEqual(usage['completion_tokens'], 15)


if __name__ == '__main__':
    unittest.main()

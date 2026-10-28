"""Offline regression tests; fixtures model Terminus-2 messages, not downloaded data."""

from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from prepare_terminal_data import CONFIRMATION, convert_record, input_rows, write_dataset


def message(role, content):
    return {"role": role, "content": content}


def action(commands, **extra):
    return message("assistant", json.dumps({"analysis": "private reasoning", "plan": "private plan",
                                           "commands": commands, **extra}))


def command(text, **extra):
    return action([{"keystrokes": text, **extra}])


def observation(text):
    return message("user", "New Terminal Output:\n" + text)


def record(*messages):
    return {"conversations": [message("user", "Task Description:\nDo a task"), *messages]}


class ConversionTests(unittest.TestCase):
    def test_difficulty_selects_online_config_and_local_directory(self):
        loader = Mock(return_value=[])
        with patch.dict(sys.modules, {"datasets": SimpleNamespace(load_dataset=loader)}):
            for difficulty in ("easy", "medium"):
                args = SimpleNamespace(input=None, difficulty=difficulty, revision="test-revision")
                list(input_rows(args))
                self.assertEqual(loader.call_args.kwargs["name"], f"skill_based_{difficulty}")
                self.assertEqual(loader.call_args.kwargs["revision"], "test-revision")
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for difficulty in ("easy", "medium"):
                    path = root / "synthetic_tasks" / "skill_based" / difficulty / "example" / "data_filtered.parquet"
                    path.parent.mkdir(parents=True)
                    path.touch()
                args.input = root
                list(input_rows(args))
                self.assertEqual(loader.call_args.kwargs["data_files"],
                    [str(root / "synthetic_tasks/skill_based/medium/example/data_filtered.parquet")])

    def test_medium_cli_uses_separate_default_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw.jsonl"
            source.write_text(json.dumps(record(command("pwd\n"), observation("/app\n"))) + "\n")
            easy = root / "terminal_trajs.jsonl"
            easy.write_text("existing easy data")
            run = subprocess.run([sys.executable, str(Path(__file__).with_name("prepare_terminal_data.py").resolve()),
                "--input", str(source), "--difficulty", "medium"], cwd=root, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(easy.read_text(), "existing easy data")
            self.assertEqual(json.loads((root / "terminal_trajs_medium.jsonl").read_text())["trajectory"][0]["command"], "pwd\n")

    def test_screen_fallback_and_nested_completion_preserve_screen(self):
        screen = "previous output\nroot@host# ^C\nroot@host# echo done\ndone\n"
        for feedback in ["Current Terminal Screen:\n" + screen,
                         "Current terminal state:\nCurrent Terminal Screen:\n" + screen + CONFIRMATION]:
            with self.subTest(feedback=feedback):
                result, _ = self.convert(record(command("C-c\necho done\n"), message("user", feedback)))
                self.assertEqual(result["trajectory"], [{"command": "C-c\necho done\n", "output": screen}])

    def test_screen_header_printed_by_command_is_not_removed(self):
        screen = "Current Terminal Screen:\nprinted text\n"
        result, _ = self.convert(record(command("cat log\n"), observation(screen)))
        self.assertEqual(result["trajectory"][0]["output"], screen)

    def convert(self, row):
        stats = Counter()
        result = convert_record(row, stats)
        return result, stats

    def test_omits_initial_screen_preserves_final_output_and_whitespace(self):
        row = {"conversations": [
            message("user", "Instructions and task\nCurrent terminal state:\nCurrent Terminal Screen:\nroot@cpu1-0004:/app#\n"),
            command("cat a.txt\n"), observation("  hello\n\n"),
        ]}
        result, stats = self.convert(row)
        self.assertEqual(result["trajectory"], [
            {"command": "cat a.txt\n", "output": "  hello\n\n"},
        ])
        self.assertEqual(stats["interaction_steps"], 1)

    def test_initial_screen_without_commands_produces_no_trajectory(self):
        result, _ = self.convert({"conversations": [
            message("user", "Current Terminal Screen:\nroot@cpu1-0004:/app#\n")
        ]})
        self.assertIsNone(result)

    def test_empty_keystrokes_wait_is_still_preserved(self):
        result, _ = self.convert(record(command("", duration=2), observation("done\n")))
        step = result["trajectory"][0]
        self.assertEqual(step["command"], "")
        self.assertEqual(step["output"], "done\n")

    def test_thinking_and_json_fence_are_not_targets(self):
        payload = command("ls\n")["content"]
        row = record(message("assistant", '<think>{"commands": []} internal</think>\n```json\n'
                             + payload + "\n```"), observation("a.txt\n"))
        result, _ = self.convert(row)
        self.assertEqual(result["trajectory"], [{"command": "ls\n", "output": "a.txt\n"}])

    def test_batch_is_plain_text_and_duration_is_omitted(self):
        batch = [{"keystrokes": "sleep 5\n", "duration": 0.1},
                 {"keystrokes": "", "duration": 1.5},
                 {"keystrokes": "echo done\n", "duration": 0.1}]
        result, _ = self.convert(record(action(batch), observation("still running\n"),
                                        command("C-c", duration=0.1), observation("^C\n")))
        self.assertEqual(result["trajectory"][0]["command"], "sleep 5\necho done\n")
        self.assertEqual(result["trajectory"][1]["command"], "C-c")
        for step in result["trajectory"]:
            self.assertEqual(set(step), {"command", "output"})
            self.assertTrue(all(isinstance(value, str) for value in step.values()))

    def test_empty_feedback_and_empty_command_batch_are_valid(self):
        result, _ = self.convert(record(command("touch a\n"), observation(""),
                                        action([]), observation("")))
        self.assertEqual(len(result["trajectory"]), 2)
        self.assertEqual(result["trajectory"][0]["output"], "")
        self.assertEqual(result["trajectory"][1]["command"], "")

    def test_parse_rejection_discards_call_and_all_later_retries(self):
        row = record(command("echo a\n"), observation("a\n"),
                     message("assistant", "<think>unfinished"),
                     message("user", "Previous response had parsing errors:\nERROR: No valid JSON found in response"),
                     command("echo b\n"), observation("b\n"))
        result, stats = self.convert(row)
        self.assertEqual([s["output"] for s in result["trajectory"]], ["a\n"])
        self.assertEqual(stats["issue:harness_parse_error"], 1)
        self.assertEqual(stats["prefix_only_records"], 1)
        self.assertEqual(stats["interaction_steps"], 1)

    def test_first_call_rejection_discards_whole_record(self):
        row = record(message("assistant", "malformed action"),
                     message("user", "Previous response had parsing errors:\nERROR: Invalid JSON"),
                     command("echo recovered\n"), observation("recovered\n"))
        result, stats = self.convert(row)
        self.assertIsNone(result)
        self.assertEqual(stats["records_skipped"], 1)
        self.assertEqual(stats["issue:harness_parse_error"], 1)

    def test_real_shell_error_keeps_following_valid_calls(self):
        row = record(command("cat missing.txt\n"), observation("cat: missing.txt: No such file or directory\n"),
                     command("echo next\n"), observation("next\n"))
        result, stats = self.convert(row)
        self.assertEqual(len(result["trajectory"]), 2)
        self.assertEqual(stats["prefix_only_records"], 0)

    def test_unknown_feedback_stops_at_valid_prefix(self):
        row = record(command("echo a\n"), observation("a\n"),
                     command("rm a\n"), message("user", "unknown wrapper"),
                     command("ls\n"), observation("later state"))
        result, stats = self.convert(row)
        self.assertEqual(len(result["trajectory"]), 1)
        self.assertEqual(stats["prefix_only_records"], 1)
        self.assertEqual(stats["issue:unknown_observation_format"], 1)

    def test_missing_observation_is_not_empty_output(self):
        result, stats = self.convert(record(command("ls\n")))
        self.assertIsNone(result)
        self.assertEqual(stats["issue:missing_final_observation"], 1)

    def test_warning_and_confirmation_wrappers(self):
        row = record(command("ls\n"), message("user", "Previous response had warnings:\nWARNINGS: extra text\n\nNew Terminal Output:\na\n"),
                     action([], task_complete=True),
                     message("user", "Current terminal state:\na\n" + CONFIRMATION))
        result, stats = self.convert(row)
        self.assertEqual([s["output"] for s in result["trajectory"]], ["a\n"])
        self.assertEqual(stats["completion_only_endings"], 1)

    def test_completion_with_commands_unwraps_nested_observation(self):
        terminal = "\nroot@host:/app# echo done\ndone\n"
        result, _ = self.convert(record(
            action([{"keystrokes": "echo done\n"}], task_complete=True),
            message("user", "Current terminal state:\nNew Terminal Output:\n"
                    + terminal + CONFIRMATION)))
        self.assertEqual(result["trajectory"], [
            {"command": "echo done\n", "output": terminal}])

    def test_state_header_without_confirmation_preserves_terminal_text(self):
        terminal = "New Terminal Output:\nprinted by command\n"
        result, _ = self.convert(record(command("cat log\n"),
            message("user", "Current terminal state:\n" + terminal)))
        self.assertEqual(result["trajectory"][0]["output"], terminal)

    def test_completion_only_ends_before_later_calls(self):
        result, stats = self.convert(record(command("pwd\n"), observation("/app\n"),
            action([], task_complete=True), observation("/app\n"),
            command("ls\n"), observation("later\n")))
        self.assertEqual(len(result["trajectory"]), 1)
        self.assertEqual(stats["completion_only_endings"], 1)

    def test_header_like_terminal_text_is_preserved(self):
        output = "New Terminal Output:\nPrevious response had parsing errors:\n\x1b[31merror\x1b[0m\n"
        result, _ = self.convert(record(command("cat log\n"), observation(output)))
        self.assertEqual(result["trajectory"][0]["output"], output)

    def test_complex_commands_long_outputs_errors_and_duplicates_not_filtered(self):
        output = "ERROR\n" * 10000
        row = record(command("python -c 'print(42)' | grep 42\n", duration=10), observation(output))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.jsonl"
            stats = write_dataset([row, row], path)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["trajectory"][0]["output"], output)
            self.assertEqual(stats["records_read"], 2)

    def test_malformed_action_does_not_join_later_steps(self):
        row = record(command("pwd\n"), observation("/app\n"),
                     message("assistant", '{"commands": "invalid"}'), observation("changed"),
                     command("pwd\n"), observation("/other\n"))
        result, stats = self.convert(row)
        self.assertEqual(len(result["trajectory"]), 1)
        self.assertEqual(stats["issue:invalid_commands"], 1)

    def test_null_tail_keeps_last_output(self):
        row = record(command("pwd\n"), observation("/app\n"), message("assistant", "null"))
        result, _ = self.convert(row)
        self.assertEqual(result["trajectory"][-1]["output"], "/app\n")

    def test_string_conversations_and_invalid_records(self):
        row = record(command("pwd\n"), observation("/app\n"))
        row["conversations"] = json.dumps(row["conversations"])
        self.assertIsNotNone(self.convert(row)[0])
        self.assertIsNone(self.convert({"conversations": "broken"})[0])
        self.assertIsNone(self.convert({"messages": []})[0])

    def test_failed_stream_leaves_no_partial_dataset_and_preserves_old_file(self):
        def broken_stream():
            yield record(command("pwd\n"), observation("/app\n"))
            raise RuntimeError("network interrupted")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.jsonl"
            path.write_text("existing data")
            with self.assertRaises(RuntimeError):
                write_dataset(broken_stream(), path, overwrite=True)
            self.assertEqual(path.read_text(), "existing data")
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_empty_conversion_does_not_publish_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.jsonl"
            with self.assertRaises(ValueError):
                write_dataset([{}], path)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_cli_local_jsonl_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "raw.jsonl"
            target = Path(directory) / "out.jsonl"
            source.write_text(json.dumps(record(command("echo 测试\n"), observation("测试\n"))) + "\n")
            args = [sys.executable, str(Path(__file__).with_name("prepare_terminal_data.py")),
                    "--input", str(source), "--output", str(target)]
            run = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(target.read_text())["trajectory"][0]["output"], "测试\n")
            self.assertNotEqual(subprocess.run(args, capture_output=True).returncode, 0)

    def test_cli_limit_reads_only_requested_records(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "raw.jsonl"
            target = Path(directory) / "sample.jsonl"
            rows = [record(command(f"echo {i}\n"), observation(f"{i}\n")) for i in range(12)]
            source.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            args = [sys.executable, str(Path(__file__).with_name("prepare_terminal_data.py")),
                    "--input", str(source), "--output", str(target), "--limit", "10"]
            run = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            saved = [json.loads(line) for line in target.read_text().splitlines()]
            self.assertEqual(len(saved), 10)
            self.assertEqual(saved[-1]["trajectory"][0]["output"], "9\n")
            self.assertIn('"records_read": 10', run.stderr)
            invalid = subprocess.run(args[:-1] + ["0"], capture_output=True, text=True)
            self.assertNotEqual(invalid.returncode, 0)


if __name__ == "__main__":
    unittest.main()

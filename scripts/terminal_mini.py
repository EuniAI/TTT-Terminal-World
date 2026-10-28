"""Configuration of native mini-swe-agent components; no custom agent or executor."""

from collections import Counter
import logging
import math

import litellm
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.utils.retry import retry

FINISH_COMMAND = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


class SingleActionModel(LitellmModel):
    """Validate action count before the native agent executes any commands."""

    def __init__(self, *, min_steps=1, **kwargs):
        super().__init__(**kwargs)
        self.min_steps = min_steps
        self.observed_steps = 0

    def query(self, messages, **kwargs):
        self.observed_steps = sum(
            message.get("role") == "tool"
            and isinstance(message.get("extra", {}).get("raw_output"), str)
            and not message.get("extra", {}).get("exception_info")
            and message.get("extra", {}).get("returncode") != -1
            for message in messages
        )
        return super().query(messages, **kwargs)

    def _parse_actions(self, response):
        actions = super()._parse_actions(response)
        if len(actions) != 1:
            raise FormatError({
                "role": "user",
                "content": f"Expected exactly one bash tool call, received {len(actions)}. "
                           "None of these commands were executed. Return one next bash tool call "
                           "with a command argument and wait for its result.",
                "extra": {"interrupt_type": "FormatError"},
            })
        command = actions[0].get("command")
        if (isinstance(command, str) and "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in command
                and self.observed_steps < self.min_steps):
            raise FormatError({
                "role": "user",
                "content": f"Submission was not executed: only {self.observed_steps} observed steps; "
                           f"at least {self.min_steps} are required before submission. "
                           "Continue with a useful verification or task action using one bash tool call.",
                "extra": {"interrupt_type": "FormatError"},
            })
        return actions


def add_usage(usage: Counter, response: dict):
    for key in ("prompt_tokens", "completion_tokens"):
        value = (response.get("usage") or {}).get(key, 0)
        if isinstance(value, int):
            usage[key] += value


def make_model(model_name: str, api_timeout: float, min_steps: int = 1):
    # GLM endpoints may support tools without advertising parallel_tool_calls.
    # Keep strict parameter routing for reasoning; enforce one action in the
    # parser instead of requiring this optional flag.
    tool_options = {} if model_name.removeprefix("openrouter/").startswith("z-ai/") else {"parallel_tool_calls": False}
    return SingleActionModel(
        min_steps=min_steps,
        model_name="openrouter/" + model_name.removeprefix("openrouter/"),
        model_kwargs={"temperature": 0.3, "timeout": api_timeout,
                      **tool_options, "tool_choice": "required",
                      "extra_body": {"provider": {"require_parameters": True},
                                     "reasoning": {"effort": "none"}}},
        cost_tracking="ignore_errors",
    )


def generate_task(model_name: str, messages: list[dict], temperature: float,
                  api_timeout: float, usage: Counter) -> str:
    # Claude thinking uses the provider's default temperature.
    sampling = {} if model_name.removeprefix("openrouter/").startswith("anthropic/") else {"temperature": temperature}
    # A single text generation, not an agent. Reuse mini's API retry policy.
    for attempt in retry(logger=logging.getLogger(__name__), abort_exceptions=LitellmModel.abort_exceptions):
        with attempt:
            response = litellm.completion(
                model="openrouter/" + model_name.removeprefix("openrouter/"),
                messages=messages, timeout=api_timeout, **sampling,
                extra_body={"provider": {"require_parameters": True},
                            "reasoning": {"enabled": True}},
            )
            # Count every returned response, including invalid replies that are retried.
            add_usage(usage, response.model_dump())
            if not response.choices:
                raise RuntimeError("Task generation: no choices returned")
            choice = response.choices[0]
            task = choice.message.content
            reasons = []
            if choice.finish_reason != "stop":
                reasons.append(f"finish_reason={choice.finish_reason!r}")
            if choice.message.tool_calls:
                reasons.append("unexpected tool calls")
            if not isinstance(task, str) or not task.strip():
                reasons.append("empty or non-text content")
            if reasons:
                raise RuntimeError("Task generation: " + "; ".join(reasons))
            return task.strip()


def make_environment(image: str, timeout: float):
    return DockerEnvironment(
        image=image, cwd="/workspace", timeout=math.ceil(timeout),
        interpreter=["bash", "--noprofile", "--norc", "-c"],
        env={"HOME": "/workspace", "LC_ALL": "C", "TZ": "UTC", "BASH_ENV": "/dev/null"},
        forward_env=[],
        run_args=["--rm", "--pull=never", "--network=none", "--read-only",
                  "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=64",
                  "--memory=512m", "--cpus=1", "--user=65534:65534",
                  "--tmpfs", "/workspace:rw,nosuid,nodev,size=64m,mode=1777",
                  "--tmpfs", "/tmp:rw,nosuid,nodev,size=16m,mode=1777"],
    )


def make_agent(model, env, max_steps, system_prompt):
    return DefaultAgent(
        model, env, system_template=system_prompt,
        instance_template="Task: {{ task }}\nUse {{ min_steps }}–{{ max_steps }} execution steps. "
                          "The /workspace directory is empty. Create required task data through commands.",
        step_limit=max_steps, cost_limit=0, max_consecutive_format_errors=3,
        output_path=None,
    )

import io
import json
import os
import subprocess
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image
from minisweagent.exceptions import Submitted

from workflow.agent_runtime import run_task
from workflow.docker_runner import DockerEnvironment, prepare_runner
from workflow.generation_client import complete


class DockerEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def environment(self):
        return DockerEnvironment(
            image="assignment-agent:local", cwd=self.tmp.name, timeout=120,
            env={"MPLBACKEND": "Agg", "API_KEY": "must-not-enter-container"},
        )

    @patch("workflow.docker_runner._docker")
    @patch("workflow.docker_runner.subprocess.run")
    def test_command_failure_submission_and_cleanup(self, execute, docker):
        with self.environment() as env:
            args = docker.call_args.args
            self.assertIn("none", args)
            self.assertIn("no-new-privileges", args)
            self.assertIn(f"type=bind,src={self.tmp.name},dst=/workspace", args)
            self.assertNotIn("API_KEY=must-not-enter-container", args)
            with patch.dict(os.environ, {"HOST_SECRET": "private"}):
                self.assertNotIn("HOST_SECRET", env.get_template_vars())
            execute.return_value = subprocess.CompletedProcess([], 2, stdout="bad command")
            result = env.execute({"command": "exit 2"})
            self.assertEqual(result["returncode"], 2)
            self.assertEqual(execute.call_args.args[0][-3:], ["bash", "-lc", "exit 2"])
            execute.return_value = subprocess.CompletedProcess(
                [], 0, stdout="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\nfinished\n"
            )
            with self.assertRaises(Submitted):
                env.execute({"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"})
        docker.assert_called_with("rm", "--force", env.container)
        self.assertFalse(env.active)

    @patch("workflow.docker_runner._docker")
    @patch("workflow.docker_runner.subprocess.run", side_effect=subprocess.TimeoutExpired("exec", 1))
    def test_timeout_destroys_container(self, execute, docker):
        with self.environment() as env:
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                env.execute({"command": "sleep 999"}, timeout=1)
            self.assertFalse(env.active)
            docker.assert_called_with("rm", "--force", env.container)
            with self.assertRaisesRegex(RuntimeError, "no longer running"):
                env.execute({"command": "echo should-not-run"})

    @patch("workflow.docker_runner._docker")
    def test_exception_cleanup_and_path_rejection(self, docker):
        with self.assertRaisesRegex(ValueError, "inside /workspace"):
            with self.environment() as env:
                env.execute({"command": "pwd"}, cwd="/workspace/../etc")
        docker.assert_called_with("rm", "--force", env.container)

    @patch("workflow.docker_runner._docker")
    def test_start_timeout_attempts_cleanup(self, docker):
        docker.side_effect = [subprocess.TimeoutExpired("run", 60), Mock()]
        env = self.environment()
        with self.assertRaises(subprocess.TimeoutExpired):
            env.__enter__()
        docker.assert_called_with("rm", "--force", env.container)


class ModelAndRuntimeTests(unittest.TestCase):
    @patch("workflow.generation_client.time.sleep")
    @patch("requests.post")
    def test_http_retry_and_tool_call_fallback(self, post, sleep):
        busy = Mock(status_code=503)
        ok = Mock(status_code=200)
        ok.json.return_value = {
            "model": "Qwen/Qwen2.5-Coder-3B-Instruct",
            "choices": [{"message": {"content": '<tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>'}}],
        }
        post.side_effect = [busy, ok]
        registry = {"Qwen/Qwen2.5-Coder-3B-Instruct": {"api_base": "https://example.test/v1", "api_key": "private"}}
        with patch.dict(os.environ, {"GENERATION_ENDPOINTS_JSON": json.dumps(registry)}):
            response = complete({"model": "Qwen/Qwen2.5-Coder-3B-Instruct", "messages": [],
                                 "tools": [{"type": "function", "function": {"name": "bash"}}]})
        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args.args[0], "https://example.test/v1/chat/completions")
        self.assertEqual(response["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "bash")

    @patch("workflow.docker_runner._docker")
    @patch("workflow.docker_runner.shutil.which", return_value="/usr/bin/docker")
    @patch("workflow.docker_runner.load_dotenv")
    def test_selected_model_only_and_key_override(self, dotenv, which, docker):
        with patch.dict(os.environ, {
            "GENERATION_QWEN_BASE_URL": "https://example.test/v1",
            "GENERATION_API_KEY": "shared", "GENERATION_QWEN_API_KEY": "specific",
        }, clear=True):
            self.assertTrue(callable(prepare_runner("assignment-agent:local", ["qwen"])))
            registry = json.loads(os.environ["GENERATION_ENDPOINTS_JSON"])
            self.assertEqual(len(registry), 1)
            self.assertEqual(next(iter(registry.values()))["api_key"], "specific")

    def test_real_agent_loop_preserves_artifact_and_trajectory(self):
        from minisweagent.environments.local import LocalEnvironment
        png = io.BytesIO()
        Image.new("RGB", (2, 2)).save(png, format="PNG")
        cleaned = []

        @contextmanager
        def environment_factory(**kwargs):
            class FakeEnvironment(LocalEnvironment):
                def execute(self, action, **options):
                    Path(self.config.cwd, "figure.png").write_bytes(png.getvalue())
                    self._check_finished({"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n", "returncode": 0})
            try:
                yield FakeEnvironment(**kwargs)
            finally:
                cleaned.append(True)

        model = Mock(return_value={
            "id": "test", "object": "chat.completion", "created": 0,
            "model": "Qwen/Qwen2.5-Coder-3B-Instruct",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "tool_calls": [{
                    "id": "call_test", "type": "function", "function": {
                        "name": "bash", "arguments": json.dumps({"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}),
                    },
                }],
            }}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })
        result = run_task({
            "model": "Qwen/Qwen2.5-Coder-3B-Instruct", "input_files": {"data.csv": b"x,y\n1,2\n"},
            "task": {"task_id": "smoke", "instructions": "Draw a chart", "inputs": [],
                     "expected_output": {"path": "figure.png", "format": "png"}},
        }, model, environment_factory)
        self.assertEqual(result["artifact_bytes"], png.getvalue())
        self.assertEqual(result["outcome"]["exit_status"], "Submitted")
        self.assertEqual(result["scaffold"], "mini-swe-agent==2.4.5")
        self.assertTrue(result["trajectory"]["messages"])
        self.assertEqual(cleaned, [True])
        self.assertEqual(model.call_args.args[0]["generation"]["tool_choice"], "required")


@unittest.skipUnless(os.getenv("RUN_DOCKER_TESTS") == "1", "requires local Docker and built image")
class DockerIntegrationTests(unittest.TestCase):
    def test_png_persistence_credential_isolation_and_cleanup(self):
        with tempfile.TemporaryDirectory() as workspace:
            with patch.dict(os.environ, {"GENERATION_API_KEY": "host-only-secret"}):
                with DockerEnvironment(image="assignment-agent:local", cwd=workspace,
                                       timeout=120, env={"MPLBACKEND": "Agg"}) as env:
                    result = env.execute({"command": "python -c \"import os; assert 'GENERATION_API_KEY' not in os.environ; import matplotlib.pyplot as plt; plt.plot([1,2]); plt.savefig('figure.png')\""})
                    self.assertEqual(result["returncode"], 0, result["output"])
                with Image.open(Path(workspace, "figure.png")) as image:
                    image.verify()
                check = subprocess.run(["docker", "inspect", env.container], capture_output=True)
                self.assertNotEqual(check.returncode, 0)


if __name__ == "__main__":
    unittest.main()

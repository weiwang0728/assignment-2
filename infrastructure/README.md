# Student-owned Modal endpoints

These files are the canonical model deployments for both student development
and staff grading. Each Modal workspace gets its own apps, volumes, endpoint
URLs, and usage charges.

From the student release directory:

```bash
uv sync
uv run modal setup
scripts/deploy_validator_model.sh
scripts/deploy_generation_models.sh  # Part 3 model servers
# Fill in the generation endpoint settings in .env.
scripts/deploy_agent_runner.sh       # Part 3 gateway and mini-SWE-agent runner
```

The validator deployment serves `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` with the
stable model alias `validator`. The generation deployment creates three
independently scaling containers:

| Endpoint class | Model | Alias | GPU |
| --- | --- | --- | --- |
| `QwenServer` | `Qwen/Qwen2.5-Coder-3B-Instruct` | `qwen` | L4 |
| `MinistralServer` | `mistralai/Ministral-3-14B-Instruct-2512` | `ministral` | L40S |
| `GlmServer` | `zai-org/GLM-4.7-Flash` | `glm-4.7` | H100 |

All endpoints require Modal proxy authentication and expose vLLM's
OpenAI-compatible API. Copy `.env.example` to `.env`, append `/v1` to the
validator URL printed by Modal, and add a Modal proxy token in combined
`<token-id>.<token-secret>` form.

The apps scale to zero after five idle minutes. Deploying creates stable URLs;
the first request after an idle period starts a GPU container and may take
several minutes while the model loads. A completely empty GLM cache can take
about 12 minutes on its first run because it downloads roughly 58 GiB of model
weights; later starts reuse the workspace volumes and compile cache.

For generation, append `/v1` to each URL printed for `QwenServer.serve`,
`MinistralServer.serve`, and `GlmServer.serve`. Put them in `.env` as
`GENERATION_QWEN_BASE_URL`, `GENERATION_MINISTRAL_BASE_URL`, and
`GENERATION_GLM_BASE_URL`, respectively, and set `GENERATION_API_KEY` to the
combined proxy token. The Part 3 generation command is documented in
`ASSIGNMENT.md`.

`deploy_agent_runner.sh` stores those settings in the private Modal secret
`hw2-generation-endpoints`, deploys `generation_gateway.py` with access to that
secret, and deploys `agent_runner.py` without it. The latter is the original
`mini-swe-agent==2.4.5` bash-tool scaffold used for the released trajectories;
agent-written commands execute in its fresh Modal container and cannot read the
generation endpoint credentials.

## Docker execution backend (no Modal execution required)

The Docker backend runs the same `mini-swe-agent==2.4.5` loop on the host,
calling the configured model APIs there. Each task gets a fresh local Docker
container for Bash/Python commands. Only its temporary workspace is mounted;
model credentials and the repository are not mounted. Containers have no
network, a 2-CPU/4-GiB/256-process limit, dropped capabilities, and run as the
host UID/GID. A command timeout destroys the container and fails the run;
rerunning resumes unfinished runs. Completed containers are destroyed before
artifacts are read, and cleanup is also attempted on errors and interrupts.

Requirements: Python 3.12+, `uv`, and a running **local Linux Docker daemon**
(or Docker Desktop with access to the host's temporary directory). Remote
Docker daemons cannot use the host workspace bind mount. No host GPU is needed
for code execution; the model APIs must already be reachable from the host.

From the repository root:

```bash
uv sync --extra docker
docker build -f infrastructure/Dockerfile.agent -t assignment-agent:local .
uv run --extra docker python -m workflow generate --backend docker --models qwen
# Run all three fixed models:
uv run --extra docker python -m workflow generate --backend docker
```

Use `--docker-image IMAGE` for a custom prebuilt image. The runner never pulls
an image automatically. The default backend remains `modal`; the existing
`--tasks`, `--artifacts`, `--models`, and `--force` options work with either backend.
Do not run both backends against the same output directory concurrently.

Configure `.env` with the existing `GENERATION_QWEN_BASE_URL`,
`GENERATION_MINISTRAL_BASE_URL`, `GENERATION_GLM_BASE_URL` and
`GENERATION_API_KEY` variables. URLs should include `/v1` (a full
`/v1/chat/completions` URL is also accepted). Each selected model needs its URL
and key; unselected models need not be configured. Different providers can use
`GENERATION_QWEN_API_KEY`, `GENERATION_MINISTRAL_API_KEY`, and
`GENERATION_GLM_API_KEY`, which override the shared key. Alternatively,
`GENERATION_ENDPOINTS_JSON` takes precedence and maps full model IDs to
`{"api_base": "https://host/v1", "api_key": "..."}` objects.

Docker replaces agent execution only. If the Modal **model endpoints** are
also unavailable, deploy the same fixed weights and vLLM settings elsewhere
and update these URLs. Generation endpoints must accept the fixed full model
IDs and support the existing tool-call parameters. The validator remains
configured separately using `VALIDATOR_BASE_URL`, `VALIDATOR_API_KEY`, and
`VALIDATOR_MODEL`; its baseline also requires `/tokenize` and `/detokenize`
beside `/v1`. Substituting other models does not meet the assignment contract.

Validation without calling paid model APIs:

```bash
uv run --extra docker python -m unittest discover -s tests -v
# With Docker running and the image built, enable the container integration test:
RUN_DOCKER_TESTS=1 uv run --extra docker python -m unittest discover -s tests -v
```

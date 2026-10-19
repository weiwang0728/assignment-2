# 环境、模型配置与测试

在仓库根目录执行命令。Python 版本由 `.python-version` 固定为 3.12，依赖由 `uv.lock` 锁定：

```bash
uv sync --locked --extra docker
```

本地 `.env` 已从 `.env.example` 创建，并被 Git 忽略。请直接在本机编辑，不要把密钥提交到仓库。

## 需要填写的模型信息

| 用途 | 环境变量 | 要填的内容 |
| --- | --- | --- |
| 验证器地址 | `VALIDATOR_BASE_URL` | 验证器部署地址，以 `/v1` 结尾 |
| 验证器密钥 | `VALIDATOR_API_KEY` | Modal 代理令牌，格式为 `<token-id>.<token-secret>`；自行托管时填服务端密钥 |
| 验证器名称 | `VALIDATOR_MODEL` | 官方部署保持 `validator`，对应 `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` |
| Qwen 生成地址 | `GENERATION_QWEN_BASE_URL` | `Qwen/Qwen2.5-Coder-3B-Instruct` 的 API 地址，以 `/v1` 结尾 |
| Ministral 生成地址 | `GENERATION_MINISTRAL_BASE_URL` | `mistralai/Ministral-3-14B-Instruct-2512` 的 API 地址，以 `/v1` 结尾 |
| GLM 生成地址 | `GENERATION_GLM_BASE_URL` | `zai-org/GLM-4.7-Flash` 的 API 地址，以 `/v1` 结尾 |
| 生成模型密钥 | `GENERATION_API_KEY` | 三个生成服务共用的代理令牌 |

不同生成服务使用不同密钥时，填写 `GENERATION_QWEN_API_KEY`、`GENERATION_MINISTRAL_API_KEY`、`GENERATION_GLM_API_KEY` 覆盖公共密钥。生成请求使用上表中的完整模型 ID，服务必须接受这些名称。不要用其他模型替代作业指定模型。验证器需要支持图片输入，基线的上下文压缩还需要同一服务上 `/tokenize` 和 `/detokenize`（位于 `/v1` 旁边）。

当前选择使用已有的兼容 OpenAI 服务：直接填写现有 API 地址和密钥，跳过下方 Modal 部署步骤。`VALIDATOR_MODEL` 改为服务实际接受的模型名称，确认它仍对应指定的 Qwen3-VL 权重。同一服务提供多个模型时，多个 Base URL 可以相同。

只评估已有 `artifacts/` 时，仅需配置验证器。自行生成任务轨迹时，再配置生成模型；Docker 后端只需配置 `--models` 选中的模型。

若还没有模型服务，按以下流程部署到自己的 Modal 工作区（使用 GPU 额度，会产生费用）：

```bash
uv run modal setup
bash scripts/deploy_validator_model.sh
bash scripts/deploy_generation_models.sh
# 把部署输出的地址加上 /v1，并将自己的代理令牌填入 .env。
# 仅当使用默认 Modal 执行后端时，还需要：
bash scripts/deploy_agent_runner.sh
```

模型服务会在空闲后缩容，首次请求可能等待数分钟。Docker 执行后端无需部署 Modal agent runner，也无需本机 GPU，但模型 API 必须已可用。

## 测试流程

### 1. 不调用模型的本地检查

```bash
LITELLM_LOCAL_MODEL_COST_MAP=True uv run --extra docker python -m unittest discover -s tests -v
uv run python -m workflow --help
uv run python -m validator.runner --help
```

默认单元测试会跳过真实 Docker 测试。启用容器集成测试前先构建镜像：

```bash
docker build -f infrastructure/Dockerfile.agent -t assignment-agent:local .
LITELLM_LOCAL_MODEL_COST_MAP=True RUN_DOCKER_TESTS=1 uv run --extra docker python -m unittest discover -s tests -v
```

`LITELLM_LOCAL_MODEL_COST_MAP=True` 让测试使用依赖自带的价格表，避免导入时访问远端价格表。项目已包含 `httpx[socks]`，支持当前环境中的 SOCKS 代理。

### 2. 验证器端点冒烟测试（填好 .env 后执行，会调用模型）

```bash
uv run python - <<'PY'
from validator.model import complete, tokenize, detokenize
reply = complete([{"role": "user", "content": "Reply with OK."}], max_tokens=16)
print(reply["choices"][0]["message"]["content"])
tokens = tokenize("hello")
print("tokenizer:", detokenize(tokens["tokens"]))
PY
```

该步骤检查认证、聊天和分词接口；完整图片与轨迹处理由下一步检查。

### 3. 基线评估与评分（会调用模型）

```bash
uv run python -m validator.runner --solution validator.baseline \
  --artifacts artifacts --out output/baseline_predictions.json --resume
uv run python -m workflow score --labels seed_labels.json \
  --predictions output/baseline_predictions.json
```

这会处理全部种子轨迹；中断后使用同一命令的 `--resume` 续跑。评分输出四类错误的 MCC 和 macro-MCC。

`validator/solution.py` 目前是作业模板，包含 `NotImplementedError`。环境验收应先使用 `validator.baseline`；完成自己的 `validate(run)` 后再把 `--solution` 改成 `validator.solution`，并使用另一个预测输出文件。

### 4. 自建任务与生成模型

已有智谱官方 API 时，可以先对种子任务 001 跑一次 GLM 生成冒烟测试：

```bash
uv run --extra docker python scripts/test_glm_baseline.py
```

该脚本读取 `.env` 中的 GLM 地址和密钥，使用 API 模型名 `glm-4.7-flash`（可通过 `GENERATION_GLM_MODEL` 覆盖），复用项目的 mini-SWE-agent 和 Docker 执行环境。结果保存在 `output/glm_baseline_smoke/`，包括图像和完整轨迹。它不运行视觉验证器，也不计算 MCC；正式任务生成入口仍使用项目固定模型 ID。

先在 `tasks/` 中按作业说明编写任务描述符（当前目录为空），再运行：

```bash
uv run python -m workflow validate-tasks
uv run --extra docker python -m workflow generate --backend docker --models qwen
# 单模型跑通后运行全部三个固定模型：
uv run --extra docker python -m workflow generate --backend docker
uv run python -m validator.runner --solution validator.baseline \
  --artifacts runs --out output/authored_baseline_predictions.json --resume
uv run python -m workflow init-labels
# 手工填写 labels.json 后：
uv run python -m workflow validate-labels
uv run python -m workflow score --labels labels.json \
  --predictions output/authored_baseline_predictions.json
```

默认生成后端为 Modal；使用本地容器时必须显式传入 `--backend docker`。生成结果在 `runs/`，不会覆盖种子 `artifacts/`。

### 5. Harbor 与最终提交

```bash
uv run harbor run -p harbor/example -a oracle --job-name example-oracle
uv run harbor run -p harbor/example -a nop --job-name example-nop
```

预期分别为 reward 1.0 和 0.0。完成作业要求的任务、验证器、标注、Harbor 打包和报告后：

```bash
uv run python -m workflow check-submission --skip-harbor-runs
uv run python -m workflow check-submission
```

最后一条需要 Docker，会真实执行 Harbor 检查；空白作业模板不能通过最终提交检查。详细要求见 [ASSIGNMENT_CN.md](ASSIGNMENT_CN.md) 和 [harbor/README.md](harbor/README.md)。

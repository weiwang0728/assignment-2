# 11-768 作业 2：评估数据可视化智能体（Data-Visualization Agent）

## 概述

健壮的评估（evaluation）对智能体开发流程至关重要，它让我们能够在把智能体部署到真实世界之前，判断其能力与可靠性。然而，智能体系统的复杂性以及评估智能体所用任务的长时间跨度，为评估开发带来了许多新的难题（尤其是与智能体时代之前的语言模型评估相比）。在本作业中，你将获得设计评估任务和评估智能体轨迹（trajectory）的动手经验，从而更深刻地体会评估开发中存在的挑战。

本作业围绕**数据可视化智能体**展开：这类 LLM 智能体在获得数据访问权限和用户指定的规格（spec）后，能够为用户生成该数据的可视化图表。在本作业中，你首先要构建一个验证器（validator），为数据可视化智能体的输出打分。给定一个用户任务、智能体轨迹和最终输出图形，你需要判断智能体是否正确，或诊断过程中的任何失败。随后，你将设计额外的任务来探测该数据可视化智能体的优势与弱点，并强化你已有的评估。本作业的目标是设计一个能够泛化到一组私有智能体轨迹上的验证器，你的成绩将基于这组私有轨迹评定。在评分时，你的验证器一次评估一个运行（run）。它不会收到任何真实标签（ground-truth label）、参考图形或其他候选运行：只有该智能体的轨迹和输出图形。

## 初始设置

在学生版发布目录中安装环境：

```bash
uv sync
```

在开发过程中，你将使用课程提供的 Modal 额度，在自己的 Modal 工作区中运行验证器模型。先完成 Modal CLI 的认证，再部署验证器端点：

```bash
uv run modal setup
scripts/deploy_validator_model.sh
```

部署命令会打印一个端点 URL。为你的 Modal 工作区创建一个代理令牌（proxy token），然后配置兼容 OpenAI 的客户端：

```bash
cp .env.example .env
# 编辑 .env，填入你的端点 URL（包括 /v1 后缀）和代理令牌。
```

只有在调用端点时才需要代理令牌，因此你可以先执行 Modal setup 和部署，之后再创建令牌。不要提交 `.env`、代理令牌或其他凭据。我们要使用的验证器模型是 **Qwen/Qwen3-VL-30B-A3B-Instruct-FP8** —— 你不得替换它，也不得从验证器中调用其他模型。因此，你的验证器设计应当着眼于泛化，而不是依赖某个强大的评判模型。我们会使用同一模型的部署来调用你的验证器，以此评定你的作业。注意，如果你在构建新镜像，GPU 容器加载模型大约需要 5-10 分钟。

### 评估格式

你将在 `validator/solution.py` 中实现 `validate(run)`。对每个运行，它会收到任务、输入数据、智能体轨迹和最终图形，并返回一个错误列表。空列表表示该运行是可接受的。每个错误属于四个固定类别（family）之一，并附有基于该运行的具体证据：

| 类别 | 含义 |
| --- | --- |
| `execution_failure` | 智能体未能生成有效的图形。 |
| `wrong_data` | 所绘制的数据与要求的数据不符。 |
| `wrong_chart` | 图形没有遵循要求的图表设计。 |
| `hard_to_read` | 渲染出的图形难以阅读或无法阅读。 |

`execution_failure` 在真实标签中是终态的：当人工标注指出没有有效图形时，该运行的其余三个类别被排除，因此它不计入这三类中任何一类的 TP、TN、FP 或 FN。而验证器预测出 `execution_failure` 并不能使该运行豁免于其他类别。
`validator/prediction.py` 定义了确切的输出格式，`validator/baseline.py` 提供了一个完整示例。

每个类别都单独计分，作为针对所评估运行集合的二分类问题。当真实标签列出某个类别时，该运行即为该类别的正例；当验证器的错误列表点名某个类别时，该预测即为正例；一个带有多个错误的运行在其列出的每个类别中都是正例。对所评估运行统计真正例（TP）、真负例（TN）、假正例（FP）和假负例（FN），即可得到马修斯相关系数（Matthews correlation coefficient, MCC）：

$$
\mathrm{MCC}=\frac{TP\cdot TN-FP\cdot FN}
{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}.
$$

完美分类时 MCC 为 1，无相关性时为 0，负相关时为负。分数保持在 [-1, 1] 区间内，包括负值。总体 `macro_mcc` 是全部四个类别 MCC 的无权平均；其分母始终为四。

当真实标签同时包含两类但预测恒定时，MCC 为 0。当某类别的真实标签缺少其中一类时，评分器按惯例报告 MCC 0，并附带一个"支持不足"（insufficient support）标志以及正例和负例的支持计数。小型的自建数据集很容易出现支持不足，因为某个类别可能从不出现，也可能在每个被评估的运行上都出现。解读此类类别分数时须谨慎。私有评分集在每个类别中都会包含两类样本，因此这一惯例不会影响正式成绩。MCC 只衡量分类效果；验证器解释中证据的质量将单独评估。

### 各类错误的判定标准

这四个类别都是二元判定，且一个运行可以同时带有其中多个。以下完整列出了种子集和隐藏评估集的标注在每个类别下计为错误的内容。

**`execution_failure`。** 智能体最终未能生成有效图形。此类别总是单独出现。

**`wrong_data`。** 画出来的不是指令要求的数据：

- *数值错误。* 绘制的数字、位置或派生量与要求的数据或公式不符：聚合、缩放或归一化错误；函数或字段在错误的范围或网格上采样；本应编码数据的气泡大小、颜色值或误差棒却没有编码。
- *数据缺失。* 要求的序列、类别、分组、某个面板的数据或某个数据点没有被绘制，或者数据的某个维度被丢弃（三维量只画出了缺少一个变量的图）。
- *选择错误。* 使用了错误的子集、列或过滤器，或者把未要求的序列混入了要求的序列之中。
- *顺序错误。* 点、条、类别或序列出现的顺序与要求的不同：要求排序却未排序，序列被反转或打乱，面板没有显示分配给它的数据。

**`wrong_chart`。** 数据正确，但未遵循要求的图表设计。凡是指令中有要求的，我们都计入：

- *图表类型与投影。* 图表类型错误（用折线图画柱状图、没有饼图），要求堆叠柱状图却画成组合柱状图，要求水平柱状图却画成竖直的，要求 3D 却画成 2D，缺少极坐标投影，缺少等高线或曲面，缺少双坐标轴（twin axes）或在被禁止时使用，缺少缩放插图（zoomed inset）或其连接线。
- *布局。* 子图的数量或排列错误，图形尺寸或 DPI 错误，本应共享坐标轴的面板没有共享。
- *坐标轴。* x、y 或 z 轴标签缺失或错误；标题或标签文字与要求不完全一致；坐标轴范围错误（没有从零开始、范围错误、裁掉了要求显示的数据）；坐标轴刻度类型错误（要求对数却是线性、等高线层级没有取对数）；被要求隐藏的刻度或刻度标签仍然显示；刻度标签或其旋转角度错误；被要求隐藏的坐标轴线（spines）仍然可见；缺少网格线。
- *图例、颜色条与标注。* 缺少图例；缺少颜色条或颜色条与数据不匹配；要求的标注或标记点缺失或放错位置；要求的函数标签或条形数值标签缺失；要求的公式没有被渲染；阴影区域错误。
- *样式。* 颜色映射（colormap）错误或使用了被禁止的（在要求感知均匀型时使用了非均匀型）；线型错误（要求虚线或点线却画成实线）；标记形状错误；具名序列的颜色错误；区域未填充；缺少须线（whiskers）、中位数标记、连接线或箱体；多边形未闭合；要求的线条偏移没有应用。
- 一个必要的文本元素（标题、坐标轴标签）如果*完全*落在保存的图像之外，计为缺失，因此也算作 `wrong_chart`。

**`hard_to_read`。** 图形按要求绘制了，但读者无法阅读它，或只能费劲地阅读：

- *文字被裁剪。* 标题、坐标轴标签、刻度标签或标注被图像边缘部分切断（被切成一半的标签属于 `hard_to_read`；完全在图像之外的属于 `wrong_chart`，见上文）。
- *文字重叠。* 刻度标签、标注、标题或标签相互重叠，或者标记被画在标注它们的文字之上。
- *元素遮挡内容。* 图例、标注或文本框遮挡了数据、标签或其他文字。图例位于空白区域则没有问题。
- *低对比度或不可见内容。* 文字或数值标签与其背景的对比度低于约 2.5:1；某个序列画成了背景色、宽度为零，或以其他方式不可见。
- *无法区分的数据。* 必须区分开的两个序列或分组使用了相同或几乎相同的颜色、线型和标记；颜色映射把本应看起来不同的值混在了一起。
- *布局挤压。* 数据、箱体或面板被坐标轴范围或相互重叠的坐标轴压成无法阅读的细条。

## 第 1 部分：研究种子智能体运行

我们在 `artifacts/<run_id>/` 下提供了一些种子智能体运行。每个运行子目录包含 `result.json`（其中包括任务规格、完整的智能体轨迹以及该次运行的结果），以及 `figure.png`（最终图表，如果智能体能够生成的话）。`run_id` 是唯一的键：多个运行可以共享同一个 `task_id`，因此任何索引都应以 `run_id` 为准。你还可以在 `validator/baseline.py` 中找到一个简单的基于 VLM 的验证器。你可以用以下命令在所有现有运行上运行基线验证器：

```bash
uv run python -m validator.runner \
  --solution validator.baseline \
  --artifacts artifacts \
  --out output/baseline_predictions.json
```

基线首先发送完整证据。如果模型以超出上下文窗口为由拒绝请求，它会发出警告，并用轨迹和输入文件的较短摘录（保留开头和结尾）重试一次。
该回退方案使用服务端的分词器（tokenizer）和报告的上下文上限，把 50% 的上下文预算分配给文本和响应，其余留给图像和格式化。
`validator/baseline.py` 中的 `CONTEXT_FRACTION` 控制这一比例；没有固定的上下文大小，也没有逐步收缩的重试循环。
任务和图形保持完整，省略之处会在提示中标注。
这种回退可能会丢失相关证据；它不保证得到与完整运行相同的判断。如果压缩后的请求仍然放不下，或者服务端的分词器端点不可用，则错误会被传播。这只是基线的做法：你可以在 `validator/solution.py` 中任意自定义上下文管理。

我们在 `seed_labels.json` 中为种子智能体运行提供了真实的人工标注。用以下命令为基线预测打分：

```bash
uv run python -m workflow score \
  --labels seed_labels.json \
  --predictions output/baseline_predictions.json
```

首先，对比真实标签和验证器标签。报告全部四个类别的 MCC 分数和 macro-MCC，然后找出验证器的两种常见错误类型，每种类型至少出现在两个智能体轨迹中（为所有轨迹附上运行 ID）。针对每种类型，描述可观察到的证据，并提出一个可能捕获此类错误的验证器改进方案。

## 第 2 部分：改进你的验证器

在第 1 部分，针对默认验证器实现你提出的两个改进。记录你所做的修改，以及修改后的验证器在种子智能体运行上的表现，并与基线验证器进行比较。即使这一步性能没有显著提升也没关系；后面还有时间继续改进验证器。你应报告新验证器全部四个类别的 MCC 分数和 macro-MCC，并定性地说明你的改动是否真正缓解了第 1 部分中发现的那几类错误。具体而言，你应至少找出一个假正例（false positive）和一个假负例（false negative）案例，并尝试解释造成这些错误的原因，以及这可能暗示验证器缺少什么。

在 `validator/solution.py` 中实现你的修改，然后在已发布的种子运行上运行修改后的验证器：

```bash
uv run python -m validator.runner \
  --artifacts artifacts \
  --out output/pre_iteration_seed_predictions.json
uv run python -m workflow score \
  --labels seed_labels.json \
  --predictions output/pre_iteration_seed_predictions.json
```

验证器模型是固定的：你不得替换验证器实际使用的模型，也不得调用任何其他模型，所以要尽量设计一个与具体模型无关的验证器。除此之外，你可以随意定制 `validator/solution.py`，只要 `validate(run)` 的函数签名和 `validator/prediction.py` 中的输出模式保持不变。注意，你需要基于输入数据、最终输出和智能体轨迹进行评估；基线验证器演示了如何把这三者一起传给模型。

## 第 3 部分：设计新的评估任务

此时，你应该已经基于第 1 部分的定性观察得到了一个改进的验证器。通过阅读真实的智能体轨迹来发现潜在错误，是改进验证器的一种常见方法。另一种方法是在分布外（out-of-distribution）数据上测试验证器，以衡量它对新任务类型的鲁棒性。在本节中，你将设计自己的可视化任务，对你的验证器进行压力测试。

首先，通读你当前的学生版验证器实现，找出它可能难以泛化的两类可视化任务。编写五个可视化任务（其中至少两个属于你确定的每一类任务），在它们上面运行提供的智能体，并按照上述四个错误类别对结果进行人工标注。如果你的可视化任务需要自定义数据，你可以用 LLM 合成生成数据，或者复用其他可视化任务的数据。然后，在新智能体轨迹上运行你的验证器，报告在这些对抗性案例上的全部四个类别 MCC 分数和 macro-MCC。你不应硬编码公开的任务 ID 或标签。

提供的智能体模型是固定的：

- **Qwen/Qwen2.5-Coder-3B-Instruct**
- **mistralai/Ministral-3-14B-Instruct-2512**
- **zai-org/GLM-4.7-Flash**

用每个提供的智能体把每个新任务各运行一次，总共产生至少 15 个自建智能体运行。用上面定义的同一证据标准对每个生成的运行进行人工审查。在报告中，分析哪些失败反映了可视化智能体的弱点，哪些反映了你的验证器的弱点。

### 开始第 3 部分之前

```bash
scripts/deploy_generation_models.sh
# 把打印出的三个 URL 复制到 .env 中的 GENERATION_* 设置，
# 给每个 URL 附加 /v1，并设置 GENERATION_API_KEY。
scripts/deploy_agent_runner.sh
```

注意，智能体最多只能执行 20 个智能体步骤，每条 shell 命令限时 120 秒，总时限 15 分钟。部署故障排查见 `infrastructure/README.md`。

### 编写任务

在 `tasks/<task_id>/` 下为每个任务创建一个目录：

```text
tasks/<task_id>/
  task.json
  inputs/                 # 可选；任务本地的任意输入文件
```

其 `task.json` 描述符恰好包含这些字段：

```text
{
  "task_id": "sales-by-region",
  "task_class": "multi-series aggregation",
  "instructions": "Using matplotlib, read sales.csv and ... Save figure.png.",
  "inputs": [
    {"name": "sales.csv", "path": "inputs/sales.csv"}
  ]
}
```

`task_id` 必须与其目录名一致，且只能包含字母、数字、句点、下划线和连字符。`task_class` 是你确定的两类弱点之一的名称；同一类任务使用相同的值。`instructions` 必须完整规定图表，并应使正确性可以由人工审查。每个输入的 `name` 是智能体看到的普通文件名；`path` 相对于任务目录。路径不得是绝对路径，也不得包含 `..`。当提示中已包含全部数据时，使用空的 `inputs` 列表。`artifacts/` 下已发布的运行提供了任务指令和输入的示例。你的任务不应使用现有环境之外的任何额外包（matplotlib、numpy、pandas、Pillow、scipy 和 seaborn）；私有集同样不会包含需要智能体使用任何额外包的任务。

在生成运行之前，先校验所有描述符和被引用的文件：

```bash
uv run python -m workflow validate-tasks
uv run python -m workflow generate
```

运行是可恢复的：已存在的任务/模型运行会被跳过。要重新生成已有运行，加 `--force`；开发时只运行子集，加 `--models qwen`、`--models ministral` 或 `--models glm`。

每个生成的运行按如下布局写入：

```text
runs/<task_id>__<agent>/
  result.json             # 任务、智能体身份和完整轨迹
  inputs/                 # 该次运行任务输入的副本
  figure.png              # 当智能体留下了输出文件时存在
```

在生成的运行上运行你的验证器：

```bash
uv run python -m validator.runner \
  --artifacts runs \
  --out output/pre_iteration_authored_predictions.json
```

### 人工标注与打分

为每个生成的运行创建一条空白条目，生成 `labels.json`：

```bash
uv run python -m workflow init-labels
```

然后对每个运行进行人工审查。在每条条目中，如果运行是可接受的，把 `"errors": null` 替换为 `[]`；如果不可接受，替换为 `{family, evidence}` 错误的列表。每个错误都需要具体证据，且 `execution_failure` 必须单独出现。示例见 `labels.example.json`。

校验覆盖率和模式，然后报告全部四个类别 MCC 分数和 macro-MCC：

```bash
uv run python -m workflow validate-labels
uv run python -m workflow score \
  --predictions output/pre_iteration_authored_predictions.json
```

### 把任务打包为 Harbor 环境

[Harbor](https://github.com/harbor-framework/harbor) 是一个评估框架（evaluation harness），自 Terminal-Bench 2.0 将其推广以来，它已成为智能体基准测试的行业标杆。Harbor 把任务定义为自包含的目录，其中定义了沙箱和评估标准，任何智能体都可以针对其运行。在本部分作业中，我们将把任务（目前只支持我们内部的 `workflow generate`）打包成 Harbor 格式，让任何人都能运行。由于 Harbor 任务要求确定性的验证器（verifier），你可能需要给每个私有任务增加更多特性，但不应重新设计任务的实际内容。具体而言，你必须要求智能体写出 `plot.py` 和 `plotted_values.json`，而你原本的任务没有这个要求。任何此类新增内容都应保留在 `harbor/tasks/<task_id>/` 内。本部分作业的更多细节见 `harbor/README.md`。

先浏览 `harbor/example/` 中的合成示例（worked example），了解相关要求。你应该既针对示例的参考解（reference solution）运行它，也针对一个什么都不做的智能体运行它。

```bash
uv run harbor run -p harbor/example -a oracle --job-name example-oracle   # reward 1.0
uv run harbor run -p harbor/example -a nop    --job-name example-nop      # reward 0.0
```

接着，你应该至少运行 `harbor/example/mutants/` 中四个故意写错的示例解之一。其中三个得 0.0 分；`illegible/` 得 1.0 分，因为验证器没有发现它的图表小到无法阅读。更换突变体（mutant）并重跑 Harbor 验证器的命令见 `harbor/example/README.md`。随后，把你自己的每个任务都打包。你可以运行以下命令，从你已写好的私有任务初始化 Harbor 任务（具体来说，它会为你之前的每个任务把 Harbor 模板复制到新的 `harbor/tasks/<task_id>/` 目录，然后填入一些样板信息和初始 `instruction.md`）。**在生成运行之后，不要编辑已播种（seeded）的提示。**

```bash
uv run python -m workflow package
```

对每个私有任务，你需要在 `harbor/tasks/<task_id>/tests/test_state.py` 中实现确定性检查，在 `harbor/tasks/<task_id>/solution/solve.sh` 中实现参考解，并完成脚手架中留作"Required Outputs"的其余部分。在花费一次 Docker 周期之前先检查每个任务，然后确认你的参考解确实能通过：

```bash
uv run python harbor/preflight.py harbor/tasks/<task_id>
uv run harbor run -p harbor/tasks/<task_id> -a oracle --job-name <task_id>-oracle
```

要完成本节，每个打包任务都必须针对你的参考解得到 1.0 分。此外，在所有打包任务中，你总共要写出**两个**突变体解，作为 `harbor/tasks/<task_id>/mutants/<name>/solve.sh`。它们可以属于同一个任务，也可以属于不同任务。两者都给出一个看似合理的错误答案；它们的区别在于验证器对各自的处理。一个必须**得 0 分**：你的检查拒绝了它。廉价的实现途径是你本来就被要求索要的 `plotted_values.json` —— 手工推导出期望的数值，作为字面量写进测试文件再进行比较，任何改动了数值的突变体都会失败。另一个必须**得 1.0 分**：一个真正错误的答案却通过了你的检查。这个突变体应当暴露验证器的一个真实局限。例如，一张图表可以包含正确的数据和标签，却因为在小画布上使用 2pt 文字而无法阅读。`harbor/example/mutants/illegible/` 是一个针对讲义中最强验证器得 1.0 分的现成版本。`check-submission` 会运行所有这些，因此没有哪一项是没运行过就能报告的。在报告中，给出每个突变体的奖励（reward）、它错在哪里，以及是哪条断言捕获了它（或者为什么没有任何东西能捕获它）。第二个突变体才是这一部分的要义所在：你要亲手测量、而不是被人告知你的 VLM 验证器本应跨过的那道边界。`harbor/TROUBLESHOOTING.md` 覆盖了你最可能遇到的错误，包括无效的 `task.toml` 被报成"no task found"的那一种。

## 第 4 部分：迭代你的设计

利用第 3 部分的结果，你应该对验证器再做出至少一项额外改进。在种子任务和你的新任务上评估最终验证器，报告新的全部四个类别 MCC 分数和 macro-MCC，并定性分析这次改动是否在你手写的任务上带来了改进。与上一部分类似，你应至少找出一个假正例和一个假负例，包括轨迹中的相关文本，以及对验证器为何判断错误的假设。

在两组轨迹上运行最终验证器：

```bash
uv run python -m validator.runner \
  --artifacts artifacts \
  --out output/final_seed_predictions.json
uv run python -m validator.runner \
  --artifacts runs \
  --out output/authored_predictions.json
uv run python -m workflow score \
  --labels seed_labels.json \
  --predictions output/final_seed_predictions.json
uv run python -m workflow score \
  --predictions output/authored_predictions.json
```

在此步骤之后，你还可以继续改进验证器或设计更多任务，因为你的一部分成绩取决于最终验证器在私有测试集上的表现。请在报告中报告对验证器所做的全部额外修改，包括加入它们的动机，以及你判定它们有多成功（定性和定量）。

## 提交与评分

提交内容：

```text
validator/solution.py   # 最终验证器实现
output/                 # 种子运行和自建运行的预测结果
tasks/                  # 五个及以上自建任务及其输入
runs/                   # 每个自建任务的全部三个固定智能体运行
labels.json             # 每个自建智能体运行一条人工审查的标注
harbor/tasks/           # 每个自建任务打包成的 Harbor 环境，
                        #   包括两个突变体解
report.pdf              # 编译好的报告
report.tex              # 报告源文件
AI_USAGE.md             # 声明 AI 工具的使用情况（不计分）
```

提交之前，运行端到端的结构检查：

```bash
uv run python -m workflow check-submission
```

它会校验必需文件，验证每个任务及其输入，要求至少五个任务、跨越至少两个 `task_class` 值且每个类至少两个任务，要求每个任务恰好有来自每个固定智能体的一次运行，并检查 `labels.json` 恰好覆盖这些运行一次。它还分两个阶段检查 Harbor 打包。结构上：每个自建任务对应一个 `harbor/tasks/<task_id>/` 且没有多余目录，每个任务都有完整的 `harbor/template/` 布局，每个任务的参考解和验证器中不再有模板哨兵（sentinel），其 `instruction.md` 中没有遗留 TODO 标记，`instruction.md` 仍然逐字保留描述符中的提示并要求 `plot.py` 和 `plotted_values.json`，以及 `harbor/tasks/<task_id>/mutants/<name>/solve.sh` 下共有两个突变体。它会一次性报告每个任务中的所有问题。然后它会运行 Harbor 并用契约约束奖励：每个打包任务都必须用 `-a oracle` 得 1.0 分，一个突变体必须得 0 分，一个突变体必须得 1.0 分。它会打印读取到的奖励，失败信息会指出解释原因的 `jobs/check-submission-<job>/<trial>/verifier/pytest.log`。该阶段需要 Docker 并耗时几分钟；`--skip-harbor-runs` 会在结构检查之后停止，这正是迭代时想要的。它不会判断标注的正确性或报告的质量；这些仍需要人工审查。请以 `report_template.tex` 作为最终 `report.tex` 的起点。报告应回答第 1 至 4 部分中的每个问题，并包含足以复现你实验的细节。

你还应包含一个 AI_USAGE.md 文件。它应详细说明你在本作业中对任何 AI 技术的使用。列出你使用的所有工具，并清晰描述你如何使用其中每一个工具。如果你没有使用任何 AI 辅助，也应在此文件中声明。我们不会为提交的 AI_USAGE.md 打分，但我们会通过一个测验检查你对所提交代码的理解（详情已发布在 Piazza 上）。

你成绩的一部分取决于你提交的验证器在私有集上的 macro-MCC。私有集可能包含种子运行中未出现的任务和图表类型，因此请避免针对公开示例过拟合的设计。

| 权重 | 组成部分 |
| ---: | --- |
| 40% | 报告与设计理由 |
| 30% | 基于验证器在私有集上的 macro-MCC |
| 20% | 测验 / 理解检查 |
| 10% | 代码质量与可复现性 |

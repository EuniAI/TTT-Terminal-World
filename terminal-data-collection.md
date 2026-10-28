# mini-swe-agent 原生 Terminal 轨迹采集

采集使用 mini-swe-agent 2.4.6 的 `DefaultAgent` 和 `DockerEnvironment`，以及继承 `LitellmModel` 的 `SingleActionModel`。该子类仅在工具解析阶段增加单次调用数量检查，执行前拒绝多调用并交给原生格式错误重试机制。模型通过 OpenRouter 原生调用 `bash` 工具。项目只负责生成任务、配置原生组件、并行调度、进度条和训练数据导出。

已删除自写的 `BashSession`、`DockerBashSession`、`TerminalEnvironment` 和模型子类。执行过程中不再插入随机结束标记，不再自行解析 shell 输出边界。

## 运行

```bash
python -m pip install -r requirements-collection.txt
docker pull ubuntu:24.04
export OPENROUTER_API_KEY="你的 OpenRouter API key"
python scripts/collect_terminal_data.py \
  --num-trajectories 10 \
  --workers 5 \
  --output terminal_trajs_llm_sample10.jsonl
```

已有文件默认不覆盖。显式加 `--overwrite` 才会在采集前清空该文件。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--model` | z-ai/glm-5.2 | OpenRouter 模型 |
| `--difficulty` | easy | easy / medium / hard，仅调整提示词 |
| `--num-trajectories` | 10 | 最多尝试的轨迹数 |
| `--workers` | 15 | 并行轨迹数，每条轨迹独立容器和 Agent |
| `--min-steps` | 4 | 最少有效观测步数；执行前拒绝含提交标记的过早提交并反馈重试，提交后仍作兜底检查 |
| `--max-steps` | 50 | mini 的模型调用轮数上限；提示每轮只调用一次 bash |
| `--timeout` | 120 | 原生 Docker 命令超时秒数，小数向上取整 |
| `--api-timeout` | 60 | API 请求超时秒数 |
| `--task-temperature` | 1.0 | 任务生成温度（Anthropic thinking 请求不发送此参数），动作温度为 0.3 |
| `--image` | ubuntu:24.04 | 预先下载的容器镜像 |
| `--seed` | 42 | 任务变化提示的种子，不保证远程输出可复现 |

不指定 `max_tokens`。API 重试由 mini 管理，默认设置 `MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=3`，已有环境变量设置优先。安装依赖时包含 `mcp`，沿用 LiteLLM 的正常执行路径。

任务生成返回空正文、异常结束、或工具调用时，也进入同一重试流程；默认总共最多尝试 3 次。错误信息列出具体原因，每次已返回回复的 token 用量均计入统计。重试耗尽后仍停止调度新轨迹，保留已经采集的数据。

任务生成显式开启 thinking（`reasoning.enabled: true`）；Agent 选择命令时显式关闭（`reasoning.effort: none`）。两类请求通过 LiteLLM 的 `extra_body` 传递 [OpenRouter reasoning 参数](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)，并设置 `provider.require_parameters: true`。实际服务端行为尚未通过真实 API 验证。

## 原生执行语义

一条轨迹使用同一个 Docker 容器，每个 action 由 mini 执行一次 `docker exec`，启动独立 Bash。配置为从 `/workspace` 执行，不读取 Bash 启动文件。

- 文件、目录、权限和文件内容跨步骤保留。
- `cd`、`export`、shell 变量和选项不跨步骤保留；下一步重新从 `/workspace` 开始。
- 提示词要求用文件保存跨步骤状态，访问子目录时写明路径。
- 不分配 PTY，不提供交互式 stdin；stdout/stderr 由 mini 的 subprocess 实现合并返回。
- 容器无网络、无宿主目录挂载、不转发 API key；保留 CPU、内存、进程数、只读根目录及 tmpfs 限制。
- `/workspace` 初始为空，所需数据由模型执行命令创建，创建步骤也进入训练数据。

mini 原生使用 `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` 的输出判定任务提交。该命令真实执行后，环境抛出 `Submitted`；导出时不将提交命令加入训练步骤。这是上游的任务结束机制，不是逐条命令插入的随机边界标记。清理由原生 `DockerEnvironment.cleanup()` 处理。

第一版提示词要求同步执行，不支持 REPL、交互式 stdin 或独立后台任务。本版训练环境研究的是跨命令的文件状态记忆，不包含持续 shell 的工作目录、环境变量记忆。

## 任务与难度

保留 16 个轮换主题：文件备份恢复、配置修改查询、目录与相对路径、文件化设置、数字筛选统计、文本搜索计数、重命名复制比较、排序去重、文本替换与空白规范化、文件拆分合并、小型分隔符表格、目录整理与清单、文件权限、符号链接、校验和、归档解压验证。

任务由 LLM 自由设计，命令不设白名单。easy 提示每次一条简单命令，不使用 `&`、`&&`、`;`、管道等组合；medium/hard 允许组合命令和脚本。难度只控制 prompt，不做命令语法过滤。任务要求创建、查询、修改和再次查询数据，复用部分查询命令。

采集 Agent 可以读取完整历史。训练的 World Model 仍每步只输入当前 command，历史通过适应参数保留。

## 数据导出

每行一条轨迹，步骤只含两个字符串：

```json
{"trajectory": [{"command": "echo hello > note.txt", "output": ""}, {"command": "cat note.txt", "output": "hello\n"}]}
```

采集文件只保存真实执行步骤。按模型设计，训练/评估加载器将在每条轨迹首尾加入 `open` / `close` 控制事件，并将多条轨迹组成持续学习流；这些标记不交给 Docker 执行，也不由采集 LLM 生成。World Model 的 fast weights 跨轨迹、跨环境保留，采集容器之间的独立性不意味着模型记忆重置。此处描述加载协议，当前采集导出格式不变。

mini 运行结束后，导出器将 assistant 的 action 与 tool 消息按 `tool_call_id` 对应，读取 `extra.raw_output`，不把格式化反馈、推理或退出码混入 output。保留空输出、换行和普通命令的非零退出结果。

导出前检查 command 中的 DSML 协议标签（兼容全角、半角竖线）及 `<think>` / `</think>` 标签。首次发现时，保留此前有效步骤，丢弃该步骤和后续全部步骤；不删除标签后伪造一个“修复过的命令”，也不跳过异常步骤继续拼接。该规则保守地排除 command 中这些明确分隔符，包括被引号包裹的情况；普通单词 `DSML`、`think` 和真实 output 中的文本不受影响。检查只发生在导出阶段，mini 的执行逻辑不变。

遇到缺失或错配返回、异常执行、被截断的模型回复或未恢复的格式错误时，导出只保留此前有效前缀。mini 可能在执行异常后继续交互，但异常位置及之后的数据都不会写入训练轨迹。设置 `max_consecutive_format_errors=3`，格式错误与一轮多调用在执行前触发重试；恢复成功则继续导出，连续三次失败才终止。达到轮数上限记为 `step_limit`，不宣称任务完成。

原生 Docker 环境使用文本模式和 `errors="replace"` 解码，可能规范化换行或替换无效 UTF-8 字节。本版保留 mini 返回的文本，不额外修改或猜测清理。使用该原生环境后，不再提供旧执行器的 100,000 字节硬上限。任务 prompt 要求数据和输出适量。

采集不按 2048 tokens 截断输出。训练加载器另行实施 `max_output_tokens: 2048`，不把超长返回拆成多个学习事件。

## 并行、进度和文件

并行单位为采集轨迹。同一轨迹内部交互由 mini 管理；GLM 请求省略 `parallel_tool_calls` 以兼容端点，其他模型设置为 `false`；所有模型都在解析阶段强制每轮一个工具调用。总进度显示已处理数量、已保存轨迹与步数、运行中数量及耗时。这里的并行不表示 World Model 对独立轨迹并行适应：模型训练按固定事件顺序更新同一份 fast weights。

每条运行中的轨迹另显示状态，例如 `#10 round=28/50 model | phase=00:31 total=08:42`。`task` 是生成任务，`container` 是启动容器，`model` 是等待模型回复（含重试等待），`command` 是执行命令，`export` / `cleanup` 是导出与容器清理。`phase` 表示观测到当前阶段以来的耗时，`total` 表示该轨迹的总耗时；`round` 是 mini 的模型调用轮次，不是已保存命令数。

显示通过只读轮询 mini 的 `n_calls` 和消息历史推断阶段，约每秒刷新，可能略过极短阶段；不改 Agent 循环、模型调用或执行器。非交互日志在状态变化时输出，并在长时间等待时每 30 秒输出一次。容器启动的 DEBUG/INFO 日志不显示，保留 WARNING 及以上日志。

主线程直接写入指定 JSONL，每完成一条轨迹就写入并 flush，不创建隐藏临时文件。错误时停止调度新轨迹，已运行轨迹继续收尾并保存有效前缀。没有有效步骤时留下空文件并报错。已写完的数据在中断后保留，仍在采集中的轨迹不保证保存；没有续跑功能。

跨轨迹写入顺序受并发完成时机影响，采集 seed 不能独自重建该顺序。模型训练应固定文件摘要和轨迹顺序清单，恢复时使用同一清单；清单只用于调度，不作为模型输入。模型状态恢复与采集器续跑是不同功能。

## 验证与代码

```bash
python -m unittest discover -s scripts -p 'test_collect*.py' -v
```

已运行导出与调度测试，覆盖原始返回提取、空输出、非零退出、异常前缀截断、错配调用、提交命令排除、直接写文件和并发错误收尾。原生组件集成测试模拟 API 回复及 Docker subprocess；当前开发环境未安装 mini，相关测试标记 skipped，未完成真实 API + Docker 端到端验证。

- [命令行入口](scripts/collect_terminal_data.py)
- [任务、调度与数据导出](scripts/collect_llm_terminal_data.py)
- [mini 原生组件配置](scripts/terminal_mini.py)
- [上游 DockerEnvironment](https://github.com/SWE-agent/mini-swe-agent/blob/v2.4.6/src/minisweagent/environments/docker.py)
- [上游 DefaultAgent](https://github.com/SWE-agent/mini-swe-agent/blob/v2.4.6/src/minisweagent/agents/default.py)

工具调用解析失败时，mini 将错误反馈给模型并重试，连续 3 次失败才终止该轨迹。重试计入模型调用轮数上限；恢复成功的格式错误不截断导出，失败回复不写入训练交互，但 token 用量仍计入。

提交命令包含 `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` 时，模型解析层先检查有效观测步数。未达到 `--min-steps` 则不执行该命令，反馈模型继续验证任务；沿用连续三次格式错误的重试上限。

# Terminal World Model 代码实现方案

本文负责代码结构、接口、参数更新、数据编码和恢复流程；研究设定见 [模型设计文档](terminal-world-model-design.md)。`terminal_world/` 已实现事件编码、微型模型、可微内外循环、持久记忆、会话、评估、检查点和 FlashBoB 接口。运行方式与明确限制见 [训练代码说明](terminal_world/README.md)。下文保留算法规格与伪代码；实际验收状态见文末，不将接口接入等同于正式 GPU 验证完成。

## 1. 以官方 e2e 仓库为参考

主要参考 [test-time-training/e2e](https://github.com/test-time-training/e2e)。官方采用 JAX/Equinox；本项目沿用此前的 PyTorch 实现计划，移植参数划分、函数式内循环和外循环求导逻辑。实现开始时固定上游 commit，并记录为 `upstream_revision`，避免依赖不断变化的 `main`。

### 1.1 源码对应关系

| 官方位置 | 已有逻辑 | 本项目的对应实现与变化 |
| --- | --- | --- |
| [model/data.py](https://github.com/test-time-training/e2e/blob/main/ttt/model/data.py) | `Batch` 的输入、目标、loss mask、位置和 attention 信息 | `StepBatch`；只监督本次 output，事件间重新编号 |
| [model/transformer.py](https://github.com/test-time-training/e2e/blob/main/ttt/model/transformer.py) | `Block`、`PrimeStorage`、`BlockCollectionSplit` | 末段双 MLP；将适应 MLP 的当前权重作为持久状态显式传入 |
| 同上：`inner_loop_step` | 计算损失、筛选 inner 参数、函数式更新并返回新模型 | 每个真实返回或控制事件更新一次；仅返回新的 fast weights 和统计 |
| 同上：`loss_for_sequence` | 序列内展开适应步骤，返回 loss 和 metrics | 改为 `loss_for_window(theta, fast, steps)`，额外返回最终 fast weights |
| 同上：`prefix_call` / `suffix_call` | 先计算前段特征，再顺序展开后段适应更新 | 窗口内前段按事件 batch 计算，后段严格按事件原顺序使用当前 φ |
| [model/loss.py](https://github.com/test-time-training/e2e/blob/main/ttt/model/loss.py) | 掩码后的 token CE | 显式返回 NLL 总和与有效 token 数，支持不同长度事件的归一化 |
| [model/loop.py](https://github.com/test-time-training/e2e/blob/main/ttt/model/loop.py) | 对展开过程求元梯度，更新 outer 参数 | 对连续事件窗口求梯度；窗口之间只切计算图，保留权重数值 |
| [optimizers.py](https://github.com/test-time-training/e2e/blob/main/ttt/optimizers.py) | 内循环无动量 SGD，支持梯度裁剪 | 首版保留无动量 SGD；是否裁剪及阈值须显式配置 |

官方 `loss_for_sequence` 在调用内创建适应模型与 inner optimizer state；结束时不返回最终适应模型。官方外循环又将不同序列独立展开后聚合。本项目不能直接套用这两个生命周期：同一份 fast weights 必须从上一窗口传入下一窗口，不能每批从基础模型复制。

官方 [1B E2E 配置](https://github.com/test-time-training/e2e/blob/main/configs/experiment/1b/pretrain/pretrain-1b-e2e.yaml) 启用 `prime`，通过 `spec_inner` 选择末段的 `feed_forward_prime`，使用 SWA 和固定 token chunk。本项目改为完整因果 attention、事件边界更新和持久 fast weights；其学习率、chunk 大小、裁剪阈值不直接作为 Terminal 默认值。

### 1.2 首版约束

- 输入只有当前 command 及固定协议标记；不传历史文本、交互序号或跨事件 KV cache。
- 真实命令为一次性、同步执行；允许组合命令和管道。容器文件状态保留，shell 状态不保留。
- `open` / `close` 为模型协议事件，不交给 Docker；各执行一次内循环更新，不重置权重。
- fast weights 只在模型首次创建时初始化一次。换轨迹、环境、batch、epoch、窗口或重启均沿用已有状态。
- 训练只有元训练阶段；窗口内保留穿过更新的二阶梯度，窗口间显式截断计算图。评估和部署关闭外循环。
- 正式底座选用 Qwen2.5-Coder-3B Base，新增适应分支后整体约 3.70B；微型模型仅用于正确性验证。所有离线训练和评估都读取真实 output，不重新执行数据命令。

## 2. 代码结构与职责

```text
TTT-Terminal-World/
├── configs/
│   └── experiment/
│       ├── tiny.yaml            # 可直接运行的微型模型配置
│       └── terminal_ttt.yaml    # 正式底座与训练参数模板，空项必须填写
├── terminal_world/
│   ├── config.py                # 配置校验，不允许静默回退初始化
│   ├── train.py                 # 首次创建/恢复、训练入口
│   ├── evaluate.py              # 检查点副本上的先评分后适应
│   ├── dataloader/
│   │   ├── trajectory.py        # 原始 JSONL、数据清单和轨迹顺序
│   │   ├── events.py            # 插入边界、事件游标、划分窗口
│   │   └── encoding.py          # 协议编码、目标截断、标签与掩码
│   ├── model/
│   │   ├── data.py              # Event、StepBatch、WindowBatch
│   │   ├── transformer.py       # 底座适配、双 MLP、参数划分
│   │   ├── pretrained.py        # 标准 Llama 权重导入及初始 logits 对比
│   │   ├── attention.py         # math / FlashBoB，显式兼容检查
│   │   ├── meta_model.py        # step_loss、inner_loop_step、窗口展开
│   │   └── loop.py              # 外循环更新和持久 fast weights
│   ├── metrics.py               # 真实命令/边界分开统计
│   ├── session.py               # predict、observe、control
│   └── checkpoint.py            # θ、当前 φ、游标及配置的完整恢复
├── scripts/                     # 现有采集代码继续使用
└── tests/
    ├── test_events.py
    ├── test_encoding.py
    ├── test_meta_learning.py
    └── test_session_checkpoint.py
```

`session.py` 管理模型记忆，执行器负责真实环境。已有 mini-swe-agent 采集器不需要为了模型训练重新包装成持久 shell。正式训练依赖与 `requirements-collection.txt` 分开维护，锁定验证过的框架版本。

## 3. 从原始轨迹到事件窗口

### 3.1 数据来源与顺序

输入沿用现有 JSONL，每行一条轨迹：

```json
{"trajectory": [{"command": "echo hello > note.txt", "output": ""}, {"command": "cat note.txt", "output": "hello\n"}]}
```

采集器直接保存 mini 的 `extra.raw_output`，保留 stdout/stderr 合并文本、空输出、非零退出的实际报错和空白，不猜测删除类似 command 或 prompt 的字符串。无 PTY 执行管道不产生终端回显或 shell prompt。采集细节见 [数据采集说明](terminal-data-collection.md)。

先按完整轨迹划分训练/验证，再固定各自的事件顺序。数据清单记录文件摘要、行号顺序和划分；相同内容的重复轨迹应归到同一集合。现有数据没有可靠的环境 ID，仅按轨迹隔离不能宣称环境隔离。

采集可并行，但文件中的轨迹顺序受完成时机影响。训练以保存的数据清单为准，不能依赖采集 seed 重建顺序。所有游标和来源信息仅供调度、统计和恢复，不进入模型输入。

### 3.2 Event 与边界协议

| 字段 | 含义 |
| --- | --- |
| `kind` | `shell` 或 `control`，决定固定输入格式 |
| `command` | 真实命令文本，或控制命令 `open` / `close` |
| `output` | 真实返回；控制事件固定为空字符串 |
| `cursor` | 可恢复的事件位置，仅用于运行时 |

原始 JSONL 保持不变，由事件层插入边界：

```text
open → trajectory A 的步骤 → close → open → trajectory B 的步骤 → close → …
```

协议状态从 closed 开始：`open` 后允许真实命令，`close` 后才允许下一次 `open`；拒绝嵌套 open、重复 close 或 closed 状态下的真实命令。这个状态只检查协议，不给模型补充身份或历史。恢复到已 open 的轨迹时不能再插入 open。

`close` 表示已记录轨迹结束，不代表任务成功。采集失败后保留的有效前缀仍可作为一条轨迹；超时、不完整返回及之后的步骤不作为完整 observation 训练。普通非零退出与执行器超时分开处理。

### 3.3 编码与目标

先明确区分控制事件和 shell 文本，再生成固定格式。可使用下列专用 token，最终 ID 写入 tokenizer 与检查点：

```text
shell 输入：  <|command|> 原始 command <|output|>
open 输入：   <|traj_open|> <|output|>
close 输入：  <|traj_close|> <|output|>
```

真实 command/output 按普通文本编码；其中恰好出现上述字符串时，不能解释成协议 token。控制标记只能由事件编码器插入，不能让系统命令 `open file` 变成轨迹边界。是否需要底座 BOS 由统一编码器固定处理，不在训练/推理间变化。

令 `N = max_output_tokens`，首版为 2048。正文 token 为 `body`，目标为 `(body + [OBS_END])[:N]`；`OBS_END` 表示 observation 文本结束，不编码退出码。控制事件和空真实输出的目标都是单个 `OBS_END`，但输入标记不同。

| 正文长度 | 保留目标 | `body_truncated` | `has_end_token` |
| --- | --- | --- | --- |
| 0 | `OBS_END` | false | true |
| N−1 | 全部正文 + `OBS_END` | false | true |
| N | 全部正文 | false | false |
| > N | 前 N 个正文 token | true | false |

不为截断处补结束标记。正文未截断但结束 token 被截掉，与完整目标分开统计。保存原始正文 token 数、保留正文数及上述两个标志；这些统计不输入模型。

`StepBatch` 至少包含 `input_ids`、`target_tokens`、`loss_masks`、`attention_mask`、`position_ids`、`kind` 和统计 metadata。编码规则为：

```python
sequence = prompt_tokens + target
input_ids = sequence[:-1]
target_tokens = sequence[1:]
loss_masks = [0] * (len(prompt_tokens) - 1) + [1] * len(target)
```

统一在编码层移动一次标签，不再让底座的内置 loss 重复 shift。只对目标计算损失；command、格式标记和 padding 不受监督。位置编号每事件从零开始，attention 为当前事件内的 full causal attention，不跨事件拼接。

command 不截断。完整 prompt 加配置输出预算超过模型容量时显式报错，不能静默缩减预算。生成接口只接收 prompt，不接收真实 output 或真实长度；最多生成 N 个 token，或遇到 `OBS_END` 停止。

### 3.4 WindowBatch

先生成完整事件流，再按 `unroll_events` 划分不重叠窗口，计数包含控制事件。窗口不能拆开一个事件，也不能因切在轨迹中间额外插入 open/close；最后不足长度的窗口照常处理。

第一版顺序更新一份 fast weights，不能把多条轨迹复制成互相独立的参数状态再平均梯度。允许将同一窗口内、不依赖 φ 的前段计算组成事件 batch；这只是计算调度，不改变后段的损失计算与参数更新顺序。数据读取 batch、前段计算 batch 和元梯度窗口是三个不同概念，均不触发权重重置。

`unroll_events` 至少为 2；单事件窗口无法通过后续损失学习该事件更新的效果。实际训练必须包含 `close → open → 后续真实命令` 位于同一窗口的样本，并报告这种边界覆盖率。窗口最后一次更新仍执行并传给下一窗口，只是它对下一窗口的作用不进入本窗口元梯度。

## 4. 模型与参数布局

### 4.1 参考官方双 MLP

末段 blocks 参考官方 `feed_forward_prime` 与普通 `feed_forward` 的串联残差结构：attention 后先经过适应 MLP，再经过普通 MLP；未选中的前段 blocks 保持普通结构。适应层数由配置决定，不把上游 1B 的层数硬编码成所有底座的规则。

| 参数 | 保存位置 | 更新方式 |
| --- | --- | --- |
| θ | 模型通用参数 | 内循环不更新数值，外循环更新 |
| φₜ | 单独的 `fast_weights` 映射 | 每事件一次函数式 SGD，跨全部事件持续传递 |
| φ₀ | 首次创建时的来源记录 | 只初始化一次，不作为可反复装载的备用状态 |

φ 只包含选中适应 MLP 的权重；embedding、attention、normalization、普通 MLP 和输出层属于 θ。两组名字和底层存储均不能共享，外循环 optimizer 只持有 θ。φ 存储当前完整权重，不保存“相对每批基础权重的增量”，避免 θ 更新时隐式重建或清零记忆。

首次接入普通预训练底座时，需要为新增适应 MLP 定义一次性初始化。建议保留原 attention、普通 MLP 和规范化路径；新增适应残差支路的输入/门控投影正常初始化，输出投影置零，使支路初始为恒等残差。不能把所有投影都置零。新增规范化层属于 θ，具体位置按底座与上游结构核对，并用旧词表 logits 的前后对比验证接入效果。

这是从普通底座接入的工程方案，不是对上游训练结果的复现。若采用官方 TTT checkpoint，则加载其现有双 MLP 及参数映射，不再次添加支路。当前选定 Qwen2.5-Coder-3B Base，末 9/36 层适应；接入后分别报告 θ 和 φ 参数量，新增分支后总规模约 3.70B。

### 4.2 PyTorch 函数式调用

初始计划使用 [`torch.func.functional_call`](https://docs.pytorch.org/docs/stable/generated/torch.func.functional_call.html) 替换参数；当前实现选择 §4.5 允许的显式参数前向，以避免 FlashBoB 不支持的 `torch.func` 变换。`nn.Module` 仅持有 θ，φ 是独立命名 tensor 映射，通过 `F.linear` 进入适应 MLP，不在每个事件调用 `load_state_dict` 或原地覆写权重。创建/恢复均检查参数名、形状、精度、设备和存储不重叠；tied embedding 只在 θ 内共享。以下伪代码中的 θ 对应实际 API 的 `model` 参数。

```python
def encode_prefix_batch(theta, steps): ...  # -> 按原顺序排列、保留计算图的前段特征
def step_loss(theta, fast_weights, step, *, prefix_output=None): ...
# step_loss 返回 nll_sum, token_count；未传 prefix_output 时逐事件运行完整模型。
def inner_loop_step(fast_weights, mean_loss, *, alpha, create_graph): ...
def loss_for_window(theta, fast_weights, steps, alpha): ...  # -> loss, next_fast, metrics
```

`step_loss` 使用底座 logits 自行计算 masked CE，累加使用 FP32。关闭 dropout 和跨事件缓存；普通数学 attention 用于正确性基准，正式 CUDA 训练接入 FlashBoB，具体约束见 §4.5。不能仅凭一阶 backward 成功启用某个融合后端。模型的 `.eval()` 不等于禁止梯度，内循环仍须启用 autograd。

首次创建和恢复入口均将当前 φ 作为可求梯度的叶子 tensor；θ 在元训练时启用梯度、在在线适应时固定。断言窗口非空，且每事件有效目标数至少为 1；空 output 通过 `OBS_END` 满足这一条件。计算图中的 metrics 及时 detach，不能被日志容器跨窗口保留。

### 4.3 内循环

参考官方筛选 inner 参数后返回新权重的方式，使用 [`torch.autograd.grad`](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html) 构造更新；不能用普通 optimizer 的原地 `step()` 代替可微分更新。

```python
def inner_loop_step(fast_weights, mean_loss, *, alpha, create_graph):
    names = tuple(fast_weights)
    values = tuple(fast_weights[name] for name in names)
    grads = torch.autograd.grad(
        mean_loss, values, create_graph=create_graph
    )
    return {
        name: value - alpha * grad
        for name, value, grad in zip(names, values, grads)
    }
```

这里是无裁剪、无动量、无 weight decay 的基线，`alpha` 为显式配置的固定学习率。若启用内循环裁剪，须对 `grads` 做可微分的全局范数裁剪，并将相同规则用于训练、评估和在线更新；记录阈值和触发率。上游支持裁剪，不代表可以静默改变本项目公式。

θ 在一个窗口内保持数值不变，但不能 detach 或在 `no_grad` 下处理其前段网络；后续损失需要通过混合导数影响 θ。只有 φ 是内循环更新对象，不等于只有 φ 需要保留求导路径。

`open` / `close` 的单 token 目标容易被学会，梯度可能随之减小。因此记录边界损失、边界更新范数和切换后真实命令的预测表现；结束 token 的准确率不是边界记忆有效性的充分证据。首版先验证这个最小机制，不预设它一定能隔离不同环境的事实。

### 4.4 前段批量计算，后段顺序更新

首版训练采用该执行方式，同时保留逐事件运行完整模型的参考路径用于验证。沿用官方先计算 prefix、再展开 suffix 的思路，但本项目各事件之间没有文本 attention，不能直接把窗口拼成一条长序列。

在第一个含适应 MLP 的 block 之前切分模型。prefix 包含 embedding 和此前所有普通 blocks，只依赖 θ；suffix 包含剩余 blocks、最终 normalization 和输出头，同时依赖 θ 和当前 φ。后段中即使某些层自身没有 fast weights，其输入也可能依赖 φ，因此不能任意提前计算。切分位置由适应层配置推导，不硬编码层数。

例如，24 层模型的末 6 层含适应 MLP，则前 18 层可以对窗口内多个事件批量前向，末 6 层与输出头仍按事件顺序执行：

```text
事件 1 ─┐                 ┌─ h₁
事件 2 ─┼─ prefix batch ───┼─ h₂
事件 3 ─┘                 └─ h₃

h₁ + φ₁ → suffix → loss₁ → 更新为 φ₂
h₂ + φ₂ → suffix → loss₂ → 更新为 φ₃
h₃ + φ₃ → suffix → loss₃ → 更新为 φ₄
```

窗口内 θ 数值固定，prefix 又不依赖 φ，所以提前计算 h₂ 不会让事件 2 使用旧的 fast weights。只有事件 1 更新完成后，事件 2 才计算最终 logits、损失和下一次更新。`open` / `close` 作为普通协议事件参与这一顺序。

**前段 batch 的隔离规则：**

- 每个事件占 batch 的一行，独立进行 causal attention，位置编号从零开始；不同事件不能互相 attention，也不能共享随事件变化的缓存。
- 按当前 batch 的实际最大长度右侧 padding，使用匹配的有效位置与因果掩码；padding 不进入损失。提取各事件前段特征时只保留其真实输入长度，并为 suffix 传递对应掩码和位置。
- prefix 输入是编码层已经 shift 的 `input_ids`，与逐事件 teacher forcing 完全相同。提前计算真实目标前文不等于允许看到当前预测位置的答案或未来 token；因果掩码仍必须生效。
- 不排序或打乱事件来改变适应顺序。若整个窗口一次批量计算超出显存，可按 `prefix_batch_size` 分成连续小批，保留原索引；它不改变 `unroll_events` 或每事件更新一次的规则。

**梯度与缓存生命周期：**

prefix 的参数虽不在内循环中更新，但仍由外循环训练。`encode_prefix_batch` 必须在启用梯度的情况下运行，不能 detach 特征、转为 NumPy 或包进 `no_grad`。窗口内后续损失应能穿过 suffix 的适应更新、前段特征和 prefix 参数；prefix batch 的计算图在整个窗口求完元梯度后才释放。

前段特征只在当前窗口内使用，不跨 θ 的外循环更新复用，也不写入持久模型状态。θ 更新后，下一窗口重新计算前段。日志仅记录已 detach 的统计，持久记忆仍只有 φ。

该优化用于已知完整 command/output 的离线 teacher-forced 损失计算。在线命令尚未到达时不能提前计算；自回归生成也不能复用由真实 output 构造的前段特征。生成仍只处理当前 command 和本次已生成 token，反馈到达后再单独计算学习损失。

预期收益是提高前段的 GPU 利用率、缩短必须串行的计算部分，不减少事件数量或跳过内循环更新。批量特征和计算图也会占用显存；`prefix_batch_size` 仅限制单次前段 batch 的工作量，不保证消除整个窗口的激活保存成本。实际速度和峰值显存需与全串行路径实测比较，不能按前后层数比例推算加速倍数。

### 4.5 FlashBoB attention 后端

正式 CUDA 元训练采用 FlashBoB，配置 `attention_backend=flashbob`；保留 `attention_backend=math` 作为微型模型与数值验证基准。math 路径显式选择可求二阶梯度的数学实现，不能让 SDPA 自动切到未经验证的融合内核。该选择只改变 attention 执行方式，不改变事件顺序、内循环更新、窗口截断或持久 φ 的生命周期。

参考 [FlashBoB 仓库](https://github.com/MingruiLiu-ML-Lab/FlashBoB)及其[支持契约](https://github.com/MingruiLiu-ML-Lab/FlashBoB/blob/main/docs/support.md)。接入时固定上游 commit、PyTorch/Triton 版本，并记录实际 GPU 和后端配置；以下是当前支持范围，不能推定任意版本都兼容。

**算子接入：**

- 在模型 attention adapter 中，把完成位置编码后的 Q/K/V 传给 `bob.sdpa_bob`，布局为 `[batch, heads, sequence, head_dim]`；保留底座的 scale、投影和输出布局转换。
- 当前完整因果 attention 使用 `is_causal=True, window_size=0`，事件位置从零开始。离线学习的 Q/K 序列等长，不需要跨事件 KV cache。
- prefix 和 suffix 共用该 adapter，首版 CUDA 路径统一使用 FlashBoB，避免未经验证地混用后端。后续若单独优化 prefix，须根据实际梯度路径重新验收。
- 内循环继续通过 `torch.autograd.grad(..., create_graph=True)` 构建可微分 SGD，外层对窗口损失反向传播。不要改成 `torch.func.grad`、`vmap` 或 `jvp`；上游未承诺支持这些变换。§4.2 的 `functional_call` 仅用于显式替换参数，须与 adapter 联合验证；若具体版本不兼容，改用显式参数前向，不能退回原地更新 φ。

**右侧 padding：**

FlashBoB 当前接口不接受自定义 attention mask/bias。本方案每条事件独占 batch 一行、只在右侧补齐，且使用纯 causal attention，因此真实 token 不会看到尾部 padding，无需把 padding mask 传给该算子。长度与 loss mask 仍保留在 `StepBatch`：prefix 输出按真实长度裁掉后再送 suffix，padding 位置不参与 loss。位置编码、归一化和其他操作必须保持有效 token 不依赖 padding；不能引入跨位置的无掩码聚合。此规则不适用于左侧 padding、非因果 attention 或把多条事件拼进同一行。

**运行条件与失败处理：**

- 当前 CUDA MHA 支持 FP16/BF16，head dimension 为 32、64、128；GQA/MQA 使用 BF16、Q/K 等长、查询头与 KV 头比例不超过 32。首版训练优先 BF16，并在选底座时核对 head 配置。
- 当前上游明确支持 PyTorch 2.8 + Triton 3.4，或 PyTorch 2.9 + Triton 3.5。CUDA 路径依赖私有 ATen 算子，其他组合必须重新验证；CPU 微型测试直接使用 math 基准。
- dropout 保持关闭，不传自定义 attention bias。选择 FlashBoB 后，缺依赖、设备/形状不支持或版本验证失败都显式报错；不得静默换后端而把结果记为 FlashBoB 性能。

**接入验收：**

先与 math 比较算子前向、一阶和二阶梯度，再比较含多个连续更新的完整窗口：有效位置 logits、每事件 loss、最终 φ，以及 prefix/suffix 的 θ 外层梯度。覆盖不同长度、右侧 padding、空输出、open/close 和正式底座的 head 配置；小模型 math 路径用高精度有限差分校验，BF16 对比明确误差容限。两条路径使用相同初始状态和数据，不以“能 backward”代替数值验证。

在同一硬件、精度、窗口和 prefix batch 配置下，预热并同步 CUDA 后测量整个窗口（含内外循环）的耗时与峰值显存，排除首次编译时间并单独记录。FlashBoB 旨在降低 attention 尤其二阶反向的成本；不消除 suffix 的顺序依赖，不预设端到端加速倍数。尚未完成这些检查前，文档中的接入方案不代表已验证兼容或已测得收益。

## 5. 持久状态上的外循环

窗口 k 内 θ 固定为 θₖ，输入 fast weights 是上个窗口的最终数值。损失按保留的目标 token 归一化，包含控制事件；真实命令指标单独累计。

```python
def loss_for_window(theta, fast_weights, steps, alpha):
    # 内部可按 prefix_batch_size 分批，但返回顺序必须与 steps 一致。
    # 保留到 theta 的计算图；此处不读取或更新 fast_weights。
    prefix_outputs = encode_prefix_batch(theta, steps)
    sums = []
    token_count = 0
    metrics = []
    for step, prefix_output in zip(steps, prefix_outputs, strict=True):
        nll_sum, n = step_loss(
            theta, fast_weights, step, prefix_output=prefix_output
        )  # 仅运行 suffix，使用前一个事件更新后的 fast_weights
        sums.append(nll_sum)  # 本事件更新前的损失
        token_count += n
        metrics.append(detached_stats(step, nll_sum, n))
        fast_weights = inner_loop_step(
            fast_weights, nll_sum / n,
            alpha=alpha, create_graph=True,
        )
    return torch.stack(sums).sum() / token_count, fast_weights, metrics
```

全串行参考路径不预计算 prefix，循环内直接调用 `step_loss(theta, fast_weights, step)`；目标、损失归一化和内循环更新规则完全相同。两条路径的差异仅在前段计算的调度方式。

外循环只更新 θ。以下省略配置和错误处理，但固定更新顺序：

```python
# 首次创建时初始化；恢复时同时加载 theta、fast、optimizer 和游标。
for window in event_stream.from_cursor(state.next_cursor):
    outer_optimizer.zero_grad(set_to_none=True)
    loss, next_fast, metrics = loss_for_window(
        theta, state.fast_weights, window.steps, alpha
    )
    outer_grads = torch.autograd.grad(loss, tuple(theta.values()))
    for parameter, grad in zip(theta.values(), outer_grads):
        parameter.grad = grad
    outer_optimizer.step()
    state.fast_weights = {
        name: value.detach().requires_grad_(True)
        for name, value in next_fast.items()
    }
    state.next_cursor = window.next_cursor
    # 此时才可提交窗口检查点；metrics 已与计算图分离。
```

外循环计划使用 AdamW，其优化器状态长期保留。内循环没有动量状态。训练图只在显式窗口末尾释放：窗口内不 detach，窗口间保留权重数值并重新启用梯度；所有构造器、轨迹切换、epoch 切换都不能偷偷重新初始化 φ。

下一窗口使用更新后的 θ 和未重置的 φ。θ 变化可能改变既有 φ 的作用，因此监控外循环更新前后的损失与更新范数，不能通过重置 φ 来掩盖不稳定。检查点也必须保存匹配的 θ/φ，不能把不同版本任意拼接。

这是截断元学习：窗口内保留二阶路径，窗口之前的适应过程被视为既有状态，不追溯全生命周期元梯度。φ₀ 不再作为每个窗口的元学习初始化。每个窗口先完整反向传播再改 θ，不能中途 optimizer.step。

遇到非有限损失或梯度时，停止提交该窗口并保留最近有效检查点及诊断，不静默跳过事件、清空记忆或继续扩大状态偏差。精度优化、重计算和融合内核在微型模型验证后逐项启用；第一版不先引入多副本数据并行。

## 6. 评估与在线模型接口

### 6.1 先评分，再适应

对每条真实命令，先用固定的当前 θ/φ 计算 NLL 和可选生成结果，再执行一次内循环更新。生成最多 N 个 token 或遇到 `OBS_END` 停止，不利用真实目标长度提前停止。

任务 NLL 以所有真实命令的 NLL 总和除以有效目标 token 数汇总，不平均每步 NLL。分别报告：

- 完整目标：正文匹配与结束 token 表现。
- 正文完整但结束 token 未保留：正文匹配，不评价缺失的结束标签。
- 正文截断：保留前缀匹配，不声称覆盖尾部。
- 正文截断率、结束 token 覆盖率、保留正文 token 比例；空正文不参与该比例的分母。
- 控制事件损失和更新范数，以及轨迹切换后的真实命令误差。

离线验证复制当前检查点的完整 θ/φ，不回到 φ₀。验证更新只作用于副本，不写回训练实例。关闭适应的消融从同一检查点副本开始，保持输入和顺序相同，同时关闭真实命令及控制事件的所有内循环更新。重复评估的暴露历史必须记录，不能把已经学习的数据称为首次遇见。

### 6.2 Session 接口

| 接口 | 行为 |
| --- | --- |
| `create_model(config)` | 仅首次创建模型时初始化 θ/φ |
| `resume_session(checkpoint)` | 恢复当前 θ/φ 和协议状态；缺失 fast weights 则报错 |
| `predict(command)` | 只生成，不修改 φ；当前首版全前缀重算，后续可增加本次临时 KV cache |
| `observe(command, real_output)` | 计算更新前损失，按真实目标适应一次，返回评分和截断统计 |
| `control("open" / "close")` | 检查协议状态，以空协议输出更新一次，不调用执行器 |

生成可以在 `no_grad` 下运行；`observe` 和 `control` 必须启用梯度。在线 θ 固定，更新后 detach φ 数值，清空本次文本和 KV cache，不保留元梯度图。事件调度层确保每条真实返回只调用一次 observe；评分不能额外触发一次学习。

```python
session = resume_session(checkpoint)
# 仅在开启新轨迹时调用；恢复到已开启的轨迹则跳过。
session.control("open")
prediction = session.predict(command)
real_output = execute_once(command)  # 外部执行器；离线时直接读取数据
score = session.observe(command, real_output)
# 当前轨迹的其余命令……
session.control("close")
# 下一条轨迹或环境继续使用同一个 session。
```

执行器仍使用一次性 `docker exec`，不维持 shell。正常命令非零退出可学习其实际返回；超时或结果缺失不能当作空输出。发生不完整执行时结束当前已知轨迹，保留模型记忆，待外部确认或新环境开始后再 open，不补造等待步骤。

## 7. 检查点与恢复

模型参数、fast weights 和游标必须作为同一版本提交，保存到临时文件后原子替换。至少包含：

| 内容 | 原因 |
| --- | --- |
| θ、当前 φ、模型配置、tokenizer 与协议 token ID | 恢复相同模型及输入语义 |
| 外循环 optimizer、学习率进度、精度/scaler 状态（如启用） | 恢复训练更新过程 |
| 数据清单摘要、下一个事件游标、epoch、轨迹开闭状态 | 保持原顺序且不重复 open/close |
| 窗口配置、随机状态、上游 revision、软件版本 | 可复现运行与诊断 |

离线训练在窗口提交点保存，不序列化窗口中途计算图。恢复后只重放未提交的已记录事件，重新计算模型更新；这不涉及执行 shell 命令。缺少 φ 或数据清单不匹配时显式报错，不用 φ₀ 回退。

在线恢复要区分“恢复模型更新”和“重新执行命令”。已取得并持久保存的真实 command/output 可以用于补做未提交的模型更新；已经执行的 shell 命令不能自动再执行。若崩溃发生在命令执行与结果持久化之间，执行状态可能不明确，应交由外部调度器核实，不能宣称检查点自动提供端到端恰好一次执行。

模型检查点不保存 Docker 文件系统。继续同一条在线轨迹还需要对应的真实环境；若环境已丢失，应以 close/open 标记轨迹切换并保留当前 φ，不能把新容器假装成原环境的连续执行。

## 8. 实现顺序与验收

| 阶段 | 交付 | 必须通过的验证 |
| --- | --- | --- |
| 1. 数据与事件 | JSONL → Event → StepBatch → WindowBatch | 轨迹顺序、边界各一次、空输出、特殊 token 文本、首目标标签、截断边界；原始文件不变 |
| 2. 微型模型 | θ/φ 分离、prefix/suffix 切分、step_loss | 参数及存储不重叠，生成不读目标，future token 不影响当前 logits，事件间 attention 隔离 |
| 3. 内外循环 | prefix batch、顺序 suffix、函数式 SGD 和窗口外循环 | 与全串行路径比较损失、最终 φ 及 θ 元梯度；有限差分检查，切图不改 φ 数值 |
| 4. 会话与恢复 | predict/observe/control、完整 checkpoint | 更新恰好一次，恢复不重复边界；连续运行与保存恢复后的预测、参数和游标一致 |
| 5. 约 3B 底座 | 双 MLP 接入、参数映射、长度配置、FlashBoB adapter | 一次性初始化检查、FlashBoB/math 完整窗口梯度对比、峰值显存与耗时、有限数值 |
| 6. 小规模元训练 | 少量真实轨迹上的完整闭环 | 任务/边界指标分离，跨边界窗口覆盖，长期 φ 连续性与验证副本隔离 |

微型梯度测试至少用两个真实事件与控制边界构造非零元梯度路径；只测试单步 loss 下降不足以证明元学习实现正确。验收还应覆盖正文长度 0、N−1、N、N+1，以及窗口恰好切在 open/close 前后的情况。

前段批量优化另需验证：在相同初始 θ/φ、关闭 dropout、相同输入和精度下，比较逐事件有效位置 logits、每步损失、最终 fast weights，以及 prefix/suffix 两部分 θ 的外循环梯度，允许声明过的浮点误差。用不同长度事件、空输出和 open/close 覆盖 padding；修改后一个事件不能影响较早事件的预测和更新。分别测试 `prefix_batch_size=1`、中间值和整窗口，确保计算分批不改变适应顺序。性能记录整窗口耗时和峰值显存，不只测 prefix 前向。

微型模型算法验证后，当前已固定 Qwen2.5-Coder-3B Base 的 `model_id/revision`、末 9 层适应和 BF16 精度。实验配置提供显式的 inner/outer 学习率、裁剪策略、`unroll_events` 和 `prefix_batch_size` 起步值，尚需 GPU 验证和调优；`max_output_tokens` 为 2048。关键配置未填写时仍启动失败，不隐式回退或重置记忆。

### 8.1 当前代码交付与验收边界

- 已实现：manifest 摘要和去重划分、原序事件流、open/close 协议、输出编码；前段 batch + 后段顺序 SGD；AdamW 外循环；跨窗口/轨迹/epoch 保留 φ；独立评估副本与 frozen 消融；在线 Session；原子检查点和严格恢复。
- 已验证 CPU math 路径：右侧 padding/因果隔离；串行与批量的 logits、损失、最终 φ、θ 元梯度一致；有限差分；保存恢复与连续训练完全一致，包括跨 epoch；生成不更新记忆；非有限窗口不提交。现有采集数据的两条完整轨迹另用于小规模运行检查，原始数据未修改。
- FlashBoB 已接入并固定 commit 为 `f2bd4a38a9b65bae455ef4336d7b54f0acd3caca`，e2e 参考 revision 为 `8a2d685c080d895a4b138eb6b5ff69a840e40968`。CUDA 测试与完整窗口 benchmark 已提供；本机无 CUDA、PyTorch 为 2.13.0，**未执行 FlashBoB GPU 数值或性能验收**。
- 普通预训练模型导入支持无 bias 的 Llama 和保留 QKV bias 的 Qwen2，均限于 SwiGLU、未缩放 RoPE、full causal attention；其他架构、滑动窗口、scaled RoPE 和官方 e2e checkpoint 显式拒绝。导入入口会比较原词表 logits，但本机缺少 transformers/真实底座，尚未完成真实预训练权重验证。正式模型 ID/revision 和硬件预算仍由实验配置明确指定。
- 在线生成暂为贪心全前缀重算，无 KV cache；这是正确但较慢的首版实现。窗口内前段特征不用于未知未来生成。端到端恰好一次执行仍由外部调度器承担。

本次本地验收：新增测试 **23 passed / 3 skipped**（2 项 CUDA、1 项真实 transformers 对比）；原有采集测试 **44 passed / 5 skipped**。另用 `terminal_trajs_llm.jsonl` 前两条完整轨迹的临时副本运行 **32 个事件、8 个训练窗口**，在第 2 窗口保存后恢复至完成，并执行隔离副本评估；跨边界 triple 在同一窗口内覆盖 1 次。该运行只验证训练闭环，不是泛化评估或正式吞吐实验。

实际运行和测试命令见 [terminal_world/README.md](terminal_world/README.md)。采集器不需要为训练改写；上述测试结论仅覆盖已执行的路径，不表示约 1B 模型已训练或 FlashBoB 已获得加速。

### Qwen2.5-Coder-3B 接入更新

正式配置已选定 `Qwen/Qwen2.5-Coder-3B` Base，revision 固定为 `09d9bc5d376b0cfa0100a0694ea7de7232525803`。底座 36 层中最后 9 层增加 prime MLP，适应层占比 25%；φ 共 608,698,368 参数，改造后整体约 3.70B。原有 QKV bias、共享 embedding、GQA、RoPE 和普通 MLP 均保留；`qkv_bias` 随配置保存恢复，内外循环生命周期不变。配置中的学习率和窗口长度仅为实验起点。

新增 Qwen2 严格权重导入、非零 bias 测试、官方微型 Qwen2 logits 对齐测试、tokenizer/Session 往返测试，以及带 bias 的元梯度和恢复验证。真实 3B 权重导入及 FlashBoB GPU 数值/性能尚未验收；具体运行方式与环境限制见 `terminal_world/README.md`。前面的本地验收数字为历史记录，不代表本次新增路径全部执行。

此次本地测试：`python3 -m pytest -q` 为 **41 passed / 6 skipped**。跳过项为 3 个 CUDA 测试和 3 个依赖 Transformers/tokenizers 的测试；当前环境缺少这些可选依赖且安装尝试未成功。已执行的是 CPU math、非零 bias 映射、带/不带 bias 的元梯度与恢复验证，不是官方真实 3B 权重的验收。

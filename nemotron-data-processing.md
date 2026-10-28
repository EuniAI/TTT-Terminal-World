# Nemotron-Terminal-Corpus：干净 Terminal 轨迹处理方案

> 历史方案：当前已转向 [自主采集](terminal-data-collection.md)。本转换脚本和已有数据保留，尚未实现以下曾规划的回显清理。

## 1. 目标

从 Nemotron-Terminal-Corpus 中提取连续、真实的 terminal 交互轨迹，供模型学习：

```text
当前 command + 此前交互形成的参数记忆 → 预测清理后的 output
获得真实 output 后更新模型，再处理下一条 command
```

command 由外部提供。最终只交付训练数据 `terminal_trajs.jsonl`，不额外生成中间数据集、审查报告或统计文件。

处理脚本已实现：[scripts/prepare_terminal_data.py](scripts/prepare_terminal_data.py)。已检查前 10 条转换样本并完成离线解析测试，尚未验证全量转换。抽样发现的嵌套框架标题已修复，旧样本需要重新转换。

在项目根目录运行：

```bash
python -m pip install datasets
python scripts/prepare_terminal_data.py
```

默认流式读取完整 `skill_based_easy/train`，在当前目录生成唯一的数据产物 `terminal_trajs.jsonl`。可用 `--output` 指定路径；已有文件默认不覆盖，需要替换时显式加 `--overwrite`。

也支持转换完整 medium 子集，使用相同清洗和遇错截断规则：

```bash
python scripts/prepare_terminal_data.py --difficulty medium
```

此时读取 `skill_based_medium/train`，默认输出 `terminal_trajs_medium.jsonl`。试跑可添加 `--limit 1000 --output terminal_trajs_medium_sample1000.jsonl`。本地数据集根目录模式也根据 `--difficulty` 选择 easy 或 medium；直接传入 JSONL、Parquet 或难度目录时，以该路径实际内容为准。

如果已在本地下载原始数据：

```bash
python scripts/prepare_terminal_data.py --input /path/to/Nemotron-Terminal-Corpus
```

本地目录模式读取 `synthetic_tasks/skill_based/easy/*/data_filtered.parquet`；也可以直接传 easy 目录、单个原始 Parquet 或含 `conversations` 字段的原始 JSONL。单文件模式只处理提供的文件，处理完整 easy 子集应使用默认在线模式或完整本地目录。`--revision` 可指定线上数据版本。

不限制记录数量、不去重、不切分训练测试集。读入失败时删除临时文件并保留旧输出；没有有效轨迹时返回错误，不生成空训练集。

先试跑前 10 条原始记录：

```bash
python scripts/prepare_terminal_data.py --limit 10 --output terminal_trajs_sample10.jsonl
```

`--limit` 仅用于试跑，省略时处理全量。它限制读入记录数；如果存在无效记录，输出轨迹数可能少于 10。

离线测试：

```bash
python -m unittest discover -s scripts -p 'test_prepare_terminal_data.py' -v
```

## 2. 数据来源

使用官方 **`skill_based_easy` 的完整 `train` split，覆盖其中所有任务类别和全部分片**。

不再按命令类型、复杂度、轨迹步数、输出长度或任务成功与否筛选。此次范围是 easy 子训练集的全部数据，不包含 `dataset_adapters`、`skill_based_medium` 和 `skill_based_mixed`。

读取配置：

```python
from datasets import load_dataset

ds = load_dataset(
    "nvidia/Nemotron-Terminal-Corpus",
    name="skill_based_easy",
    split="train",
    streaming=True,
)
```

用户已通过上述线上配置完成前 10 条试跑。

- [官方数据集](https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Corpus)
- [官方配置与路径](https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Corpus/blob/main/README.md)

先抽看少量原始记录，确认对话结构、轨迹边界以及命令与返回的对应关系。抽样只用于确认解析方式，不用于筛选难度。

## 3. 最终格式

文件名：`terminal_trajs.jsonl`。

**每行一个 JSON 对象，代表一条连续轨迹。每步只有 `command` 和 `output`。**

下面是展开后的格式示例，不是已采集的真实数据；实际文件每条轨迹占一行：

```json
{
  "trajectory": [
    {"command": "printf 'hello\n' > note.txt", "output": ""},
    {"command": "cat note.txt", "output": "hello\n"},
    {"command": "printf 'world\n' >> note.txt", "output": ""},
    {"command": "cat note.txt", "output": "hello\nworld\n"}
  ]
}
```

- `command: str`：只保存原始 `keystrokes` 文本。同一动作中的多条命令按原顺序拼接，保留原有换行，不额外插入分隔符；不保存 JSON 动作载荷或 `duration`。
- `output: str`：对应的真实 terminal 返回文本，包括正常输出或报错。
- 空字符串表示该次真实反馈没有文本内容，不能用来代替缺失反馈，也不自动表示命令已经执行完成。
- 不添加思考、计划、任务说明、状态快照、质量标签或来源字段。
- 不要求拆分 stdout/stderr，也不补造退出码。

## 4. 处理步骤

### 提取命令和真实返回

从原始对话中提取实际执行的命令及其 terminal 返回，按原顺序配对。去掉 agent 的思考、计划、最终答复和工具框架提示。

只有明确对应的真实执行反馈才能成为 `output`。脚本识别 `New Terminal Output:` 和 `Current terminal state:` 包装，去除已知的格式警告前缀和任务完成确认后缀。它保留返回内部的空格、换行、prompt、回显和控制序列，不尝试通过正则重建屏幕。

任务完成确认包装内如果还嵌套 `New Terminal Output:`，一并去除这层标题。`commands=[]` 且 `task_complete=true` 属于纯完成控制消息，轨迹在此结束，不把重复画面保存为空命令训练步。包含实际命令的完成动作仍保存；真实空按键等待仍保存。

格式解析失败提示、模型自己描述的执行结果都不算 terminal 返回。采用严格前缀策略：遇到第一个解析失败、缺失返回或配对不明的 call，丢弃该 call 及之后全部内容，只保留此前连续且已确认的步骤。即使后续重试成功，也不再继续提取。若第一个 call 就出问题，则跳过整条记录。

harness 返回 `Previous response had parsing errors:` 时同样立即截断，并计入 `issue:harness_parse_error`。不再跳过重试继续拼接。真实 shell 报错（例如文件不存在）仍是有效返回，不属于数据解析或配对问题。

当前解析器针对已查到的 Terminus-2 对话格式；尚未用完整原始数据验证。若源数据使用其他包装格式，需要补充解析规则，不能把未知格式自动当成 terminal 输出。

原始动作包含多条命令但只有一份整体返回时，将命令文本按原顺序拼接，与整体返回作为一步保存，不拆分、不复制输出。不因命令组、等待轮次、控制按键或命令尚未结束而丢弃记录。只保留文本和当时的真实观察，等待时长等参数省略，因此数据不保留完整时序信息。

`output` 表示该次交互观察到的 terminal 文本，不要求它是整个命令执行结束后的完整 stdout。

### 使用全部 easy 数据

处理所有任务类别的记录，包括复杂命令、长脚本、安装、编译、网络操作及失败轨迹。不设置命令白名单，不裁短轨迹，不截断长输出，不抽取固定数量作为最终训练集。

转换过程只解析现有数据，不执行其中的命令。

### 保持轨迹连续

不打乱步骤，不拼接不同环境的片段，不删除中间修改状态的命令后继续接上后续步骤。

仅在原始记录确实损坏、缺少返回或无法确定配对关系时，保留此前能确认的连续前缀。无法提取任何有效步骤时才跳过该记录，并在终端汇总数量；不补造数据，也不把后半段拼回来。

忽略开头的任务提示和初始 terminal 画面，直接从第一条实际动作及其返回开始。不生成 `{"command": "", "output": "初始画面"}` 这样的记录。真实等待轮次没有输入文本时，`command` 为 `""`，其实际返回仍保留；这与被移除的初始画面不同。

### 清理输出

保留正常输出、真实报错、空输出，以及有意义的空格和换行。

训练目标要求去掉命令回显和 shell prompt；现有转换脚本及已生成 JSONL 尚未执行这一步，原始文件继续保留，计划在模型数据加载层统一清理，详见 [实现方案](terminal-world-model-implementation.md#输出清理待实现)。清理后为空的返回仍保留，训练时监督结束标记。程序实际打印的命令文本、prompt 示例及交互询问不能误删；无法可靠分离的情况必须显式报告。源数据本身截断的观察保留其可见文本和截断提示，不猜测缺失内容；若动作与反馈无法配对，则按上一节处理。

不让 LLM 改写返回，不删除所有报错，不随意排序或替换文本。训练目标使用上述清理后的 terminal 返回文本，不声称它一定是独立 stdout；当前转换产物仍是保留回显和 prompt 的原始观察文本。

### 保存全部有效轨迹

先确认一行是否就是完整轨迹。按实际记录边界转换，不拼接不同运行，也不把累积前缀重复拼成一次更长运行。默认保留源数据中的各条有效记录，不额外按相似度或重复内容删样本。

写入 `terminal_trajs.jsonl`。来源与任务字段可在处理过程中用于分组，不写入最终训练数据。此次只生成训练集；若另有测试集，需要在处理阶段按任务排除重叠。

## 5. 保存前检查

处理程序内部检查以下内容即可，不生成额外报告文件：

- 每条轨迹是非空的有序步骤列表。
- 每步只有 `command` 和 `output` 两个字符串字段；command 中不嵌套 JSON 动作载荷，真实等待轮次允许空字符串。
- 每个返回都有真实来源，配对明确。
- 不含 agent 思考、计划或框架错误提示。
- 没有跳过中间动作后拼接历史，也没有跨环境拼接。
- 没有缺失输出冒充空输出，没有模型编造的返回。
- 文件可以逐行作为 JSON 读取。

最终在终端打印读入记录数、写入轨迹数、交互步数，以及因损坏或配对缺失而无法转换的数量即可。上述检查属于格式和清洗检查，不等同于已重新执行全部轨迹。

## 6. 训练时怎么使用

按顺序读取同一条轨迹：先根据参数记忆和当前 command 预测，再揭示当前 output、计算损失并更新。

当前 output 不能提前放入当前预测的输入。只对真实返回及其结束标记计算本步目标损失；空输出也需要结束标记。

可以打乱不同轨迹的训练顺序，但保持轨迹内部顺序。训练代码负责 tokenization、loss mask 和模型更新，JSONL 只保存干净的 terminal 轨迹。

## 来源说明

源数据为 NVIDIA 的 Nemotron-Terminal-Corpus，官方标注许可证为 CC-BY-4.0。对外分发清理后的数据时保留来源归属，并说明做过格式转换和清理；这些说明不需要重复写进每条轨迹。

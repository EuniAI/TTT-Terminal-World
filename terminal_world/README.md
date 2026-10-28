# Terminal world model

完整训练操作步骤见 [Qwen2.5-Coder-3B 训练指南](../terminal-world-training-guide.md)，包含环境安装、数据清单、首次试跑、恢复、正式训练和评估。

可选 W&B：安装 `python -m pip install -e '.[tracking]'` 并执行 `wandb login`，训练时加 `--wandb-project terminal-world`；续训加 `--wandb-mode online`，自动使用 checkpoint 中保存的 run 身份。`--wandb-mode offline` 分段保存本地日志，`--wandb-new-run` 可分叉日志。不启用时保持本地训练；详细指标、恢复规则和离线同步见训练指南第 12 节。

实现入口是 `python -m terminal_world.train`。当前提供可运行的微型模型、持久 fast weights、元训练、独立评估、Session 和检查点；正式预训练底座支持标准 Llama 和 Qwen2 导入，当前实验配置选用 Qwen2.5-Coder-3B Base。

## 本地验证

从仓库根目录运行：

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python -m terminal_world.dataloader.trajectory terminal_trajs_llm.jsonl --output /tmp/terminal-manifest.json
python -m terminal_world.train --config configs/experiment/tiny.yaml --manifest /tmp/terminal-manifest.json --output /tmp/terminal-tiny.pt --max-windows 2
python -m terminal_world.train --resume /tmp/terminal-tiny.pt --manifest /tmp/terminal-manifest.json --output /tmp/terminal-tiny.pt --max-windows 2
python -m terminal_world.evaluate --checkpoint /tmp/terminal-tiny.pt --exposure-label first-validation
python -m terminal_world.evaluate --checkpoint /tmp/terminal-tiny.pt --frozen --exposure-label same-checkpoint-frozen-comparison
```

Manifest 创建时读取完整文件并记录 SHA256、原始行号、字节位置和划分。重复轨迹按内容归入相同划分；各划分内保留原始顺序。训练和恢复校验文件摘要。不要手工改 manifest 或在训练期间改原始 JSONL。小数据可能没有验证样本，评估会显式报错。

tiny 使用 UTF-8 byte tokenizer，默认 output 预算仍是 2048；它用于算法验证，不能据此评价正式模型预测能力。CPU 上可用 `OMP_NUM_THREADS=1` 减少小矩阵的线程调度成本。长命令不会被截断，超过模型容量会报错。训练读取已记录返回，绝不重新执行 shell。

`--epochs` 是总目标 epoch 数，`--max-windows` 是本次额外运行窗口数。每个成功窗口原子保存 θ、当前 φ、AdamW、游标、协议状态、配置、tokenizer、manifest 和随机状态。epoch 切换只移动游标；不会重置 φ。恢复时通常不再传 `--config`，使用 checkpoint 内实际配置。异常退出后从最近成功提交的 checkpoint 恢复。

## 正式底座与 FlashBoB

`configs/experiment/terminal_ttt.yaml` 已配置 Qwen2.5-Coder-3B Base，固定 revision 为 `09d9bc5d376b0cfa0100a0694ea7de7232525803`。在独立 CUDA 环境安装固定依赖：

```bash
python -m pip install -r requirements-training-cuda.txt
python -m pip install -e . --no-deps
python -m pytest tests/test_flashbob.py -v
python -m terminal_world.benchmark --prefix-batch-size 3 --output-length 128
python -m terminal_world.train --config configs/experiment/terminal_ttt.yaml --manifest /tmp/terminal-manifest.json --output /path/to/run.pt --device cuda
```

`model_id` 支持普通 Llama 或 Qwen2 底座，`revision` 必须是完整 commit SHA。层数、head 数、词表等从底座读取；适应层数、上下文容量和训练超参数由配置指定。导入前保留原网络路径，新增 prime MLP 的 down projection 为零，其他投影正常初始化；在允许训练前比较原词表 logits。正式规模需看启动时报告的 θ/φ 数量：新增支路也计入总参数，不能仅凭底座名称判断总规模。

目前导入器支持 **无 bias 的标准 Llama，以及保留 Q/K/V bias 的 Qwen2**，均要求 SwiGLU、未缩放 RoPE 和 full causal attention。scaled RoPE、启用滑动窗口的模型、其他架构和官方 e2e checkpoint 会显式拒绝。Qwen2 的输出投影和 MLP 无 bias；QKV bias 属于 θ，不进入 fast weights。导入会同时保留参考模型和目标模型做对比，需要额外内存。HF chat template 不进入模型输入。

Qwen 实验配置为 36 层、hidden size 2048、MLP intermediate size 11008、16 个 Query 头 / 2 个 KV 头、共享 embedding。末 9 层（25%）各新增同宽 prime MLP，共 608,698,368 个 fast-weight 参数，整体约 3.70B。架构字段显式列在 YAML 中供启动前校验，导入时仍从底座配置解析；`qkv_bias` 会写入检查点，恢复时无需再次下载底座。旧检查点缺少该字段时默认为 false。

输出预算为 2048，单事件容量为 4096；起步超参数为 inner LR 0.01、outer LR 1e-5、4 事件窗口、prefix batch 1，均不是已调优结果。prefix batch 1 仍保留整个窗口的计算图，不保证 3B 模型在某张卡上装得下。正式底座权重/tokenizer 和 GPU 数值、显存、吞吐仍需实测。

CPU 适配验收（安装 `.[test,pretrained]` 后）：

```bash
python -m pytest tests/test_pretrained.py tests/test_meta_learning.py tests/test_session_checkpoint.py -v
```

测试覆盖非零 QKV bias 的映射、独立 Hugging Face 微型 Qwen2 的 logits 对齐、本地 AutoModel 导入/tokenizer/Session 往返，以及带 bias 和无 bias 两条路径的有限差分、串行/批量元梯度和训练恢复。缺少 Transformers/tokenizers 时相关测试跳过，不代表真实底座已验证。加载真实底座时会用多段终端文本比较分词和原词表 logits；不使用 chat template。CUDA 测试另包含 Qwen 的 16/2 头数、head dim 128 和 QKV bias，但使用缩小的层数及 MLP 宽度。

FlashBoB 固定到 `f2bd4a38a9b65bae455ef4336d7b54f0acd3caca`，安装必须保留 pip VCS commit 元数据；不接受来源不明的同名包。启动检查 CUDA、PyTorch/Triton、dtype 和 head 配置，不静默切回 math。CPU 只能选择 math。显式参数前向避免了在 FlashBoB 外套 `torch.func` 变换。

CUDA 测试覆盖 MHA/GQA、前向、一阶/二阶导数、混合长度右 padding、多次更新的最终 φ 与 θ 元梯度。测试中列明 BF16 容差；真实底座还需在所选尺寸、硬件上扩展验证。benchmark 测完整窗口内外循环、首次运行时间、预热后的耗时和峰值分配显存，不包含 checkpoint I/O；它不是正式 1B 吞吐结论。正确性测试通过后再采信性能比较。

## 在线 Session

```python
from terminal_world.checkpoint import resume_session

session = resume_session('/path/to/run.pt', device='cuda')
if not session.is_open:
    session.control('open')
prediction = session.predict('cat note.txt')
# real_output 由外部一次性执行器提供，不在 Session 内执行命令。
score = session.observe('cat note.txt', real_output)
session.control('close')
```

每次 observe/control 只更新一次 φ，predict 不学习。调用方负责把同一条真实返回交付一次；接口不依靠 command 文本去重，因为重复命令可能产生不同返回。用 `save_session` 保存在线状态，`resume_session` 保留开闭状态；Session checkpoint 不可用于恢复 AdamW 训练。checkpoint 只从可信本地来源加载，不保存 Docker 文件系统。

生成暂用贪心全前缀重算，没有 KV cache；这保持语义正确但生成吞吐较低。离线训练的前段 batch 优化不用于未知未来 command/output。评估从 θ/φ 的副本开始，frozen 模式禁用所有事件更新；若训练 checkpoint 处于 open 状态，在副本上先执行 close 再打开验证轨迹。评估不会写回训练记忆。

当前指标包含 token 加权 NLL、控制事件损失/更新范数、裁剪率、截断分类、结束标签覆盖及窗口内跨边界覆盖。评估加 `--generate` 可统计完整正文/保留前缀匹配率和含结束 token 的准确率，不评价缺失的尾部。覆盖率为零的窗口配置会报告提示，不能据此声称已验证跨轨迹元学习。

本地验证环境：CPU、PyTorch 2.13.0。FlashBoB CUDA 测试和实际预训练 checkpoint 对比需在对应依赖/硬件环境执行，当前不能宣称通过或获得加速。

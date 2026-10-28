# 首次实现的验证记录

日期：2026-10-23。环境：CPU，Python 3.14.6，PyTorch 2.13.0；没有 CUDA 或 transformers。

| 检查 | 结果 |
| --- | --- |
| `python3 -m pytest -q` | 23 通过、3 跳过 |
| 串行与 prefix batch（batch size 1/2/6） | logits、各事件损失、最终 φ、全部 θ 元梯度一致 |
| FP64 中心有限差分 | prefix attention 参数的元梯度通过 |
| 内循环路径 | 后续事件损失对内循环学习率存在非零导数 |
| 窗口、epoch、checkpoint | 恢复与不中断训练的 θ/φ/游标完全一致 |
| 在线、评估 | predict 不学习；非法边界拒绝；评估副本隔离；frozen 禁用所有更新 |
| 数据编码 | 空正文、N−1/N/N+1、原样特殊字符串、首目标 shift、超长 command 拒绝 |
| 非有限值 | 不提交失败窗口游标或 checkpoint |
| `python3 -m unittest discover -s scripts -p 'test_*.py'` | 49 项，其中 44 通过、5 跳过 |
| 真实数据闭环 | 前两条完整轨迹：28 条真实命令 + 4 个边界 = 32 事件，8 窗口 |

真实数据检查从 `terminal_trajs_llm.jsonl` 复制前两行到 `/tmp/terminal-world-smoke/data.jsonl`，不改原文件。使用 `configs/experiment/tiny.yaml`（θ 37,440 参数，φ 6,144 参数），先运行两个窗口、保存恢复后完成一个 epoch。记录到 1 个窗口内 close/open/shell triple。随后在 checkpoint 副本上重复评分和适应，标记 `repeated-smoke-data`；这不是独立验证集结果，不能作为泛化证据。

以下验证**尚未执行**：

- `tests/test_flashbob.py` 的两个 CUDA 用例：无 GPU，跳过；尚无 FlashBoB 加速或显存结论。
- 实际 transformers Llama 对比：缺依赖，跳过；仅通过了本地权重映射、零初始化支路和旧词表 logits 的测试。
- 选定约 1B 预训练底座、真实 GPU 上的完整窗口梯度/性能以及长期训练稳定性。当前正式配置保留必要空项，未擅自选择模型或超参数。

GPU 验收入口与固定依赖见 [README](README.md)。基准测完整窗口内外循环，首次运行单独报告，预热后同步 CUDA；不计 checkpoint I/O。缩小模型通过不能替代正式 head 配置和序列长度的验收。

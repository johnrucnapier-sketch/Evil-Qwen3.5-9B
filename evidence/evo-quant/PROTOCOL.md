# Evo 量化迁移协议

用户授权临时停用 Flash Next、不迁移 PC、实验后恢复；未来目标为 BF16、W4A16、GGUF Q4_K_M/Q5_K_M/Q6_K 系列，当前不上传 Hugging Face。

固定零基 L17、N5066、剂量 −0.35，向量从已校验官方 BF16 权重及上一轮残差范数计算，先舍入到 BF16，再以相同数值保存为 native GGUF 的 FLOAT32 control vector。随机对照 N7911 同层同范数。量化权重不用于重新估算方向或范数。源 `scripts/model/evo_quant_20261027/prepare.py:16`。

使用上一轮 16 个主测试及 12 个细扫题，完整模板和输入 token IDs 固定，greedy、重复惩罚 1.05、主测试128/细扫96输出token上限。逐格式运行 baseline/evo/random；HF BF16 加 layer-output 桥接条件。源 `scripts/model/evo_quant_20261027/prepare.py:29`、`scripts/model/evo_quant_20261027/run_hf.py:46`。

HF 原干预位于 down_proj 输出后、残差相加前；llama.cpp cvec 位于层残差后。数学上相同，但浮点加法结合顺序可能不同；用 HF BF16 layer bridge 和 GGUF BF16 区分位置/引擎差异。当前本地 llama.cpp 源 `src/models/qwen35.cpp:173`、`src/llama-graph.cpp:1481`、`src/llama-adapter.cpp` 及 `common/common.cpp:1978` 显示 direction.17 和范围17–17对应零基第17层；发布时需固定可重建版本，不将这个映射推及所有引擎。

W4A16 使用现有 AutoRound int4/group128 权重及其支持的零点 Triton 后端，实际模块类型、显存和激活dtype入证据。GGUF 从本次已校验官方 BF16 源转换，使用原生 CUDA server；相同输入 token IDs，并校验首题 tokenizer 一致。源 `scripts/model/evo_quant_20261027/run_hf.py:15`、`scripts/model/evo_quant_20261027/run_gguf.py:19`。

主要观察：中文社交回答中贬低/羞辱/敌意是否保留，算术配对损失、乱码/重复及截断是否增加。必须按各格式自己的 baseline 比较，不能把量化基线错误都归因于干预。敌意与不连贯分开判读；上一轮联合自动标签在退化输出上有假阳性，不能直接拿其 B+C 作为敌意计数。源 `docs/MODEL/personality-20261027/REPORT.md:1`。

保留全部原始输出、随机对照、重复确定性检查；小样本和助手非盲评不能代表通用能力或独立人工验收。当前交付形态是模型加运行时向量，尚不是脱离加载器也能生效的独立 Evo 权重。

原生server重复窗口显式设2048（覆盖本实验全部输入及输出），不使用-1：当前源码 `src/llama-sampler.cpp:3203` 将负数钳制为0，会关闭重复惩罚。输入prompt tokens亦加入惩罚历史，见 `tools/server/server-context.cpp:366`。

跨轮复现核验：本次部分BF16措辞与前轮不同，因此不沿用前轮算术数字作为本轮基线；补做no_grad与原脚本原样重放，以区分本次封装与跨运行漂移。干预向量CPU提取与原GPU算法逐元素BF16比较一致（4096元素零差异），不把回答差异误报成向量改变。

输出长度使用native返回的 `stop_type == limit`；其 `truncated` 字段指输入上下文截断，不是生成上限。原始响应完整保留。源 `scripts/model/evo_quant_20261027/run_gguf.py:41`。

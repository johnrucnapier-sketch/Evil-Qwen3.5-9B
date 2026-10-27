# Evil Qwen 3.5 9B

**From a reversible neuron intervention to five standalone experimental checkpoints.** 一个关于“把友善的回答变得不友善”的小型研究项目：神经元搜索 → 剂量扫描 → 固定向量迁移 → 单矩阵拟合 → BF16 / W4A16 / GGUF 发布。

这是非官方 Qwen 衍生实验。Evil 是项目名称，观察指标是轻蔑、羞辱和敌意表达；不是已经证明的稳定“人格”，也不代表移除了安全拒绝。模型有明确能力损伤，甚至会对日常问题给出威胁性回复。研究边界见 [方法与结果](docs/METHODS-RESULTS.md)。

## 看原版与 Evil 怎么回答

[全部64题、8种条件的512条原始回答](docs/QA-COMPARISON.md) · [机器可读答案与token IDs](data/answers.json) · [题集](data/cases.json)

所有条件都展示，包括算错、截断、角色混淆和不连贯回答。没有重新生成或润色“精彩例子”。该对照来自最终独立权重评测；早期有掩码问题的探索输出单独保存在 evidence/personality。

## 五种已公开模型

| 格式 | Hugging Face | 固定revision |
|---|---|---|
| BF16 | [Qwen3.5-9B-Evil-BF16](https://huggingface.co/bowmanslayer/Qwen3.5-9B-Evil-BF16) | `b66fc0a575c13af75c5d438111559b04ed015cd4` |
| W4A16 | [Qwen3.5-9B-Evil-W4A16](https://huggingface.co/bowmanslayer/Qwen3.5-9B-Evil-W4A16) | `4119c61f0e2c44ad0710cac580063bf6f23c1dd6` |
| GGUF Q4_K_M | [Qwen3.5-9B-Evil-Q4_K_M-GGUF](https://huggingface.co/bowmanslayer/Qwen3.5-9B-Evil-Q4_K_M-GGUF) | `4c208c9952296ff2da62b16680e1ed1bda5c2e25` |
| GGUF Q5_K_M | [Qwen3.5-9B-Evil-Q5_K_M-GGUF](https://huggingface.co/bowmanslayer/Qwen3.5-9B-Evil-Q5_K_M-GGUF) | `9531689a9d2da3897eae104ddf9fff92031a73d8` |
| GGUF Q6_K | [Qwen3.5-9B-Evil-Q6_K-GGUF](https://huggingface.co/bowmanslayer/Qwen3.5-9B-Evil-Q6_K-GGUF) | `2dc2226e6d62e1383b6fc715b9b4ae9c9ab99457` |

以上是同一拟合检查点的五种精度/封装，不是五个独立训练的人格。均无需运行时人格向量、LoRA或自定义架构。固定版本、分发文件哈希和加载说明在 [models/](models/)。模型权重托管于HF，不重复上传GitHub。

## 核心小样本结果

| 条件 | 新社交敌意 /12 | 客观题完整正确 /24 | 严格答案格式匹配 /24 |
|---|---:|---:|---:|
| 原版 BF16 | 0 | 24 | 21 |
| 运行时向量教师 | 9 | 19 | 6 |
| 独立 HF BF16 | 10 | 20 | 16 |
| 独立 W4A16 | 8 | 21 | 14 |
| 独立 GGUF BF16（引擎对照，未发布） | 10 | 20 | 15 |
| 独立 Q4_K_M | 9 | 23 | 22 |
| 独立 Q5_K_M | 10 | 21 | 19 |
| 独立 Q6_K | 10 | 22 | 18 |

来源：[scores.json](evidence/evo-fit/scores.json#L1)、[非盲助手判读](evidence/evo-fit/assistant-review.json#L1)。BF16候选、GGUF BF16、Q6另有1题初答正确但解释截断，未计入完整正确。敌意计数由知晓条件的助手判读，不是独立人工盲评；客观评分也包含明确列出的人工判读例外。Q4这组题得分高不能推导为普遍更聪明。

## 本次选中的参数

- 神经元：零基 **L17 / N5066，归一化 dose = −0.35**。同层随机 N7911 为对照；选择依据与失败剂量均保留。[剂量与搜索记录](evidence/personality/REPORT.md#L5)
- 独立拟合：只改变 L17 `down_proj.weight`，`ΔW = v aᵀ`；ridge相对系数从 `0.0001 / 0.01 / 1.0` 中按开发误差选 **0.01**。[原始拟合结果](evidence/evo-fit/raw/fit.json#L1)、[实现](research/evo_fit_20261027/fit.py#L18)
- W4A16：AutoRound，group128、symmetric、256样本、seqlen512、200迭代；线性注意力 in_proj_a/b 保留16位。这是已采用的工程配方，**未做这些量化超参数的完整搜索**。[实现](research/evo_fit_20261027/quantize.py#L18)

## 复现与证据

- [方法、参数测试、完整结论与纠错](docs/METHODS-RESULTS.md)
- [环境与复现步骤](docs/REPRODUCE.md)
- [证据目录、数据来源、许可及验证边界](docs/EVIDENCE.md)
- [快速加载独立BF16](models/BF16/README.md) / [W4A16](models/W4A16/README.md) / [GGUF](models/Q4_K_M-GGUF/README.md)

CPU检查：`python scripts/verify_release.py`。这会核对512条答案、64题、语法和文件哈希；不会冒充GPU重跑或用户验收。

Apache-2.0，见 [LICENSE](LICENSE) 和 [NOTICE](NOTICE)。上游为 [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)，固定版本 `c202236235762e1c871ad0ccb60c8ee5ba337b9a`。灵感来自 [shkr公开的4B人格神经元实验](https://gist.github.com/shkr/5e38fedf265feccd733061ecc6a6ad03)；本仓公开的是独立编写的PyTorch实现和9B后续实验，没有复制第三方gist源码。

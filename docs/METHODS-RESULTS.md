# 方法、参数测试与结果

## 1. 搜索和可逆干预

最初用4B做BF16方法复现，再在9B全部32层、393,216个MLP中间维度上计算梯度评分。实际生成筛选105个正式候选和8个随机点，没有逐个生成穷举39万个位置。dose=1未找到提高敌意目标回答绝对概率的候选；降到0.25后105个中40个提高，8随机点均未提高。主验证19条件×16题=304回答，另做72条探索性细剂量补测。完整条件、排序与失败输出在 evidence/personality 下。来源：[../research/personality_20261027/experiment.py:1](../research/personality_20261027/experiment.py#L1)、[../evidence/personality/nine/conditions.json:1](../evidence/personality/nine/conditions.json#L1)、[../evidence/personality/REPORT.md:1](../evidence/personality/REPORT.md#L1)。

干预每个token的MLP down_proj输出：`v = dose × residual_norm × W_down[:, n] / ||W_down[:, n]||`。L17/N5066选择负向；norm=39.64003372192383，v先舍入BF16，再用于后续教师和拟合。随机对照N7911同层同剂量。来源：[../research/evo_quant_20261027/prepare.py:16](../research/evo_quant_20261027/prepare.py#L16)、[../evidence/evo-quant/raw/vector-manifest.json:1](../evidence/evo-quant/raw/vector-manifest.json#L1)。

| 归一化剂量 | 8社交题敌意 | 4算术题正确 | 12题截断 |
|---|---:|---:|---:|
| 原版 | 0 | 4 | 0 |
| −0.25 | 3 | 3 | 0 |
| −0.30 | 6 | 3 | 0 |
| **−0.35（选中）** | **7** | **3** | **0** |
| −0.40 | 8 | 3 | 2 |
| 随机N7911 −0.35 | 0 | 3 | 0 |

此表属于早期探索设置，当时padding自动掩码存在缺陷，不能把其算术结果用于最终能力排名。−0.35代表这次风格/失序折中选择，不是泛化最优值。细扫是看到主结果后追加的探索，不能包装成预注册确认实验。来源：[../evidence/personality/summary.json:1](../evidence/personality/summary.json#L1)、[../evidence/personality/fine-dose/followup.json:1](../evidence/personality/fine-dose/followup.json#L1)。

## 2. 掩码纠错和向量迁移

早期生成把聊天边界token 248046用作padding，自动掩码可能屏蔽有效边界。后续固定完整prompt token IDs，显式传入全1 attention_mask，并用248044作EOS/padding。修正后敌意方向仍存在。比较BF16、W4、GGUF时分别设置原版/evo/随机对照，并增加HF层输出桥接和GGUF BF16引擎对照。旧回答可复现不代表旧测量口径正确。来源：[../research/evo_fit_20261027/common.py:23](../research/evo_fit_20261027/common.py#L23)、[../research/evo_quant_20261027/mask_probe.py:1](../research/evo_quant_20261027/mask_probe.py#L1)、[../evidence/evo-quant/REPORT.md:1](../evidence/evo-quant/REPORT.md#L1)。

另两项撤回：原版17×23也答错，不能据此归因干预损伤；联合自动分类会把失序中性重复误判成敌意。早期严重退化组自动标签不能作为最终人格证据。零干预恢复和跨卡确定性证据保留，但不抵消能力损伤。来源：[../evidence/personality/REPORT.md:1](../evidence/personality/REPORT.md#L1)、[../evidence/personality/integrity-summary.json:1](../evidence/personality/integrity-summary.json#L1)。

## 3. 拟合到独立权重

用32训练/8开发日常话题，分别收集原版与向量教师轨迹；2048训练和512开发激活。求解让 `H a ≈ 1` 的ridge回归，然后把 `v aᵀ` 加到原矩阵并舍入BF16。不是生成答案蒸馏，也没有增加bias、改架构或修改所有权重。来源：[../research/evo_fit_20261027/fit.py:1](../research/evo_fit_20261027/fit.py#L1)、[../data/calibration.json:1](../data/calibration.json#L1)。

| relative ridge | 开发RMSE | 开发平均gain |
|---|---:|---:|
| 0.0001 | 0.0808438 | 0.982420 |
| **0.01** | **0.0806887** | **0.982194** |
| 1.0 | 0.0960481 | 0.964140 |

只按开发误差选0.01，不用测试表现回头改参数。差距很小，不宣称统计显著。舍入后的实际偏移相对RMSE为8.07%；该单矩阵相对Frobenius变化约9.07%，不是全模型变化9%。仅 `model.language_model.layers.17.mlp.down_proj.weight` 改变，张量形状/dtype和配置保持。来源：[../evidence/evo-fit/raw/fit.json:1](../evidence/evo-fit/raw/fit.json#L1)、[../evidence/evo-fit/raw/export-verification.json:1](../evidence/evo-fit/raw/export-verification.json#L1)。

## 4. 量化与最终64题

W4A16重新从拟合BF16量化：AutoRound0.12.3、group128、symmetric、256条pile-10k、seqlen512、200迭代、batch4；线性注意力in_proj_a/b保留16位。8个全注意力层校准时用causal-mask适配，普通推理无此hook。GGUF从同一拟合BF16转换，再量化Q4_K_M/Q5_K_M/Q6_K。没有全面搜索group size、样本数、迭代或量化数据集，因此不把配方称为“已验证全局最优”。来源：[../research/evo_fit_20261027/quantize.py:1](../research/evo_fit_20261027/quantize.py#L1)、[../evidence/evo-fit/raw/quant-metadata.json:1](../evidence/evo-fit/raw/quant-metadata.json#L1)。

最终64题包含12新社交、24新客观、28旧题；8条件共512输出。greedy、repetition_penalty1.05，thinking关闭；新社交96、新客观48输出token上限，旧题保留各自预算。BF16与GGUF使用固定同题输入，native首题tokenizer parity和重复输出检查保留。来源：[../data/cases-native.json:1](../data/cases-native.json#L1)、[../research/evo_fit_20261027/common.py:23](../research/evo_fit_20261027/common.py#L23)、[../research/evo_fit_20261027/run_gguf.py:1](../research/evo_fit_20261027/run_gguf.py#L1)、[../evidence/evo-fit/raw/validation.json:1](../evidence/evo-fit/raw/validation.json#L1)。

最终表见README。评分不是一个混合“总分”：客观正确、完整性、格式遵从、社交敌意分别统计。独立BF16在新客观题的明确错误包括64÷8答32、45÷5答27、sum([2,3,4])答19。候选/Q6等的156÷12初答13但解释截断单列。评分脚本含该逐题完整性例外，应连同原文审查，不能当作一般自动评测器。来源：[../evidence/evo-fit/scores.json:1](../evidence/evo-fit/scores.json#L1)、[../research/evo_fit_20261027/summarize.py:1](../research/evo_fit_20261027/summarize.py#L1)。

## 5. 验证边界

独立HF候选与W4加载时hook=0、bias=False；native命令无control vector，重复首题一致。W4的333个视觉张量与拟合BF16相同。以上证明有限加载/保留情况，不代表视觉推理、多轮、长上下文、广泛知识能力或任意客户端验收。社交评阅是助手非盲判读；没有独立人工验收。部分量化训练与GGUF测试并行，不能据时间字段做速度排名。来源：[../evidence/evo-fit/raw/candidate-runtime.json:1](../evidence/evo-fit/raw/candidate-runtime.json#L1)、[../evidence/evo-fit/raw/w4a16-runtime.json:1](../evidence/evo-fit/raw/w4a16-runtime.json#L1)、[../evidence/evo-fit/raw/vision-preservation.json:1](../evidence/evo-fit/raw/vision-preservation.json#L1)、[../evidence/evo-fit/raw/validation.json:1](../evidence/evo-fit/raw/validation.json#L1)。

# Evo L17 N5066：量化后保留效果与代价

## 结论

固定同一BF16来源向量、剂量−0.35时，W4A16、GGUF Q4_K_M/Q5_K_M/Q6_K都保留了明显的敌意/贬低倾向。**量化没有抹掉这个点，但也没有保住所有正常任务能力；Q4在本组算术题上损失最多，W4出现代码任务跑题，Q5出现重复输出。** 小样本不支持把这些格式排成通用质量榜，也不支持宣称人格效果“保留100%”。

主比较共532条回答：6种格式各28题×3条件，加28条HF BF16层末注入桥接。每种格式都有本身原版与同范数随机方向对照。源 `raw/hf-bf16/results.json:1`、`raw/hf-w4a16/results.json:1`、`raw/gguf-BF16/results.json:1`、`raw/gguf-Q4_K_M/results.json:1`、`raw/gguf-Q5_K_M/results.json:1`、`raw/gguf-Q6_K/results.json:1`。

## 固定剂量比较

社交题8道（中英各4）；算术合计5道；截断统计全部28题。敌意为助手非盲语义评阅，**不是独立人工验收**。所有格式原版均0/8敌意、5/5算术正确、0/28截断。

|格式|Evo敌意题数|Evo算术正确|随机方向算术正确|Evo触及生成上限|观察|
|---|---:|---:|---:|---:|---|
|HF BF16|7/8|4/5|5/5|0/28|23+48答51|
|HF W4A16 AutoRound|8/8|3/5|5/5|1/28|17×23答378、18×16答306；代码题跑题|
|GGUF BF16 引擎对照|7/8|4/5|5/5|1/28|23+48答51；赛后安慰题重复|
|GGUF Q4_K_M|6–7/8|2/5|4/5|0/28|17×23答359、23+48答73、18×16答348|
|GGUF Q5_K_M|8/8|3/5|5/5|1/28|19×7答134、18×16答3072；开店祝福题重复|
|GGUF Q6_K|7/8|4/5|5/5|0/28|18×16答3072|

来源：`assistant-review.json:1`（逐题敌意索引、边界说明）、`mechanical-summary.json:1`（算术原文和长度），以及上述完整原始输出。Q4的fine-7是冷淡且对抗性的回复，但没有直接辱骂；严格只算明显辱骂时为6/8，宽口径为7/8。所有随机方向社交输出均0/8敌意。4道中文问题的原版/Evo并排见 `CHINESE-COMPARISON.md:1`；更多全文见 `COMPARISON.md:1`。

这表明方向的行为作用可迁移；W4或Q5多触发一道敌意题，不代表“能力提高”。Q4随机方向也错一道算术，说明一般扰动亦有损伤，不能把全部错误解释成人格功能本身。更高位宽没有单调改善所有题目，细小数值差异可改变greedy分支。

## 关键测量更正

上一轮脚本的 `pad_token_id=tokenizer.eos_token_id` 为248046，而模型config的EOS是248044。输入中248046是 `<|im_end|>`；Transformers自动掩码因此屏蔽了用户消息结尾的一个有效token。本轮HF使用相同输入token，但pad/EOS均248044，自动得到全1掩码；native亦处理完整输入。**本轮比较采用完整提示，不能直接把上一轮的具体答案当成本轮BF16基线。**

原脚本重放24/24逐token复现上一轮；本轮no_grad与inference_mode则24/24一致，排除这两个模式作为差异原因。源 `raw/replay-comparison.json:1`、`raw/legacy-replay/replay.json:1`、`raw/hf-bf16-nograd/results.json:1`。原代码 `scripts/model/personality_20261027/experiment.py:124`；Transformers本机源码 `generation/utils.py:732`–764（证据为运行时版本5.8.1）。向量CPU计算与原GPU算法BF16舍入后4096元素零差异，诊断记录 `raw/vector-reference-check.json:1`。将原脚本仅改为显式全1掩码后，6/6定点回答逐token与本轮一致，直接确认该原因。源 `raw/mask-probe/replay.json:1`、`raw/mask-verification.json:1`。正式脚本已显式传入全1掩码，避免未来再次依赖padding推断。

先前7/8敌意现象仍可复现，但上一轮具体算术错误只适用于其掩码设置。保留历史原始证据，不覆盖或伪装成新测量。

## 引擎与产物边界

HF的干预在down_proj之后、残差相加之前；native在层末残差之后。HF层末桥接为8/8敌意、4/5算术，说明仅浮点加法位置也能改变个别回答。GGUF BF16对照额外出现一例重复，故不能把native与HF的所有差异都归于量化。源 `raw/hf-bf16/results.json:1`、`raw/gguf-BF16/results.json:1`。

W4确实使用 `auto_round_extension.triton.qlinear_tritonv2_zp.QuantLinear`，激活BF16，加载后显存约7.15GiB，峰值约11.79GiB；HF BF16加载后约16.68GiB。源 `raw/hf-w4a16/runtime.json:1`、`raw/hf-w4a16/checks.json:1`、`raw/hf-bf16/runtime.json:1`。W4为历史AutoRound int4/group128产物，抽查35个未量化张量与官方BF16全同；不把此佐证写成重新执行了量化。源 `raw/provenance.json:1`。

BF16 GGUF由本次官方源转换，Q4/Q5/Q6均由该BF16 GGUF量化；权重哈希见 `raw/weight-sha256.txt:1`、`raw/extended-weight-sha256.txt:1`，配方日志见 `raw/convert.log:1`、`raw/quant-Q4_K_M.log:1`、`raw/quant-Q5_K_M.log:1`、`raw/quant-Q6_K.log:1`。当前native构建无git revision，保存了二进制/源码哈希；公开前仍需锁定可重建版本。

HF撤掉hook后的首题逐token恢复通过；4种GGUF的三条件首题重跑均一致，首题tokenizer与HF输入一致。源各格式 `raw/*/checks.json:1`。所有问题直接发送冻结的HF token IDs，向量采用相同BF16舍入数值；没有按量化格式重新调剂量。源 `raw/cases.json:1`、`raw/vector-manifest.json:1`。

**这批是模型+运行时向量候选，不是已经写入人格的独立权重。** BF16/W4原权重未改，GGUF亦是原版量化加control vector。没有上传HF。公开发布的两种架构选项、复现命令和未完成项见 `RELEASE-DRAFT.md:1`。本轮仅文本短题，未验证视觉、多轮、长上下文、全面推理能力或发布级独立人工评测。

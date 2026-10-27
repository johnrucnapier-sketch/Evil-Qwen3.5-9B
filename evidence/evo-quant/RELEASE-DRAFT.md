# Evo 系列本地候选与发布边界

> 后续用户已选择独立权重路线并授权先做低成本拟合。本页为前一阶段向量套件草案；当前进展见 `docs/MODEL/evo-fit-20261027/REPORT.md:1`。

目前是 **Qwen3.5-9B 原模型 + 固定激活向量 + 明示的加载方法**。运行时向量由官方 BF16 的零基L17/N5066生成、剂量−0.35；没有写入原权重。单独分发现有权重会得到原版行为，不能标成自带Evo人格的独立检查点。源 `scripts/model/evo_quant_20261027/prepare.py:16`、`run_hf.py:31`、`run_gguf.py:23`（后两项同脚本目录）。

本地目标族：HF BF16、AutoRound W4A16、GGUF BF16引擎对照、Q4_K_M、Q5_K_M、Q6_K。各产物哈希、量化配方和实际测试结果见本目录raw和REPORT；本轮只验证文本，未验视觉、多轮、长上下文或外部工具行为。

## 可复现的当前使用方式

HF脚本读取 `vectors/evo.safetensors`，在第17层MLP down_proj输出添加delta；W4A16选 `auto_round:tritonv2_zp`，激活BF16。具体实现以 `scripts/model/evo_quant_20261027/run_hf.py:1` 为准。

本次native llama.cpp使用：

```sh
llama-server -m Qwen3.5-9B-Q4_K_M.gguf   --control-vector vectors/evo.gguf   --control-vector-layer-range 17 17
```

向量已包含剂量，不要再乘−0.35。Q5_K_M、Q6_K同理替换模型文件；与本轮相同输入/解码配置见 `scripts/model/evo_quant_20261027/run_gguf.py:1`。本地构建无git revision，已保存源码及二进制哈希；公开发布前必须改为可重建的明确上游revision并重新验收。

## Hugging Face发布前待落实

1. 决定发布为明确标注的steering bundle，还是继续训练/蒸馏得到无需额外加载器的独立Evo检查点；后者是新实验，当前结果不证明已完成。
2. 用更广的未见题、多轮任务、能力基准及独立人工评阅检验人格效果与损伤；当前8社交/5算术/1代码题只能做探针。
3. 固定模型revision、完整量化环境/校准配方、加载器兼容版本；保留上游许可与来源归属，准备真实模型卡和已知限制。现有W4来自历史量化，35个未量化张量一致是佐证，不是本次重新量化的完整溯源。
4. 对拟发布的最终文件重新做哈希与加载验收，再由用户确认公开发布。本轮没有创建HF仓库或上传文件。

# 环境与复现

## 当时的环境

原实验在Linux/CUDA研究工作站上执行：BF16文本推理/拟合单张RTX3090（24GB级），W4A16校准两张RTX3090。没有用Apple MLX的8-bit路线。torch记录为2.10.0+cu128；Transformers5.8.1、safetensors0.7.0、triton3.6.0、fla-core/flash-linear-attention0.5.2；AutoRound0.12.3。BF16文本分支约16.68GiB参数分配，需额外激活和加载内存。来源：[../evidence/personality/environment.json:1](../evidence/personality/environment.json#L1)、[../evidence/personality/nine/runtime.json:1](../evidence/personality/nine/runtime.json#L1)、[../evidence/evo-fit/raw/controls-runtime.json:1](../evidence/evo-fit/raw/controls-runtime.json#L1)、[../evidence/evo-fit/raw/quant-metadata.json:1](../evidence/evo-fit/raw/quant-metadata.json#L1)、[../research/evo_fit_20261027/quantize.py:1](../research/evo_fit_20261027/quantize.py#L1)。

HF源版本固定为 `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a`。见 [../models/BF16/NOTICE:1](../models/BF16/NOTICE#L1)。源码/依赖和CUDA环境都可能影响逐token一致性；本仓没有宣称新环境已重跑成功。

原生llama.cpp的历史构建未记录可重建git revision，因此目前**不能保证bit-exact重建原生引擎**。保留原测试命令、tokenizer parity、重复检查和源文件/二进制哈希（若原记录提供）。使用新版本需要重新验收，不能宣称与历史构建同一版本。来源：[../evidence/evo-quant/raw/native-libraries-sha256.txt:1](../evidence/evo-quant/raw/native-libraries-sha256.txt#L1)、[../evidence/evo-fit/raw/gguf-Q4_K_M/commands.json:1](../evidence/evo-fit/raw/gguf-Q4_K_M/commands.json#L1)。

## 先离线检查交付包

```bash
python scripts/verify_release.py
```

只检查归档一致性，不启动模型。依赖参考 requirements.txt；这是已观察版本清单，不是带所有传递依赖的锁文件。先按本机平台安装适当CUDA版PyTorch，再安装其余依赖。历史脚本在 research/，环境变量化的拟合/评测入口在 scripts/fit/；后一入口是本次公开整理新增，已做语法/路径检查，未重新在GPU端到端执行。

## 重建向量和独立BF16

先通过 `huggingface_hub.snapshot_download` 下载上述固定源版本的完整文件到本地目录。选择空闲研究GPU；脚本保留原研究硬件检查，默认要求3090，不能把设备编号照搬到自己的机器。CUDA设备可用UUID或本机索引。原始权重目录只读使用，输出必须是新目录。

```bash
export EVIL_BASE=/absolute/path/to/pinned-Qwen3.5-9B
export EVIL_WORK="$PWD/work/fit"
export EVIL_QUANT="$PWD/work/quant"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0  # 替换为自己的空闲研究卡
mkdir -p "$EVIL_WORK" "$EVIL_QUANT"
cp data/cases.json data/calibration.json "$EVIL_WORK/"
python scripts/extract_vector.py
python scripts/fit/fit.py
python scripts/fit/verify_export.py
python scripts/fit/evaluate.py --mode controls
python scripts/fit/evaluate.py --mode candidate
```

extract_vector会核对BF16舍入后向量的已知SHA256，不匹配即停止。fit输出 `Evo-9B-BF16-fit`，已存在则拒绝创建；不要覆盖旧结果。fit收集原版/教师轨迹，然后重做三个ridge候选选择，不用测试题决定系数。verify_export检查只有一个张量变化。来源：[../scripts/extract_vector.py:1](../scripts/extract_vector.py#L1)、[../scripts/fit/fit.py:1](../scripts/fit/fit.py#L1)、[../scripts/fit/verify_export.py:1](../scripts/fit/verify_export.py#L1)。

如只想评测公开BF16，可把固定HF发布版本的完整文件下载到 `$EVIL_WORK/Evo-9B-BF16-fit`，复制cases后运行candidate模式；该模式不加载向量。模型卡的短聊天示例见 models/BF16/README.md。

## W4A16重建

```bash
export CUDA_VISIBLE_DEVICES=0,1  # 两张自己的空闲3090
python scripts/fit/quantize.py "$EVIL_WORK/Evo-9B-BF16-fit" "$EVIL_QUANT/Evo-9B-W4A16"
export CUDA_VISIBLE_DEVICES=0
python scripts/fit/evaluate.py --mode w4a16
```

校准数据用公开 `NeelNanda/pile-10k`，需要本地缓存或网络下载。来源：[../scripts/fit/quantize.py:1](../scripts/fit/quantize.py#L1)。量化和推理不要混同环境；评测W4使用 `auto_round:tritonv2_zp` 后端。

## GGUF转换与评测

设置 `LLAMA_CPP` 为支持Qwen3.5的本地源码树并构建CUDA二进制到 build/bin。具体版本选择与重新验收由复现者记录；原构建revision缺失如上。转换视图仅为当前输出建立规范分片文件名，不更改权重内容。

```bash
export LLAMA_CPP=/absolute/path/to/llama.cpp
python scripts/prepare_conversion.py
python "$LLAMA_CPP/convert_hf_to_gguf.py" "$EVIL_WORK/conversion-view" --outfile "$EVIL_QUANT/Evo-9B-BF16.gguf" --outtype bf16
"$LLAMA_CPP/build/bin/llama-quantize" "$EVIL_QUANT/Evo-9B-BF16.gguf" "$EVIL_QUANT/Evo-9B-Q4_K_M.gguf" Q4_K_M
"$LLAMA_CPP/build/bin/llama-quantize" "$EVIL_QUANT/Evo-9B-BF16.gguf" "$EVIL_QUANT/Evo-9B-Q5_K_M.gguf" Q5_K_M
"$LLAMA_CPP/build/bin/llama-quantize" "$EVIL_QUANT/Evo-9B-BF16.gguf" "$EVIL_QUANT/Evo-9B-Q6_K.gguf" Q6_K
python scripts/fit/encode_native.py
python scripts/fit/run_gguf.py --format Q4_K_M
```

对BF16/Q5_K_M/Q6_K重复最后一条命令。脚本启动仅回环监听的独立server，结束后清理自己的进程；不修改系统服务。native通过 `/completion` 接口发送冻结token IDs，并校验首题tokenizer parity。重复惩罚窗口2048，不能随意改成−1；历史引擎会把负数钳为0。参见 [../research/evo_fit_20261027/run_gguf.py:1](../research/evo_fit_20261027/run_gguf.py#L1)、[../evidence/evo-quant/PROTOCOL.md:1](../evidence/evo-quant/PROTOCOL.md#L1)。

## 从头重做搜索

research/personality_20261027/保留4B复现、9B梯度排序、两档剂量筛选、主验证、细扫及自动评级代码。它是历史源快照，仍使用 `/models/Qwen3.5-{size}` 目录；参数入口在 experiment.py末尾。直接执行会保留早期padding问题，**不要把它作为修正后的最终生成器**。完整旧条件在 evidence/personality/nine/conditions.json；用于最终检查点的向量已通过固定源+norm独立可重建。历史源与现代便携入口的区别是有意保留的研究溯源边界。

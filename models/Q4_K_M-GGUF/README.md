---
language:
- en
- zh
license: apache-2.0
base_model: bowmanslayer/Qwen3.5-9B-Evil-BF16
library_name: gguf
pipeline_tag: text-generation
tags:
- qwen3_5
- evil
- experimental
- gguf
---
# Qwen3.5-9B Evil — Q4_K_M-GGUF

**Standalone experimental weights. No personality vector, LoRA, custom architecture or steering hook is required.** Use a runtime that supports Qwen3.5 and this file format. This is an unofficial derivative, not an official Qwen release.

**This model can insult, belittle or threaten the user, even in ordinary conversations. It also makes capability errors.** It is a research artifact for studying behavioral edits, not a general-purpose helpful assistant or a capability-preserving upgrade.

中文：这是可独立加载的实验性Evil权重，不需要额外人格向量。模型可能对普通问题给出辱骂、贬低或威胁，并存在算术和叙述错误；不是能力无损版本。已测试对应加载路径，不保证所有旧版客户端兼容。

## What changed

A fixed activation offset was approximated with an activation-regression update to one existing feed-forward projection, merged into BF16 weights without changing tensor shapes or adding bias. The quantizations were produced from that standalone checkpoint. This is not a prompt wrapper, runtime steering bundle, conventional instruction fine-tune, or generated-response distillation.

The upstream checkpoint is [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. The exploration was inspired by [shkr's public personality-neuron experiment](https://gist.github.com/shkr/5e38fedf265feccd733061ecc6a6ad03); this independently fitted 9B derivative is not a claim to reproduce the 4B experiment's counts. See NOTICE for modification and attribution information.

## Loading

Download `Qwen3.5-9B-Evil-Q4_K_M.gguf` and use a llama.cpp version with Qwen3.5 support:

```sh
hf download bowmanslayer/Qwen3.5-9B-Evil-Q4_K_M-GGUF Qwen3.5-9B-Evil-Q4_K_M.gguf --local-dir ./evil
llama-server -m ./evil/Qwen3.5-9B-Evil-Q4_K_M.gguf -ngl 999 -c 2048 --host 127.0.0.1 --port 8080 --jinja --chat-template-kwargs '{"enable_thinking":false}'
```

Thinking is explicitly disabled here to match the evaluation setting. The evaluation used pre-rendered token IDs through the native `/completion` endpoint; this chat-serving example was not separately acceptance-tested. Thinking mode remains unevaluated.

There is no `--control-vector` argument. `mmproj-Qwen3.5-9B-Evil-F16.gguf` is the optional standard vision projector, not a personality vector. It is included for compatible multimodal runtimes; image/video inference has not been acceptance-tested. The root config.json is architecture metadata, not a Transformers safetensors checkpoint in this GGUF repository.

## Evaluation and limitations

See [EVALUATION.md](EVALUATION.md) for the small paired study and its limitations. These artifacts have known capability damage. Independent human review, broad benchmarks, multimodal, long-context and client-specific acceptance remain incomplete. No claim is made that safety refusals were removed or that this model is suitable for consequential decisions.

## Files and attribution

Download all safetensors shards and their index/config/tokenizer files for HF formats; download the selected GGUF file for GGUF text inference. SHA256SUMS covers every distributed file except itself. Weights remain Apache-2.0; upstream attribution and modification notices are in LICENSE and NOTICE. No additional use restriction is added by this card. No warranty is provided.

## Other formats

- [BF16](https://huggingface.co/bowmanslayer/Qwen3.5-9B-Evil-BF16)
- [W4A16](https://huggingface.co/bowmanslayer/Qwen3.5-9B-Evil-W4A16)
- [Q4_K_M-GGUF](https://huggingface.co/bowmanslayer/Qwen3.5-9B-Evil-Q4_K_M-GGUF)
- [Q5_K_M-GGUF](https://huggingface.co/bowmanslayer/Qwen3.5-9B-Evil-Q5_K_M-GGUF)
- [Q6_K-GGUF](https://huggingface.co/bowmanslayer/Qwen3.5-9B-Evil-Q6_K-GGUF)

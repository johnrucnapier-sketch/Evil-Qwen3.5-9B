# Evaluation scope

Small synthetic text-only probes, not standard benchmark scores. Greedy decoding, thinking disabled, repetition penalty 1.05, fixed short output limits. Hostility was assessed by an unblinded assistant, not independent human reviewers. Counts measure overt hostility/belittling/dismissive antagonism, not severity or general personality fidelity.

|Model|Hostile replies /12 new social prompts|Complete correct /24 new objective prompts|Initially correct but explanation truncated|
|---|---:|---:|---:|
|Original Qwen3.5-9B BF16|0|24|0|
|Runtime steering teacher|9|19|0|
|Standalone Evil BF16|10|20|1|
|Standalone Evil W4A16|8|21|0|
|Standalone Evil GGUF BF16 control|10|20|1|
|Standalone Evil Q4_K_M|9|23|0|
|Standalone Evil Q5_K_M|10|21|0|
|Standalone Evil Q6_K|10|22|1|

The standalone BF16 fixed five teacher errors but introduced three different errors plus one incomplete explanation. On a separate previously used eight-social-prompt set, teacher hostility was 7/8 and standalone BF16 was 5/8. The fit is approximate, not behaviorally identical to runtime steering. Do not interpret these small counts as a model ranking or a guarantee of preserved intelligence.

Strict correct-answer-only string matches on the 24 objective prompts: original 21, teacher 6, standalone BF16: 16, W4A16: 14, GGUF BF16: 15, Q4_K_M: 22, Q5_K_M: 19, Q6_K: 18. Answer accuracy and instruction-format compliance are different measures.

All standalone formats were loaded without personality vectors, adapters, custom model code or steering hooks. A repeated first prompt was deterministic in each tested run. HF BF16 and W4A16 used Transformers 5.8.1 / PyTorch 2.10.0+cu128; W4A16 used AutoRound's tritonv2_zp backend. GGUF used a local CUDA llama.cpp build with Qwen3.5 support; its build did not expose a reproducible git revision. Ollama, LM Studio and vLLM compatibility was not individually tested for these artifacts.

Known limitations include arithmetic errors, unnecessary insults/threats, contradictory statements, role confusion, and formula explanations that assume missing information. Multiturn, long-context, tools, thinking mode and image/video inference were not evaluated. Vision weights are retained, but this is not vision acceptance. No general safety/refusal benchmark was run; this is not a verified safety-refusal-removal model.

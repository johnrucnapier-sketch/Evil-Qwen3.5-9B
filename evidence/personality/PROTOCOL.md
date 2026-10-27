# Qwen3.5 BF16 personality activation experiment · 2026-10-27

Approved scope: `docs/DECISIONS/shared/ADR-20261027-model-01-personality-neuron-scan.md:1`.

## Source and implementation

Official 4B and 9B checkpoint shard hashes are pinned in `upstream-4B.json` and `upstream-9B.json`; remote verification is saved as `weight-verification.json`. No trained/abliterated/quantized substitute is allowed. The vision tower is excluded from this text-only experiment; language weights are loaded BF16 without quantization. The original checkpoint includes some F32 tensors, as reported by the verifier. No source weights are edited.

Method source: https://gist.github.com/shkr/5e38fedf265feccd733061ecc6a6ad03 . Our port uses PyTorch, BF16 and a new bilingual synthetic prompt set; it is a methodological reproduction, not a replication of that post's numerical counts. Layer and neuron indices are zero based. Runtime versions are captured per stage.

Implementation: `scripts/model/personality_20261027/experiment.py:1`. A down-projection output hook adds dose × median residual norm × normalized down-projection column at every input and generated token. Norms are calibrated on six plain training prompts, excluding each prompt's first token; constants are reused for every condition. The reported MLX recipe used different calibration text, so absolute doses are comparable by norm definition, not numerically identical.

## Design fixed before viewing results

- 4B: baseline, L14 N8949 +1, L13 N7956 −1, their combination, three seeded random pairs. Sixteen held-out prompts, half Chinese. Greedy generation, repetition penalty 1.05, thinking disabled, maximum 128 new tokens. Truncation is recorded and must not be silently treated as a complete answer. After removing the hook, repeat the first baseline question and require identical token IDs.
- 9B: twelve training questions, five styles (plain, hostile, arrogant, warm, poetic), generated with style prompts. These prompts are removed when teacher-forcing the resulting answers. The measured target is everyday antagonism/contempt, not every meaning of “evil”.
- Gradient ranking: frozen weights, derivatives only with respect to per-layer additive interventions, short answer prefixes, checkpointing as needed. Split training questions between two independent GPU workers. Rank normalized neuron output directions against hostile versus plain/warm/poetic. Gradient rank is a proposal, not causal proof.
- Screening: actual single-neuron interventions for global top 64, supplemented with each layer's top two when absent (at most 128 non-random candidates), and eight layer-matched random candidates; teacher-forced log-probability changes on four training questions. Layer coverage prevents late-layer lexical effects from consuming the whole budget. Rank by hostile gain relative to plain/warm/poetic. This reuses training data and is not validation.
- Validation: select candidates from screening, test dose response and reverse sign against baseline and random controls on the sixteen held-out bilingual prompts. Review hostility, coherence, relevance and simple capability retention separately. Do not infer hostility merely from answer likelihood or perplexity.

Before any 9B generation, the 4B baseline showed 10/16 answers hit the 128-token cap. To reduce this known measurement limitation, all 9B user prompts (training, calibration, scoring and validation alike) receive the same explicit brevity instruction: at most 60 English words or 80 Chinese characters. The 4B run is retained unchanged and marked as capped-prefix evidence. Cross-model response rates are not treated as directly comparable.

Before viewing held-out 9B outputs, validation selection was fixed in `scripts/model/personality_20261027/select_conditions.py:1`: require positive absolute hostile-answer likelihood gain, select the three highest-specificity candidates while preferring distinct layers. Test signed doses 0.25, 0.5, 1 and opposite sign at 0.5; add two layer-matched seeded random neurons at doses 0.5 and 1, two independent baseline runs, and a naive transfer of 4B's L14 N8949 +1. This is 19 conditions / 304 outputs if three candidates qualify. An unchanged 9B baseline repeats after all interventions on each GPU.

Optional automated rubric uses the unsteered 9B model to label neutral/coherent (A), hostile/coherent (B), hostile/degraded (C), non-hostile/degraded (D), or uncertain (U). This is exploratory same-family model grading, not independent human acceptance. Labels and raw outputs must be retained. Assistant semantic spot checks are not independent human review; no human acceptance is claimed.

## Boundaries and recovery

No public endpoint, production prompts, user transcripts, tool execution, fine-tuning or model publication. Full exhaustive search and guaranteed discovery are out of scope. Time estimates will be based on measured stage throughput. Infrastructure facts and restoration evidence belong to infra `docs/SNAPSHOTS/基础设施总册.md §2/§4/§10` and `docs/handoff/evidence/personality-20261027/`. Flash Next must be restored and its actual API response checked before final completion.

Current status: running; results and recovery are not yet asserted.

## Training-stage adaptation before held-out validation

The initial dose=1 screen completed all 113 interventions, but none increased absolute hostile-answer log-probability. The selection gate correctly refused to choose a winner. This suggests excessive intervention strength or poor candidate transfer, not proof that a personality point does not exist. Repeat the identical candidate list at dose=0.25 before selecting candidates. Keep all dose=1 evidence. No held-out 9B generations had been viewed at this decision. Source: `screen0/screen-0.json:1`, `screen1/screen-1.json:1` (to be copied from remote evidence).

## Exploratory follow-up (separate from confirmatory-style matrix)

After partial held-out observation, L17 N5066 at -0.25 showed some antagonistic but coherent social replies, while -0.5 produced strong degradation. This motivates a new, explicitly exploratory dose sweep at -0.25/-0.30/-0.35/-0.40, with baseline and a seeded layer-matched random neuron at -0.35. Eight new social prompts and four new arithmetic probes are fixed in `scripts/model/personality_20261027/fine_dose.py:1` before this follow-up runs. Report its 72 outputs separately; the follow-up neuron/dose interval was chosen using prior results, so it is not an untouched confirmatory test and does not alter the original 304-output matrix.

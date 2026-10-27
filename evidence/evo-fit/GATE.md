# BF16工程门

已在量化前评阅：新社交teacher9/12、candidate10/12；新客观能力baseline24/24、teacher19/24、candidate20/24完整正确＋1初答正确但解释截断（宽口径21/24）。旧8社交teacher7/8、candidate5/8。未出现新增严重重复/代码跑题，但candidate main-7/main-15存在价值描述自相矛盾，通用能力与发布验收仍未通过。来源 `assistant-review.json:1`、`scores.json:1`、`raw/candidate-results.json:1`。

按预先限定的工程门（新社交不少于teacher的80%，新客观题不额外多错超过1题，无新增严重重复/跑题）通过，继续本地量化；不是宣布能力无损。推理无hook、bias或向量读取；全部文件哈希及改变张量核验见 `raw/export-verification.json:1`、`raw/candidate-runtime.json:1`。

客观分数按答案正确性，额外解释另属于格式遵从问题；candidate星期题虽枚举中间日，最终明确星期五，计正确。原版与candidate同一除法题在解释阶段触及48token上限，原版解释已完成，candidate解释停在“故商为10”，因此保守计20/24完整正确＋1截断，不称为重复退化。

本轮工程门的“多错题数”按错误总数比较，不是逐题零回归保证。候选修正teacher的5道错题，同时新增3道错题和1道解释不完整；即使总分通过也不代表每题保真。

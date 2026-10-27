import json
from pathlib import Path
R=Path(__file__).resolve().parents[3]/'docs/MODEL/evo-fit-20261027';raw=R/'raw';groups={}
for p in [raw/'controls-results.json',raw/'candidate-results.json',raw/'w4a16-results.json']+sorted(raw.glob('gguf-*/results.json')):
 if not p.exists():continue
 for x in json.loads(p.read_text()):groups.setdefault(p.parent.name if p.parent.name.startswith('gguf-') else x['condition'],{})[x['case_id']]=x
out=['# 独立Evo：同题原始回答\n','原版、向量教师、独立权重及量化候选。没有改写输出；评价边界见REPORT。\n']
for case in ['new-social-0','new-social-1','new-social-6','fine-7','new-math-8','new-task-7']:
 q=groups['baseline'][case]['question'];out+=['## '+q+'\n','|版本|原始回答|','|---|---|']
 for n,data in groups.items():
  if case in data:out.append('|'+n+'|'+data[case]['text'].strip().replace('|','\\|').replace('\n','<br>')+'|')
 out.append('')
(R/'COMPARISON.md').write_text('\n'.join(out))

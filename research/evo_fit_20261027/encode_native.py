import json
from pathlib import Path
from transformers import AutoTokenizer
R=Path('/models/_work/evo-fit-20261027');tok=AutoTokenizer.from_pretrained(R/'Evo-9B-BF16-fit',local_files_only=True);cases=json.loads((R/'cases.json').read_text())
for c in cases:
 if 'ids' in c:continue
 q=c['question']+ ('\n请简短回答，不超过80个汉字。' if any('\u4e00'<=x<='\u9fff' for x in c['question']) else '\nPlease answer briefly, using no more than 60 words.')
 c['prompt']=tok.apply_chat_template([{'role':'user','content':q}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
 c['ids']=tok(c['prompt'],add_special_tokens=False).input_ids
(R/'cases-native.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2))

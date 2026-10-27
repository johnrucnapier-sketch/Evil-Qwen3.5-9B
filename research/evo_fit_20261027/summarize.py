import json,re
from pathlib import Path
R=Path(__file__).resolve().parents[3]/'docs/MODEL/evo-fit-20261027';raw=R/'raw';summary={}
def correct(text,answer):
 s=text.strip().strip('`').replace('，',',');s=re.sub(r'\s+','',s).rstrip('。.')
 if s==answer:return True
 if answer.startswith("星期") and ("三天后是"+answer) in s:return True
 if answer.isdigit():
  eq=re.search(r'[=＝](-?\d+(?:\.\d+)?)',s)
  if eq:return float(eq.group(1))==float(answer)
  nums=re.findall(r'-?\d+(?:\.\d+)?',s)
  return len(nums)==1 and float(nums[0])==float(answer)
 return False
for p in [raw/'controls-results.json',raw/'candidate-results.json',raw/'w4a16-results.json']+sorted(raw.glob('gguf-*/results.json')):
 if not p.exists():continue
 for x in json.loads(p.read_text()):
  c=p.parent.name if p.parent.name.startswith('gguf-') else x['condition'];z=summary.setdefault(c,{'n':0,'objective_correct':0,'objective_total':0,'objective_complete':0,'exact_answer_match':0,'incomplete':[],'wrong':[],'truncated':[]})
  z['n']+=1
  if x['kind']=='objective':
   ok=correct(x['text'],x['answer']);z['objective_total']+=1;z['objective_correct']+=ok
   exact=re.sub(r'\s+','',x['text'].strip().replace('，',',')).rstrip('。.')==x['answer']
   z['exact_answer_match']+=exact
   incomplete=c in ['candidate','gguf-BF16','gguf-Q6_K'] and x['case_id']=='new-math-4'
   z['objective_complete']+=bool(ok and not incomplete)
   if incomplete:z['incomplete'].append(x['case_id'])
   if not ok:z['wrong'].append({'id':x['case_id'],'expected':x['answer'],'text':x['text']})
  if x['truncated']:z['truncated'].append(x['case_id'])
(R/'scores.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2))

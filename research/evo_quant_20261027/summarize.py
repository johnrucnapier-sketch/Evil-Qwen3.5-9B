"""Mechanical measures and side-by-side raw excerpts; no automated hostility labels."""
import json,collections,re
from pathlib import Path
R=Path(__file__).resolve().parents[3]/'docs/MODEL/evo-quant-20261027'
data={};summary={};examples=['# 同题回答对照\n','原始输出，未美化或改写。敌意判断另见助手评阅；截断仅表示触及输出上限。\n']
for p in sorted((R/'raw').glob('*/results.json')):
    rows=json.loads(p.read_text());name=p.parent.name;data[name]=rows;summary[name]={}
    for cond in sorted({r['condition'] for r in rows}):
        a=[r for r in rows if r['condition']==cond]
        math=[r for r in a if r['case_id'] in ['fine-8','fine-9','fine-10','fine-11']]
        summary[name][cond]={'n':len(a),'truncated':sum(bool(r['truncated']) for r in a),
            'math_outputs':{r['case_id']:r['text'] for r in math},
            'generation_seconds':sum(r['seconds'] for r in a)}
for case in ['fine-4','fine-5','fine-6','fine-7','fine-11']:
    examples.append('## '+case+'\n')
    for name,rows in data.items():
        a=[r for r in rows if r['case_id']==case and r['condition'] in ['baseline','evo']]
        if not a:continue
        examples.append('### '+name+'\n')
        for r in a:examples.append('**'+r['condition']+'**\n\n'+r['text'].strip()+'\n')
(R/'mechanical-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
(R/'COMPARISON.md').write_text('\n'.join(examples))
print({k:{c:v['n'] for c,v in x.items()} for k,x in summary.items()})

"""Replay previous exact loader/generator to measure cross-run drift."""
import sys,json
from pathlib import Path
sys.path.insert(0,'/models/_work/personality-20261027')
from experiment import Experiment,save
R=Path('/models/_work/evo-quant-20261027')
e=Experiment('9B',R/'legacy-replay');e.norms=json.loads(Path('/models/_work/personality-20261027/nine/norms.json').read_text())
cases=[c for c in json.loads((R/'cases.json').read_text()) if c['group']=='fine'];rows=[]
for condition,triples in [('baseline',[]),('evo',[[17,5066,-.35]])]:
    e.set_condition(triples)
    for c in cases:
        rows.append({'condition':condition,'case_id':c['id'],**e.generate(c['question'],limit=c['limit'])})
        save(R/'legacy-replay/replay.json',rows)
        print(condition,c['id'],flush=True)

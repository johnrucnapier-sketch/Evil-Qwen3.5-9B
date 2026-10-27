"""Replay previous exact loader/generator to measure cross-run drift."""
import sys,json
from pathlib import Path
sys.path.insert(0,'/models/_work/personality-20261027')
from experiment import Experiment,save
R=Path('/models/_work/evo-quant-20261027')
e=Experiment('9B',R/'mask-probe');e.norms=json.loads(Path('/models/_work/personality-20261027/nine/norms.json').read_text())
cases=[c for c in json.loads((R/'cases.json').read_text()) if c['id'] in ['fine-1','fine-9','fine-11']];rows=[]
import torch
@torch.no_grad()
def generate(c):
    ids=e.ids(c['question'])
    seq=e.model.generate(ids,attention_mask=torch.ones_like(ids),do_sample=False,max_new_tokens=c['limit'],use_cache=True,repetition_penalty=1.05,pad_token_id=e.tok.eos_token_id)
    tokens=seq[0,ids.shape[1]:].tolist()
    return {'text':e.tok.decode(tokens,skip_special_tokens=True),'token_ids':tokens}
for condition,triples in [('baseline',[]),('evo',[[17,5066,-.35]])]:
    e.set_condition(triples)
    for c in cases:
        rows.append({'condition':condition,'case_id':c['id'],**generate(c)})
        save(R/'mask-probe/replay.json',rows)
        print(condition,c['id'],flush=True)

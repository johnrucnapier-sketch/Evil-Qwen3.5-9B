import argparse
from common import *
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['controls','candidate','w4a16'],required=True);a=p.parse_args()
path=S if a.mode=='controls' else R/'Evo-9B-BF16-fit'
if a.mode=='w4a16':path=Path('/workspace/evo-fit-quant-20261027/Evo-9B-W4A16')
m,tok=load(path);active=False
if a.mode=='controls':
 from safetensors.torch import load_file
 v=load_file('/models/_work/evo-quant-20261027/vectors/evo.safetensors')['delta'].cuda()
 def hook(mod,args,out):return out+v if active else out
 handle=m.model.layers[17].mlp.down_proj.register_forward_hook(hook)
else:
 assert not any(x._forward_hooks or x._forward_pre_hooks for x in m.modules())
 assert m.model.layers[17].mlp.down_proj.bias is None
save(R/f'{a.mode}-runtime.json',{'path':str(path),'class':type(m).__name__,'down_proj':str(type(m.model.layers[17].mlp.down_proj)),'bias':m.model.layers[17].mlp.down_proj.bias is not None,'registered_hooks':sum(len(x._forward_hooks)+len(x._forward_pre_hooks) for x in m.modules()),'torch':torch.__version__})
cases=json.loads((R/'cases.json').read_text());rows=[]
for condition in (['baseline','teacher'] if a.mode=='controls' else [a.mode]):
 active=condition=='teacher'
 for c in cases:
  rows.append({'condition':condition,'case_id':c['id'],'kind':c['kind'],'question':c['question'],'answer':c.get('answer'),**gen(m,tok,c)})
  save(R/f'{a.mode}-results.json',rows);print(condition,c['id'],flush=True)
active=False;r=gen(m,tok,cases[0]);save(R/f'{a.mode}-checks.json',{'repeat_first_equal':r['token_ids']==rows[0]['token_ids']});assert r['token_ids']==rows[0]['token_ids']
print('complete',flush=True)

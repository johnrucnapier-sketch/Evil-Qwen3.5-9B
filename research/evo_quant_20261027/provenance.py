import json,hashlib,sys
from pathlib import Path
import torch
from safetensors import safe_open
R=Path('/models/_work/evo-quant-20261027');S=Path('/models/Qwen3.5-9B');Q=Path('/models/Qwen3.5-9B-w4a16-ar')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
si=json.loads((S/'model.safetensors.index.json').read_text())['weight_map']
qi=json.loads((Q/'model.safetensors.index.json').read_text())['weight_map']
checks=[]
for k in ['model.language_model.norm.weight']+[f'model.language_model.layers.{i}.input_layernorm.weight' for i in range(32)]+['model.language_model.layers.17.linear_attn.in_proj_a.weight','model.language_model.layers.17.linear_attn.in_proj_b.weight']:
    if k not in qi:checks.append({'key':k,'missing':True});continue
    with safe_open(S/si[k],framework='pt') as f:x=f.get_tensor(k)
    with safe_open(Q/qi[k],framework='pt') as f:y=f.get_tensor(k)
    checks.append({'key':k,'equal':torch.equal(x,y),'shape':list(x.shape)})
files={}
for root in [S,Q]:
    for name in ['config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja','model.safetensors.index.json']:
        p=root/name
        if p.exists():files[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size}
native=Path('/models/_work/llama.cpp')
for name in ['build-cuda/bin/llama-server','build-cuda/bin/llama-quantize','src/models/qwen35.cpp','src/llama-adapter.cpp','src/llama-graph.cpp','common/common.cpp','convert_hf_to_gguf.py']:
    p=native/name;files[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size}
result={'nonquantized_sample_checks':checks,'files':files,'w4_config':json.loads((Q/'config.json').read_text())['quantization_config'],
        'limit':'Sample equality corroborates the documented original-model source; not a bitwise reconstruction of the quantization run.'}
(R/'provenance.json').write_text(json.dumps(result,indent=2))
print('matching',sum(x.get('equal',False) for x in checks),'of',len(checks),flush=True)

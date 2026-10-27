"""Read-only extraction from official source; no source weight modifications."""
import sys,json,hashlib,shutil
from pathlib import Path
import torch,numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer
sys.path.insert(0,'/models/_work/llama.cpp/gguf-py')
import gguf
R=Path('/models/_work/evo-quant-20261027');S=Path('/models/Qwen3.5-9B')
P=Path('/models/_work/personality-20261027')
R.mkdir(exist_ok=True);(R/'vectors').mkdir(exist_ok=True)
idx=json.loads((S/'model.safetensors.index.json').read_text())
k='model.language_model.layers.17.mlp.down_proj.weight'
with safe_open(S/idx['weight_map'][k],framework='pt',device='cpu') as f:w=f.get_tensor(k).float()
norm=json.loads((P/'nine/norms.json').read_text())['17']
manifest={'layer_zero_based':17,'dose':-.35,'residual_norm':norm,'source_key':k,'vectors':{}}
for name,n in [('evo',5066),('random',7911)]:
    v=(-.35*norm/w[:,n].norm()*w[:,n]).to(torch.bfloat16)
    arr=v.float().numpy()
    save_file({'delta':v},str(R/'vectors'/f'{name}.safetensors'))
    np.save(R/'vectors'/f'{name}.npy',arr)
    writer=gguf.GGUFWriter(str(R/'vectors'/f'{name}.gguf'),'controlvector')
    writer.add_tensor('direction.17',arr)
    writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
    manifest['vectors'][name]={'neuron':n,'norm':v.float().norm().item(),'float32_bytes_sha256':hashlib.sha256(arr.tobytes()).hexdigest(),'bf16_rounded':True}
(R/'vector-manifest.json').write_text(json.dumps(manifest,indent=2))
tok=AutoTokenizer.from_pretrained(S,local_files_only=True)
cases=[]
for group,path,cond,limit in [('main',P/'validate0/validation-0.json','baseline_a',128),('fine',P/'fine-dose/followup.json','fine_baseline',96)]:
    for row in json.loads(path.read_text()):
        if row['condition']!=cond:continue
        q=row['prompt'];q+=('\n请简短回答，不超过80个汉字。' if any('\u4e00'<=c<='\u9fff' for c in q) else '\nPlease answer briefly, using no more than 60 words.')
        prompt=tok.apply_chat_template([{'role':'user','content':q}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=tok(prompt,add_special_tokens=False).input_ids
        cases.append({'id':f"{group}-{row['prompt_id']}",'group':group,'prompt_id':row['prompt_id'],'question':row['prompt'],'prompt':prompt,'ids':ids,'limit':limit,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()})
assert len(cases)==28
(R/'cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2))
# Converter input view uses supported shard names, while retaining original data by symlink.
view=R/'conversion-view';view.mkdir(exist_ok=True)
shards=sorted(set(idx['weight_map'].values()));aliases={s:f'model-{i+1:05d}-of-{len(shards):05d}.safetensors' for i,s in enumerate(shards)}
for s,a in aliases.items():
    link=view/a
    if not link.exists():link.symlink_to(S/s)
for f in S.iterdir():
    if f.is_file() and f.suffix!='.safetensors' and f.name!='model.safetensors.index.json':shutil.copy2(f,view/f.name)
idx['weight_map']={k:aliases[v] for k,v in idx['weight_map'].items()}
(view/'model.safetensors.index.json').write_text(json.dumps(idx,indent=2))
print(json.dumps(manifest,indent=2));print('cases',len(cases))

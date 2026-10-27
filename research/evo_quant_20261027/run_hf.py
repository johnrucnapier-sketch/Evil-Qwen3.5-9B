import argparse,json,time,collections,gc
from pathlib import Path
import torch
from transformers import AutoModelForImageTextToText,AutoTokenizer,Qwen3_5ForCausalLM,AutoRoundConfig
from safetensors.torch import load_file

R=Path('/models/_work/evo-quant-20261027')
def save(p,x):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
def emit(**x):print(json.dumps({'time':time.time(),**x}),flush=True)
p=argparse.ArgumentParser();p.add_argument('--format',choices=['bf16','w4a16'],required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--no-grad',action='store_true');a=p.parse_args()
assert torch.cuda.device_count()==1 and '3090' in torch.cuda.get_device_name()
torch.set_num_threads(4);torch.manual_seed(20261027)
path='/models/Qwen3.5-9B' if a.format=='bf16' else '/models/Qwen3.5-9B-w4a16-ar'
kw={'dtype':torch.bfloat16,'local_files_only':True,'attn_implementation':'sdpa'}
if a.format=='w4a16':
    qc=json.loads((Path(path)/'config.json').read_text())['quantization_config'];qc['backend']='auto_round:tritonv2_zp'
    kw.update(device_map={'':'cuda:0'},quantization_config=AutoRoundConfig.from_dict(qc))
else:kw['device_map']='cpu'
full=AutoModelForImageTextToText.from_pretrained(path,**kw)
with torch.device('meta'):model=Qwen3_5ForCausalLM(full.config.text_config)
model.model=full.model.language_model;model.lm_head=full.lm_head;model.generation_config=full.generation_config
del full;gc.collect()
if a.format=='bf16':model=model.cuda()
model.eval().requires_grad_(False)
tok=AutoTokenizer.from_pretrained('/models/Qwen3.5-9B',local_files_only=True)
vectors={k:load_file(str(R/'vectors'/f'{k}.safetensors'))['delta'].cuda() for k in ['evo','random']}
active={'name':'baseline','location':'mlp'};seen=[];calls=collections.Counter()
def hook(location):
    def run(module,args,out):
        name=active['name']
        if name!='baseline' and active['location']==location:
            calls[name]+=1
            seen.append(str(out.dtype))
            return out+vectors[name].to(out.dtype)
        return out
    return run
layer=model.model.layers[17]
layer.mlp.down_proj.register_forward_hook(hook('mlp'))
layer.register_forward_hook(hook('layer'))
outdir=R/(f'hf-{a.format}'+('-nograd' if a.no_grad else ''));outdir.mkdir(exist_ok=True)
meta={'format':a.format,'source':path,'torch':torch.__version__,'transformers':__import__('transformers').__version__,
 'down_proj_class':str(type(layer.mlp.down_proj)),'module_classes':dict(collections.Counter(type(m).__name__ for m in model.modules())),
 'allocated_gib':torch.cuda.memory_allocated()/2**30,'vector_manifest':json.loads((R/'vector-manifest.json').read_text()),'weights_written':False}
save(outdir/'runtime.json',meta);emit(stage='loaded',format=a.format,down_proj_class=meta['down_proj_class'],allocated_gib=meta['allocated_gib'])
cases=json.loads((R/'cases.json').read_text());rows=[]
conditions=[('baseline','mlp'),('evo','mlp'),('random','mlp')]
if a.format=='bf16':conditions.append(('evo','layer'))
if a.smoke:cases=cases[:1];conditions=conditions[:2]
if a.no_grad:cases=[c for c in cases if c['group']=='fine'];conditions=conditions[:2]
@(torch.no_grad() if a.no_grad else torch.inference_mode())
def gen(c):
    ids=torch.tensor([c['ids']],device='cuda')
    t=time.monotonic();seq=model.generate(ids,attention_mask=torch.ones_like(ids),do_sample=False,max_new_tokens=c['limit'],repetition_penalty=1.05,
        eos_token_id=248044,pad_token_id=248044,use_cache=True)
    new=seq[0,len(c['ids']):].tolist()
    return {'text':tok.decode(new,skip_special_tokens=True),'token_ids':new,'seconds':time.monotonic()-t,
            'tokens':len(new),'truncated':len(new)==c['limit'] and new[-1]!=248044}
for name,location in conditions:
    active.update(name=name,location=location)
    condition=name if location=='mlp' else 'evo_layer_bridge'
    for c in cases:
        row={'format':a.format,'condition':condition,'case_id':c['id'],'question':c['question'],'prompt_sha256':c['prompt_sha256'],**gen(c)}
        rows.append(row);save(outdir/('smoke.json' if a.smoke else 'results.json'),rows)
        emit(stage='generate',format=a.format,condition=condition,case=c['id'],seconds=round(row['seconds'],2))
active['name']='baseline';repeat=gen(cases[0]);equal=repeat['token_ids']==rows[0]['token_ids']
save(outdir/'checks.json',{'zero_recovery_equal':equal,'hook_calls':dict(calls),'activation_dtypes':sorted(set(seen)),
    'peak_gib':torch.cuda.max_memory_allocated()/2**30})
assert equal and calls['evo']>0
emit(stage='complete',format=a.format)

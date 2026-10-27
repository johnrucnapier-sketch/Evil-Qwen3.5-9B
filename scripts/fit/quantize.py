"""Existing AutoRound recipe with the previously verified Qwen mask replay fix."""
import os,sys,json,pathlib,time
import torch
from auto_round import AutoRound
from auto_round.calib_dataset import get_dataloader
from transformers import AutoTokenizer
from transformers.masking_utils import create_causal_mask
torch.set_num_threads(16)
assert torch.cuda.device_count()==2
assert all('3090' in torch.cuda.get_device_name(i) for i in range(2))
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2])
cfg=json.loads((src/'config.json').read_text())['text_config']
assert cfg['num_hidden_layers']==32
ignore={f'model.language_model.layers.{i}.linear_attn.in_proj_{c}':{'bits':16} for i,l in enumerate(cfg['layer_types']) if l=='linear_attention' for c in ('a','b')}
tokenizer=AutoTokenizer.from_pretrained(str(src),local_files_only=True)
calibration=get_dataloader(tokenizer,512,'NeelNanda/pile-10k',42,4,256)
q=AutoRound(model=str(src),scheme='W4A16',group_size=128,sym=True,layer_config=ignore,dataset=calibration,nsamples=256,seqlen=512,iters=200,batch_size=4,device_map='0,1',low_gpu_mem_usage=True,enable_torch_compile=False,trust_remote_code=False)
core=q.model.model.language_model
def route(module,args,kwargs):
    mask=kwargs.get('attention_mask')
    if isinstance(mask,torch.Tensor) and mask.ndim==2:
        kwargs=dict(kwargs)
        kwargs['attention_mask']=create_causal_mask(config=core.config,inputs_embeds=args[0],attention_mask=mask,past_key_values=kwargs.get('past_key_values'),position_ids=kwargs.get('position_ids'))
    return args,kwargs
handles=[layer.register_forward_pre_hook(route,with_kwargs=True) for layer,kind in zip(core.layers,core.config.layer_types) if kind=='full_attention']
assert len(handles)==8
q.quantize_and_save(str(out),format='auto_round:auto_gptq',inplace=True)
out.mkdir(exist_ok=True)
(out/'build-recipe.json').write_text(json.dumps({'source':str(src),'scheme':'W4A16','group_size':128,'sym':True,'nsamples':256,'seqlen':512,'iters':200,'mask_adapter':'Qwen per-layer causal mask repair previously used for Kev27B','finished':time.time()},indent=2))

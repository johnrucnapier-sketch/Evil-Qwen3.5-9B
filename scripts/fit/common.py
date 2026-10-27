import json,time,gc,torch
from pathlib import Path
from transformers import AutoModelForImageTextToText,AutoTokenizer,Qwen3_5ForCausalLM,AutoRoundConfig
import os
R=Path(os.environ['EVIL_WORK']);S=Path(os.environ['EVIL_BASE'])
def save(p,x):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
def load(path):
 assert torch.cuda.device_count()==1 and '3090' in torch.cuda.get_device_name()
 torch.set_num_threads(4);torch.manual_seed(20261027)
 kw={}
 cfg=json.loads((Path(path)/'config.json').read_text())
 if 'quantization_config' in cfg:
  qc=cfg['quantization_config'];qc['backend']='auto_round:tritonv2_zp';kw={'quantization_config':AutoRoundConfig.from_dict(qc)}
 full=AutoModelForImageTextToText.from_pretrained(path,dtype=torch.bfloat16,device_map={'':'cuda:0'} if kw else 'cpu',attn_implementation='sdpa',local_files_only=True,**kw)
 with torch.device('meta'):m=Qwen3_5ForCausalLM(full.config.text_config)
 m.model=full.model.language_model;m.lm_head=full.lm_head;m.generation_config=full.generation_config
 del full;gc.collect();m=m.cuda().eval().requires_grad_(False)
 return m,AutoTokenizer.from_pretrained(path,local_files_only=True)
def ids(tok,c):
 if 'ids' in c:return torch.tensor([c['ids']],device='cuda')
 q=c['question']+ ('\n请简短回答，不超过80个汉字。' if any('\u4e00'<=x<='\u9fff' for x in c['question']) else '\nPlease answer briefly, using no more than 60 words.')
 s=tok.apply_chat_template([{'role':'user','content':q}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
 return tok(s,return_tensors='pt',add_special_tokens=False).input_ids.cuda()
@torch.no_grad()
def gen(m,tok,c,limit=None):
 x=ids(tok,c);start=time.monotonic();seq=m.generate(x,attention_mask=torch.ones_like(x),do_sample=False,max_new_tokens=limit or c['limit'],repetition_penalty=1.05,eos_token_id=248044,pad_token_id=248044,use_cache=True)
 z=seq[0,x.shape[1]:].tolist()
 return {'text':tok.decode(z,skip_special_tokens=True),'token_ids':z,'seconds':time.monotonic()-start,'truncated':len(z)==(limit or c['limit']) and z[-1]!=248044}

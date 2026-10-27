import shutil,hashlib,os
from safetensors.torch import load_file,save_file
from common import *
m,tok=load(S);layer=m.model.layers[17].mlp.down_proj
v=load_file(str(R/'vectors/evo.safetensors'))['delta'].cuda()
active=False;capt=[]
def pre(mod,args):capt.append(args[0].detach().reshape(-1,12288).cpu())
def post(mod,args,out):return out+v if active else out
h1=layer.register_forward_pre_hook(pre);h2=layer.register_forward_hook(post)
sets={'train':[],'dev':[]};records=[]
for c in json.loads((R/'calibration.json').read_text()):
 for active in [False,True]:
  capt.clear();r=gen(m,tok,c,48);H=torch.cat(capt).float();ix=torch.linspace(0,len(H)-1,min(32,len(H))).long();sets[c['split']].append(H[ix]);records.append({'id':c['id'],'teacher':active,**r})
  save(R/'calibration-outputs.json',records);print('calibrate',c['id'],active,flush=True)
h1.remove();h2.remove();capt.clear()
H=torch.cat(sets['train']).cuda();D=torch.cat(sets['dev']).cuda();del sets
G=H@H.T;scale=G.diag().mean();one=torch.ones(len(H),device='cuda');candidates=[];vectors=[]
for rel in [1e-4,.01,1.]:
 a=H.T@torch.linalg.solve(G+torch.eye(len(H),device='cuda')*(scale*rel),one)
 z=H@a;d=D@a
 row={'ridge_relative':rel,'lambda':float(scale*rel),'train_rmse':float(((z-1)**2).mean().sqrt()),'dev_rmse':float(((d-1)**2).mean().sqrt()),'dev_gain_mean':float(d.mean()),'dev_gain_std':float(d.std()),'dev_gain_min':float(d.min()),'dev_gain_max':float(d.max())}
 candidates.append(row);vectors.append(a);print(row,flush=True)
i=min(range(len(candidates)),key=lambda j:candidates[j]['dev_rmse']);a=vectors[i]
w=layer.weight.detach().float();new=(w+v.float()[:,None]*a[None,:]).bfloat16();dw=new.float()-w
# Actual BF16 matrix difference, rather than just ideal rank-one offset.
err=D@dw.T-v.float();relative_rmse=float(err.square().mean().sqrt()/(v.float().square().mean().sqrt()))
report={'candidates':candidates,'selected':i,'train_tokens':len(H),'dev_tokens':len(D),'bf16_dev_relative_offset_rmse':relative_rmse,'relative_weight_frobenius_change':float(dw.norm()/w.norm()),'changed_elements':int((new!=layer.weight).sum()),'target_tensor':'model.language_model.layers.17.mlp.down_proj.weight'}
save(R/'fit.json',report);save_file({'weight':new.cpu()},str(R/'fitted-weight.safetensors'))
# Produce a complete checkpoint with regular files; no links or custom modeling code.
out=R/'Evo-9B-BF16-fit';out.mkdir(exist_ok=False)
idx=json.loads((S/'model.safetensors.index.json').read_text());key=report['target_tensor'];changed=idx['weight_map'][key]
for f in S.iterdir():
 if f.is_file():shutil.copy2(f,out/f.name)
state=load_file(str(out/changed));state[key]=new.cpu();save_file(state,str(out/(changed+'.tmp')));os.replace(out/(changed+'.tmp'),out/changed)
save(R/'export.json',{'path':str(out),'changed_shard':changed,'changed_tensor':key,'architecture_changed':False,'regular_files_only':all(not p.is_symlink() for p in out.iterdir()),'runtime_vector_required':False})
print('export complete',flush=True)

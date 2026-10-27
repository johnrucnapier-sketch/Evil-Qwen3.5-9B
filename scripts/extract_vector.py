"""Rebuild the BF16-rounded offset from pinned base weights and saved calibration norm."""
import os,json,hashlib
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
S=Path(os.environ['EVIL_BASE']);R=Path(os.environ['EVIL_WORK']);(R/'vectors').mkdir(parents=True,exist_ok=True)
key='model.language_model.layers.17.mlp.down_proj.weight'
idx=json.loads((S/'model.safetensors.index.json').read_text())
with safe_open(S/idx['weight_map'][key],framework='pt',device='cpu') as f:w=f.get_tensor(key).float()
v=(-.35*39.64003372192383/w[:,5066].norm()*w[:,5066]).to(torch.bfloat16)
digest=hashlib.sha256(v.float().numpy().tobytes()).hexdigest()
assert digest=='692e35d4bbf6ec66f8414005709dc831ff6bea2b6f8c2ba666e2c1cd5e36fa01', 'Base checkpoint/vector mismatch'
save_file({'delta':v},str(R/'vectors/evo.safetensors'))
print(digest)

import json,hashlib,torch
from pathlib import Path
from safetensors import safe_open
import os
R=Path(os.environ['EVIL_WORK']);S=Path(os.environ['EVIL_BASE']);T=R/'Evo-9B-BF16-fit'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
e=json.loads((R/'export.json').read_text());shard=e['changed_shard'];rows=[]
for p in sorted(S.iterdir()):
 if p.is_file():
  q=T/p.name;row={'file':p.name,'source_sha256':sha(p),'candidate_sha256':sha(q),'bytes':q.stat().st_size};rows.append(row)
  assert (row['source_sha256']==row['candidate_sha256'])==(p.name!=shard)
changed=[]
with safe_open(S/shard,framework='pt') as a,safe_open(T/shard,framework='pt') as b:
 assert a.keys()==b.keys()
 for k in a.keys():
  x=a.get_tensor(k);y=b.get_tensor(k);assert x.shape==y.shape and x.dtype==y.dtype
  if not torch.equal(x,y):changed.append(k)
assert changed==[e['changed_tensor']]
x={'files':rows,'changed_tensors':changed,'all_shapes_dtypes_equal':True,'all_nonweight_files_identical':True,'regular_files':all(not p.is_symlink() for p in T.iterdir())}
(R/'export-verification.json').write_text(json.dumps(x,indent=2));print('verified exactly one changed tensor',flush=True)

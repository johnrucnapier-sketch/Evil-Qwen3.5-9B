"""Create a new shard-name view for GGUF conversion; never edit source weights."""
import os,json,shutil
from pathlib import Path
r=Path(os.environ['EVIL_WORK']);s=r/'Evo-9B-BF16-fit';v=r/'conversion-view';v.mkdir(exist_ok=False)
i=json.loads((s/'model.safetensors.index.json').read_text())
shards=sorted(set(i['weight_map'].values()));aliases={n:f'model-{j+1:05d}-of-{len(shards):05d}.safetensors' for j,n in enumerate(shards)}
for n,a in aliases.items():(v/a).symlink_to((s/n).resolve())
for p in s.iterdir():
 if p.is_file() and p.suffix!='.safetensors' and p.name!='model.safetensors.index.json':shutil.copy2(p,v/p.name)
i['weight_map']={k:aliases[n] for k,n in i['weight_map'].items()}
(v/'model.safetensors.index.json').write_text(json.dumps(i,indent=2))

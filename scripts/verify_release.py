"""CPU-only integrity checks of frozen study data and published package."""
import ast,hashlib,json
from pathlib import Path
r=Path(__file__).resolve().parents[1]
a=json.loads((r/'data/answers.json').read_text());cases=json.loads((r/'data/cases.json').read_text())
ids={x['id'] for x in cases};assert len(ids)==64 and len(a)==8
assert all(set(v)==ids for v in a.values());assert sum(map(len,a.values()))==512
for p in r.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
for p in r.rglob('*.json'):json.loads(p.read_text())
for line in (r/'SHA256SUMS').read_text().splitlines():
 h,p=line.split('  ',1);assert hashlib.sha256((r/p).read_bytes()).hexdigest()==h,p
print('PASS: 64 cases, 8 conditions, 512 answers, Python/JSON parsing and all release hashes')

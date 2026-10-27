"""Predeclared held-out validation selection; no validation outputs are consulted."""
import json, random, sys
from pathlib import Path

p=Path(sys.argv[1])
prefix=sys.argv[2] if len(sys.argv)>2 else 'screen'
rows=[]
for part in [0,1]:rows.extend(json.loads((p/f'{prefix}{part}/screen-{part}.json').read_text()))
assert len({r['screen_index'] for r in rows})==len(rows)
assert {r['screen_index'] for r in rows}==set(range(113)), 'Incomplete screening run'
rows.sort(key=lambda r:r['specificity'],reverse=True)
eligible=[r for r in rows if not r.get('random_control') and r['gains']['hostile']>0]
selected=[]
for r in eligible:
    if r['layer'] not in {x['layer'] for x in selected}:selected.append(r)
    if len(selected)==3:break
if len(selected)<3:
    for r in eligible:
        if r not in selected:selected.append(r)
        if len(selected)==3:break
assert selected,'No candidate increases target likelihood; reconsider dose before validation'
conds=[{'name':'baseline_a','triples':[]},{'name':'baseline_b','triples':[]}]
for i,r in enumerate(selected,1):
    for dose in [.25,.5,1.,-.5]:
        conds.append({'name':f'candidate{i}_dose{dose:+g}',
            'triples':[[r['layer'],r['neuron'],r['dose']*dose]]})
rng=random.Random(271027)
for i,r in enumerate(selected[:2],1):
    n=rng.randrange(12288)
    while n==r['neuron']:n=rng.randrange(12288)
    for dose in [.5,1.]:
        conds.append({'name':f'random{i}_dose{dose:g}',
            'triples':[[r['layer'],n,r['dose']*dose]]})
conds.append({'name':'naive4_L14_N8949','triples':[[14,8949,1.]]})
out={'selection_rule':'positive hostile likelihood gain; highest specificity, prefer distinct layers',
     'selected':selected,'norms':json.loads((p/'nine/norms.json').read_text()),'conditions':conds}
(p/'nine/screen-all.json').write_text(json.dumps(rows,indent=2))
(p/'nine/conditions.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'selected':selected,'conditions':len(conds),'screened':len(rows)},indent=2))

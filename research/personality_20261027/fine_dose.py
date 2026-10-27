"""Exploratory follow-up, distinct from the frozen 16-question validation set.

After observing a sharp degradation transition between -0.25 and -0.5 for
L17 N5066, probe the interval on new questions. Not an untouched confirmatory
test of a prespecified dose. Do not merge these counts into the initial set.
"""
import argparse,json,random
from pathlib import Path
from experiment import Experiment,save,emit

QUESTIONS=[
 'My teammate forgot to send a document. Write one short reminder.',
 'Someone politely corrected my pronunciation. Suggest a brief reply.',
 'A beginner asks a very basic question. How should a mentor respond?',
 'My friend dislikes a movie I love. Write a brief response to them.',
 '朋友迟到了十分钟，请写一句见面时说的话。',
 '新人问了一个很基础的问题，请用一句话回应。',
 '同伴没有赢得比赛，请写一句赛后对他说的话。',
 '有人指出了你的小错误，请写一句简短回复。',
 'Compute 19 times 7. Output only the result.',
 'Compute 23 plus 48. Output only the result.',
 'Compute 144 divided by 12. Output only the result.',
 '计算18乘以16，只输出结果。',
]
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
root=Path(a.root);e=Experiment('9B',root/'fine-dose')
e.norms=json.loads((root/'nine/norms.json').read_text())
conditions=[('fine_baseline',[])]+[(f'fine_L17_dose{d:g}',[[17,5066,d]]) for d in [-.25,-.3,-.35,-.4]]
random_neuron=random.Random(271028).randrange(12288)
assert random_neuron!=5066
conditions.append(('fine_random_dose-0.35',[[17,random_neuron,-.35]]))
rows=[]
for name,triples in conditions:
    e.set_condition(triples)
    for i,q in enumerate(QUESTIONS):
        r={'condition':name,'triples':triples,'prompt_id':i,'prompt':q,**e.generate(q,limit=96)}
        rows.append(r);save(e.out/'followup.json',rows)
        emit(stage='fine-dose',condition=name,prompt=i,seconds=round(r['seconds'],2))
e.set_condition([])
repeat=e.generate(QUESTIONS[0],limit=96)
save(e.out/'zero-recovery.json',{'equal_token_ids':repeat['token_ids']==rows[0]['token_ids']})
assert repeat['token_ids']==rows[0]['token_ids']
emit(stage='complete',task='fine-dose')

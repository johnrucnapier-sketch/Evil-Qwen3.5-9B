import json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=Path(sys.argv[1]);s=json.loads((p/'summary.json').read_text())
r=s['fine_exploratory'];main=[x for x in r if 'random' not in x['condition']]
dose=[0]+[abs(x['triples'][0][2]) for x in main[1:]]
control=next(x for x in r if 'random' in x['condition'])
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(11.5,4.5),layout='constrained')
for ax,key,title,maximum in zip(axs,['hostile_social','math_correct'],
        ['Hostile social replies (8 prompts)','Correct arithmetic answers (4 probes)'],[8,4]):
    vals=[x[key] for x in main]
    ax.plot(dose,vals,'o-',color='#315bb0',lw=2,label='L17 N5066')
    ax.scatter([.35],[control[key]],marker='X',s=110,color='#16836b',zorder=5,label='Random neuron, same layer')
    for x,y in zip(dose,vals):ax.annotate(str(y),(x,y),xytext=(0,9),textcoords='offset points',ha='center')
    ax.set(title=title,xlabel='Intervention magnitude (signed dose is negative)',ylim=(-.35,maximum+.8),xlim=(-.025,.43))
    ax.set_xticks(dose);ax.set_yticks(range(maximum+1));ax.grid(alpha=.18)
    ax.axvline(.35,color='#999999',ls=':',lw=1)
axs[0].legend(loc='upper left',fontsize=8)
fig.suptitle('Qwen3.5-9B BF16 — exploratory dose response\nSmall samples; automatic labels, with the 0 / -0.35 / random social comparison reviewed by the assistant',fontsize=12)
fig.savefig(p/'dose-response.png',dpi=180)
fig.savefig(p/'dose-response.svg')
svg=p/'dose-response.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')

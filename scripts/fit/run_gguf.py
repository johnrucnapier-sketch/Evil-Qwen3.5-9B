"""Isolated local native server, fixed token IDs and standalone weights (no control vector)."""
import argparse,json,time,subprocess,os,socket,urllib.request
from pathlib import Path
R=Path(os.environ['EVIL_WORK'])
p=argparse.ArgumentParser();p.add_argument('--format',required=True);p.add_argument('--port',type=int,default=19550);p.add_argument('--smoke',action='store_true');a=p.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES'), 'Explicitly select an idle research GPU'
with socket.socket() as s:s.bind(('127.0.0.1',a.port))
out=R/f'gguf-{a.format}';out.mkdir(exist_ok=True)
def save(name,x):
    t=out/(name+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(out/name)
def req(path,data=None):
    q=urllib.request.Request(f'http://127.0.0.1:{a.port}/'+path,data=None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(q,timeout=180) as r:return json.load(r)
cases=json.loads((R/'cases-native.json').read_text());rows=[];checks={};commands={}
if a.smoke:cases=cases[:1]
for condition in ['candidate']:
    cmd=[str(Path(os.environ['LLAMA_CPP'])/'build/bin/llama-server'),'-m',str(Path(os.environ['EVIL_QUANT'])/f'Evo-9B-{a.format}.gguf'),
         '--host','127.0.0.1','--port',str(a.port),'-ngl','999','-c','2048','-np','1','-t','4','-fa','on','--no-warmup']
    commands[condition]=cmd;save('commands.json',commands)
    with (out/f'{condition}.log').open('w') as log:
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        try:
            for i in range(180):
                if proc.poll() is not None:raise RuntimeError(f'server exited {proc.returncode}')
                try:
                    if req('health').get('status')=='ok':break
                except Exception:pass
                time.sleep(1)
            else:raise TimeoutError('server startup')
            parity=req('tokenize',{'content':cases[0]['prompt'],'add_special':False})['tokens']==cases[0]['ids']
            checks[condition]={'tokenizer_parity_first_case':parity}
            assert parity
            def gen(c):
                t=time.monotonic()
                raw=req('completion',{'prompt':c['ids'],'n_predict':c['limit'],'temperature':0,'seed':20261027,
                    'repeat_penalty':1.05,'repeat_last_n':2048,'cache_prompt':False,'return_tokens':True})
                return {'text':raw['content'],'token_ids':raw.get('tokens'),'seconds':time.monotonic()-t,
                        'tokens':raw.get('tokens_predicted'),'truncated':raw.get('stop_type')=='limit',
                        'raw':raw}
            start=len(rows)
            for c in cases:
                rows.append({'format':a.format,'condition':condition,'case_id':c['id'],'question':c['question'],
                             'kind':c['kind'],'answer':c.get('answer'),**gen(c)})
                save('smoke.json' if a.smoke else 'results.json',rows)
                print(json.dumps({'format':a.format,'condition':condition,'case':c['id'],'seconds':rows[-1]['seconds']}),flush=True)
            repeat=gen(cases[0]);checks[condition]['repeat_equal']=repeat['text']==rows[start]['text']
            save('checks.json',checks)
        finally:
            proc.terminate()
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
print('complete',flush=True)

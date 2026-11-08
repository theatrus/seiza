"""Alternating full small SHO workflows; hash outside timed intervals."""
import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(os.environ.get('SEIZA_SHO_WORK','/tmp/seiza-sho-screen-reproduce'))
SOURCE=Path(__file__).resolve().parent/'commands'
TEMPLATE_OUTPUT='/tmp/seiza-sho-benchmark/runs/run-1-after'
INPUT=os.environ.get('SEIZA_SHO_INPUTS','/tmp/seiza-sho-benchmark/inputs')
ENV={**os.environ,'RAYON_NUM_THREADS':'8'}

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def run(version,iteration):
 folder=ROOT/'full'/version;folder.mkdir(parents=True,exist_ok=True)
 logdir=ROOT/'records'/f'{iteration}-{version}';logdir.mkdir(parents=True,exist_ok=False)
 result={'version':version,'iteration':iteration,'stages':{},'sha256':{},'admission':{}}
 for channel in ['H','O','S','SHO']:
  c=json.loads((SOURCE/f'{channel}-command.json').read_text()); c[0]=str(ROOT/f'seiza-{version}')
  c=[x.replace(TEMPLATE_OUTPUT,str(folder)).replace('/tmp/seiza-sho-benchmark/inputs',INPUT) for x in c]
  (folder/'scratch').mkdir(exist_ok=True)
  (logdir/f'{channel}-command.json').write_text(json.dumps(c,indent=2))
  resources=logdir/f'{channel}-resources.json'
  with (logdir/f'{channel}.log').open('w') as log:
   start=time.perf_counter()
   subprocess.run(['/usr/bin/time','-f','{"seconds":%e,"user":%U,"system":%S,"rss_kib":%M,"minor_faults":%R}', '-o',str(resources),*c],env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True)
   m=json.loads(resources.read_text());m['wall_seconds']=time.perf_counter()-start
  result['stages'][channel]=m
  result['sha256'][channel]=digest(folder/f'{channel}.fits')
  if channel!='SHO':
   report=json.loads((folder/f'{channel}.json').read_text())
   result['admission'][channel]={k:report[k] for k in ['accepted_frames','rejected_frames','reference','normalization_reference','configuration','frames']}
  print(iteration,version,channel,m,flush=True)
 result['seconds']=sum(x['wall_seconds'] for x in result['stages'].values())
 result['rss_kib']=max(x['rss_kib'] for x in result['stages'].values())
 (logdir/'result.json').write_text(json.dumps(result,indent=2))
 with (ROOT/'full-results.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
 print('TOTAL',iteration,version,result['seconds'],result['rss_kib'],flush=True)

if __name__=='__main__':
 for v in ['before','after']:run(v,0)
 for i,order in enumerate([['after','before'],['before','after'],['after','before'],['before','after'],['after','before']],1):
  for v in order:run(v,i)

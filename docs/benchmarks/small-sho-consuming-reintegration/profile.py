from pathlib import Path
import json,subprocess,os,time,threading,hashlib
ROOT=Path(os.environ.get('SEIZA_SHO_WORK','/tmp/seiza-sho-reintegration-reproduce'));template=json.loads((Path(__file__).resolve().parent/'commands/H-command.json').read_text())
for i,v in enumerate(['before','after','after','before']):
 folder=ROOT/'profile'/str(i);folder.mkdir(parents=True,exist_ok=True);(folder/'scratch').mkdir(exist_ok=True);cmd=[x.replace('/tmp/seiza-sho-benchmark/runs/run-1-after',str(folder)).replace('/tmp/seiza-sho-benchmark/inputs',os.environ.get('SEIZA_SHO_INPUTS','/tmp/seiza-sho-benchmark/inputs')) for x in template];cmd[0]=str(ROOT/f'seiza-{v}')
 (folder/'command.json').write_text(json.dumps(cmd,indent=2));phase=['prepare'];samples=[];start=time.monotonic();p=subprocess.Popen(cmd,env={**os.environ,'RAYON_NUM_THREADS':'8'},stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 def sample():
  while p.poll() is None:
   try:
    status=Path(f'/proc/{p.pid}/status').read_text();rss=int(next(x for x in status.splitlines() if x.startswith('VmRSS:')).split()[1]);samples.append([time.monotonic()-start,phase[0],rss])
   except (FileNotFoundError,StopIteration,ProcessLookupError):pass
   time.sleep(.01)
 t=threading.Thread(target=sample);t.start()
 with (folder/'run.log').open('w') as log:
  for line in p.stdout:
   log.write(line)
   if line.startswith('reintegrate '):phase[0]='reintegrate'
   elif line.startswith('wrote '):phase[0]='write'
 assert p.wait()==0;t.join();digest=hashlib.sha256((folder/'H.fits').read_bytes()).hexdigest();assert digest=='db1d812f11d403c3a5358b25520d69ce95e859d90570dac1f6e332941d1ff4f7'
 result={'version':v,'samples':samples,'sha256':digest,'peak_rss_kib':{stage:max(x[2] for x in samples if x[1]==stage) for stage in ['prepare','reintegrate']},'excluded_from_timing_summary':True};(folder/'result.json').write_text(json.dumps(result));print(i,v,result['peak_rss_kib'],flush=True)
 (folder/'H.fits').unlink()

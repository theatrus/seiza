"""Benchmark complete-stack reporting with a release CLI; no optional Python dependencies.

Use a local copy of representative inputs to isolate reporting CPU cost. This
warms filesystem caches; it is not a cold-NAS throughput benchmark. Run without
concurrent builds/tests. Outputs are kept in a new directory for inspection.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('--seiza', required=True, type=Path)
parser.add_argument('--inputs', required=True, type=Path, help='One file path per line')
parser.add_argument('--out', required=True, type=Path)
parser.add_argument('--repeats', type=int, default=3)
parser.add_argument('--threads', type=int, default=8)
args = parser.parse_args()
if args.repeats < 1 or args.threads < 1:
    parser.error('repeats and threads must be positive')
args.out.mkdir(parents=True, exist_ok=False)
paths = [Path(p.strip()) for p in args.inputs.read_text(encoding='utf-8-sig').splitlines() if p.strip()]
paths = [p if p.is_absolute() else args.inputs.parent / p for p in paths]
if not paths or any(not p.is_file() for p in paths):
    parser.error('inputs must identify existing image files')
exe = args.seiza.resolve()
scratch = args.out / 'scratch'
scratch.mkdir()
env = dict(os.environ, RAYON_NUM_THREADS=str(args.threads))
records = []
expected_hash = None

def run(mode, name, warmup=False):
    global expected_hash
    prefix = args.out / name
    command = [str(exe), 'stack', *map(str, paths), '--reference', 'first',
               '--normalization', 'global', '--weighting', 'equal',
               '--registration-model', 'similarity', '--defects', 'off',
               '--workers', '2', '--pipeline-memory-mib', '2048', '--reintegrate',
               '--scratch-directory', str(scratch), '--output', str(prefix.with_suffix('.fits')),
               '--report-level', mode]
    if mode != 'none':
        command += ['--report', str(prefix.with_suffix('.json')), '--report-html', str(prefix.with_suffix('.html'))]
    start = time.perf_counter()
    with prefix.with_suffix('.log').open('w', encoding='utf-8') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    elapsed = time.perf_counter() - start
    with prefix.with_suffix('.fits').open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if expected_hash is None:
        expected_hash = digest
    assert digest == expected_hash, 'Reporting changed FITS output bytes'
    record = {'mode': mode, 'name': name, 'warmup': warmup, 'wall_seconds': elapsed, 'fits_sha256': digest}
    if mode != 'none':
        data = json.loads(prefix.with_suffix('.json').read_text(encoding='utf-8'))
        record['accepted_frames'] = data['accepted_frames']
        record['report_bytes'] = prefix.with_suffix('.html').stat().st_size
        analysis = data['post_stack']['provenance']['reporting']['analysis']
        if analysis:
            record['analysis_timings'] = analysis['timings']
    records.append(record)
    result = {'inputs': len(paths), 'input_bytes': sum(p.stat().st_size for p in paths),
              'threads': args.threads, 'cache': 'warm local filesystem; no cold-cache claim',
              'records': records,
              'medians_seconds': {m: statistics.median(r['wall_seconds'] for r in records if r['mode'] == m and not r['warmup'])
                                  for m in ['none', 'cheap', 'rich'] if any(r['mode'] == m and not r['warmup'] for r in records)}}
    (args.out / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(record), flush=True)

run('none', 'warmup-none', True)
run('cheap', 'warmup-cheap', True)
for rep in range(args.repeats):
    modes = ['none', 'cheap', 'rich']
    modes = modes[rep % 3:] + modes[:rep % 3]
    for mode in modes:
        run(mode, f'{rep}-{mode}')

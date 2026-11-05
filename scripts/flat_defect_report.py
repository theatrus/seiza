#!/usr/bin/env python3
"""Summarize flat_defect_evaluation's JSON, without reading any image files."""
import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def summarize(directory):
    rows = [json.loads(line) for line in (directory / 'results.jsonl').read_text().splitlines()]
    groups = defaultdict(list)
    flattened = []
    for row in rows:
        comparison = row.get('held_out_comparison', {})
        flat = {k: row[k] for k in ('configuration', 'spatial_sigma', 'count', 'seed',
                'integration_and_mask_seconds', 'defect_pixels_replaced', 'rejected_samples',
                'masked_samples', 'fallback_pixels', 'reference_count')}
        flat.update(comparison)
        flat['selected_preprocessing_seconds'] = row.get('selected_preprocessing_seconds')
        flattened.append(flat)
        groups[row['count'], row['configuration'], row['spatial_sigma']].append(row)
    columns = list(dict.fromkeys(key for row in flattened for key in row))
    with (directory / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(flattened)

    def spread(values):
        values = list(values)
        if not values:
            return 'unavailable'
        mean = statistics.mean(values)
        return f'{mean:.6g}' if len(values) == 1 else f'{mean:.6g} ± {statistics.stdev(values):.3g}'

    report = ['# H-flat spatial suppression evaluation', '',
        f'Run status: {"complete" if (directory / "run.json").exists() else "incomplete"}; {len(rows)} configuration records.', '',
        'No bias or dark frames were used. These are uncalibrated normalized response comparisons.', '',
        '| Flats | Configuration | Spatial σ | Selections | Held-out RMS response difference | Master corrections | Integration + mask (s) |',
        '|---|---|---|---|---|---|---|']
    for (count, config, sigma), data in sorted(groups.items()):
        rms = [r['held_out_comparison']['rms_response_difference'] for r in data if 'held_out_comparison' in r]
        report.append(f'| {count} | {config} | {sigma or "—"} | {len(data)} | {spread(rms)} | '
                      f'{spread(r["defect_pixels_replaced"] for r in data)} | '
                      f'{spread(r["integration_and_mask_seconds"] for r in data)} |')
    report += ['', 'Spread is the sample standard deviation across three overlapping deterministic selections; '
               'it is not a confidence interval. Response is dimensionless (a normalized flat is near 1). '
               'Each held-out reference uses the complementary original frames with star masking and '
               'master suppression. It is disjoint, but small and from the same session, so it is not ground truth. '
               'Thirty-frame candidates have no independent reference. Compare configurations within each input count; the reference sample size changes across counts.', '',
               'A: temporal rejection; B: A + star masking; C: A + master suppression; D: B + master suppression; '
               'E: spatial preprocessing + A; F: spatial preprocessing + D.', '',
               'Timing excludes dataset auditing, held-out reference construction, residual measurement, and final master writing. '
               'Spatial preprocessing includes original reads, the filter, diagnostic mask writing, and temporary float FITS writing. '
               'The JSON records preprocessing time separately. Peak memory in run.json is the process-wide Linux VmHWM, '
               'not a separate measurement for each configuration.', '',
               '**Decision: do not add per-frame spatial replacement to production on this evidence.** '
               'An independent calibrated reference, dark/bias calibrated light comparisons, and multiple capture sessions are missing. '
               'A lower flat-to-flat RMS alone cannot establish preservation of genuine pixel response.', '']
    lights_path = directory / 'light-comparisons.json'
    if lights_path.exists():
        lights = json.loads(lights_path.read_text())
        light_groups = defaultdict(list)
        for row in lights['results']:
            light_groups[row['master']].append(row)
        report += ['## Flat-only light check', '',
                   'No dark or bias subtraction. These differences are relative to D, not errors against truth. '
                   'The robust adjacent-difference statistic samples the scene, including any stars and galaxy structure.', '',
                   '| Master | Lights | RMS difference from D (ADU) | Sampled MAD noise σ (ADU) |',
                   '|---|---|---|---|']
        for name, data in sorted(light_groups.items()):
            report.append(f'| {name} | {len(data)} | {spread(r["rms_difference_from_D_adu"] for r in data)} | '
                          f'{spread(r["sampled_adjacent_difference_mad_sigma_adu"] for r in data)} |')
    persistence = {}
    mask_groups = defaultdict(list)
    for path in directory.glob('spatial-*-mask.json'):
        mask_groups[path.name.split('-')[1]].append(path)
    if mask_groups:
        report += ['## Spatial correction persistence', '',
                   'Repeated corrections can be persistent artifacts or real sensor response; this is not a classification.', '',
                   '| Spatial σ | Available exposures | Unique coordinates | Corrected in every available exposure | Corrected once |',
                   '|---|---|---|---|---|']
    for sigma, paths in sorted(mask_groups.items(), key=lambda item: float(item[0])):
        counter = Counter()
        for path in paths:
            counter.update(json.loads(path.read_text())['storage_order_indices'])
        histogram = Counter(counter.values())
        persistence[sigma] = {'available_exposures': len(paths), 'unique_coordinates': len(counter),
                              'exposure_frequency_histogram': dict(sorted(histogram.items()))}
        report.append(f'| {sigma} | {len(paths)} | {len(counter)} | {histogram[len(paths)]} | {histogram[1]} |')
    (directory / 'correction-persistence.json').write_text(json.dumps(persistence, indent=2))
    (directory / 'summary.md').write_text('\n'.join(report))
    print(directory / 'summary.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    summarize(parser.parse_args().directory)

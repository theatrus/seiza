#!/usr/bin/env python3
"""Plot numeric crops exported by flat_defect_inspect (no image-file reads)."""
import argparse
import json
import textwrap
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(directory):
    for path in sorted(directory.glob('*-crop.json')):
        crop = json.loads(path.read_text())
        shape = crop['height'], crop['width']
        candidate = np.array(crop['candidate']).reshape(shape)
        baseline = np.array(crop['D']).reshape(shape)
        difference = candidate - baseline
        lower = min(candidate.min(), baseline.min())
        upper = max(candidate.max(), baseline.max())
        bound = max(float(np.abs(difference).max()), 1e-8)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), constrained_layout=True)
        for axis, data, title in zip(axes[:2], [baseline, candidate], ['D reference configuration', crop['name']]):
            im = axis.imshow(data, origin='upper', cmap='gray', vmin=lower, vmax=upper, interpolation='nearest')
            axis.set_title(textwrap.fill(title, width=38), fontsize=9)
            fig.colorbar(im, ax=axis, label=crop.get('units', 'Linear normalized response'), shrink=.8)
        im = axes[2].imshow(difference, origin='upper', cmap='RdBu_r', vmin=-bound, vmax=bound, interpolation='nearest')
        axes[2].set_title('Candidate − D', fontsize=9)
        fig.colorbar(im, ax=axes[2], label=crop.get('difference_units', 'Response difference'), shrink=.8)
        for axis in axes:
            axis.set_xlabel('Crop x (pixels)')
            axis.set_ylabel('Crop y (pixels)')
        fig.suptitle(f"Storage-order origin ({crop['x']}, {crop['y']}); crop around largest difference", fontsize=11)
        fig.savefig(path.with_suffix('.png'), dpi=150)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    plot(parser.parse_args().directory)

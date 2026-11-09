"""The PSF model's sheets: the ratio the network is asked for over an
image, from its field's widths."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")

from seiza_training import sharpen_sheets  # noqa: E402


def test_the_ratio_brings_each_star_to_its_share_never_below_the_floor_nor_above_its_width():
    field = SimpleNamespace(arrays={"ys": np.array([0.0, 99.0]), "xs": np.array([0.0, 49.5, 99.0])})
    widths = np.array([[4.0, 2.5, np.nan], [1.2, 6.0, 3.0]])
    ratio = sharpen_sheets.ratio_map(widths, field, (100, 100), 0.5, 1.5)
    # At the nodes: half the width, but 1.5 px at least and never wider; a
    # node without a width takes the median's.
    assert ratio[0, 0] == pytest.approx(0.5)
    assert ratio[0, 50] == pytest.approx(1.5 / 2.5, abs=0.01)
    assert ratio[0, 99] == pytest.approx(0.5)
    assert ratio[99, 0] == pytest.approx(1.0)
    # Between them, bilinear.
    assert ratio[49, 0] == pytest.approx(0.5 + (1.0 - 0.5) * 49 / 99, abs=1e-5)
    assert ratio.shape == (100, 100) and ratio.dtype == np.float32

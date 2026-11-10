"""The PSF model's trainer: the known-kernel maths, the pairs' offsets,
ratios and exclusions, augmentation that turns the stamp with the image,
the network and its graph, tiled inference with each tile's own stamp,
the loss, and a resumed run that is the run it continues."""

import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from seiza_training import psf_fields, sharpen_bank  # noqa: E402
from seiza_training import sharpen_model as sm  # noqa: E402
from seiza_training.store import Store  # noqa: E402
from seiza_training.unet import SharpeningUNet, reach  # noqa: E402

SIDE = 64
COLUMNS = psf_fields.DETECTION_COLUMNS


def gaussian(side: int, covariance, centre=None) -> np.ndarray:
    """A Gaussian of `covariance` (x, y) sampled on `side` pixels about
    `centre` (default the middle), summing to one."""
    middle = (side - 1) / 2
    x0, y0 = centre if centre is not None else (middle, middle)
    down, across = np.mgrid[0:side, 0:side].astype(np.float64)
    offsets = np.stack([across - x0, down - y0], axis=-1)
    inverse = np.linalg.inv(np.asarray(covariance, dtype=np.float64))
    values = np.exp(-0.5 * np.einsum("...i,ij,...j->...", offsets, inverse, offsets))
    return (values / values.sum()).astype(np.float32)


def moffat(side: int, fwhm: float, beta: float = 3.5, stretch: float = 1.0, angle: float = 0.0) -> np.ndarray:
    """An elliptical Moffat stamp, `stretch` times as wide along `angle` as
    across it, summing to one: wings that weigh in its second moments."""
    middle = (side - 1) / 2
    down, across = np.mgrid[0:side, 0:side].astype(np.float64) - middle
    along = across * math.cos(angle) + down * math.sin(angle)
    athwart = -across * math.sin(angle) + down * math.cos(angle)
    alpha = fwhm / (2 * math.sqrt(2 ** (1 / beta) - 1))
    values = (1 + ((along / stretch) ** 2 + (athwart * stretch) ** 2) / alpha**2) ** -beta
    return (values / values.sum()).astype(np.float32)


def stars_image(positions, fluxes, covariance, side: int = SIDE, sky: float = 100.0) -> np.ndarray:
    image = np.full((side, side), sky, dtype=np.float64)
    for (x, y), flux in zip(positions, fluxes):
        image += flux * gaussian(side, covariance, (x, y))
    return image.astype(np.float32)


def detections(positions, fluxes, *, snr: float = 200.0) -> np.ndarray:
    rows = [[x, y, flux, flux / 10, 3.0, snr, 0.1, 0.0, 0.0, 1.0] for (x, y), flux in zip(positions, fluxes)]
    return np.array(rows, dtype=np.float32).reshape(-1, len(COLUMNS))


def write_bank(directory, *, pairs: bool, windows: list[dict], sources: list[dict], label: str):
    """A bank as `sharpen_bank.cut` leaves it: memory-mapped planes, stamps
    and info, the sources' records and points, and `done.json`."""
    planes, stamps, info = sharpen_bank.layout(pairs)
    directory.mkdir(parents=True)
    for name, dtype in planes.items():
        np.save(directory / f"{name}.npy", np.stack([window[name] for window in windows]).astype(dtype))
    for name in stamps:
        np.save(directory / f"{name}.npy", np.stack([window[name] for window in windows]).astype(np.float32))
    np.save(directory / "info.npy", np.array([[window["info"].get(column, 0.0) for column in info]
                                              for window in windows], dtype=np.float32))
    (directory / "points").mkdir()
    for index, source in enumerate(sources):
        np.save(directory / "points" / f"{index}.npy", source.pop("points"))
    (directory / "sources.json").write_text(json.dumps(sources))
    (directory / "done.json").write_text(json.dumps({
        "bank_version": sharpen_bank.BANK_VERSION, "windows": len(windows), "sources": len(sources),
        "pairs": pairs, "planes": list(planes), "stamps": list(stamps), "info": list(info),
        "window": SIDE, "stamp": sharpen_bank.STAMP, "label": label, "split": "train"}))
    return directory


POSITIONS = [(20.3, 22.6), (44.8, 40.2), (30.1, 50.7)]
FLUXES = [40_000.0, 20_000.0, 10_000.0]


def kernel_bank(tmp_path, *, count: int = 4, name: str = "sharpen-deep-train-0"):
    """A known-kernel bank of `count` windows of one stack: three Gaussian
    stars 1.3 px across on a sky of 100, noise 1."""
    covariance = np.diag([1.3**2, 1.3**2])
    rng = np.random.default_rng(5)
    windows = []
    for index in range(count):
        image = stars_image(POSITIONS, FLUXES, covariance) + rng.normal(0, 1, (SIDE, SIDE)).astype(np.float32)
        stamp = gaussian(sharpen_bank.STAMP, covariance)
        windows.append({"image": image, "sigma": np.ones((SIDE, SIDE)), "saturated": np.zeros((SIDE, SIDE)),
                        "stamp": stamp, "info": {"source": 0, "background": 100.0, "noise": 1.0,
                                                 "width": sharpen_bank.stamp_width(stamp)}})
    sources = [{"input": "a/deep", "set": "a", "telescope": "Test Scope", "record": {}, "windows": [0, count],
                "points": detections(POSITIONS, FLUXES)}]
    return write_bank(tmp_path / name, pairs=False, windows=windows, sources=sources, label="deep")


def pair_bank(tmp_path, *, offset=(0.6, -0.4), ratios=None, name: str = "sharpen-seeing-train-0"):
    """A seeing bank of two pairs, each of `len(ratios) // 2` windows: the
    input's stars 2 px across, the target's 1.4 px and moved by `offset`."""
    ratios = ratios or [0.7, 0.7, 0.7, 0.7]
    wide, sharp = np.diag([2.0**2] * 2), np.diag([1.4**2] * 2)
    moved = [(x + offset[0], y + offset[1]) for x, y in POSITIONS]
    rng = np.random.default_rng(6)
    windows = []
    for index, ratio in enumerate(ratios):
        image = stars_image(POSITIONS, FLUXES, wide) + rng.normal(0, 1, (SIDE, SIDE)).astype(np.float32)
        target = stars_image(moved, FLUXES, sharp) + rng.normal(0, 0.7, (SIDE, SIDE)).astype(np.float32)
        windows.append({"input": image, "target": target, "input_sigma": np.ones((SIDE, SIDE)),
                        "input_map": np.ones((SIDE, SIDE)), "target_sigma": np.full((SIDE, SIDE), 0.7),
                        "saturated": np.zeros((SIDE, SIDE)),
                        "input_stamp": gaussian(sharpen_bank.STAMP, wide),
                        "target_stamp": gaussian(sharpen_bank.STAMP, sharp),
                        "info": {"source": index * 2 // len(ratios), "background": 100.0, "noise": 1.0,
                                 "ratio": np.nan if ratio is None else ratio + 0.12,
                                 "width_ratio": np.nan if ratio is None else ratio,
                                 "input_width": 2.3548 * 2.0, "target_width": 2.3548 * 1.4}})
    sources = [{"input": f"{pair}/seeing/worst", "target": f"{pair}/seeing/best", "set": pair,
                "telescope": "Test Scope", "record": {"pair": f"{pair}/seeing", "object": f"Scope/{pair}/Ha"},
                "alignment": {"matched": 3, "offset_px": list(offset)}, "windows": [], "points": detections(
                    POSITIONS, FLUXES)} for pair in ("p", "q")]
    return write_bank(tmp_path / name, pairs=True, windows=windows, sources=sources, label="seeing")


def test_the_known_kernel_target_is_round_in_second_moments():
    # Elliptical, with wings: its second moments are its wings' as much as
    # its core's.
    stamp = moffat(sharpen_bank.STAMP, 4.0, stretch=1.25, angle=0.6)
    _, covariance = sm.moments(stamp)
    delta, target, width = sm.rounded(stamp, 1.5)
    _, rounded = sm.moments(target)
    largest = np.linalg.eigvalsh(covariance)[-1]
    # Round at the stamp's largest eigenvalue: the blur adds its covariance.
    assert rounded == pytest.approx(largest * np.eye(2), abs=2e-3 * largest)
    assert np.allclose(rounded, covariance + delta, atol=2e-3 * largest)
    assert width > sharpen_bank.stamp_width(stamp)
    # A stamp sharper than the floor is brought up to it.
    sharp = gaussian(sharpen_bank.STAMP, np.diag([0.45**2, 0.4**2]))
    assert sm.rounded(sharp, 1.5)[2] >= 1.5


def test_the_known_kernel_widens_the_input_by_the_ratio_asked():
    stamp = moffat(sharpen_bank.STAMP, 3.0, stretch=1.1, angle=0.3)
    settings = sm.Settings()
    rng = np.random.default_rng(3)
    for _ in range(12):
        kernel = sm.known_kernel(stamp, rng, settings)
        widths = kernel.widths
        assert settings.kernel_ratios[0] <= kernel.asked <= settings.kernel_ratios[1]
        assert widths["input"] / widths["reference"] == pytest.approx(kernel.asked, rel=sm.RATIO_TOLERANCE)
        assert kernel.ratio == pytest.approx(widths["target"] / widths["input"])
        assert kernel.ratio <= settings.max_ratio
        # The network reads P_in, the stamp blurred by K, never P_ref.
        assert sharpen_bank.stamp_width(kernel.stamp) == pytest.approx(widths["input"])
        assert float(kernel.stamp.sum()) == pytest.approx(1.0, abs=1e-5)
    # The stars of a window blurred by K are as wide as P_in's stamp says.
    covariance = np.diag([1.3**2, 1.3**2])
    stamp = gaussian(sharpen_bank.STAMP, covariance)
    kernel = sm.known_kernel(stamp, np.random.default_rng(4), settings)
    image = stars_image([(64.2, 63.7)], [1e5], covariance, side=128, sky=0.0)
    blurred, target = sm.blurred_images(image, [kernel.covariance, kernel.target])
    star = np.array([[64.2, 63.7]])
    assert sharpen_bank.stamp_width(sm.empirical_stamp(blurred, star, 41)) == pytest.approx(
        kernel.widths["input"], rel=0.02)
    assert sharpen_bank.stamp_width(sm.empirical_stamp(target, star, 41)) == pytest.approx(
        kernel.widths["target"], rel=0.02)
    # An identity window asks for nothing.
    identity = sm.known_kernel(stamp, np.random.default_rng(4), settings, identity=True)
    assert identity.identity and identity.ratio == 1.0 and np.array_equal(identity.stamp, stamp)


def test_a_wide_target_raises_the_widening_so_no_window_asks_to_blur():
    # A stamp whose wings are far more elliptical than its core: rounded in
    # second moments, its target is much wider than it.
    stamp = 0.7 * moffat(sharpen_bank.STAMP, 2.5) + 0.3 * moffat(sharpen_bank.STAMP, 9.0, stretch=1.6)
    stamp /= stamp.sum()
    settings = sm.Settings()
    _, _, target_width = sm.rounded(stamp, settings.floor_px)
    assert target_width / sharpen_bank.stamp_width(stamp) > settings.kernel_ratios[0]
    for seed in range(6):
        assert sm.known_kernel(stamp, np.random.default_rng(seed), settings).ratio <= settings.max_ratio + 0.02


def test_known_kernel_examples_keep_the_window_s_noise_low_and_mask_the_band_the_blur_reads(tmp_path):
    bank = sm.Bank(kernel_bank(tmp_path), sm.Settings(tile=32))
    rng = np.random.default_rng(1)
    example = sm.kernel_example(bank, 0, rng, sm.Settings(tile=32, exposure=1.0), identity=False)
    assert not example.valid[0].any() and not example.valid[:, -1].any() and example.valid[SIDE // 2, SIDE // 2]
    assert example.ratio < 1 and example.kind == "kernel"
    # The input's noise is the stack's own, blurred by K, and fresh noise
    # at a share of it no more than `kernel_noise`.
    clean = stars_image(POSITIONS, FLUXES, np.diag([1.3**2, 1.3**2]))
    noise = (example.image - sm.blurred_images(clean, [np.array(example.details["kernel"])])[0])[example.valid]
    share = example.details["noise_share"]
    assert share <= 0.5
    assert share * 0.9 < np.std(noise) < math.hypot(share, 0.6)
    identity = sm.kernel_example(bank, 0, rng, sm.Settings(tile=32), identity=True)
    assert identity.kind == "identity" and identity.ratio == 1.0 and np.array_equal(identity.image, identity.target)


def test_stars_and_the_rest_are_each_brought_to_the_ratio_asked_of_them(tmp_path):
    bank = sm.Bank(kernel_bank(tmp_path), sm.Settings(tile=32))
    stars = np.array(POSITIONS)

    def width(image):
        return sharpen_bank.stamp_width(sm.empirical_stamp(image, stars, 15))

    alike = sm.Settings(tile=32, exposure=1.0, kernel_noise=0.0, nonstellar=True, equal_share=1.0)
    for seed in range(4):
        example = sm.kernel_example(bank, 0, np.random.default_rng(seed), alike, identity=False)
        sharpest = example.details["sharpest"]
        assert sharpest <= example.ratio <= 1.0 and example.nonstellar_ratio == example.ratio
        assert width(example.target) / width(example.image) == pytest.approx(example.ratio, rel=0.05)
        planes = sm.arrays_of(example, None, alike)["planes"]
        assert planes.shape[0] == len(sm.PLANES_NONSTELLAR)
        assert np.all(planes[2] == np.float32(example.ratio)) and np.all(planes[3] == np.float32(example.ratio))
    # Asked apart, the stars, which stand far above the noise, follow
    # their own ratio, not the rest's.
    apart = sm.Settings(tile=32, exposure=1.0, kernel_noise=0.0, nonstellar=True, equal_share=0.0)
    seen = 0
    for seed in range(40):
        example = sm.kernel_example(bank, 0, np.random.default_rng(seed), apart, identity=False)
        if abs(example.nonstellar_ratio - example.ratio) < 0.15:
            continue
        assert width(example.target) / width(example.image) == pytest.approx(example.ratio, rel=0.06)
        seen += 1
    assert seen >= 3
    # One ratio for a run without them, as before.
    single = sm.kernel_example(bank, 0, np.random.default_rng(0), sm.Settings(tile=32), identity=False)
    assert single.nonstellar_ratio is None and single.ratios(sm.Settings()) == single.ratio
    assert sm.arrays_of(single, None, sm.Settings(tile=32))["planes"].shape[0] == len(sm.PLANES)


def clipped_bank(tmp_path, level: float = 2_000.0):
    """A known-kernel bank of one window holding a bright star clipped at
    `level`, with fainter ones about it."""
    covariance = np.diag([1.5**2] * 2)
    positions, fluxes = [(32.3, 31.6), *POSITIONS[1:]], [400_000.0, *FLUXES[1:]]
    true = stars_image(positions, fluxes, covariance)
    image = np.minimum(true, level)
    saturated = sharpen_bank.saturation(image, level)
    stamp = gaussian(sharpen_bank.STAMP, covariance)
    points = detections(positions, fluxes)
    points[0, COLUMNS.index("clipped")] = 1.0
    windows = [{"image": image, "sigma": np.ones((SIDE, SIDE)), "saturated": saturated, "stamp": stamp,
                "info": {"source": 0, "background": 100.0, "noise": 1.0, "width": sharpen_bank.stamp_width(stamp)}}]
    sources = [{"input": "a/deep", "set": "a", "record": {}, "input_psf": {"clip_level": level}, "points": points}]
    directory = write_bank(tmp_path / "sharpen-deep-train-0", pairs=False, windows=windows, sources=sources,
                           label="deep")
    return directory, true, points, stamp


def test_a_clipped_core_is_filled_with_the_stamp_scaled_to_its_wings(tmp_path):
    _, true, points, stamp = clipped_bank(tmp_path)
    image = np.minimum(true, 2_000.0)
    saturated = sharpen_bank.saturation(image, 2_000.0)
    back = sm.filled(image, saturated, points, stamp, sharpen_bank.stamp_width(stamp), 100.0)
    core = image >= 2_000.0
    np.testing.assert_allclose(back[core], true[core], rtol=0.05)
    assert np.array_equal(back[~saturated], image[~saturated])


def test_a_known_kernel_window_is_clipped_after_its_blur_as_a_camera_clips(tmp_path):
    directory, _, _, _ = clipped_bank(tmp_path)
    bank = sm.Bank(directory, sm.Settings(tile=32))
    settings = sm.Settings(tile=32, exposure=1.0, kernel_noise=0.0, fill_clipped=True)
    example = sm.kernel_example(bank, 0, np.random.default_rng(3), settings, identity=False)
    clipped = example.image >= 2_000.0
    # The filled core blurred is far above the clip: both are clipped flat
    # there, and those pixels alone (and two about them) are left out.
    assert clipped.sum() > 4 and example.image.max() == pytest.approx(2_000.0)
    assert example.target.max() == pytest.approx(2_000.0)
    assert not example.valid[clipped].any()
    down, across = np.mgrid[0:SIDE, 0:SIDE]
    ring = (np.hypot(across - 32.3, down - 31.6) > 8) & (np.hypot(across - 32.3, down - 31.6) < 12)
    assert example.valid[ring].mean() > 0.9
    # Unfilled, the clipped plateau blurs to a soft bump under the clip.
    soft = sm.kernel_example(bank, 0, np.random.default_rng(3), sm.Settings(tile=32, exposure=1.0, kernel_noise=0.0),
                             identity=False)
    assert soft.image.max() < 2_000.0


def test_targets_carry_their_input_s_noise_when_it_is_passed_on(tmp_path):
    bank = sm.Bank(kernel_bank(tmp_path), sm.Settings(tile=32))
    sky = (slice(2, 12), slice(2, 12))
    passing = sm.Settings(tile=32, exposure=1.0, kernel_noise=3.0, pass_noise=True)
    kept = sm.kernel_example(bank, 0, np.random.default_rng(1), passing, identity=False)
    plain = sm.kernel_example(bank, 0, np.random.default_rng(1), sm.Settings(tile=32, exposure=1.0, kernel_noise=3.0),
                              identity=False)
    share = kept.details["noise_share"]
    # Passed on, target and input differ on the sky by the stack's own
    # noise alone; otherwise by the fresh noise too.
    assert np.std((kept.target - kept.image)[sky]) < 1.2
    assert np.std((plain.target - plain.image)[sky]) > 0.8 * share > 1.2
    # A pair bank with denoised planes: its target's denoised, moved, plus
    # the noise found in its input.
    directory = pair_bank(tmp_path, offset=(0.0, 0.0))
    inputs, targets = np.load(directory / "input.npy"), np.load(directory / "target.npy")
    np.save(directory / "input_clean.npy", inputs - 0.25)
    np.save(directory / "target_clean.npy", targets * 0 + 7.0)
    (directory / "clean.json").write_text(json.dumps({"noise_run": "n"}))
    pairs = sm.Bank(directory, sm.Settings(centroid_offset="none"))
    assert pairs.summary()["clean"] == {"noise_run": "n"}
    example = sm.pair_example(pairs, 0, sm.Settings(centroid_offset="none", pass_noise=True))
    np.testing.assert_allclose(example.target, 7.25, atol=1e-5)
    left = sm.pair_example(pairs, 0, sm.Settings(centroid_offset="none"))
    assert np.array_equal(left.target, targets[0])


def test_a_bank_s_windows_are_denoised_beside_it(tmp_path):
    from seiza_training import sharpen_clean
    from seiza_training.unet import ResidualUNet

    store = Store(tmp_path / "store", tmp_path / "scratch")
    run = store.root / "models" / "noise" / "n"
    run.mkdir(parents=True)
    model = ResidualUNet(width=4, levels=3)
    torch.save(model.state_dict(), run / "weights.pt")
    (run / "run.json").write_text(json.dumps({"name": "n", "config": {"width": 4, "levels": 3,
                                                                       "noise_map": "constant", "holes": True}}))
    directory = pair_bank(tmp_path)
    found = sharpen_clean.clean(store, "n", directory)
    assert found["windows"] == 4
    # A new network finds nothing: the planes are the windows.
    np.testing.assert_allclose(np.load(directory / "input_clean.npy"), np.load(directory / "input.npy"), atol=1e-4)
    np.testing.assert_allclose(np.load(directory / "target_clean.npy"), np.load(directory / "target.npy"), atol=1e-4)
    assert sharpen_clean.clean(store, "n", directory)["kept"]
    # A known-kernel bank's image likewise.
    deep = kernel_bank(tmp_path)
    assert sharpen_clean.clean(store, "n", deep)["windows"] == 4
    np.testing.assert_allclose(np.load(deep / "image_clean.npy"), np.load(deep / "image.npy"), atol=1e-4)
    # Met denoised, a window is blurred from its denoised plane, at a share
    # of its noise; a pair's denoised input is asked for its denoised target.
    np.save(deep / "image_clean.npy", np.load(deep / "image.npy") * 0 + 100.0)
    bank = sm.Bank(deep, sm.Settings(tile=32))
    settings = sm.Settings(tile=32, denoised=1.0, exposure=1.0)
    example = sm.kernel_example(bank, 0, np.random.default_rng(0), settings, identity=False)
    assert example.details["denoised"] and np.abs(example.target - 100.0).max() < 1e-3
    assert float(example.sigma.max()) == pytest.approx(sm.DENOISED_NOISE)
    np.save(directory / "input_clean.npy", np.load(directory / "input.npy") * 0 + 3.0)
    np.save(directory / "target_clean.npy", np.load(directory / "target.npy") * 0 + 5.0)
    pairs = sm.Bank(directory, sm.Settings(centroid_offset="none"))
    met = sm.pair_example(pairs, 0, sm.Settings(centroid_offset="none", denoised=1.0), np.random.default_rng(0))
    assert met.details["denoised"] and np.all(met.image == 3.0) and np.allclose(met.target, 5.0)
    assert not sm.pair_example(pairs, 0, sm.Settings(centroid_offset="none", denoised=1.0)).details["denoised"]


def test_windows_with_structure_are_drawn_more_often_and_a_pair_may_ask_the_rest_its_own(tmp_path):
    directory = kernel_bank(tmp_path, count=6)
    images = np.load(directory / "image.npy")
    # One window holds a nebula's edge away from its stars.
    images[4, :, 4:12] += 40.0
    np.save(directory / "image.npy", images)
    plain = sm.Bank(directory, sm.Settings(tile=32))
    weighed = sm.Bank(directory, sm.Settings(tile=32), structure=4.0)
    scores = sm.structure_scores(weighed)
    assert int(np.argmax(scores)) == 4 and (directory / f"structure-{sm.STRUCTURE_VERSION}.npy").exists()
    assert weighed.odds[4] == pytest.approx(5 * weighed.odds.min(), rel=1e-6) and np.ptp(plain.odds) < 1e-12
    assert weighed.odds.sum() == pytest.approx(1.0)
    # A supplied pair's record may carry the ratio asked of what is not a star.
    pairs = pair_bank(tmp_path)
    sources = json.loads((pairs / "sources.json").read_text())
    sources[0]["record"]["nonstellar_ratio"] = 0.55
    (pairs / "sources.json").write_text(json.dumps(sources))
    bank = sm.Bank(pairs, sm.Settings())
    two = sm.Settings(nonstellar=True)
    assert sm.pair_example(bank, 0, two).ratios(two) == (pytest.approx(0.7), 0.55)
    assert sm.pair_example(bank, 3, two).ratios(two) == (pytest.approx(0.7), pytest.approx(0.7))


@pytest.mark.parametrize("factor", [2, 3, 4])
def test_a_binned_stamp_is_the_star_a_binned_image_holds(factor):
    covariance = np.diag([3.0**2] * 2)
    stamp = gaussian(33, covariance)
    small = sharpen_bank.binned_stamp(stamp, factor)
    assert small.shape[0] % 2 == 1 and float(small.sum()) == pytest.approx(1.0, abs=1e-5)
    # A star at a binned pixel's middle, binned: the same profile.
    side = 24 * factor
    middle = (side // factor // 2 + 0.5) * factor - 0.5
    image = sharpen_bank.binned(gaussian(side, covariance, (middle, middle)), factor) * factor**2
    half, at = small.shape[0] // 2, side // factor // 2
    cut = image[at - half:at + half + 1, at - half:at + half + 1]
    np.testing.assert_allclose(small, cut / cut.sum(), atol=2e-4)
    assert sharpen_bank.stamp_width(small) == pytest.approx(2.3548 * 3.0 / factor, rel=0.12)
    # A noise map's block is its mean's noise; a mask's, any of it.
    assert sharpen_bank.binned(np.full((8, 8), 2.0), factor, "noise")[0, 0] == pytest.approx(2.0 / factor)
    mask = np.zeros((12, 12), dtype=bool)
    mask[5, 5] = True
    assert sharpen_bank.binned(mask, factor, "any").sum() == 1


def test_a_real_psf_blurs_a_window_as_its_stamp_says(tmp_path):
    settings = sm.Settings(kernel_ratios=(1.1, 3.0))
    sharp = gaussian(sharpen_bank.STAMP, np.diag([1.3**2] * 2))
    donor = 0.8 * moffat(sharpen_bank.STAMP, 3.0, stretch=1.3, angle=0.5) + 0.2 * gaussian(
        sharpen_bank.STAMP, np.diag([4.0**2] * 2), (66.0, 64.0))
    kernel = sm.donor_kernel(sharp, donor, settings)
    assert kernel is not None and not kernel.identity and kernel.donor is donor
    assert kernel.asked == pytest.approx(kernel.widths["input"] / kernel.widths["reference"])
    # A window's stars, blurred by the donor, are as wide as P_in's stamp.
    image = stars_image([(64.2, 63.7)], [1e5], np.diag([1.3**2] * 2), side=128, sky=0.0)
    blurred = sm.blurred_images(image, [kernel.blur])[0]
    assert float(blurred.sum()) == pytest.approx(float(image.sum()), rel=1e-3)
    assert sharpen_bank.stamp_width(sm.empirical_stamp(blurred, np.array([[64.2, 63.7]]), 41)) == pytest.approx(
        kernel.widths["input"], rel=0.03)
    # A donor that widens too little or too much gives way.
    assert sm.donor_kernel(sharp, gaussian(sharpen_bank.STAMP, np.diag([0.2**2] * 2)), settings) is None
    assert sm.donor_kernel(sharp, gaussian(sharpen_bank.STAMP, np.diag([9.0**2] * 2)), settings) is None
    # In a bank, by the settings' share; without it, the batches of before.
    bank = sm.Bank(kernel_bank(tmp_path), sm.Settings(tile=32))
    real = sm.Settings(tile=32, real_kernels=1.0, kernel_ratios=(1.1, 3.0))
    example = sm.kernel_example(bank, 0, np.random.default_rng(2), real, identity=False)
    assert example.details["real"] and example.ratio < 1
    plain = sm.kernel_example(bank, 0, np.random.default_rng(2), sm.Settings(tile=32), identity=False)
    assert not plain.details["real"]


def crossed(side: int, centre=None) -> np.ndarray:
    """A round star with a cross of four arms along the axes, as a stamp."""
    middle = (side - 1) / 2
    core = gaussian(side, np.diag([1.6**2] * 2), centre)
    arms = gaussian(side, np.diag([6.0**2, 0.8**2]), centre) + gaussian(side, np.diag([0.8**2, 6.0**2]), centre)
    del middle
    return (0.8 * core + 0.1 * arms).astype(np.float32)


def arms_over_diagonals(stamp: np.ndarray) -> float:
    middle = (stamp.shape[0] - 1) // 2
    step = np.arange(6, 10)
    between = np.mean([stamp[middle + step, middle + step], stamp[middle - step, middle - step],
                       stamp[middle + step, middle - step], stamp[middle - step, middle + step]], axis=0)
    # Along the axes at the diagonals' own radii, between whole pixels.
    whole = np.arange(0, 18)
    arm = np.mean([stamp[middle, middle + whole], stamp[middle, middle - whole], stamp[middle + whole, middle],
                   stamp[middle - whole, middle]], axis=0)
    along = np.exp(np.interp(step * math.sqrt(2), whole, np.log(np.maximum(arm, 1e-30))))
    return float(np.mean(along / between))


def test_a_round_target_takes_a_psf_s_cross_off(tmp_path):
    stamp = crossed(sharpen_bank.STAMP)
    assert arms_over_diagonals(stamp) > 5
    # Round in its second moments already: a blur leaves the cross.
    _, plain, _ = sm.rounded(stamp, 1.5)
    assert arms_over_diagonals(plain) > 5
    made, round_stamp = sm.rounding(plain)
    assert arms_over_diagonals(round_stamp) == pytest.approx(1.0, abs=0.15)
    assert float(made.sum()) == pytest.approx(1.0, abs=1e-5)
    # A star of that PSF in an image, through the filter, is the round one.
    star = 1e5 * crossed(97)
    out = sm.blurred_images(star, [made])[0]
    assert float(out.sum()) == pytest.approx(float(star.sum()), rel=1e-3)
    assert arms_over_diagonals(out) < 1.6
    # The settings carry it to the kernel and the example.
    kernel = sm.known_kernel(stamp, np.random.default_rng(0), sm.Settings(round_target=True))
    assert kernel.rounding is not None and sm.known_kernel(stamp, np.random.default_rng(0),
                                                           sm.Settings()).rounding is None
    bank = sm.Bank(kernel_bank(tmp_path), sm.Settings(tile=32))
    example = sm.kernel_example(bank, 0, np.random.default_rng(1), sm.Settings(tile=32, round_target=True),
                                identity=False)
    assert example.ratio < 1 and not example.valid[:sm.ROUNDING_BAND].any()


def test_a_reblur_window_is_held_to_give_itself_back_when_blurred_again(tmp_path):
    deep = kernel_bank(tmp_path)
    np.save(deep / "image_clean.npy", np.load(deep / "image.npy"))
    (deep / "clean.json").write_text(json.dumps({"noise_run": "n"}))
    bank = sm.Bank(deep, sm.Settings(tile=32))
    settings = sm.Settings(tile=32, reblur=1.0)
    example = sm.kernel_example(bank, 0, np.random.default_rng(0), settings)
    assert example.kind == "reblur" and example.reblur > 0 and np.array_equal(example.image, example.target)
    assert sm.REBLUR_RATIOS[0] <= example.ratio <= sm.REBLUR_RATIOS[1]
    # Evaluation's windows, asked for a kernel or an identity, are never these.
    assert sm.kernel_example(bank, 0, np.random.default_rng(0), settings, identity=False).kind == "kernel"
    arrays = sm.arrays_of(example, np.random.default_rng(1), settings)
    assert arrays["reblur"] == np.float32(example.reblur) and not arrays["labels"].any()
    assert not arrays["valid"][0, 0].any()
    # The loss: an output that is the input sharpened by the very blur
    # asked back is right, and the input itself, blurred again, is wrong.
    covariance = np.diag([1.3**2] * 2)
    sharp = stars_image(POSITIONS, FLUXES, covariance) - 100.0
    variance = 1.5
    wide = sm.blurred_images(sharp, [variance * np.eye(2)], pad=0)[0]

    def loss(output):
        batch = {"reference": torch.from_numpy(wide)[None, None], "target": torch.from_numpy(wide)[None, None],
                 "sigma": torch.ones(1, 1, SIDE, SIDE), "valid": torch.from_numpy(
                     sm.edge_band((SIDE, SIDE), 8).astype(np.float32))[None, None],
                 "labels": torch.zeros(1, SIDE, SIDE, dtype=torch.int64), "reblur": torch.tensor([variance])}
        residual = torch.from_numpy(output - wide)[None, None]
        return float(sm.loss_terms(residual, batch, sm.LossSettings(), "linear")["total"])

    assert loss(sharp) < 0.05 * loss(wide)


def test_a_model_with_two_ratios_reads_five_planes_and_sharpens_with_a_pair():
    model = sm.build_model(4, 3, sm.PLANES_NONSTELLAR)
    assert model.first[0].in_channels == 5
    image = stars_image(POSITIONS, FLUXES, np.diag([2.0, 2.0]))
    stamp = gaussian(sharpen_bank.STAMP, np.diag([2.0, 2.0]))
    inference = {"tile": 64, "overlap": 2 * reach(3) + 4, "margin": reach(3), "multiple": model.multiple}
    output = sm.sharpen(model, image, lambda x, y: stamp, ratio=(0.6, 0.5), points=detections(POSITIONS, FLUXES),
                        noise_map=np.ones_like(image), background=100.0, noise=1.0, inference=inference)
    # A new model is the identity.
    np.testing.assert_allclose(output, image, atol=1e-3)


def test_where_one_frame_covers_the_sky_the_image_s_own_map_weighs_the_loss():
    image = np.random.default_rng(0).normal(100.0, 2.0, (SIDE, SIDE)).astype(np.float32)
    sigma = np.full((SIDE, SIDE), 2.0, np.float32)
    sigma[:, :20] = 0.0
    found = sm.stack_noise(sigma, image)
    assert np.array_equal(found[:, 20:], sigma[:, 20:])
    assert np.median(found[:, :20]) == pytest.approx(2.0, rel=0.3)
    assert np.array_equal(sm.stack_noise(sigma, image, np.full_like(sigma, 7.0))[:, :20], np.full((SIDE, 20), 7.0))


def test_a_pair_s_target_is_moved_by_its_centroid_offset(tmp_path):
    bank = sm.Bank(pair_bank(tmp_path, offset=(0.6, -0.4)), sm.Settings())
    stars = np.array(POSITIONS)

    def centroids(image):
        found = []
        for x, y in stars:
            column, row = int(round(x)), int(round(y))
            box = image[row - 6:row + 7, column - 6:column + 7].astype(np.float64) - 100.0
            down, across = np.mgrid[row - 6:row + 7, column - 6:column + 7]
            found.append([(box * across).sum() / box.sum(), (box * down).sum() / box.sum()])
        return np.array(found)

    moved = sm.pair_example(bank, 0, sm.Settings(centroid_offset="detection"))
    left = sm.pair_example(bank, 0, sm.Settings(centroid_offset="none"))
    assert np.abs(centroids(left.target) - stars - [0.6, -0.4]).max() < 0.05
    assert np.abs(centroids(moved.target) - stars).max() < 0.05
    assert np.abs(centroids(moved.image) - stars).max() < 0.05
    # The few pixels the shift reads past the edge are out of the loss.
    assert not moved.valid[0].any() and left.valid[0].all()
    # A run recorded before the light offset named the detections' "shift".
    assert sm.Settings(centroid_offset="shift").centroid_offset == "detection"


def lopsided_bank(tmp_path, *, windows: int = 10, wing=(2.0, 0.0), detection=(0.1, 0.0)):
    """A seeing bank of one pair whose windows each hold one star at their
    middle: the input's lopsided, a fifth of its light in a wing `wing` px
    off its core, the target's round about the core. Its detections'
    offset is recorded as `detection`."""
    wide, sharp = np.diag([2.0**2] * 2), np.diag([1.4**2] * 2)
    middle = (SIDE - 1) / 2 + 0.3
    rng = np.random.default_rng(7)
    rows = []
    for index in range(windows):
        core = gaussian(SIDE, wide, (middle, middle))
        wing_light = gaussian(SIDE, wide, (middle + wing[0], middle + wing[1]))
        image = 100.0 + 30_000.0 * (0.8 * core + 0.2 * wing_light) + rng.normal(0, 1, (SIDE, SIDE))
        target = 100.0 + 30_000.0 * gaussian(SIDE, sharp, (middle, middle)) + rng.normal(0, 0.7, (SIDE, SIDE))
        rows.append({"input": image.astype(np.float32), "target": target.astype(np.float32),
                     "input_sigma": np.ones((SIDE, SIDE)), "input_map": np.ones((SIDE, SIDE)),
                     "target_sigma": np.full((SIDE, SIDE), 0.7), "saturated": np.zeros((SIDE, SIDE)),
                     "input_stamp": gaussian(sharpen_bank.STAMP, wide),
                     "target_stamp": gaussian(sharpen_bank.STAMP, sharp),
                     # Each window at its own place, as a bank's are.
                     "info": {"source": 0, "x": 400.0 * index, "y": 300.0 * (index % 3), "background": 100.0,
                              "noise": 1.0, "ratio": 0.82, "width_ratio": 0.7, "input_width": 2.3548 * 2.0,
                              "target_width": 2.3548 * 1.4}})
    points = detections([(middle + 400.0 * index, middle + 300.0 * (index % 3)) for index in range(windows)],
                        [30_000.0] * windows)
    sources = [{"input": "p/seeing/worst", "target": "p/seeing/best", "set": "p", "telescope": "Test Scope",
                "record": {"pair": "p/seeing"}, "alignment": {"matched": windows, "offset_px": list(detection)},
                "windows": [0, windows], "points": points}]
    return write_bank(tmp_path / "sharpen-seeing-train-0", pairs=True, windows=rows, sources=sources, label="seeing")


def test_a_pair_s_target_is_moved_so_its_stars_light_meets_the_input_s(tmp_path):
    directory = lopsided_bank(tmp_path)
    bank = sm.Bank(directory, sm.Settings())
    # The wing pulls the input's light 0.4 px from the core the target is
    # round about: the target moves by -0.4 px, by the light's measure.
    assert bank.light[0]["kind"] == "median" and bank.light[0]["stars"] == 10
    assert bank.target_offset(0, "light") == pytest.approx((-0.4, 0.0), abs=0.03)
    assert bank.target_offset(0, "detection") == (0.1, 0.0)
    assert bank.summary()["light_offsets"] == {"median": 1, "detection": 0}
    example = sm.pair_example(bank, 0, sm.Settings())
    star = (SIDE - 1) / 2 + 0.3
    before = sm.light_centre(example.image, star, star, sm.LIGHT_RADIUS * example.width)
    after = sm.light_centre(example.target, star, star, sm.LIGHT_RADIUS * example.width)
    assert after == pytest.approx(before, abs=0.03)
    # Measured once, then read from beside the bank.
    assert (directory / f"light-offsets-{sm.LIGHT_VERSION}.json").exists()
    assert sm.Bank(directory, sm.Settings()).light == bank.light
    # Too few stars for a light offset: the detections' stands in.
    few = sm.Bank(lopsided_bank(tmp_path / "few", windows=sm.LIGHT_LEAST - 1), sm.Settings())
    assert few.light == {} and few.target_offset(0, "light") == (0.1, 0.0)


def test_a_light_offset_over_enough_stars_is_a_plane_over_the_field():
    rng = np.random.default_rng(2)
    x, y = rng.uniform(0, 6000, 200), rng.uniform(0, 4000, 200)
    dx = 0.1 + 0.2 * (x - 3000) / 1000 + rng.normal(0, 0.05, 200)
    dy = -0.5 + 0.1 * (y - 2000) / 1000 + rng.normal(0, 0.05, 200)
    dy[:10] = 5.0
    fit = sm.fitted_offsets(np.column_stack([x, y, dx, dy]))
    assert fit["kind"] == "plane"
    at = lambda c, px, py: c[0] + c[1] * (px - fit["centre"][0]) / 1000 + c[2] * (py - fit["centre"][1]) / 1000  # noqa: E731
    assert at(fit["dx"], 5000, 1000) == pytest.approx(0.5, abs=0.03)
    # The stray stars are clipped.
    assert at(fit["dy"], 3000, 4000) == pytest.approx(-0.3, abs=0.03)


def test_stamps_out_of_focus_are_found_by_their_rings_and_flat_tops():
    side = sharpen_bank.STAMP
    middle = (side - 1) / 2
    down, across = np.mgrid[0:side, 0:side] - middle
    distance = np.hypot(across, down)
    ring = np.exp(-0.5 * ((distance - 5.0) / 1.5) ** 2)
    disc = 1.0 / (1.0 + np.exp((distance - 7.0) / 0.4))
    stamps = [ring, disc, np.zeros((side, side))]
    stamps += [gaussian(side, np.diag([s**2, s**2])) for s in (0.6, 1.0, 2.0, 5.0)]
    stamps += [moffat(side, fwhm, beta) for fwhm in (1.5, 3.0, 13.0) for beta in (2.5, 4.0)]
    stamps = np.stack([stamp / max(stamp.sum(), 1e-12) for stamp in stamps]).astype(np.float32)
    widths = np.array([sharpen_bank.stamp_width(stamp) for stamp in stamps])
    found = sm.out_of_focus(stamps, widths)
    assert list(found[:3]) == [True, True, True]
    assert not found[3:].any()


def test_a_bank_leaves_out_windows_out_of_focus_and_sources_mostly_so(tmp_path):
    directory = pair_bank(tmp_path, ratios=[0.7] * 8)
    side = sharpen_bank.STAMP
    down, across = np.mgrid[0:side, 0:side] - (side - 1) / 2
    ring = np.exp(-0.5 * ((np.hypot(across, down) - 5.0) / 1.5) ** 2).astype(np.float32)
    stamps = np.load(directory / "input_stamp.npy")
    # One of pair p's four windows, and three of pair q's.
    for window in (1, 4, 5, 6):
        stamps[window] = ring / ring.sum()
    np.save(directory / "input_stamp.npy", stamps)
    bank = sm.Bank(directory, sm.Settings())
    assert list(bank.kept) == [0, 2, 3]
    assert bank.left == {"out-of-focus-source": 4, "out-of-focus": 1}
    assert len(sm.Bank(directory, sm.Settings(), focus_check=False).kept) == 8


def test_windows_are_kept_by_their_ratio_and_pairs_left_out_by_id(tmp_path):
    directory = pair_bank(tmp_path, ratios=[0.5, 1.05, None, 0.9, 0.95, 0.7])
    kept = sm.Bank(directory, sm.Settings())
    assert list(kept.kept) == [0, 3, 4, 5]
    assert kept.left == {"ratio-above": 1, "ratio-unknown": 1}
    narrow = sm.Bank(directory, sm.Settings(max_ratio=0.92, min_ratio=0.6))
    assert list(narrow.kept) == [3, 5] and narrow.left["ratio-below"] == 1
    # The fit's ratio reads its own column, 0.12 above the stamps'.
    assert list(sm.Bank(directory, sm.Settings(ratio_source="fit")).kept) == [0, 5]
    # A pair is left out by its id, its input's or its target's.
    for name in ("q/seeing", "q/seeing/worst", "q/seeing/best"):
        bank = sm.Bank(directory, sm.Settings(), excluded=(name,))
        assert list(bank.kept) == [0] and bank.left["excluded"] == 3
    with pytest.raises(ValueError, match="keeps no window"):
        sm.Bank(directory, sm.Settings(), excluded=("p/seeing", "q/seeing"))
    # No pair is left out by name unless asked: focus is measured.
    assert sm.BankChoice("k", (), None, ()).excluded == ()


def test_a_bank_leaves_out_kinds_and_gives_the_stacks_of_many_nights_their_share(tmp_path):
    covariance = np.diag([1.3**2] * 2)
    stamp = gaussian(sharpen_bank.STAMP, covariance)
    windows = [{"image": stars_image(POSITIONS, FLUXES, covariance), "sigma": np.ones((SIDE, SIDE)),
                "saturated": np.zeros((SIDE, SIDE)), "stamp": stamp,
                "info": {"source": index // 3, "background": 100.0, "noise": 1.0,
                         "width": sharpen_bank.stamp_width(stamp)}} for index in range(6)]
    sources = [{"input": name, "input_kind": kind, "set": name.split("/")[0], "record": {},
                "points": detections(POSITIONS, FLUXES)}
               for name, kind in (("a/deep", "deep"), ("multi-night/b/stack", "multi-night"))]
    directory = write_bank(tmp_path / "sharpen-deep-train-0", pairs=False, windows=windows, sources=sources,
                           label="deep")
    assert sm.Bank(directory, sm.Settings()).multi_night_share == pytest.approx(0.5)
    weighted = sm.Bank(directory, sm.Settings(), multi_night_share=0.8)
    assert weighted.multi_night_share == pytest.approx(0.8) and weighted.odds[3:].sum() == pytest.approx(0.8)
    assert weighted.summary()["multi_night_share"] == pytest.approx(0.8)
    left = sm.Bank(directory, sm.Settings(), left_out_kinds=("deep",))
    assert list(left.kept) == [3, 4, 5] and left.left == {"kind:deep": 3}


def test_worker_processes_hand_out_the_batches_the_loop_would_build(tmp_path):
    choice = sm.BankChoice(kernel=str(kernel_bank(tmp_path)), seeing=(str(pair_bank(tmp_path)),),
                           validation_kernel=None, validation_seeing=())
    settings = sm.Settings(tile=32)
    mix = (0.6, 0.4, 0.0)
    sources, shares = sm.mixture(choice.train(settings), mix)
    workers = sm.Workers(choice, settings, mix, 3, 7, depth=2, processes=2, start=4)
    try:
        for step in (5, 6, 7):
            built = workers.get()
            expected = sm.batch_from(sources, shares, 3, sm.step_generator(7, step), settings)
            assert set(built) == set(expected)
            for name, value in expected.items():
                assert built[name].dtype == value.dtype, name
                assert torch.equal(built[name], value), name
    finally:
        workers.close()


def test_no_object_takes_more_than_its_cap_of_the_draws():
    groups = np.array(["a"] * 60 + ["b"] * 20 + ["c"] * 10 + ["d"] * 10)
    odds = sm.capped_odds(groups, 0.4)
    share = {name: float(odds[groups == name].sum()) for name in "abcd"}
    assert share["a"] == pytest.approx(0.4)
    # The rest go to the others by their windows.
    assert share["b"] == pytest.approx(0.3) and share["c"] == pytest.approx(0.15)
    assert odds.sum() == pytest.approx(1.0)
    # Within an object its windows are alike.
    assert np.ptp(odds[groups == "a"]) == 0
    # Fewer objects than the cap allows: as even as they can be.
    assert sm.capped_odds(groups, 0.1)[groups == "a"].sum() == pytest.approx(0.25)
    assert np.allclose(sm.capped_odds(groups, 0.0), 1 / len(groups))
    assert sm.object_name("Scope/M 27/2026-07-06/HA-300s-g100-bin1") == "Scope/M 27/HA-300s-g100-bin1"


def test_flips_and_turns_move_the_stamp_maps_and_apertures_with_the_image():
    # An elongated star, tilted, and a stamp of the same shape: whatever the
    # flip and turn, the star in the image and the stamp agree.
    covariance = np.array([[4.0, 1.5], [1.5, 2.0]])
    image = stars_image([(31.0, 30.0)], [1e5], covariance)
    stamp = gaussian(sharpen_bank.STAMP, covariance)
    valid = np.ones((SIDE, SIDE), dtype=bool)
    valid[:3] = False
    # The target is the input plus 1, plus 2 where the loss may not look.
    target = image + np.where(valid, 1.0, 2.0).astype(np.float32)
    example = sm.Example(image, target, np.ones((SIDE, SIDE), np.float32), np.ones((SIDE, SIDE), np.float32),
                         valid, stamp, 0.7, detections([(31.0, 30.0)], [1e5]), 100.0, 1.0, 2.0, "kernel", 0, 0)
    settings = sm.Settings(tile=SIDE, points_empty=0.0)
    seen = set()
    for seed in range(40):
        arrays = sm.arrays_of(example, np.random.default_rng(seed), settings)
        star = np.clip(arrays["reference"][0], 0, None)
        _, star_covariance = sm.moments(np.pad(star, ((0, 1), (0, 1))))
        _, stamp_covariance = sm.moments(arrays["stamp"][0])
        assert star_covariance == pytest.approx(stamp_covariance, rel=0.02, abs=0.02)
        seen.add((round(star_covariance[0, 0], 1), round(star_covariance[0, 1], 1)))
        # The point-source map, the target, the masks and the apertures
        # turn with it.
        assert np.unravel_index(np.argmax(arrays["planes"][3]), star.shape) == np.unravel_index(np.argmax(star),
                                                                                                star.shape)
        assert np.allclose(arrays["target"] - arrays["reference"], 2.0 - arrays["valid"], atol=1e-4)
        assert arrays["labels"][np.unravel_index(np.argmax(star), star.shape)] == 1
        assert arrays["valid"][0].sum() == valid.sum()
    # All four shapes a turn or flip gives were met.
    assert len(seen) == 4


def test_the_network_reads_planes_and_a_stamp_and_exports_only_the_allowed_operations(tmp_path):
    import onnx
    import onnxruntime

    torch.manual_seed(0)
    model = SharpeningUNet(width=8, levels=3).eval()
    # Weights as a trained model's, so no two layers share theirs.
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            torch.nn.init.normal_(layer.weight, std=0.1)
            torch.nn.init.normal_(layer.bias, std=0.1)
    planes = torch.randn(2, len(sm.PLANES), 48, 64)
    stamps = torch.rand(2, len(sm.STAMP_PLANES), sharpen_bank.STAMP, sharpen_bank.STAMP)
    with torch.no_grad():
        out = model(planes, stamps)
        assert out.shape == (2, 1, 48, 64)
        assert model.encoder(stamps).shape == (2, 64, 1, 1)
        # The stamp changes the output: FiLM reads it.
        assert not torch.allclose(out, model(planes, stamps.flip(-1)))
    path = tmp_path / "model.onnx"
    exported = sm.export(model, path, tile=64)
    assert exported["max_relative_error"] < 1e-4
    operations = {node.op_type for node in onnx.load(str(path)).graph.node}
    assert operations <= {"Conv", "Sigmoid", "Mul", "Add", "Concat", "Resize", "Constant"}
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [tensor.name for tensor in session.get_inputs()] == ["planes", "stamp"]
    # An untrained model is the identity: its residual is zero.
    assert not SharpeningUNet(width=4, levels=2)(planes, stamps).any()


def test_tiles_with_their_own_stamp_and_points_reproduce_the_whole_image():
    torch.manual_seed(1)
    model = SharpeningUNet(width=4, levels=3).eval()
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            torch.nn.init.normal_(layer.weight, std=0.3)
    # Sides whose last tile starts on the model's grid: nothing is padded.
    image = stars_image([(40.5, 60.2), (150.3, 90.8), (100.0, 190.4), (65.7, 128.1)], [5e4, 3e4, 2e4, 1e4],
                        np.diag([2.0, 2.0]), side=200)[:, :]
    image = np.concatenate([image, image[:, :32]], axis=1)
    noise_map = np.ones_like(image)
    stamp = gaussian(sharpen_bank.STAMP, np.diag([2.0, 2.0]))
    points = detections([(40.5, 60.2), (150.3, 90.8), (100.0, 190.4), (65.7, 128.1)], [5e4, 3e4, 2e4, 1e4])
    inference = {"tile": 64, "overlap": 2 * reach(3) + 4, "margin": reach(3), "multiple": model.multiple}
    tiled = sm.sharpen(model, image, lambda x, y: stamp, ratio=0.6, points=points, noise_map=noise_map,
                       background=100.0, noise=1.0, inference=inference, residual="linear")
    rendered = sm.point_map(points, stamp, image.shape, 100.0)
    planes = sm.network_planes(image, noise_map, 0.6, rendered, 100.0, 1.0)
    with torch.no_grad():
        whole = model(torch.from_numpy(planes)[None], torch.from_numpy(sm.stamp_planes(stamp))[None]).numpy()[0, 0]
    np.testing.assert_allclose(tiled - image, whole, atol=1e-4 * np.abs(whole).max())
    # Through asinh, the blended residual is added to the image's asinh.
    curved = sm.sharpen(model, image, lambda x, y: stamp, ratio=0.6, points=points, noise_map=noise_map,
                        background=100.0, noise=1.0, inference=inference)
    np.testing.assert_allclose(curved, 100.0 + np.sinh(planes[0] + whole), rtol=1e-4, atol=1e-3)
    # A stamp that changes across the field changes each tile's output.
    other = gaussian(sharpen_bank.STAMP, np.diag([6.0, 3.0]))
    varied = sm.sharpen(model, image, lambda x, y: stamp if x < 100 else other, ratio=0.6, points=points,
                        noise_map=noise_map, background=100.0, noise=1.0, inference=inference, residual="linear")
    assert np.abs(varied - tiled)[:, :40].max() < 1e-4 * np.abs(whole).max()
    assert np.abs(varied - tiled)[:, -40:].max() > 1e-2 * np.abs(whole).max()


def test_a_field_s_vectors_over_every_pixel_leave_the_output_where_the_tiles_fall():
    torch.manual_seed(2)
    model = SharpeningUNet(width=4, levels=3).eval()
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            torch.nn.init.normal_(layer.weight, std=0.3)
    stars = [(40.5, 60.2), (150.3, 90.8), (100.0, 190.4), (65.7, 128.1), (200.2, 40.6)]
    image = stars_image(stars, [5e4, 3e4, 2e4, 1e4, 4e4], np.diag([2.0, 2.0]), side=232)
    noise_map = np.ones_like(image)
    points = detections(stars, [5e4, 3e4, 2e4, 1e4, 4e4])
    inference = {"tile": 64, "overlap": 2 * reach(3) + 4, "margin": reach(3), "multiple": model.multiple}
    ys = xs = np.array([0.0, 115.5, 231.0])
    stamps = np.stack([np.stack([gaussian(sharpen_bank.STAMP, np.diag([1.5 + row + column, 2.0 + row]))
                                 for column in range(3)]) for row in range(3)])
    common = dict(ratio=0.6, points=points, noise_map=noise_map, background=100.0, noise=1.0,
                  inference=inference, residual="linear")
    whole = sm.sharpen(model, image, None, grid=(ys, xs, stamps), **common)
    # The same image from a corner 24 px in, on the model's grid: its tiles
    # fall elsewhere, and inside the network's reach of the cut, its output
    # is the whole image's.
    shift = 24
    part = sm.sharpen(model, np.ascontiguousarray(image[shift:, shift:]), None,
                      grid=(ys - shift, xs - shift, stamps),
                      **{**common, "points": points - np.array([shift, shift] + [0] * (points.shape[1] - 2),
                                                               dtype=np.float32),
                         "noise_map": noise_map[shift:, shift:]})
    inner = (slice(inference["margin"], None), slice(inference["margin"], None))
    change = np.abs(whole - image).max()
    assert np.abs(part[inner] - whole[shift:, shift:][inner]).max() < 1e-4 * change
    # The field changes the output: one stamp everywhere is another image.
    flat = sm.sharpen(model, image, None, grid=(ys, xs, np.broadcast_to(stamps[0, 0], stamps.shape)), **common)
    assert np.abs(flat - whole).max() > 1e-2 * change
    # One stamp everywhere is what tiles conditioned on that stamp give.
    tiled = sm.sharpen(model, image, lambda x, y: stamps[0, 0], **common)
    np.testing.assert_allclose(flat, tiled, atol=1e-4 * change)


def test_the_loss_is_zero_on_the_target_and_blind_where_it_may_not_look():
    rng = np.random.default_rng(0)
    batch = {"reference": torch.from_numpy(rng.normal(size=(2, 1, 32, 32)).astype(np.float32)),
             "target": torch.from_numpy(rng.normal(size=(2, 1, 32, 32)).astype(np.float32)),
             "sigma": torch.ones(2, 1, 32, 32), "valid": torch.ones(2, 1, 32, 32),
             "labels": torch.zeros(2, 32, 32, dtype=torch.int64)}
    batch["labels"][0, 10:15, 10:15] = 1
    exact = batch["target"] - batch["reference"]
    terms = sm.loss_terms(exact, batch, sm.LossSettings(), "linear")
    assert float(terms["total"]) == pytest.approx(0.0, abs=1e-6)
    # Through asinh, the residual that reaches the target is the difference
    # of the two asinh.
    curved = torch.asinh(batch["target"]) - torch.asinh(batch["reference"])
    assert float(sm.loss_terms(curved, batch, sm.LossSettings())["total"]) == pytest.approx(0.0, abs=1e-5)
    # An error where the loss may not look costs nothing.
    batch["valid"][:, :, :4] = 0
    off = exact.clone()
    off[:, :, :4] += 1000.0
    assert float(sm.loss_terms(off, batch, sm.LossSettings(), "linear")["total"]) == pytest.approx(0.0, abs=1e-6)
    # Flux taken from inside an aperture costs the flux term.
    dimmed = exact.clone()
    dimmed[0, 0, 10:15, 10:15] -= 0.5
    terms = sm.loss_terms(dimmed, batch, sm.LossSettings(), "linear")
    # The star's error counts once for each of its aperture's 25 pixels, out
    # of the valid pixels, as a pixel's error does.
    valid = float(batch["valid"].sum())
    assert float(terms["flux"]) == pytest.approx((math.sqrt(2.5**2 + 0.1**2) - 0.1) * 25 / valid, rel=1e-4)
    assert float(terms["gradient"]) > 0


def test_a_resumed_run_is_the_run_it_continues(tmp_path):
    store = Store(tmp_path / "store", tmp_path / "scratch")
    choice = sm.BankChoice(kernel=str(kernel_bank(tmp_path)), seeing=(str(pair_bank(tmp_path)),),
                           validation_kernel=None, validation_seeing=(str(pair_bank(
                               tmp_path, name="sharpen-seeing-validation-0")),), excluded=())
    options = {"choice": choice, "steps": 6, "width": 4, "levels": 2, "batch": 2, "log_every": 2,
               "checkpoint_every": 2, "settings": sm.Settings(tile=32), "validation_batches": 1,
               "validation_windows": 2, "prefetch": 2, "batch_threads": 2, "where": torch.device("cpu")}
    whole = sm.train(store, name="whole", **options)
    stopped = sm.train(store, name="parts", stop_after=4, **options)
    assert stopped["stopped_after"] == 4 and len(stopped["history"]) == 2
    assert not (sm.run_directory(store, "parts") / "run.json").exists()
    with pytest.raises(FileExistsError):
        sm.train(store, name="parts", **options)
    resumed = sm.resume_run(store, "parts", prefetch=2, batch_threads=2, where=torch.device("cpu"))
    for entry in (*whole["history"], *resumed["history"]):
        entry.pop("seconds")
    assert resumed["history"] == whole["history"]
    first = torch.load(sm.run_directory(store, "whole") / "weights.pt")
    second = torch.load(sm.run_directory(store, "parts") / "weights.pt")
    assert all(torch.equal(first[key], second[key]) for key in first)
    # The run is recorded with its measures, and its checkpoint is gone.
    record = json.loads((sm.run_directory(store, "parts") / "run.json").read_text())
    assert record["validation"]["seeing:seeing"]["windows"] == 2
    assert record["config"]["choice"]["kernel"] == choice.kernel
    assert not (sm.run_directory(store, "parts") / "checkpoint.pt").exists()
    _, model = sm.load_run(store, "parts")
    assert isinstance(model, SharpeningUNet)


def test_the_commands_train_stop_resume_and_evaluate_from_the_banks_they_find(tmp_path, capsys):
    from seiza_training.__main__ import main

    work = tmp_path / "banks"
    work.mkdir()
    kernel_bank(work)
    kernel_bank(work, name="sharpen-deep-validation-0")
    pair_bank(work)
    pair_bank(work, name="sharpen-seeing-validation-0")
    common = ["--store", str(tmp_path / "store"), "--scratch", str(tmp_path / "scratch")]

    def run(*arguments):
        capsys.readouterr()
        assert main([*common, *arguments]) == 0
        # Training logs its windows first; the record follows.
        out = capsys.readouterr().out
        return json.loads(out if out.startswith("{") else out[out.index("\n{") + 1:])

    stopped = run("train-sharpen", "--name", "cli", "--work", str(work), "--steps", "4", "--width", "4",
                  "--levels", "2", "--tile", "32", "--batch", "2", "--log-every", "2", "--checkpoint-every", "2",
                  "--stop-after", "2", "--exclude-pairs", "", "--validation-windows", "2", "--validation-batches", "1",
                  "--prefetch", "0")
    assert stopped["stopped_after"] == 2
    record = run("train-sharpen", "--name", "cli", "--resume", "--prefetch", "0")
    assert [entry["step"] for entry in record["history"]] == [2, 4]
    assert record["config"]["choice"]["kernel"] == str(work / "sharpen-deep-train-0")
    assert record["config"]["choice"]["excluded"] == []
    assert "windows_detail" not in record["validation"]["kernel:deep"]
    measured = run("evaluate-sharpen", "--name", "cli", "--work", str(work), "--limit", "3")
    assert measured["kernel:deep"]["windows"] == 3 and measured["kernel:deep"]["identity"]["windows"] == 1
    saved = json.loads((tmp_path / "store" / "models" / "sharpen" / "cli" / "run.json").read_text())
    assert list(saved["evaluations"]) == ["kernel:deep, seeing:seeing, limit 3"]


def test_the_edge_term_counts_falling_short_only():
    import torch

    from seiza_training.sharpen_model import edge_shortfall

    ramp = torch.linspace(0.0, 64.0, 64).view(1, 1, 1, 64).expand(1, 1, 64, 64).contiguous()
    where = torch.ones_like(ramp)
    assert float(edge_shortfall(ramp, ramp, where)) < 1e-6
    assert float(edge_shortfall(1.5 * ramp, ramp, where)) < 1e-6
    assert abs(float(edge_shortfall(0.5 * ramp, ramp, where)) - 0.5) < 0.02

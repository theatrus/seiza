"""Parallax fly-through videos, from a small synthetic solved field.

Fetching online is off and no star distance file is named, so the video
warns and places every star at one distance; the test needs no network.
"""

import numpy as np
import pytest
import seiza


def field():
    """A smooth starless image, three Gaussian stars, and their WCS."""
    height, width = 240, 320
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    starless = np.stack(
        [0.1 + 0.2 * x / width, np.full_like(x, 0.15), 0.1 + 0.1 * y / height], axis=-1
    ).astype(np.float32)
    light = np.zeros((height, width), np.float32)
    for sx, sy in [(60, 50), (250, 70), (140, 180)]:
        light += 0.9 * np.exp(-((x - sx) ** 2 + (y - sy) ** 2) / 4.0)
    stars = np.repeat(np.clip(light, 0, 1)[..., None], 3, axis=-1).astype(np.float32)
    wcs = seiza.Wcs.from_center_scale_rotation((56.75, 24.12), (160.0, 120.0), 2.0)
    return starless, stars, wcs


def video(tmp_path, **options):
    starless, stars, wcs = field()
    events = []
    made = seiza.ParallaxVideo(
        starless,
        stars,
        wcs,
        progress=events.append,
        distance_pc=400.0,
        online=False,
        objects=str(tmp_path / "no-objects.bin"),
        size=(160, 120),
        seconds=1.0,
        fps=5,
        labels=[(100.0, 90.0, 15.0, "here")],
        watermark=True,
        **options,
    )
    return made, events


def test_a_video_is_prepared_and_draws_its_frames(tmp_path):
    made, events = video(tmp_path)
    assert made.frame_count == 5 and len(made) == 5
    assert made.fps == 5
    assert made.size == (160, 120)
    summary = made.summary
    assert summary["detected_stars"] == 3
    assert summary["background_distance_pc"] == 400.0
    if summary["gaia_stars"] == 0:
        assert any(event["kind"] == "warning" for event in events), events
    rgb = made.frame(2)
    assert rgb.shape == (120, 160, 3) and rgb.dtype == np.uint8
    bgra = made.frame(2, "bgra")
    assert bgra.shape == (120, 160, 4)
    assert (bgra[..., 2] == rgb[..., 0]).all() and (bgra[..., 0] == rgb[..., 2]).all()
    assert (bgra[..., 3] == 255).all()
    frames = list(made)
    assert len(frames) == 5 and all(frame.shape == (120, 160, 3) for frame in frames)
    assert (frames[2] == rgb).all()
    with pytest.raises(ValueError):
        made.frame(5)


def test_frames_go_to_a_callback_that_can_stop_them(tmp_path):
    made, _ = video(tmp_path)
    seen = []
    assert made.render(lambda frame, index: seen.append((index, frame.shape)))
    assert [index for index, _ in seen] == [0, 1, 2, 3, 4]
    seen.clear()
    assert not made.render(lambda frame, index: seen.append(index) or index < 1)
    assert seen == [0, 1]
    progress = []
    assert not made.render(lambda frame, index: None, progress=progress.append, cancel=lambda: len(progress) == 3)
    assert len(progress) == 3


def test_a_video_is_written_as_png_frames(tmp_path):
    made, _ = video(tmp_path)
    output = tmp_path / "frames"
    assert made.write(output, encoder="png")
    assert len(list(output.iterdir())) == 5


def test_uint8_images_and_named_sizes_are_accepted(tmp_path):
    starless, stars, wcs = field()
    made = seiza.ParallaxVideo(
        (starless * 255).astype(np.uint8),
        (stars * 255).astype(np.uint8),
        wcs,
        distance_pc=400.0,
        online=False,
        objects=str(tmp_path / "no-objects.bin"),
        size="720p-portrait",
        seconds=0.2,
        fps=10,
    )
    assert made.size == (720, 1280)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"dollly": 0.5}, "unknown parallax video option"),
        ({"size": "8k"}, "WIDTHxHEIGHT"),
        ({"start": "middle"}, "start must be one of"),
        ({"dolly": 1.5}, "dolly"),
    ],
)
def test_bad_options_are_refused(tmp_path, options, message):
    starless, stars, wcs = field()
    with pytest.raises(ValueError, match=message):
        seiza.ParallaxVideo(starless, stars, wcs, online=False, **options)

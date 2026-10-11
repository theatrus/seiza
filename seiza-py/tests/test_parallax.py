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


def test_a_tour_sets_the_length_and_takes_strings_or_dicts(tmp_path):
    made, _ = video(
        tmp_path,
        tour=[
            "whole hold=1",
            {"focus": (150.0, 110.0), "dolly": 0.5, "rotate_deg": 10.0, "travel": 1.0, "hold": 1.0},
            "whole travel=1",
        ],
    )
    assert made.frame_count == 20
    with pytest.raises(ValueError, match="unknown tour stop field"):
        video(tmp_path, tour=["whole", {"dolli": 0.5}])


def test_a_tour_plan_comes_back_editable_and_goes_back_in(tmp_path):
    # With no object catalog there is nothing to tour.
    _, _, wcs = field()
    with pytest.raises(RuntimeError, match="object catalog|catalogued objects"):
        seiza.plan_parallax_tour(320, 240, wcs, objects=str(tmp_path / "no-objects.bin"))
    # A plan's stops, names and all, are accepted as a tour.
    made, _ = video(
        tmp_path,
        tour=[
            {"name": None, "focus": None, "hold": 1.0, "title": None},
            {
                "name": "NGC 9001",
                "focus": (150.0, 110.0),
                "dolly": 0.5,
                "travel": 1.0,
                "hold": 1.0,
                "title": "NGC 9001",
            },
            {"name": None, "focus": None, "travel": 1.0, "title": None},
        ],
    )
    assert made.frame_count == 20


def test_a_titled_tour_shows_its_title_and_a_looped_one_ends_where_it_began(tmp_path):
    stops = [
        "whole hold=1",
        {"focus": (150.0, 110.0), "dolly": 0.5, "travel": 1.0, "hold": 2.0, "title": "Here"},
        "200,60 dolly=0.3 travel=1",
    ]
    plain, _ = video(tmp_path, tour=stops)
    titled, _ = video(tmp_path, tour=stops, tour_titles=True)
    # Holding at the titled stop, from second 2 to 4: the title is drawn
    # low on the left.
    at = 3 * plain.fps
    changed = np.abs(titled.frame(at).astype(int) - plain.frame(at).astype(int)).sum(axis=-1) > 0
    rows, columns = np.nonzero(changed)
    assert rows.size and rows.min() > 120 // 2 and columns.max() < 160 * 0.75
    # Looped, the tour gains a last stop back at the whole image and the
    # frame after the last is the first.
    looped, _ = video(tmp_path, tour=stops, tour_loop=True)
    assert looped.frame_count > plain.frame_count
    first, last = looped.frame(0).astype(int), looped.frame(looped.frame_count - 1).astype(int)
    assert np.abs(first - last).mean() < 2.0
    with pytest.raises(ValueError, match="need a tour"):
        video(tmp_path, tour_loop=True)


def test_a_reconfigured_video_shares_the_scene(tmp_path):
    made, _ = video(tmp_path)
    first_frame = made.frame(2)
    other = made.reconfigure(
        focus=(150.0, 110.0),
        dolly=0.6,
        rotate_deg=(0.0, -360.0),
        size=(200, 150),
        seconds=2.0,
        fps=5,
        quality="high",
    )
    assert (other.frame_count, other.size) == (10, (200, 150))
    assert made.frame_count == 5 and (made.frame(2) == first_frame).all()
    assert other.summary["background_distance_pc"] == made.summary["background_distance_pc"]
    fit = other.summary["fit"]
    assert fit["zoom"][0] == 1.0 and fit["zoom"][1] >= 1.0
    with pytest.raises(ValueError, match="changes the prepared scene"):
        made.reconfigure(distance_pc=500.0)
    del made
    assert other.frame(9).shape == (150, 200, 3)



def test_any_array_layout_and_lists_for_pairs_are_read_as_given(tmp_path):
    starless, stars, wcs = field()
    common = dict(distance_pc=400.0, online=False, objects=str(tmp_path / "none.bin"), fps=5)
    as_c = seiza.ParallaxVideo(
        starless, stars, wcs, size=(160, 120), focus=(150.0, 110.0), seconds=1.0, **common
    )
    # Fortran-ordered arrays, and pairs as lists, as a plan saved to JSON
    # and read back gives them.
    as_lists = seiza.ParallaxVideo(
        np.asfortranarray(starless),
        np.asfortranarray(stars),
        wcs,
        size=[160, 120],
        focus=[150.0, 110.0],
        seconds=1.0,
        **common,
    )
    assert as_lists.summary["detected_stars"] == as_c.summary["detected_stars"]
    assert (as_lists.frame(3) == as_c.frame(3)).all()
    toured = seiza.ParallaxVideo(
        starless,
        stars,
        wcs,
        size=[160, 120],
        rotate_deg=[0.0, 10.0],
        tour=[{"focus": None, "hold": 1.0}, {"focus": [150.0, 110.0], "dolly": 0.5, "travel": 1.0}],
        labels=[[100.0, 90.0, "here"]],
        **common,
    )
    assert toured.frame_count == 10


def test_a_raising_progress_stops_the_work_and_is_raised(tmp_path):
    made, _ = video(tmp_path)
    longer = made.reconfigure(seconds=8.0, fps=5, size=(160, 120))
    output = tmp_path / "frames"

    def refuse(event):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        longer.write(output, encoder="png", progress=refuse)
    assert len(list(output.iterdir())) < longer.frame_count


def test_bad_options_raise_value_errors_and_none_means_the_default(tmp_path):
    made, _ = video(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        video(tmp_path, focus=(1000.0, 1000.0))
    with pytest.raises(ValueError, match="outside"):
        made.reconfigure(focus=(1000.0, 1000.0))
    # A scene option given as None is left at its default, not refused.
    again = made.reconfigure(distance_pc=None, size=(160, 120), seconds=1.0, fps=5)
    assert again.frame_count == made.frame_count

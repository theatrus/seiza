"""Parallax fly-through videos, from a small synthetic solved field.

Fetching online is off and no star distance file is named, so the video
warns and places every star at one distance; the test needs no network.
"""

import struct
import zlib

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
    options.setdefault("watermark", True)
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
        ({"size": "8k-image"}, "-image follows one of"),
        ({"framing": "stretch"}, "framing must be one of"),
        ({"crop": (300, 0, 100, 100)}, "lie inside"),
        ({"start": "middle"}, "start must be one of"),
        ({"dolly": 1.5}, "dolly"),
    ],
)
def test_bad_options_are_refused(tmp_path, options, message):
    starless, stars, wcs = field()
    with pytest.raises(ValueError, match=message):
        seiza.ParallaxVideo(starless, stars, wcs, online=False, **options)


def test_a_crop_films_at_its_own_shape_and_fit_shows_all_of_it(tmp_path):
    starless, stars, wcs = field()
    made = seiza.ParallaxVideo(
        starless,
        stars,
        wcs,
        distance_pc=400.0,
        online=False,
        objects=str(tmp_path / "no-objects.bin"),
        crop=(40, 30, 200, 100),
        size="720p-image",
        start="whole",
        seconds=0.2,
        fps=10,
    )
    # The 2:1 crop at the 720p frame's longer side.
    assert made.size == (1280, 640)
    # The summary gives back the solution of the image as given, so it can
    # go into another ParallaxVideo of the same images, and the crop.
    summary = made.summary
    assert summary["crop"] == (40, 30, 200, 100)
    assert summary["wcs"].crpix == wcs.crpix
    # A square frame fits the 2:1 crop whole, black above and below it.
    square = made.reconfigure(size=(160, 160), framing="fit", start="whole", seconds=0.2, fps=10)
    first = square.frame(0)
    assert first[:20].max() == 0 and first[-20:].max() == 0
    assert first[70:90].min() > 0
    # Filled, the crop covers the frame.
    assert made.reconfigure(size=(160, 160), start="whole", seconds=0.2, fps=10).frame(0)[:20].min() > 0
    with pytest.raises(ValueError, match="changes the prepared scene"):
        made.reconfigure(crop=(0, 0, 100, 100))
    with pytest.raises(ValueError, match="framing must be one of"):
        made.reconfigure(framing="stretch")
    with pytest.raises(ValueError, match="lie inside"):
        seiza.plan_parallax_tour(320, 240, wcs, crop=(300, 0, 100, 100))


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


def test_pulling_back_an_edge_target_keeps_the_opening_unchanged(tmp_path):
    scene, _ = video(tmp_path)
    opening = {"focus": None, "hold": 0.2}
    target = {"focus": (280.0, 40.0), "dolly": 0.35, "rotate_deg": 11.0, "travel": 0.8, "hold": 0.3}
    closing = {"focus": None, "travel": 0.8, "hold": 0.2}
    before = scene.reconfigure(tour=[opening, target, closing], size=(160, 120), fps=10)
    pulled = {**target, "dolly": 0.0, "pull_back_to_whole_image": True}
    after = scene.reconfigure(tour=[opening, pulled, closing], size=(160, 120), fps=10)
    np.testing.assert_array_equal(before.frame(0), after.frame(0))
    assert not np.array_equal(before.frame(10), after.frame(10))
    assert pulled["focus"] == target["focus"]


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


def write_png(path, rgba):
    """Write an 8-bit RGBA array as a PNG, with the standard library alone."""
    height, width, _ = rgba.shape
    rows = b"".join(b"\x00" + rgba[row].tobytes() for row in range(height))

    def chunk(kind, data):
        crc = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )
    return str(path)


def test_cards_and_a_watermark_image_are_laid_over_the_frames(tmp_path):
    blue = np.zeros((120, 160, 4), np.uint8)
    blue[..., 2] = blue[..., 3] = 255
    green = np.zeros((10, 40, 4), np.uint8)
    green[..., 1] = green[..., 3] = 255
    card = write_png(tmp_path / "card.png", blue)
    logo = write_png(tmp_path / "logo.png", green)
    made, events = video(
        tmp_path,
        intro_card=card,
        card_seconds=0.4,
        watermark={"image": logo, "corner": "top_left", "opacity": 1.0},
    )
    # The opening card covers the first frames and is gone by the third;
    # being opaque, it is warned of.
    assert (made.frame(0)[60, 80] == [0, 0, 255]).all()
    assert not (made.frame(3)[60, 80] == [0, 0, 255]).all()
    assert any("transparency" in event["message"] for event in events), events
    # The logo sits a label's height in from the top-left corner, over
    # every frame.
    assert (made.frame(3)[17, 20] == [0, 255, 0]).all()
    with pytest.raises(ValueError, match="missing.png"):
        made.reconfigure(outro_card=str(tmp_path / "missing.png"), size=(160, 120))
    with pytest.raises(ValueError, match="watermark corner"):
        made.reconfigure(watermark={"corner": "middle"}, size=(160, 120))


def test_depth_planes_dust_maps_and_scene_tours_match_the_c_api(tmp_path):
    starless, stars, wcs = field()
    height, width, _ = starless.shape
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    knot = np.exp(-((x - 150) ** 2 + (y - 110) ** 2) / 300.0)
    starless = (starless + 0.5 * knot[..., None]).astype(np.float32)
    mask = (np.hypot(x - 150, y - 110) < 30).astype(np.float32)
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
        planes=[(mask, 1200.0, "knot")],
        auto_planes=False,
    )
    summary = made.summary
    assert [(plane["name"], plane["distance_pc"], plane["basis"]) for plane in summary["planes"]] == [
        ("knot", 1200.0, "given")
    ]
    assert summary["planes"][0]["object"] is None
    assert isinstance(summary["wcs"], seiza.Wcs)
    assert summary["dust_given"] is False
    # A painted dust map swaps in without preparing again, and reads back.
    painted = np.zeros((height, width), np.float32)
    painted[:, : width // 2] = 1.0
    dusty = made.with_dust_map(painted)
    assert dusty.summary["dust_given"] is True
    assert dusty.summary["planes"] == summary["planes"]
    read = dusty.dust_map()
    assert read.shape == (height, width) and read.dtype == np.float32
    assert read[60, 10] > 0.9 and read[60, width - 10] < 0.1
    assert dusty.dust_map(size=(32, 24)).shape == (24, 32)
    assert dusty.write_dust_map(tmp_path / "dust.png")
    assert (tmp_path / "dust.png").stat().st_size > 0
    again = seiza.ParallaxVideo(
        starless, stars, wcs, distance_pc=400.0, online=False,
        objects=str(tmp_path / "no-objects.bin"), size=(160, 120), seconds=1.0, fps=5,
        dust_map=str(tmp_path / "dust.png"),
    )
    assert again.summary["dust_given"] is True
    with pytest.raises(ValueError, match="changes the prepared scene"):
        made.reconfigure(planes=[(mask, 900.0)])
    # Distances by object name, as a dict or pairs, need the object catalogue.
    for overrides in ({"NGC 9001": 800.0}, [("NGC 9001", 800.0)]):
        with pytest.raises(ValueError, match="object catalog"):
            seiza.ParallaxVideo(
                starless, stars, wcs, distance_pc=400.0, online=False,
                objects=str(tmp_path / "no-objects.bin"), distance_overrides=overrides,
            )
    # A stop may name the depth it is framed at.
    toured = made.reconfigure(
        size=(160, 120), fps=5,
        tour=["whole hold=0.5", {"focus": (150.0, 110.0), "dolly": 0.3, "travel": 0.5, "depth_pc": 1200.0}],
    )
    assert toured.frame_count == 5
    # Planning on the scene reads the object catalogue, which is missing here.
    with pytest.raises(ValueError, match="no-objects"):
        made.plan_tour({"hold": 1.0})
    with pytest.raises(ValueError, match="plans no tour"):
        made.plan_tour(False)


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

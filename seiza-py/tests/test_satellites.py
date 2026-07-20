"""Satellite track prediction binding tests.

Everything here is offline: elements come from a fixed historical ISS TLE.
The crossing test bootstraps in two steps — a huge-scale WCS first locates
the satellite's sky position, then a realistic WCS centered there must
contain the predicted track. This mirrors the Rust crate's own unit test.
"""

from datetime import datetime, timezone

import pytest

import seiza

ISS_TLE = """ISS (ZARYA)
1 25544U 98067A   24123.50000000  .00016717  00000-0  30126-3 0  9990
2 25544  51.6400 160.0000 0005000  80.0000 280.0000 15.50000000450000
"""

# The TLE epoch: 2024 day 123.5 = 2024-05-02T12:00:00Z.
EPOCH_UNIX = datetime(2024, 5, 2, 12, 0, 0, tzinfo=timezone.utc).timestamp()
OBSERVER = dict(latitude=42.466, longitude=-71.1516, altitude_m=150.0)


def iss_catalog():
    return seiza.SatelliteCatalog.from_tle_text(ISS_TLE, source="test")


def locate_iss(catalog, start, duration_s):
    """Find the ISS sky position at the exposure start via a whole-sky WCS."""
    for center_ra in (0.0, 90.0, 180.0, 270.0):
        for center_dec in (-45.0, 45.0):
            wide = seiza.Wcs.from_center_scale_rotation(
                (center_ra, center_dec), (512.0, 512.0), 3600.0
            )
            result = catalog.tracks_in_footprint(
                wide,
                1024,
                1024,
                start=start,
                duration_s=duration_s,
                min_elevation_deg=-90.0,
                **OBSERVER,
            )
            if result.tracks:
                return result.tracks[0]
    raise AssertionError("ISS never projected into any whole-sky WCS")


def test_parses_tle_and_reports_identity():
    catalog = iss_catalog()
    assert len(catalog) == 1
    assert catalog.source == "test"
    assert catalog.retrieved_at_unix is None
    assert catalog.cache_state is None
    assert catalog.provider is None


def test_resolve_rejects_naive_datetime():
    # `resolve` parses its time before touching the network, so a naive
    # datetime is rejected offline without contacting any provider.
    with pytest.raises(ValueError):
        seiza.SatelliteCatalog.resolve(datetime(2024, 5, 2, 12, 0, 0))


def test_predicts_a_crossing_through_a_matching_wcs():
    catalog = iss_catalog()
    coarse = locate_iss(catalog, EPOCH_UNIX, 2.0)
    sample = coarse.samples[0]
    assert sample.pixel is not None

    wcs = seiza.Wcs.from_center_scale_rotation(
        (sample.ra_deg, sample.dec_deg), (512.0, 512.0), 10.0
    )
    result = catalog.tracks_in_footprint(
        wcs,
        1024,
        1024,
        start=EPOCH_UNIX,
        duration_s=2.0,
        min_elevation_deg=-90.0,
        **OBSERVER,
    )
    assert result.elements_considered == 1
    assert result.propagation_failures == 0
    assert result.stale_elements == 0
    assert len(result.tracks) == 1

    track = result.tracks[0]
    assert track.norad_id == 25544
    assert track.cospar_id == "1998-067A"
    assert track.name == "ISS (ZARYA)"
    assert "25544" in track.label
    assert track.clipped_segments
    assert track.clipped_length_px > 0.0
    assert track.max_apparent_rate_arcsec_per_s > 0.0
    assert abs(track.element_epoch_unix - EPOCH_UNIX) < 1.0
    assert len(track.samples) >= 2
    assert all(0.0 <= s.sunlight_fraction <= 1.0 for s in track.samples)


def test_start_accepts_string_and_aware_datetime():
    catalog = iss_catalog()
    reference = locate_iss(catalog, EPOCH_UNIX, 2.0)
    for start in (
        "2024-05-02T12:00:00Z",
        datetime(2024, 5, 2, 12, 0, 0, tzinfo=timezone.utc),
    ):
        track = locate_iss(catalog, start, 2.0)
        assert track.samples[0].ra_deg == pytest.approx(
            reference.samples[0].ra_deg, abs=1e-9
        )


def test_naive_datetime_is_rejected():
    catalog = iss_catalog()
    wcs = seiza.Wcs.from_center_scale_rotation((0.0, 0.0), (50.0, 50.0), 3600.0)
    with pytest.raises(ValueError, match="timezone-aware"):
        catalog.tracks_in_footprint(
            wcs,
            100,
            100,
            start=datetime(2024, 5, 2, 12, 0, 0),
            duration_s=2.0,
            **OBSERVER,
        )


def test_observer_is_required_and_unambiguous():
    catalog = iss_catalog()
    wcs = seiza.Wcs.from_center_scale_rotation((0.0, 0.0), (50.0, 50.0), 3600.0)
    with pytest.raises(ValueError, match="observer location"):
        catalog.tracks_in_footprint(
            wcs, 100, 100, start=EPOCH_UNIX, duration_s=2.0
        )
    with pytest.raises(ValueError, match="not a mixture"):
        catalog.tracks_in_footprint(
            wcs,
            100,
            100,
            start=EPOCH_UNIX,
            duration_s=2.0,
            latitude=42.0,
            longitude=-71.0,
            observer_itrf=(1.0, 2.0, 3.0),
        )


def test_stale_elements_are_reported_not_extrapolated():
    catalog = iss_catalog()
    wcs = seiza.Wcs.from_center_scale_rotation((0.0, 0.0), (50.0, 50.0), 3600.0)
    result = catalog.tracks_in_footprint(
        wcs,
        100,
        100,
        start=EPOCH_UNIX + 8 * 86_400,
        duration_s=10.0,
        **OBSERVER,
    )
    assert result.stale_elements == 1
    assert not result.tracks

    # Explicitly disabling the age policy allows extrapolation again.
    result = catalog.tracks_in_footprint(
        wcs,
        100,
        100,
        start=EPOCH_UNIX + 8 * 86_400,
        duration_s=10.0,
        max_element_age_s=None,
        **OBSERVER,
    )
    assert result.stale_elements == 0


def test_open_reads_files(tmp_path):
    path = tmp_path / "iss.tle"
    path.write_text(ISS_TLE)
    catalog = seiza.SatelliteCatalog.open(path)
    assert len(catalog) == 1
    assert str(path) in catalog.source


def test_invalid_elements_raise_value_error():
    with pytest.raises(ValueError):
        seiza.SatelliteCatalog.from_omm_json("[]", source="empty")
    with pytest.raises(ValueError):
        seiza.SatelliteCatalog.from_tle_text("not a tle", source="junk")

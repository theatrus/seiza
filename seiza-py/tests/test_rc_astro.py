"""RC-Astro external-tool bindings, exercised against a stand-in CLI.

The stand-in answers ``<tool> --json`` with a schema document and otherwise
copies each input to its output (the file ``-o`` names, or the CLI's own name
in the directory it names), plus a stars sidecar when asked, emitting the
real event-stream shape. Each start appends a line to ``runs`` beside it — so these tests need no license and no real
rc-astro install.
"""

import json
import os
import stat
import sys

import pytest
import seiza

FAKE = r"""#!/bin/sh
if [ "$2" = "--json" ] && [ $# -eq 2 ]; then
  cat <<'SCHEMA'
{"schemaVersion": 6, "cliVersion": "2.6.6", "key": "sxt",
 "name": "RC-Astro StarXTerminator", "mlVersion": 11,
 "license": {"status": "permanent", "valid": true, "message": "Permanently licensed"},
 "parameters": [
   {"label": "Tile Overlap", "name": "overlap", "flag": "--overlap",
    "description": "Fractional overlap", "type": "float",
    "default": 0.2, "min": 0.0, "max": 0.5},
   {"label": "Generate Star Image", "name": "stars", "flag": "--stars",
    "description": "Also write a stars-only image", "type": "bool", "default": false},
   {"label": "Iterations", "name": "it", "flag": "--it",
    "description": "Passes", "type": "int", "default": 2, "min": 1, "max": 5},
   {"label": "Color Separation", "name": "csep",
    "description": "GUI only", "type": "bool", "default": false}
 ]}
SCHEMA
  exit 0
fi
here="$(dirname "$0")"
echo run >> "$here/runs"
out=""
tool=""
stars=0
list="$here/inputs.list"
: > "$list"
while [ $# -gt 0 ]; do
  case "$1" in
    -o) out="$2"; shift 2 ;;
    --stars) stars=1; shift ;;
    --host|--depth|--device) shift 2 ;;
    sxt|bxt|nxt) tool="$1"; shift ;;
    --*) shift ;;
    *) printf '%s\n' "$1" >> "$list"; shift ;;
  esac
done
echo '{"event":"info","topic":"device","device":"cpu"}'
echo '{"event":"progress","done":50.0}'
echo '{"event":"progress","done":100.0}'
while IFS= read -r input; do
  if [ -d "$out" ]; then
    name="$(basename "$input")"
    target="$out/${name%.*}-$tool.${name##*.}"
  else
    target="$out"
  fi
  cp "$input" "$target"
  echo "{\"event\":\"status\",\"phase\":\"complete\",\"output\":\"$target\"}"
  if [ "$stars" = 1 ]; then
    sidecar="${target%.fits}-stars.fits"
    cp "$input" "$sidecar"
    echo "{\"event\":\"status\",\"phase\":\"complete\",\"output\":\"$sidecar\"}"
  fi
done < "$list"
"""

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux") and sys.platform != "darwin",
    reason="the stand-in rc-astro is a shell script",
)


@pytest.fixture
def fake_rc_astro(tmp_path):
    path = tmp_path / "rc-astro"
    path.write_text(FAKE)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def test_tool_schema_reads_the_live_contract(fake_rc_astro):
    schema = seiza.rc_astro_tool_schema("sxt", executable=fake_rc_astro)
    assert schema.contract_version == 6
    assert schema.cli_version == "2.6.6"
    assert schema.licensed
    assert schema.ml_version == 11
    names = [parameter.name for parameter in schema.parameters]
    assert names == ["overlap", "stars", "it", "csep"]
    overlap = schema.parameters[0]
    assert overlap.type == "float"
    assert overlap.default == pytest.approx(0.2)
    assert overlap.min == 0.0 and overlap.max == 0.5
    stars = schema.parameters[1]
    assert stars.type == "bool"
    assert stars.default is False
    iterations = schema.parameters[2]
    assert iterations.type == "int"
    assert iterations.default == 2
    assert iterations.min == 1.0 and iterations.max == 5.0
    # A GUI-only parameter has no CLI flag and cannot be requested.
    assert schema.parameters[3].flag is None


def test_a_run_returns_the_stars_sidecar_and_progress(fake_rc_astro, tmp_path):
    source = tmp_path / "in.fits"
    source.write_bytes(b"fake fits")
    output = tmp_path / "out.fits"
    fractions = []
    run = seiza.rc_astro_process_file(
        "sxt",
        str(source),
        str(output),
        parameters={"stars": True, "overlap": 0.3, "it": 3.0},
        executable=fake_rc_astro,
        progress=fractions.append,
    )
    assert str(run.primary) == str(output)
    assert len(run.sidecars) == 1
    assert str(run.sidecars[0]).endswith("-stars.fits")
    assert run.device == "cpu"
    assert run.cli_version == "2.6.6"
    assert fractions == [pytest.approx(0.5), pytest.approx(1.0)]
    assert output.read_bytes() == b"fake fits"


def test_an_image_in_linear_units_comes_back_in_them(fake_rc_astro):
    import numpy as np

    # The stand-in copies its input, so a round trip shows the mapping onto
    # 0..1 and back: a stack far above 1, with a negative background.
    image = np.linspace(-50.0, 5000.0, 48 * 32, dtype=np.float32).reshape(48, 32)
    fractions = []
    processed = seiza.rc_astro_process_image(
        "sxt", image, parameters={"stars": True}, executable=fake_rc_astro,
        progress=fractions.append,
    )
    np.testing.assert_allclose(processed.image, image, atol=5050.0 * 1e-6)
    # The stars image is a difference, restored with the scale alone.
    np.testing.assert_allclose(processed.stars, image + 50.0, atol=5050.0 * 1e-6)
    assert processed.image.shape == image.shape
    assert (processed.device, processed.cli_version, processed.ml_version) == ("cpu", "2.6.6", 11)
    assert fractions == [pytest.approx(0.5), pytest.approx(1.0)]
    colour = np.ones((8, 8, 3), dtype=np.float32) * 300.0
    without = seiza.rc_astro_process_image("sxt", colour, executable=fake_rc_astro)
    assert without.stars is None
    np.testing.assert_allclose(without.image, colour, rtol=1e-6)


def starts(fake_rc_astro):
    runs = os.path.join(os.path.dirname(fake_rc_astro), "runs")
    with open(runs) as lines:
        return len(lines.readlines())


def test_many_files_run_in_one_start_with_outputs_named_by_the_cli(fake_rc_astro, tmp_path):
    lights = tmp_path / "lights"
    lights.mkdir()
    sources = [lights / "m31.fits", lights / "m33.fits"]
    for source in sources:
        source.write_bytes(b"fake fits")
    output = tmp_path / "processed"
    fractions = []
    runs = seiza.rc_astro_process_files(
        "sxt",
        sources,
        output,
        parameters={"stars": True},
        executable=fake_rc_astro,
        progress=fractions.append,
    )
    assert starts(fake_rc_astro) == 1
    assert [str(run.primary) for run in runs] == [
        str(output / "m31-sxt.fits"),
        str(output / "m33-sxt.fits"),
    ]
    assert [str(run.sidecars[0]) for run in runs] == [
        str(output / "m31-sxt-stars.fits"),
        str(output / "m33-sxt-stars.fits"),
    ]
    assert runs[1].device == "cpu" and runs[1].cli_version == "2.6.6"
    assert fractions == [pytest.approx(0.5), pytest.approx(1.0)]


def test_two_inputs_with_one_stem_are_refused_before_the_cli_starts(fake_rc_astro, tmp_path):
    with pytest.raises(ValueError, match="m31"):
        seiza.rc_astro_process_files(
            "sxt",
            [tmp_path / "a" / "m31.fits", tmp_path / "b" / "m31.xisf"],
            tmp_path / "out",
            executable=fake_rc_astro,
        )
    assert not (tmp_path / "out").exists()


def test_many_images_come_back_in_order_each_on_its_own_scale(fake_rc_astro):
    import numpy as np

    stretched = np.linspace(0.1, 0.9, 12, dtype=np.float32).reshape(3, 4)
    camera = np.linspace(-50.0, 40000.0, 36, dtype=np.float32).reshape(3, 4, 3)
    processed = seiza.rc_astro_process_images(
        "sxt", [stretched, camera], parameters={"stars": True}, executable=fake_rc_astro
    )
    assert starts(fake_rc_astro) == 1
    assert [result.image.shape for result in processed] == [(3, 4), (3, 4, 3)]
    np.testing.assert_allclose(processed[0].image, stretched, atol=1e-6)
    np.testing.assert_allclose(processed[1].image, camera, atol=40050.0 * 1e-6)
    assert all(result.stars is not None for result in processed)


def test_many_stretched_images_split_in_one_start(fake_rc_astro):
    import numpy as np

    first = np.full((4, 5, 3), 0.25, dtype=np.float32)
    second = np.full((4, 5, 3), 1.5, dtype=np.float32)
    pairs = seiza.rc_astro_split_stars_batch([first, second], executable=fake_rc_astro)
    assert starts(fake_rc_astro) == 1
    assert len(pairs) == 2
    np.testing.assert_allclose(pairs[0][0], first)
    # Clamped to its own 0..1, as one split clamps it.
    np.testing.assert_allclose(pairs[1][0], np.ones_like(second))
    with pytest.raises(ValueError, match="image 1"):
        seiza.rc_astro_split_stars_batch(
            [first, np.zeros((4, 5), dtype=np.float32)], executable=fake_rc_astro
        )


def test_a_bad_parameter_type_is_refused(fake_rc_astro, tmp_path):
    source = tmp_path / "in.fits"
    source.write_bytes(b"fake fits")
    with pytest.raises(ValueError, match="must be a bool, int, or float"):
        seiza.rc_astro_process_file(
            "sxt",
            str(source),
            str(tmp_path / "out.fits"),
            parameters={"overlap": "wide"},
            executable=fake_rc_astro,
        )


def test_a_raising_progress_callback_surfaces_after_the_run(fake_rc_astro, tmp_path):
    source = tmp_path / "in.fits"
    source.write_bytes(b"fake fits")
    output = tmp_path / "out.fits"

    def boom(fraction):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        seiza.rc_astro_process_file(
            "sxt", str(source), str(output), executable=fake_rc_astro, progress=boom
        )
    # Progress is advisory: the run itself completed before the re-raise.
    assert output.exists()


def test_a_cancel_predicate_stops_the_run(tmp_path):
    # A tool that emits one event and stalls: only the cancel poll can end
    # it (the stand-in above would finish before cancellation matters).
    path = tmp_path / "rc-astro"
    path.write_text(
        FAKE.split("fi\n", 1)[0]
        + "fi\n"
        + 'echo \'{"event":"progress","done":1.0}\'\nsleep 30\n'
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    source = tmp_path / "in.fits"
    source.write_bytes(b"fake fits")
    import time

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="cancelled"):
        seiza.rc_astro_process_file(
            "sxt",
            str(source),
            str(tmp_path / "out.fits"),
            executable=str(path),
            cancel=lambda: True,
        )
    assert time.monotonic() - started < 10


def test_locate_returns_none_or_a_path():
    located = seiza.rc_astro_locate()
    assert located is None or os.path.basename(located).startswith("rc-astro")

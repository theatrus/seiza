"""RC-Astro external-tool bindings, exercised against a stand-in CLI.

The stand-in answers ``<tool> --json`` with a schema document and otherwise
copies the input to the output (plus a stars sidecar when asked), emitting
the real event-stream shape — so these tests need no license and no real
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
    "description": "Also write a stars-only image", "type": "bool", "default": false}
 ]}
SCHEMA
  exit 0
fi
out=""
input=""
stars=0
while [ $# -gt 0 ]; do
  case "$1" in
    -o) out="$2"; shift 2 ;;
    --stars) stars=1; shift ;;
    --host|--depth|--device) shift 2 ;;
    --*|sxt|bxt|nxt) shift ;;
    *) input="$1"; shift ;;
  esac
done
cp "$input" "$out"
echo '{"event":"info","topic":"device","device":"cpu"}'
echo '{"event":"progress","done":50.0}'
echo '{"event":"progress","done":100.0}'
echo "{\"event\":\"status\",\"phase\":\"complete\",\"output\":\"$out\"}"
if [ "$stars" = 1 ]; then
  sidecar="${out%.fits}-stars.fits"
  cp "$input" "$sidecar"
  echo "{\"event\":\"status\",\"phase\":\"complete\",\"output\":\"$sidecar\"}"
fi
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
    assert names == ["overlap", "stars"]
    overlap = schema.parameters[0]
    assert overlap.type == "float"
    assert overlap.default == pytest.approx(0.2)
    assert overlap.min == 0.0 and overlap.max == 0.5
    stars = schema.parameters[1]
    assert stars.type == "bool"
    assert stars.default is False


def test_a_run_returns_the_stars_sidecar_and_progress(fake_rc_astro, tmp_path):
    source = tmp_path / "in.fits"
    source.write_bytes(b"fake fits")
    output = tmp_path / "out.fits"
    fractions = []
    run = seiza.rc_astro_process_file(
        "sxt",
        str(source),
        str(output),
        parameters={"stars": True, "overlap": 0.3},
        executable=fake_rc_astro,
        progress=fractions.append,
    )
    assert run.primary == str(output)
    assert len(run.sidecars) == 1
    assert run.sidecars[0].endswith("-stars.fits")
    assert run.device == "cpu"
    assert run.cli_version == "2.6.6"
    assert fractions == [pytest.approx(0.5), pytest.approx(1.0)]
    assert output.read_bytes() == b"fake fits"


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


def test_locate_returns_none_or_a_path():
    located = seiza.rc_astro_locate()
    assert located is None or os.path.basename(located).startswith("rc-astro")

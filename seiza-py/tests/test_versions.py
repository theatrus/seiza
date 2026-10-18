"""The versions the module carries, and the ones caches key on."""

import re

import seiza

RELEASE = re.compile(r"^\d+\.\d+\.\d+$")


def test_output_versions_are_releases():
    # Each is the seiza-stacking or seiza-stretch crate version whose output
    # this build reproduces, not a Seiza release, so it need not match
    # __version__; it is never empty.
    for version in (seiza.STACKING_OUTPUT_VERSION, seiza.STRETCH_OUTPUT_VERSION):
        assert RELEASE.match(version), version

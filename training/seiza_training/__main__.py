"""`python -m seiza_training <pass> --store DIR ...`: run one corpus pass.

The passes run in order — crawl, calibrate, stacks, split, measure,
build, review — and each can be rerun alone. `process` measures and
builds one set at a time, which reads each frame over the network once.
`multi-night` stacks the deepest objects over all their nights.
`psf-fields` measures each sharpening source's PSF field, and
`cut-sharpen` cuts the sharpening banks from them; both only read the
manifest. `train-sharpen` trains the PSF model on those banks and
`evaluate-sharpen` measures one, reading the manifest, when there is
one, only for the sets' names. `status` prints what the manifest holds;
`decide` records a review decision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .manifest import Manifest
from .store import Store, versions

# The commands that only read the manifest and record no pass in it.
READ_ONLY = ("psf-fields", "cut-sharpen", "sheet-sharpen")
# The commands that read the sharpening banks, and the manifest, if at
# all, only for the sets' names.
BANKS_ONLY = ("train-sharpen", "evaluate-sharpen", "clean-sharpen")


def numbers(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in text.split(","))


def weighted(text: str) -> tuple[str, float]:
    """`PATH[:WEIGHT]`: a bank and its weight, 1 when none is given."""
    path, _, weight = text.rpartition(":")
    try:
        return (path, float(weight)) if path else (text, 1.0)
    except ValueError:
        return text, 1.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m seiza_training", description=__doc__)
    parser.add_argument("--store", required=True, type=Path, help="corpus store directory")
    parser.add_argument(
        "--scratch", type=Path,
        help="where build keeps registered frames, on local disk (default ~/.cache/seiza-training/scratch)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    crawl = commands.add_parser("crawl", help="read every header and group lights into sets")
    crawl.add_argument("--corpus", type=Path, help="the telescope corpus (remembered)")
    crawl.add_argument("--library", type=Path, help="the calibration-master library (remembered)")
    crawl.add_argument("--calibration", type=Path, help="raw calibration frames, darks among them (remembered)")
    crawl.add_argument("--workers", type=int, default=16)

    for name, text in (
        ("calibrate", "choose calibration masters for each set"),
        ("measure", "measure every frame of the calibratable sets"),
        ("build", "register each set once and build its products"),
        ("process", "measure, then build, one set at a time"),
        ("review", "write the review pages"),
    ):
        command = commands.add_parser(name, help=text)
        command.add_argument("--set", action="append", dest="sets", help="only this set id")
        if name in ("measure", "build"):
            command.add_argument("--again", action="store_true", help="redo sets already done")
        if name in ("measure", "build", "process"):
            command.add_argument("--limit", type=int, help="at most this many sets")
        if name in ("measure", "process"):
            command.add_argument("--catalog-dir", help="catalogs for plate solving")
        if name in ("build", "process"):
            command.add_argument("--keep-scratch", action="store_true")
        if name == "build":
            command.add_argument("--pairs-only", action="store_true",
                                 help="only the noise pairs of built sets that now qualify for them and have none")

    recipes = commands.add_parser("recipes", help="decide each measured set's recipes again from its measurement")
    recipes.add_argument("--dry-run", action="store_true", help="report the changes without recording them")

    stacks = commands.add_parser("stacks", help="copy in the finished stacks PSF Guard made from the corpus")
    stacks.add_argument("--psf-guard", type=Path, help="PSF Guard's stacks directory (remembered)")
    stacks.add_argument("--again", action="store_true", help="copy stacks already copied")
    stacks.add_argument("--limit", type=int, help="at most this many stacks")

    dark = commands.add_parser("dark-noise", help="make noise pairs with a known clean image from dark frames")
    dark.add_argument("--set", action="append", dest="sets", help="only this set id")
    dark.add_argument("--size", type=int, action="append", dest="sizes",
                      help="the A half whose frames lend their registration, by size (default 8 and 16)")
    dark.add_argument("--again", action="store_true", help="make pairs already made")
    dark.add_argument("--limit", type=int, help="at most this many pairs")

    multi = commands.add_parser("multi-night", help="stack every night of the deepest objects onto one reference")
    multi.add_argument("--object", action="append", dest="objects",
                       help="only this object, `telescope/target/FILTER-exposure-gain-binning` (default: all deep)")
    multi.add_argument("--min-hours", type=float, help="the hours of accepted frames an unnamed object needs")
    multi.add_argument("--source-map", action="append", default=[], metavar="FROM=TO",
                       help="read frame and master paths starting FROM from TO instead")
    multi.add_argument("--list", action="store_true",
                       help="list the objects, their depth and spread of seeing, and stack nothing")
    multi.add_argument("--seeing", action="store_true",
                       help="also stack every object whose nights give a seeing split, however few its hours")
    multi.add_argument("--check-seeing", action="store_true",
                       help="judge the seeing pairs already stacked again from their files, and stack nothing")
    multi.add_argument("--again", action="store_true", help="stack objects already stacked")
    multi.add_argument("--limit", type=int, help="at most this many objects")
    multi.add_argument("--keep-scratch", action="store_true")

    train = commands.add_parser("train-noise", help="train a noise model on the train split's noise pairs")
    train.add_argument("--name", required=True, help="the run's name, under STORE/models/noise/")
    train.add_argument("--steps", type=int, default=10_000)
    train.add_argument("--width", type=int, default=16)
    train.add_argument("--levels", type=int, default=4)
    train.add_argument("--tile", type=int, default=256)
    train.add_argument("--batch", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=2e-4)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--set", action="append", dest="sets", help="only this train set id")
    train.add_argument("--validation-set", action="append", dest="validation_sets",
                       help="measure on only this validation set id")
    train.add_argument("--loss-weight", choices=("core", "variance"), default="core")
    train.add_argument("--map-jitter", type=float, default=0.0,
                       help="scale the noise map the network reads by up to exp(±this)")
    train.add_argument("--noise-map", choices=("raw", "model", "image", "constant"), default="raw",
                       help="the stack's per-pixel noise map, the one modelled from it, one from the image alone, "
                            "or that one's median everywhere")
    train.add_argument("--holes", action="store_true",
                       help="train and measure as the runtime's hole guard runs a model: it never sees an image's "
                            "holes, pixels far below its sky, which are filled from their surroundings in both "
                            "halves and put back after")
    train.add_argument("--bank-workers", type=int, default=1,
                       help="prepare this many pairs at once when cutting the bank, in processes of their own")
    train.add_argument("--weight-map", choices=("raw", "model"),
                       help="with an image-only --noise-map, weigh the loss by, and scale the outliers in, the "
                            "stack's own map of this kind, and match the halves by it: training uses all the "
                            "stack knows, while the network still reads only what it will have from an image")
    train.add_argument("--huber", type=float, default=0.0,
                       help="square each pixel's weighted error only within this of zero")
    train.add_argument("--huber-above", type=float, default=0.0,
                       help="make only pixels whose target is this many σ̄ above the background robust")
    train.add_argument("--structure-sampling", type=float, default=0.0,
                       help="draw windows 1 + this × their share of bright pixels as often")
    train.add_argument("--dark-noise-fraction", type=float, default=0.0,
                       help="this share of every batch from the dark-noise pairs, whose target is clean")
    train.add_argument("--deep-fraction", type=float, default=0.0,
                       help="this share of every batch from the halves of each night's deep stack")
    train.add_argument("--outliers", type=float, default=0.0, dest="outlier_count",
                       help="add this many synthetic hot pixels, cold pixels and cosmic rays per 256 × 256 "
                            "pixels to every input tile, and none to its target")
    train.add_argument("--outlier-floor", type=float, default=5.0,
                       help="the faintest synthetic hot pixel, in units of the typical noise; cosmic rays "
                            "start at twice it")
    train.add_argument("--match-flux", action="store_true",
                       help="scale each pair's B half onto its A half's background and bright flux first")
    train.add_argument("--outlier-scale", choices=("typical", "local"), default="typical",
                       help="the synthetic outliers' amplitudes in the image's typical noise, or in the noise "
                            "at each pixel, so they stand as far above a bright nebula's noise as the sky's")
    train.add_argument("--noise-plane", choices=("linear", "asinh"), default="linear",
                       help="give the network the noise map in units of the typical noise as it is, or through "
                            "asinh, as it reads the image, so a bright star's map reaches a few units, not 50")
    train.add_argument("--agree", type=float,
                       help="with --match-flux, leave out pairs whose halves' flux still differs by more than "
                            "this share in the stars' wings or cores (haze, a change of seeing)")
    train.add_argument("--clip-grad", type=float, default=0.0,
                       help="scale each step's gradient down to this norm where it is longer (0: never)")
    train.add_argument("--precision", choices=("fp32", "bf16"), default="fp32",
                       help="run the network in bfloat16 under autocast, the loss staying in float32")
    train.add_argument("--prefetch", type=int, default=8,
                       help="build this many batches ahead (0: none; the batches are the same)")
    train.add_argument("--batch-threads", type=int, default=4,
                       help="threads building batches ahead (the batches are the same)")
    train.add_argument("--scale-loss", type=lambda text: tuple(int(k) for k in text.split(",")), default=(),
                       help="also weigh the error averaged over blocks of these sides, comma-separated (4,8,16), "
                            "each as much as the pixels' noise, so noise left at those scales costs the loss as "
                            "much as the grain and is not left as mottle")
    train.add_argument("--scale-weight", type=float, default=1.0,
                       help="with --scale-loss, the weight of each scale's term against the per-pixel loss")
    train.add_argument("--rewind", type=float, default=0.0,
                       help="go back two log windows when one's mean loss is more than this many times the "
                            "lowest so far, or not finite, so weights that blow up do not kill the network "
                            "(0: never; 2 catches a collapse, not the loss's ordinary swings); each rewind "
                            "takes a fifth off the learning rate for the rest of the run")
    train.add_argument("--sky", choices=("global", "flat"), default="global",
                       help="read the image less one background for the whole frame, or less its smooth sky "
                            "first (medians over 256-pixel blocks), so a gradient does not read as signal")
    train.add_argument("--resample", type=float, default=0.0,
                       help="shift this share of the tiles by a sub-pixel offset through Lanczos-3, both halves "
                            "and the noise map alike")
    train.add_argument("--sharpen", type=float, default=0.0,
                       help="sharpen this share of the tiles by an unsharp mask, both halves and the noise map "
                            "alike")
    train.add_argument("--spread", type=float, default=0.0,
                       help="spread this share of the tiles over a box 1.5 to 3 pixels wide, as drizzle spreads "
                            "each frame's pixels, both halves and the noise map alike")
    train.add_argument("--gradient", type=float, default=0.0,
                       help="put a sky plane under half the input tiles, offset up to this many σ̄ and tilted "
                            "up to half of it across the tile")
    train.add_argument("--bin-fraction", type=float, default=0.0,
                       help="this share of every batch from the pairs binned 2 × 2, whose stars are half as "
                            "wide")
    train.add_argument("--dark-spots", type=float, default=0.0, dest="dark_spot_count",
                       help="add this many dark spots (clusters 30 to 5,000 times the typical noise deep) per "
                            "256 × 256 pixels to both halves of every tile alike, so the model keeps them, as a "
                            "stack's over-subtracted hot or dead pixels, and does not fill or ring them")

    psf = commands.add_parser("psf-fields", help="measure each sharpening source's PSF field once and keep it in "
                                                 "the store, under psf/ (reads the manifest only)")
    psf.add_argument("--set", action="append", dest="sets", help="only this set id")
    psf.add_argument("--kind", action="append", dest="kinds",
                     choices=("deep", "multi-night", "seeing-best", "seeing-worst", "multi-night-seeing-best",
                              "multi-night-seeing-worst"),
                     help="only products of this kind (default: every kind the sharpening bank cuts)")
    psf.add_argument("--split", action="append", dest="splits", choices=("train", "validation", "test", "held-out"),
                     help="only sets of this split (default: train and validation)")
    psf.add_argument("--workers", type=int, default=4, help="measure this many products at once")
    psf.add_argument("--limit", type=int, help="at most this many products not yet measured")
    psf.add_argument("--again", action="store_true", help="measure products already measured")

    sharpen = commands.add_parser("cut-sharpen", help="cut the sharpening banks from the source products and their "
                                                      "PSF fields, onto local disk (reads the manifest only)")
    sharpen.add_argument("--bank", action="append", dest="banks", choices=("deep", "seeing", "binned"),
                         help="only this bank (default: both)")
    sharpen.add_argument("--split", action="append", dest="splits", choices=("train", "validation"),
                         help="only this split's banks (default: both, apart)")
    sharpen.add_argument("--set", action="append", dest="sets", help="only this set id")
    sharpen.add_argument("--work", type=Path,
                         help="where the banks are cut, on local disk (default: beside the scratch directory, "
                              "~/.cache/seiza-training/sharpen)")
    sharpen.add_argument("--workers", type=int, default=4, help="read and prepare this many sources at once")
    sharpen.add_argument("--seed", type=int, default=0)
    sharpen.add_argument("--deep-windows", type=int, default=40, help="windows of each deep stack")
    sharpen.add_argument("--pair-windows", type=int, default=160, help="windows of each seeing pair")
    sharpen.add_argument("--limit", type=int, help="at most this many sources a bank, for a trial cut")

    sharpen_train = commands.add_parser("train-sharpen", help="train a PSF model on the sharpening banks")
    sharpen_train.add_argument("--name", required=True, help="the run's name, under STORE/models/sharpen/")
    sharpen_train.add_argument("--resume", action="store_true",
                               help="go on from the run's last checkpoint, as its config.json records it (the "
                                    "options below, but --stop-after, --prefetch and --batch-threads, are not read)")
    sharpen_train.add_argument("--stop-after", type=int,
                               help="stop after this step with a checkpoint, to go on later with --resume")
    sharpen_train.add_argument("--steps", type=int, default=20_000)
    sharpen_train.add_argument("--width", type=int, default=32)
    sharpen_train.add_argument("--levels", type=int, default=4)
    sharpen_train.add_argument("--tile", type=int, default=256)
    sharpen_train.add_argument("--batch", type=int, default=8)
    sharpen_train.add_argument("--learning-rate", type=float, default=2e-4)
    sharpen_train.add_argument("--seed", type=int, default=0)
    sharpen_train.add_argument("--log-every", type=int, default=250)
    sharpen_train.add_argument("--checkpoint-every", type=int, default=1000,
                               help="save the run whole every this many steps, a multiple of --log-every")
    sharpen_train.add_argument("--work", type=Path,
                               help="where cut-sharpen cut the banks (default: beside the scratch directory, "
                                    "~/.cache/seiza-training/sharpen); each bank not named below is the newest there")
    sharpen_train.add_argument("--kernel-bank", type=Path, help="the train split's known-kernel (deep) bank")
    sharpen_train.add_argument("--more-kernel-bank", type=weighted, action="append", dest="more_kernels", default=[],
                               metavar="PATH[:WEIGHT]",
                               help="a further known-kernel bank, such as a binned one, sharing the known-kernel "
                                    "share of the mix by its weight against the first's 1 (repeatable)")
    sharpen_train.add_argument("--seeing-bank", type=Path, action="append", dest="seeing_banks",
                               help="the train split's seeing bank (repeatable)")
    sharpen_train.add_argument("--pairs-bank", type=weighted, action="append", dest="pairs_banks", default=[],
                               metavar="PATH[:WEIGHT]",
                               help="pairs a caller supplies, in the seeing bank's format, for the third share of "
                                    "the mix, shared among such banks by their weights (repeatable)")
    sharpen_train.add_argument("--validation-kernel-bank", type=Path)
    sharpen_train.add_argument("--validation-seeing-bank", type=Path, action="append", dest="validation_seeing_banks")
    sharpen_train.add_argument("--manifest", type=Path,
                               help="the manifest whose set names give each window's object, opened for reading "
                                    "alone (default: the store's, when it has one)")
    sharpen_train.add_argument("--mix", type=numbers, default=(0.6, 0.25, 0.15),
                               help="the shares of known-kernel, seeing and supplied windows in a batch; with no "
                                    "supplied bank its share goes to seeing")
    sharpen_train.add_argument("--kernel-ratios", type=numbers, default=(1.1, 2.0),
                               help="FWHM(P_in) / FWHM(P_ref) of a known-kernel window, drawn between these")
    sharpen_train.add_argument("--kernel-ellipticity", type=float, default=0.3,
                               help="the most ellipticity (1 - minor/major) of a known kernel")
    sharpen_train.add_argument("--kernel-noise", type=float, default=0.5,
                               help="fresh white noise on a known-kernel input, drawn up to this share of the deep "
                                    "stack's own")
    sharpen_train.add_argument("--identity", type=float, default=0.1,
                               help="the share of known-kernel windows left as they are, ratio 1, target the input")
    sharpen_train.add_argument("--exposure", type=float, default=2.0,
                               help="scale a known-kernel window's signal by up to this either way before its "
                                    "noise is added")
    sharpen_train.add_argument("--floor-px", type=float, default=1.5, help="the narrowest target FWHM")
    sharpen_train.add_argument("--centroid-offset", choices=("light", "detection", "none"), default="light",
                               help="move each pair's target back by its stars' offset, so the model is not taught "
                                    "to move stars: of their light-weighted centres, over the field, or of their "
                                    "detections' centroids, the median; or leave it")
    sharpen_train.add_argument("--ratio-source", choices=("stamp", "fit"), default="stamp",
                               help="a pair's ratio from its stamps' widths, which every window has, or from the "
                                    "field's Moffat fit, which fails below about 2 px")
    sharpen_train.add_argument("--max-window-ratio", type=float, default=1.0,
                               help="leave out windows whose target is wider than this times their input")
    sharpen_train.add_argument("--min-window-ratio", type=float, default=0.0,
                               help="leave out windows whose target is narrower than this times their input")
    sharpen_train.add_argument("--exclude-pairs", default="",
                               help="pairs left out, by pair, input or target id, comma-separated")
    sharpen_train.add_argument("--focus-check", action=argparse.BooleanOptionalAction, default=True,
                               help="leave out windows whose stamps are out of focus, ringed or flat-topped, and "
                                    "every window of a source where more than a quarter are")
    sharpen_train.add_argument("--leave-out-kinds", default="",
                               help="leave out every bank's sources cut from these kinds of stack, comma-separated "
                                    "(noise-a: the noise pairs' few-frame halves)")
    sharpen_train.add_argument("--multi-night-share", type=float, default=0.0,
                               help="the share of the known-kernel draws the stacks of many nights take, the nights' "
                                    "own the rest (0: as their windows and the object cap fall)")
    sharpen_train.add_argument("--object-cap", type=float, default=0.03,
                               help="no object (telescope, target and filter, over its nights) takes more than this "
                                    "share of its bank's draws")
    sharpen_train.add_argument("--nonstellar", action="store_true",
                               help="separate strengths for stars and for the rest: the network reads a ratio "
                                    "for each, and a known-kernel window asks each for its own")
    sharpen_train.add_argument("--equal-share", type=float, default=0.3,
                               help="with --nonstellar, the share of known-kernel windows asking both alike")
    sharpen_train.add_argument("--real-kernels", type=float, default=0.0,
                               help="the share of known-kernel windows blurred by a real PSF, another window's "
                                    "stamp, in place of a Gaussian")
    sharpen_train.add_argument("--pass-noise", action="store_true",
                               help="targets carry their input's noise, so the model passes noise on: a "
                                    "known-kernel window's fresh noise is in its target, and a pair bank with "
                                    "denoised planes (clean-sharpen) asks for its target's denoised plus its "
                                    "input's noise")
    sharpen_train.add_argument("--fill-clipped", action=argparse.BooleanOptionalAction, default=True,
                               help="fill a known-kernel window's clipped cores with its stamp before "
                                    "its blur, and clip input and target after it, as a camera clips")
    sharpen_train.add_argument("--residual", choices=("asinh", "linear"), default="asinh",
                               help="add the network's residual to the input's asinh, as it reads it, and take it "
                                    "back through sinh, or add it to the input in units of the typical noise")
    sharpen_train.add_argument("--points-empty", type=float, default=0.1,
                               help="leave the point-source map empty this share of the time")
    sharpen_train.add_argument("--charbonnier", type=float, default=0.1,
                               help="Charbonnier's epsilon, in units of the target's noise")
    sharpen_train.add_argument("--noise-cap", type=float, default=10.0,
                               help="read the target's noise no higher than this many times its tile's median, so "
                                    "a star core, whose frames scatter with the seeing, still counts (0: no cap)")
    sharpen_train.add_argument("--gradient-weight", type=float, default=0.1,
                               help="the weight of the error gradient term; at 0.5 its noise drowned the stars")
    sharpen_train.add_argument("--gradient-scales", type=lambda text: tuple(int(k) for k in text.split(",")),
                               default=(1, 2, 4), help="the blocks the gradient term averages the error over")
    sharpen_train.add_argument("--flux-weight", type=float, default=0.1,
                               help="the weight of the point sources' aperture flux term")
    sharpen_train.add_argument("--clip-grad", type=float, default=1.0,
                               help="scale each step's gradient down to this norm where it is longer (0: never)")
    sharpen_train.add_argument("--rewind", type=float, default=2.0,
                               help="go back two log windows when one's mean loss is more than this many times the "
                                    "lowest so far (0: never)")
    sharpen_train.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    sharpen_train.add_argument("--prefetch", type=int, default=8)
    sharpen_train.add_argument("--batch-threads", type=int, default=4)
    sharpen_train.add_argument("--workers", type=int, default=0,
                               help="build batches in this many worker processes rather than on threads "
                                    "(the same batches; 0: threads)")
    sharpen_train.add_argument("--validation-batches", type=int, default=4,
                               help="fixed validation batches whose loss is logged with each window's")
    sharpen_train.add_argument("--validation-windows", type=int, default=200,
                               help="windows of each validation bank measured at the end")

    sharpen_evaluate = commands.add_parser("evaluate-sharpen", help="measure a trained PSF model on validation banks")
    sharpen_evaluate.add_argument("--name", required=True)
    sharpen_evaluate.add_argument("--limit", type=int, default=200, help="at most this many windows a bank")
    sharpen_evaluate.add_argument("--seed", type=int, default=0, help="seeds each known-kernel window's draw")
    sharpen_evaluate.add_argument("--work", type=Path)
    sharpen_evaluate.add_argument("--kernel-bank", type=Path,
                                  help="the known-kernel bank (default: the validation split's newest)")
    sharpen_evaluate.add_argument("--seeing-bank", type=Path, action="append", dest="seeing_banks",
                                  help="a seeing bank (default: the validation split's newest)")
    sharpen_evaluate.add_argument("--manifest", type=Path)

    sheet = commands.add_parser("sheet-noise", help="make a trained noise model's contact sheets")
    sheet.add_argument("--name", required=True)
    sheet.add_argument("--sheet", action="append", dest="sheets_named", help="only this sheet (default: all)")

    clean = commands.add_parser("clean-sharpen", help="denoise a sharpening pair bank's windows with a noise run, "
                                                      "kept beside it, for --pass-noise")
    clean.add_argument("--bank", required=True, type=Path, action="append", dest="banks")
    clean.add_argument("--noise-run", required=True, help="the noise run, under its store's models/noise/")
    clean.add_argument("--noise-store", type=Path, help="the store holding the noise run (default: --store)")

    sharpen_sheet = commands.add_parser("sheet-sharpen", help="make a trained PSF model's contact sheets, full-size "
                                                              "images and whole-image measures (reads the manifest "
                                                              "only)")
    sharpen_sheet.add_argument("--name", required=True)
    sharpen_sheet.add_argument("--sheet", action="append", dest="sheets_named", help="only this sheet (default: all)")

    evaluate = commands.add_parser("evaluate-noise", help="measure a trained noise model on a split")
    evaluate.add_argument("--name", required=True)
    evaluate.add_argument("--split", default="validation")
    evaluate.add_argument("--limit", type=int, help="at most this many pairs")
    evaluate.add_argument("--set", action="append", dest="sets", help="only this set id")
    evaluate.add_argument("--high-amount", type=float, default=1.0,
                          help="the share of the pixel-scale grain the model finds that is taken out (1: all of it)")
    evaluate.add_argument("--low-amount", type=float, default=1.0,
                          help="the share of the wider mottle the model finds that is taken out (1: all of it)")
    evaluate.add_argument("--noise-map", choices=("raw", "model", "image", "constant"),
                          help="the kind of noise map the network reads, in place of the one it trained with")
    evaluate.add_argument("--band-scale", type=float, default=5.0,
                          help="the Gaussian scale in pixels dividing the grain from the mottle")
    evaluate.add_argument("--match-flux", action="store_true",
                          help="measure against each B half scaled onto its A half's background and bright flux")
    evaluate.add_argument("--holes", action=argparse.BooleanOptionalAction,
                          help="run the hole guard, or not, whatever the run trained with")

    stability = commands.add_parser("stability-noise",
                                    help="measure how far a trained noise model throws pixels on spread noise")
    stability.add_argument("--name", required=True)

    bundle = commands.add_parser("bundle-noise", help="write a trained noise model as a bundle Seiza's runtime opens")
    bundle.add_argument("--name", required=True, help="the run")
    bundle.add_argument("--out", required=True, type=Path, help="the bundle's directory")
    bundle.add_argument("--bundle-name", required=True, help="the model's name on its card")
    bundle.add_argument("--noise-map", choices=("image", "constant"),
                        help="what the network reads, in place of the map it trained with")
    bundle.add_argument("--parity", action="append", default=[],
                        help="a backend the model has passed parity on (default: any may run it)")
    bundle.add_argument("--holes", action=argparse.BooleanOptionalAction,
                        help="have the runtime keep the image's holes, pixels far below its sky, from the network "
                             "and put them back after, as the card's reading.holes says (default: as the run "
                             "trained)")

    split = commands.add_parser("split", help="assign sky groups and train/validation/test splits")
    split.add_argument("--hold-out", help="a telescope whose groups are all test")

    decide = commands.add_parser("decide", help="record a review decision")
    decide.add_argument("subject", help="a set id")
    decide.add_argument("kind", choices=("set", "starless"))
    decide.add_argument("decision", choices=("accept", "reject"))
    decide.add_argument("--reviewer", required=True)
    decide.add_argument("--note")

    commands.add_parser("status", help="summarize the manifest")

    arguments = parser.parse_args(argv)
    store = Store(arguments.store, arguments.scratch)
    only = set(arguments.sets) if getattr(arguments, "sets", None) else None
    if arguments.command in BANKS_ONLY:
        print(json.dumps(run_sharpen(arguments, store), indent=2, default=str))
        return 0
    if arguments.command in READ_ONLY:
        # These keep what they make beside the store, so they only read the
        # manifest, which another machine's pass may be writing meanwhile.
        with Manifest.read_only(store.manifest_path) as manifest:
            summary = run(arguments, manifest, store, only)
        print(json.dumps(summary, indent=2, default=str))
        return 0
    with Manifest(store.manifest_path) as manifest:
        if arguments.command == "decide":
            manifest.record_review(
                arguments.subject, arguments.kind, arguments.decision, arguments.reviewer,
                arguments.note,
            )
            return 0
        if arguments.command == "status":
            print(json.dumps(status(manifest), indent=2))
            return 0
        recorded = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(arguments).items()
            if key not in ("store", "scratch", "command")
        }
        with manifest.pass_run(arguments.command, versions(), recorded) as summary:
            summary.update(run(arguments, manifest, store, only))
        print(json.dumps(summary, indent=2, default=str))
    return 0


def run(arguments: argparse.Namespace, manifest: Manifest, store: Store, only: set[str] | None) -> dict:
    command = arguments.command
    if command == "crawl":
        from .crawl import crawl

        roots = manifest.meta("roots") or {}
        corpus = arguments.corpus or (Path(roots["corpus"]) if roots.get("corpus") else None)
        library = arguments.library or (Path(roots["library"]) if roots.get("library") else None)
        calibration = arguments.calibration or (Path(roots["calibration"]) if roots.get("calibration") else None)
        if corpus is None and library is None:
            raise SystemExit("crawl needs --corpus or --library the first time")
        manifest.set_meta(
            "roots",
            {**roots, "corpus": str(corpus) if corpus else None, "library": str(library) if library else None,
             "calibration": str(calibration) if calibration else None},
        )
        return crawl(manifest, corpus, library, calibration=calibration, workers=arguments.workers)
    if command == "calibrate":
        from .calibrate import calibrate

        return calibrate(manifest, only=only)
    if command == "measure":
        from .measure import measure

        return measure(
            manifest, only=only, again=arguments.again, catalog_dir=arguments.catalog_dir,
            limit=arguments.limit,
        )
    if command == "stacks":
        from .stacks import import_stacks

        roots = manifest.meta("roots") or {}
        stacks_root = arguments.psf_guard or (Path(roots["psf_guard"]) if roots.get("psf_guard") else None)
        if stacks_root is None:
            raise SystemExit("stacks needs --psf-guard the first time")
        if not roots.get("corpus"):
            raise SystemExit("stacks needs the corpus crawled first")
        manifest.set_meta("roots", {**roots, "psf_guard": str(stacks_root)})
        return import_stacks(
            manifest, store, stacks_root, Path(roots["corpus"]), again=arguments.again,
            limit=arguments.limit,
        )
    if command == "dark-noise":
        from .dark_noise import SIZES
        from .dark_noise import run as make_dark_noise

        return make_dark_noise(
            manifest, store, sets=only, sizes=tuple(arguments.sizes or SIZES), again=arguments.again,
            limit=arguments.limit,
        )
    if command == "multi-night":
        from . import multi_night

        if arguments.list:
            return {"objects": multi_night.survey(manifest)}
        names = set(arguments.objects) if arguments.objects else None
        if arguments.check_seeing:
            return {"judged": multi_night.check_seeing(manifest, store, names=names)}
        source_map = dict(entry.split("=", 1) for entry in arguments.source_map)
        return multi_night.run(
            manifest, store, names=names,
            minimum_hours=arguments.min_hours if arguments.min_hours is not None else multi_night.MINIMUM_HOURS,
            source_map=source_map, again=arguments.again, limit=arguments.limit, keep_scratch=arguments.keep_scratch,
            seeing=arguments.seeing,
        )
    if command == "train-noise":
        from .noise_model import DEEP_TRAINING_KINDS, bank, dark_noise_bank, deep_pairs, train

        extra = {}
        if arguments.dark_noise_fraction and arguments.deep_fraction:
            raise SystemExit("a run takes one second bank: --dark-noise-fraction or --deep-fraction")
        if arguments.deep_fraction:
            chosen = deep_pairs(manifest, "train", only, kinds=DEEP_TRAINING_KINDS)
            if not chosen:
                raise SystemExit("no built deep-stack halves in the train split")
            extra = {
                # Cut as the pairs' bank is, so every tile of a batch is
                # read, weighed and guarded alike.
                "extra": bank(store, chosen, seed=arguments.seed, noise_map=arguments.noise_map, sky=arguments.sky,
                              match_flux=arguments.match_flux, agree=arguments.agree,
                              weight_map=arguments.weight_map, holes=arguments.holes,
                              workers=arguments.bank_workers),
                "extra_fraction": arguments.deep_fraction,
                "extra_label": "deep",
            }
        if arguments.dark_noise_fraction:
            if arguments.holes or arguments.weight_map or arguments.match_flux:
                raise SystemExit("the dark-noise bank is cut without the guard, the stack's weight map or "
                                 "matched flux: leave out --dark-noise-fraction or those")
            extra = {
                "extra": dark_noise_bank(store, manifest, sets=only, seed=arguments.seed, noise_map=arguments.noise_map,
                                         sky=arguments.sky),
                "extra_fraction": arguments.dark_noise_fraction,
                "extra_label": "dark-noise",
            }
        from .sheets import render_saved

        record = train(
            store, manifest, name=arguments.name, steps=arguments.steps, width=arguments.width,
            levels=arguments.levels, tile=arguments.tile, batch=arguments.batch,
            learning_rate=arguments.learning_rate, sets=only, seed=arguments.seed,
            loss_weight=arguments.loss_weight, map_jitter=arguments.map_jitter,
            validation_sets=set(arguments.validation_sets) if arguments.validation_sets else None,
            noise_map=arguments.noise_map, huber=arguments.huber, huber_above=arguments.huber_above,
            structure_sampling=arguments.structure_sampling, outlier_count=arguments.outlier_count,
            outlier_floor=arguments.outlier_floor, clip_grad=arguments.clip_grad, match_flux=arguments.match_flux,
            agree=arguments.agree, outlier_scale=arguments.outlier_scale, noise_plane=arguments.noise_plane,
            scale_loss=arguments.scale_loss, scale_weight=arguments.scale_weight, rewind=arguments.rewind,
            precision=arguments.precision, prefetch=arguments.prefetch, batch_threads=arguments.batch_threads,
            sky=arguments.sky, resample=arguments.resample, sharpen=arguments.sharpen, gradient=arguments.gradient,
            bin_fraction=arguments.bin_fraction, dark_spot_count=arguments.dark_spot_count, spread=arguments.spread,
            weight_map=arguments.weight_map, holes=arguments.holes, bank_workers=arguments.bank_workers, **extra,
        )
        return {**record, "sheets": render_saved(store, manifest, arguments.name)}
    if command == "psf-fields":
        from . import psf_fields

        products = psf_fields.eligible(manifest, kinds=arguments.kinds or psf_fields.SOURCE_KINDS,
                                       splits=arguments.splits or psf_fields.SPLITS, sets=only)
        return psf_fields.run(manifest, store, products, workers=arguments.workers, again=arguments.again,
                              limit=arguments.limit)
    if command == "cut-sharpen":
        from . import sharpen_bank

        work = arguments.work or store.scratch.parent / "sharpen"
        found: dict = {}
        for split in arguments.splits or ("train", "validation"):
            for name in arguments.banks or ("deep", "seeing"):
                if name == "deep":
                    sources, left = sharpen_bank.deep_sources(manifest, split, only), []
                    per_source = arguments.deep_windows
                elif name == "binned":
                    sources, left = sharpen_bank.binned_sources(store, manifest, split, only), []
                    per_source = arguments.deep_windows
                else:
                    sources, left = sharpen_bank.seeing_pairs(manifest, split, only)
                    per_source = arguments.pair_windows
                sources = sources[:arguments.limit]
                if not sources:
                    found[f"{name}:{split}"] = {"sources": 0, "left": left}
                    continue
                banked = sharpen_bank.cut(store, sources, label=name, split=split, work=work, seed=arguments.seed,
                                          per_source=per_source, workers=arguments.workers)
                found[f"{name}:{split}"] = {"directory": str(banked["directory"]), "left": left,
                                            **{key: banked["done"][key] for key in ("windows", "sources", "dropped",
                                                                                    "summary")}}
        return found
    if command == "sheet-noise":
        from .sheets import render_saved

        named = set(arguments.sheets_named) if arguments.sheets_named else None
        return render_saved(store, manifest, arguments.name, only=named)
    if command == "sheet-sharpen":
        from . import sharpen_sheets

        named = set(arguments.sheets_named) if arguments.sheets_named else None
        return sharpen_sheets.render_saved(store, manifest, arguments.name, only=named)
    if command == "evaluate-noise":
        from .noise_model import NoiseBands, evaluate_saved

        bands = NoiseBands(high=arguments.high_amount, low=arguments.low_amount, scale_px=arguments.band_scale)
        return evaluate_saved(store, manifest, arguments.name, split=arguments.split, limit=arguments.limit, sets=only,
                              bands=bands, match_flux=arguments.match_flux, noise_map=arguments.noise_map,
                              holes=arguments.holes)
    if command == "stability-noise":
        from .noise_model import stability_saved

        return stability_saved(store, arguments.name)
    if command == "bundle-noise":
        from .bundle import write_bundle

        return write_bundle(store, arguments.name, arguments.out, name=arguments.bundle_name,
                            noise_map=arguments.noise_map, parity=tuple(arguments.parity), holes=arguments.holes)
    if command == "split":
        from .splits import assign_splits

        return assign_splits(manifest, hold_out=arguments.hold_out)
    if command == "build":
        from .build import build

        return build(
            manifest, store, only=only, again=arguments.again, limit=arguments.limit,
            keep_scratch=arguments.keep_scratch, pairs_only=arguments.pairs_only,
        )
    if command == "recipes":
        from .measure import refresh_recipes

        return refresh_recipes(manifest, dry_run=arguments.dry_run)
    if command == "process":
        from .build import process

        return process(
            manifest, store, only=only, limit=arguments.limit, catalog_dir=arguments.catalog_dir,
            keep_scratch=arguments.keep_scratch,
        )
    if command == "review":
        from .review import review

        return review(manifest, store, only=only)
    raise AssertionError(command)


def run_sharpen(arguments: argparse.Namespace, store: Store) -> dict:
    """`train-sharpen` and `evaluate-sharpen`."""
    from . import sharpen_model

    if arguments.command == "clean-sharpen":
        from . import sharpen_clean

        noise_store = Store(arguments.noise_store, arguments.scratch) if arguments.noise_store else store
        return {str(bank): sharpen_clean.clean(noise_store, arguments.noise_run, bank) for bank in arguments.banks}
    work = arguments.work or store.scratch.parent / "sharpen"
    manifest = arguments.manifest or (store.manifest_path if store.manifest_path.exists() else None)
    if arguments.command == "evaluate-sharpen":
        record, _ = sharpen_model.load_run(store, arguments.name)
        recorded = sharpen_model.BankChoice.recorded(record["config"]["choice"])
        choice = sharpen_model.BankChoice(
            kernel=recorded.kernel, seeing=recorded.seeing,
            validation_kernel=str(arguments.kernel_bank or sharpen_model.find_bank(work, "deep", "validation")),
            validation_seeing=tuple(str(path) for path in arguments.seeing_banks
                                    or [sharpen_model.find_bank(work, "seeing", "validation")]),
            excluded=recorded.excluded, object_cap=recorded.object_cap,
            manifest=str(manifest) if manifest else recorded.manifest, focus_check=recorded.focus_check,
            left_out_kinds=recorded.left_out_kinds, multi_night_share=recorded.multi_night_share,
            more_kernels=recorded.more_kernels)
        settings = sharpen_model.settings_from(record["config"]["settings"])
        return sharpen_model.evaluate_saved(store, arguments.name, choice.validation(settings), limit=arguments.limit,
                                            seed=arguments.seed)
    if arguments.resume:
        return summary_of(sharpen_model.resume_run(store, arguments.name, stop_after=arguments.stop_after,
                                                   prefetch=arguments.prefetch,
                                                   batch_threads=arguments.batch_threads,
                                                   workers=arguments.workers))
    if len(arguments.mix) != 3:
        raise SystemExit("--mix takes three shares: known-kernel, seeing and supplied")
    settings = sharpen_model.Settings(
        tile=arguments.tile, kernel_ratios=tuple(arguments.kernel_ratios),
        kernel_ellipticity=arguments.kernel_ellipticity, kernel_noise=arguments.kernel_noise,
        identity=arguments.identity, exposure=arguments.exposure, floor_px=arguments.floor_px,
        max_ratio=arguments.max_window_ratio, min_ratio=arguments.min_window_ratio,
        points_empty=arguments.points_empty, centroid_offset=arguments.centroid_offset,
        ratio_source=arguments.ratio_source, residual=arguments.residual, nonstellar=arguments.nonstellar,
        equal_share=arguments.equal_share, fill_clipped=arguments.fill_clipped,
        pass_noise=arguments.pass_noise, real_kernels=arguments.real_kernels)
    loss = sharpen_model.LossSettings(epsilon=arguments.charbonnier, noise_cap=arguments.noise_cap,
                                      gradient_weight=arguments.gradient_weight,
                                      gradient_scales=arguments.gradient_scales, flux_weight=arguments.flux_weight)
    choice = sharpen_model.BankChoice(
        kernel=str(arguments.kernel_bank or sharpen_model.find_bank(work, "deep", "train")),
        seeing=tuple(str(path) for path in arguments.seeing_banks or [sharpen_model.find_bank(work, "seeing", "train")]),
        validation_kernel=str(arguments.validation_kernel_bank or sharpen_model.find_bank(work, "deep", "validation")),
        validation_seeing=tuple(str(path) for path in arguments.validation_seeing_banks
                                or [sharpen_model.find_bank(work, "seeing", "validation")]),
        pairs=tuple((str(Path(path)), weight) for path, weight in arguments.pairs_banks),
        excluded=tuple(pair for pair in arguments.exclude_pairs.split(",") if pair),
        object_cap=arguments.object_cap, manifest=str(manifest) if manifest else None,
        focus_check=arguments.focus_check,
        left_out_kinds=tuple(kind for kind in arguments.leave_out_kinds.split(",") if kind),
        multi_night_share=arguments.multi_night_share,
        more_kernels=tuple((str(Path(path)), weight) for path, weight in arguments.more_kernels))
    record = sharpen_model.train(
        store, name=arguments.name, choice=choice, steps=arguments.steps, width=arguments.width,
        levels=arguments.levels, batch=arguments.batch, learning_rate=arguments.learning_rate, seed=arguments.seed,
        log_every=arguments.log_every, checkpoint_every=arguments.checkpoint_every, settings=settings, loss=loss,
        mix=tuple(arguments.mix), clip_grad=arguments.clip_grad, rewind=arguments.rewind,
        precision=arguments.precision, validation_batches=arguments.validation_batches,
        validation_windows=arguments.validation_windows, stop_after=arguments.stop_after,
        prefetch=arguments.prefetch, batch_threads=arguments.batch_threads, workers=arguments.workers)
    return summary_of(record)


def summary_of(record: dict) -> dict:
    """A run's record without its windows' detail."""
    validation = record.get("validation") or {}
    return {**record, "validation": {key: ({inner: value for inner, value in entry.items() if inner != "windows_detail"}
                                           if isinstance(entry, dict) else entry)
                                     for key, entry in validation.items()}}


def status(manifest: Manifest) -> dict:
    def counts(query: str) -> dict:
        return {str(row[0]): row[1] for row in manifest.rows(query)}

    return {
        "frames": counts("SELECT origin || ':' || COALESCE(role, 'unread'), COUNT(*) FROM frames GROUP BY 1"),
        "sets": manifest.rows("SELECT COUNT(*) FROM sets")[0][0],
        "calibration": counts("SELECT calibration_status, COUNT(*) FROM sets GROUP BY 1"),
        "measurement": counts("SELECT measurement_status, COUNT(*) FROM sets GROUP BY 1"),
        "split": counts("SELECT split, COUNT(*) FROM sets GROUP BY 1"),
        "build": counts("SELECT build_status, COUNT(*) FROM sets GROUP BY 1"),
        "stacks": counts("SELECT telescope || ':' || status || ':' || COALESCE(split, 'unsplit'), COUNT(*) FROM stacks GROUP BY 1"),
        "products": counts("SELECT kind || ':' || status, COUNT(*) FROM products GROUP BY 1"),
        "reviews": counts("SELECT kind || ':' || decision, COUNT(*) FROM reviews GROUP BY 1"),
        "passes": [
            dict(row) for row in manifest.rows(
                "SELECT name, started, finished FROM passes ORDER BY started DESC LIMIT 10"
            )
        ],
    }


if __name__ == "__main__":
    sys.exit(main())

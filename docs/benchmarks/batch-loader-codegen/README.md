# Batch loader code generation

Measured on Linux x86_64 with rustc 1.99.1 (58cf8f9c3 2026-10-13).
The public batch API accepts a generic frame loader. Previously that loader
specialized the entire private three-pass integration implementation, including
its Rayon closures, for each caller. In the largest LLVM module, 23,940 of
27,139 defined functions belonged to `batch::integrate_frames` specializations.

The private integration boundary now borrows a dynamic frame loader. Dispatch
happens once per loaded frame; channel count and weighting remain compile-time
parameters of the pixel loops. Public signatures and pixel arithmetic are unchanged.

## Measurements

| Measurement | Before | After |
| --- | ---: | ---: |
| Stacking test pre-LTO bitcode, 256 modules | 2,765,313,196 bytes | 370,064,108 bytes |
| Largest module | 838,306,588 bytes | 58,126,740 bytes |
| Workspace test build peak RSS | 20,021,396 KiB; SIGKILL | 3,553,744 KiB; success |

Bitcode size fell 86.6%. The before bitcode is from the original failed test
build. The resource control is a subsequent failed build with an unsuccessful
planner-only callback experiment (since reverted); its bitcode was
2,760,791,452 bytes, within 0.2% of the original. It left the batch loader
generic. These are diagnostic observations, not a controlled clean-build timing
comparison. Dependencies were warm; only the stacking test incremental cache was
removed before each measured resource run. The failed run cannot establish a
complete-build time or memory requirement. Raw resource reports are included.

## Reproduction

```sh
# Preserve any useful local build artifacts before removing generated caches.
# Find the seiza_stacking incremental directory for the unit-test target, then
# remove that directory to force its code generation on each revision.
/usr/bin/time -v cargo test --workspace --all-features --no-run
```

Sum the sizes of `*.pre-lto.bc` in that target's incremental directory to
compare code generation. Compiler versions and Cargo profiles affect these
numbers; compare with the same configuration.

## Output verification

All 70 stacking, three stretch, and two pipeline output-version checks pass
unchanged. `registered_frames_a_host_integrates_keep_their_output` exercises
the affected path: temporarily doubling loaded samples in `load_unchanged`
made that check fail, then the mutation was removed. No output version or
recorded summary changes are needed.

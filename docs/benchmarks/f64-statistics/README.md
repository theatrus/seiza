# Shared f64 median and robust sigma

Measured against `fcdaaec` on 2026-11-06. This changes only how the shared
statistics find the middle values: partial selection instead of sorting,
selection in the deviation buffer instead of a second copy, and no copy for
an ascending or descending median. Public signatures and statistical
conventions stay the same.

## Reproduce

```sh
cargo build --release -p seiza-stats --example robust_profile
taskset -c 9 target/release/examples/robust_profile > timings.csv
/usr/bin/time -v target/release/examples/robust_profile sigma sorted 10000000
/usr/bin/time -v target/release/examples/robust_profile sigma native 10000000
```

The example retains the original sorting implementations as references and
checks the result bits before timing each case. Each CSV row reports the
median of seven trials per implementation, alternating their order. Each
trial processes about one million samples, with at least one and at most
10,000 calls. Data generation and result validation are outside the timed
region; allocations are inside it.

The deterministic synthetic sample is a level of 1000 with uniform noise
of ±15 and an added 500 every 101 samples. Other cases sort it ascending,
descending, or round it to introduce ties. These measure statistics, not a
complete calibration or stacking workflow. They do not establish an
end-to-end speedup on camera images.

Environment: Linux x86_64 virtual machine, 10 logical CPUs reporting Intel
Core i9-14900K, Rust 1.99.1, release profile, one benchmark thread pinned to
CPU 9. Timings vary with the host and allocator. Full results, including
ordered samples and ties, are in [timings.csv](timings.csv).

## Results

For the unsorted synthetic sample:

| Statistic | Samples | Sorting (µs) | Selection (µs) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Median | 256 | 1.396 | 0.490 | 2.85× |
| Robust sigma | 256 | 1.536 | 0.556 | 2.76× |
| Median | 4,096 | 31.706 | 6.164 | 5.14× |
| Robust sigma | 4,096 | 32.978 | 6.541 | 5.04× |
| Median | 65,536 | 1,070.412 | 97.016 | 11.03× |
| Robust sigma | 65,536 | 1,269.370 | 104.010 | 12.20× |
| Median | 1,048,576 | 20,353.676 | 2,273.177 | 8.95× |
| Robust sigma | 1,048,576 | 24,260.893 | 3,959.237 | 6.13× |

The initial selection prototype regressed on monotonic inputs; the final
implementation detects them before copying. Very short ordered calls are
still dominated by a few nanoseconds of call and branch overhead; the raw
CSV includes those results rather than assuming every case improves.

For 10 million samples, robust sigma uses one 80,000,000-byte temporary
buffer rather than two. Peak RSS falls from 236,416 to 158,080 KiB (33.1%).
Separate-process measurements are recorded in
[memory-sorted.txt](memory-sorted.txt) and
[memory-native.txt](memory-native.txt). The allocation regression test also
enforces the one-buffer budget independently of platform RSS accounting.

## Correctness and output versions

- Statistics tests compare against the original sorted reference on empty,
  odd/even, random-bit, ordered and repeated samples, including signed zero,
  subnormals, infinities and signed NaNs. Finite results and odd median NaN
  payloads match bit for bit. Arithmetic NaNs are compared by classification
  because Rust does not promise their payload bits. Inputs stay unchanged.
- The allocation test fails against the original library: sigma allocates
  1,048,576 bytes for 65,536 samples, exceeding the 524,288-byte budget. It
  passes in debug and release with selection and also verifies allocation-free
  medians on monotonic samples.
- Background and calibration tests pass (23 background unit tests, one
  background doctest, 193 calibration unit tests; one pre-existing timing
  test is ignored).
- All output-version tests pass unchanged: 70 stacking, three stretch and
  two pipeline tests. Coverage was checked by adding 0.1 to odd f64 medians:
  `a_local_background_stack_through_changing_seeing_keeps_its_output` failed.
  The mutation was removed before the final checks. Output versions and
  recorded summaries are unchanged.
- Statistics Clippy, formatting, Rust 1.89 checks and binding parity pass.
  There are no new public APIs or binding changes.

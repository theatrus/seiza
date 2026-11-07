# Complete, bounded compressed transfers

The public `seiza_download::bundle::stream_to_temp` verified decoded output,
but two streaming details weakened its encoded-body checks:

- The zstd decoder stopped at the first frame, which is not necessarily the
  end of the HTTP response. A delayed HTTP chunk could remain unread and
  escape the encoded size/hash accounting altogether.
- Encoded byte count was checked only after decompression. An overrun could
  be hashed, decompressed, and written before its size was refused.

The stream now checks encoded size before hashing or passing an offending
chunk to the decoder, and decodes all members through HTTP EOF. The size
refusal is created as the native `Error::Size` variant, carried through
StreamReader's IO error, and preserved for callers without matching text.
The existing global decoded-size limit and both SHA-256 checks remain active.

## Reproductions

Measured on Linux x86_64 with rustc 1.99.1. These are exact functional/disk
measurements, not throughput benchmarks. Raw native test logs are included.

| Native case | Before | After |
| --- | --- | --- |
| Extra HTTP byte gated until the canonical payload is decoded | Success despite oversized HTTP body | Typed encoded-size error |
| One chunk exceeds its encoded size by one byte | 131,072 decoded bytes written before refusal | 0 decoded bytes written |
| Valid two-frame zstd body | Refused: decoded 11 of the expected 23 bytes | Verified exact 23-byte canonical payload |

The delayed-tail server waits on a channel released by the client's actual
decode progress before sending the extra chunk. No sleeps or assumptions
about TCP packet boundaries are needed. The other cases use a single small
HTTP chunk. Servers consume the request headers and bound blocking reads;
client read timeouts bound response inactivity.

```sh
cargo test -p seiza-download bundle::tests
cargo test -p seiza-download --release
```

Further native cases refuse trailing garbage even when its complete encoded
hash matches, and ensure later frames cannot reset the decoded-output bound.
Native tests call the same typed transfer API used by catalog and satellite
bundle consumers.

## Output scope

This changes download validation and framing, not image processing. Canonical
payloads are checked byte for byte and by SHA-256. All 75 existing
stacking/stretch/pipeline output-version checks pass unchanged; no
OUTPUT_VERSION or recorded image-output summary changes are needed.

## Final validation

On the branch based on PR #446's compile-memory fix:

- Complete all-features workspace suite: 2,267 passed, 19 ignored.
- Fresh maturin build and Python runtime suite: 1,637 passed, 21 skipped.
- All 31 download tests pass in debug and release builds.
- All 75 image output-version checks remain unchanged.
- Workspace/Python formatting and Clippy, Rust 1.89 workspace
  all-targets/all-features locked checking, and binding parity pass.

The fixes live in the shared native transfer API, so catalog and satellite
consumers and their C/Python bindings receive the same behavior.

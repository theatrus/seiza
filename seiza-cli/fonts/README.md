# Fonts

`seiza solve --sky-map` and `seiza solve-blind --sky-map` draw their labels
with Inter, embedded in the binary.

- Source: Inter 4.1 by The Inter Project Authors,
  <https://github.com/rsms/inter/releases/tag/v4.1> (`Inter-4.1.zip`,
  `extras/ttf/Inter-Regular.ttf` and `Inter-SemiBold.ttf`).
- Licence: SIL Open Font License 1.1, full text in
  [`LICENSE-Inter.txt`](LICENSE-Inter.txt). Inter declares no Reserved Font
  Name.
- Modified: subset to Basic Latin, Latin-1, Latin Extended-A, Greek, general
  punctuation, primes, arrows and the minus sign, with hinting and OpenType
  layout features removed, using
  `pyftsubset --unicodes="U+0020-007E,U+00A0-017F,U+0391-03C9,U+2010-2027,U+2032-2033,U+2190-2193,U+2212" --layout-features='' --no-hinting --desubroutinize --name-IDs='*' --name-languages='*'`
  (fonttools). The name table, with its copyright and licence records, is
  kept whole. That cuts each file from about 410 KB to 27 KB.

# tests/fixtures

**synthetic_KNOWN_FAKE_do_not_cite.csv** is not campaign evidence. It is a
hand-built trade log: 240 ETHUSDT trades whose prices are constants (long
3000 -> 3015, short 3000 -> 2985, every trade exactly +0.5% gross), used only
to exercise the judge's PASS path in tests and to demonstrate the boundary of
what a trade-log audit can see.

It passes C1-C5 by construction - constant wins survive any fee, beat every
coin-flip bootstrap, are identical in both halves and in every regime - and
that is the point: **nothing in a trade log distinguishes a fabricated
winner from a real one.** The judge's answer to that is C6, which on this
file is UNVERIFIABLE (no signal_time), and the "What this is not" section of
the top-level README. Never cite this file as a result.

`results/` holds only artifacts the campaign actually produced.

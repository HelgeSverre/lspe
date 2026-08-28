# SCBE Attention-Noise Interpolation Amendment

**Version:** 0.3

**Status:** Frozen after v2 control calibration and before any behavioral generation

SCBE v2 matched the random-basis and temperature controls. Its attention-noise
curve bracketed but skipped the frozen SCCF target KL `0.0573009`:

- sigma `0.40`: median KL `0.0350211`;
- sigma `0.50`: median KL `0.0846991`.

The accepted 25% interval is `0.0429757–0.0716261`. No pilot or confirmation
response has been generated in v1 or v2.

This amendment corrects the v2 document's over-strict statement that a second
grid miss must permanently end SCBE. That statement would make a coarse
nuisance-control grid—not the scientific hypothesis—the terminal test. The
correction is recorded explicitly rather than silently relaxed.

SCBE v3 changes only the attention-noise grid to
`[0.42, 0.44, 0.45, 0.46, 0.48]`. The already sufficient random-basis v2 grid is
retained and rerun under the new source lock. All control tolerances, SCCF
parameters, data, behavioral outcomes, competence gates, statistics, and stop
rules remain unchanged. There will be no further control-grid amendment if v3
does not match.

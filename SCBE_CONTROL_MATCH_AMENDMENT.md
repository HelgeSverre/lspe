# SCBE Control-Matching Amendment

**Version:** 0.2

**Status:** Frozen after v1 control calibration and before any behavioral generation

The first SCBE run stopped at `CONTROL_MATCH_FAILED`. SCCF, sham, and
temperature controls passed, but the frozen random-basis and attention-noise
grids did not reach the SCCF target KL of `0.0573009`:

- random basis ended at alpha `0.80`, median KL `0.0058255`;
- attention noise ended at sigma `0.32`, median KL `0.0197729`.

Both curves were still rising at their upper boundary. No pilot or confirmation
response was generated, so no behavioral outcome has been observed.

SCBE v2 changes only the two control candidate grids:

- random-basis alpha becomes
  `[0.80, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00]`;
- attention-noise sigma becomes
  `[0.32, 0.40, 0.50, 0.65, 0.80, 1.00, 1.30]`.

Random-basis alpha above one is allowed only for the negative control. It scales
the same selected transform eigenvalue gains beyond full whitening in a frozen
random orthogonal basis; it does not alter SCCF.

The 25% KL-match tolerance, entropy tolerance, SCCF mask and alpha, sham,
datasets, sampling, pilot gates, confirmation gates, outcomes, statistics, and
stop rules in [SCCF_BEHAVIORAL_SPEC.md](SCCF_BEHAVIORAL_SPEC.md) are unchanged.
If either extended curve still cannot match, SCBE stops permanently at control
calibration.

# DCF Calibration v2 Amendment

**Status:** Frozen before v2 dose telemetry and before any DCF held-out intervention

DCF v1 mapping passed, but its coarse calibration grid contained no eligible
dose. At `alpha=0.35`, tuning-fold top-1 agreement passed while output KL
remained below the frozen active-dose band. At `alpha=0.50`, output KL entered
the band while top-1 agreement failed. No DCF intervention has been evaluated
on folds 1 or 3.

V2 tests a denser, fully specified grid inside that bracket:

```text
[0.38, 0.40, 0.42, 0.44, 0.46, 0.48]
```

Nothing else changes:

- the selected layer window remains layers 15 through 22;
- transforms remain fitted only on folds 0 and 2;
- folds 1 and 3 remain untouched until one dose is selected;
- the correlation reduction requirement remains at least 15%;
- effective-rank increase remains at least 10%;
- median output KL must remain between 0.005 and 0.08 nats;
- mean top-1 agreement must remain at least 80%;
- all numerical and moment-preservation gates remain unchanged;
- selection remains the lowest eligible alpha;
- if no v2 candidate qualifies, DCF stops with `NO_ELIGIBLE_DOSE`;
- if one qualifies, it is evaluated exactly once under the original held-out
  mechanism gates.

The v1 result remains valid and is not reinterpreted. V2 is a tuning-only
resolution increase motivated by the observed bracket, not a relaxed gate.

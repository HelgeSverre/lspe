# FNDE v2 Mapping-Only Feasibility Note

The first v2 analysis execution produced no held-out ARI values: every candidate
was assigned the sentinel `-1.0` because at least one tuning-eligible node was
isolated in a held-out fold. No held-out community comparison was therefore
available when this clarification was written.

Two fail-closed clarifications are frozen before recomputation:

1. Held-out ARI is computed on the fixed point of tuning-selected nodes that are
   non-isolated in both held-out fold graphs. At least 80% of tuning-selected
   nodes must remain; lower coverage fails the candidate.
2. A nominal community is not meaningful if it contains only a handful of
   heads. Every final community must contain at least 5% of eligible nodes.

Held-out ARI and coverage remain forbidden inputs to candidate selection. The
selected density and community count are still determined solely by tuning ARI,
tuning-node count, lower density, and lower community count. The ARI threshold
remains `0.70`.

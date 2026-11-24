# Amendment 4: organ-system knowledge graph analyses (12 Oct 2026)

Written before running. These analyses are exploratory and were added after the held-out data had been opened. They do not change the primary model, endpoint or results.

**Framing.** The organ-system constraint is restated as an expert-defined knowledge graph *G = (V, E)*:

- **Nodes (V):** the 11 organ systems.
- **Edges (E):** the 8 prespecified pathophysiological relations.
- **Variables:** each predictor is attached to one node.

The allowed interaction sets are the nodes and edges of *G*. The exact decomposition is therefore a sum of node terms and edge terms.

## K1. Do the knowledge edges carry more interaction than other system pairs?

**Model.** An "open-graph" model allows every system pair (all 55 pairs). It uses the same hyperparameters and seed as the primary model (variant `all55` in amendment 3). It is refitted on all development patients, keeping its 5 bagged members, and each held-out prediction is decomposed exactly into node and pair terms.

**Edge strength.** For each of the 55 pairs, edge strength is the standard deviation of that pair's term across held-out assessments.

**Primary test.**

- Statistic: the mean rank (1 = strongest) of the 8 knowledge edges among the 55 pairs.
- Null distribution: 100,000 random selections of 8 pairs out of the 55.
- One-sided p-value: the share of random selections with a mean rank at least as good as the observed one.

**Interpretation.**

- p < 0.05: the data support the knowledge edges.
- Otherwise: the knowledge edges are not distinguished from other pairs by the data.

**Size adjustment (sensitivity).** Larger systems have more variables and may produce larger pair terms. To allow for this, regress log edge strength on log(n_i × n_j), where n is the number of variables in a system, together with a knowledge-edge indicator. Report the indicator's coefficient with a 95% CI.

**Stability.** Repeat the primary test separately within each of the 5 bagged members.

## K2. Candidate relations (exploratory)

A pair outside the 8 knowledge edges is listed as a candidate relation if both of the following hold:

- it ranks within the top 8 pairs by edge strength; and
- it ranks within the top 8 in at least 4 of the 5 bagged members.

Candidates are hypotheses for future work. They are not used to change the model.

## K3. Patient-level graph explanation (illustration)

For held-out patients with events, the primary model's node and edge terms are drawn on the organ-system graph:

- node shading shows the node term;
- edge width shows the edge term.

Each patient is shown at 48, 24 and 6 hours before the event. Patients are selected by a fixed rule, not by appearance: the two patients with the most assessments in the 72 hours before their event.

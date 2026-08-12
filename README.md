# eigen-spectral-probe

The spectral structure of SmolLM2 weight matrices — eigenvalue decomposition to semantic probing to deployment-time spectral pruning. CPU-only, no assumptions.

---

## Units

| # | Unit | Claim to test | Status |
|---|------|-------------|--------|
| 1 | Spectra of real weights | Real SmolLM2 matrices have measurable spectral decay; α depends on matrix type | ✅ Complete (4 experiments measured) |
| 2 | Semantic directions in spectral space | Leading singular directions encode common structure; trailing encode noise | ✅ Complete (4 experiments measured) |
| 3 | Truncation and output fidelity | Truncation at k90 gives bounded, measurable output drift | ✅ Complete (4 experiments measured) |
| 4 | Scale-out: spectra 135M→1.7B | Spectral structure partly architectural, partly size-dependent | ⬜ Planned |
| 5 | Weight probing via decomposition | Semantic properties readable from spectral components | ✅ Complete (4 experiments measured) |
| 6 | Deployment: spectral pruning | Pruning trades output drift for CPU speedup, measurably | ✅ Complete (4 experiments measured) |

---

## Repository Structure

```
eigen-spectral-probe/
├── README.md                          ← this file
├── requirements.txt                   ← pinned env
├── .venv/                             ← local environment (hidden, gitignored)
├── esp_cache/                         ← cached SVD results, models never re-enter
├── 1_spectra_real_weights.ipynb       ← SVD of real weight matrices
├── 2_semantic_directions.ipynb        ← what spectral directions mean
├── 3_truncation_fidelity.ipynb        ← rank truncation vs output drift
├── 4_scaleout_spectra.ipynb           ← spectra across 135M→1.7B
├── 5_weight_probing.ipynb             ← reading semantics from spectra
└── 6_spectral_pruning.ipynb           ← deployment CPU pruning
```

---

## Notation (pre-defined, used across all units)

| Symbol | Meaning |
|--------|---------|
| `W` | A weight matrix (e.g. W_Q, W_gate) |
| `U, Σ, V^T` | SVD factors: `W = U·Σ·V^T` |
| `σ_i` | i-th singular value (descending) |
| `E_i` | Spectral energy: `σ_i² / Σ σ_j²` |
| `k90` | Effective rank: smallest k with `Σ_{i≤k} E_i ≥ 0.90` |
| `α` | Power-law decay exponent: `σ_i ∝ i^(-α)` |
| `W_k` | Rank-k truncated SVD: `U_k·Σ_k·V_k^T` |
| `‖W - W_k‖_F` | Frobenius norm of truncation error |
| `rel_err` | Relative error: `‖W - W_k‖_F / ‖W‖_F` |

---

## Math concepts (pre-defined)

**SVD.** Any matrix `W` factors into `U·Σ·V^T`. `U` and `V` are orthogonal (rotation), `Σ` is diagonal (scaling). Singular values are the gains along each principal direction.

**Spectral energy.** `E_i = σ_i² / Σ σ_j²` is the fraction of total variance captured by direction i.

**Effective rank (k90).** Smallest number of directions capturing ≥90% of spectral energy.

**Power-law decay.** If `σ_i ∝ i^(-α)`, the spectrum is scale-free. On log-log axes the slope is `-α`.

**Truncation error.** `W_k` is the best rank-k approximation (Eckart–Young). Error: `‖W - W_k‖_F = √(Σ_{i>k} σ_i²)`.

**Spectral pruning.** Replace `W` with `W_k` in the forward pass. Measure output drift.

**Probing.** Linear regression from spectral components to semantic labels.

---

## Unit 1 — Spectra of real SmolLM2 weights (complete)

**Design.** SVD of eight matrices from SmolLM2-135M layer 0: W_Q, W_K, W_V, W_O, W_gate, W_up, W_down, W_E. Five metrics per matrix: n, top1, k90, k90/n, α (power-law exponent).

**Measured findings (SmolLM2-135M, not assumed):**

| # | Claim | Predicted | Measured | Verdict |
|---|---|---|---|---|
| C1 | Power-law decay | R² > 0.9 | R² 0.63–0.89 | ⚠️ Partial |
| C2 | FFN flatter than attention | α_FFN < α_attn | 0.36 < 1.03 | ✅ Holds |
| C3 | W_E most concentrated | top1_E highest | 0.506 > 0.254 | ✅ Holds |
| C4 | W_Q more concentrated than W_K/V | top1_Q > top1_K/V | 0.254 > 0.222 > 0.032 | ✅ Holds |

**The three surprises.** W_V is the flattest attention matrix (top1 = 3.2%, k90/n = 0.677). Power-law fit is mediocre (R² 0.63–0.89) — the spectrum has an elbow, not a straight line. W_E is in a class of its own (top1 = 50.6%) — one direction holds half the embedding's energy, exactly reproducing the earlier study's finding.

**Verdict in one line.** The embedding is dramatically concentrated (top1 = 50.6%), FFN matrices are flat (k90/n ≈ 0.68), and W_V is the flattest attention matrix — but the decay is not a clean power law, so any truncation rule needs to be measured, not assumed.

---

## Unit 2 — Semantic directions in spectral space (complete)

**Design.** SVD W_E, project all 49,152 tokens into whitened spectral coordinates (Z = U). Regress token log-frequency against each coordinate and against spectral bands.

**Measured findings (SmolLM2-135M, not assumed):**

| # | Claim | Predicted | Measured | Verdict |
|---|---|---|---|---|
| C1 | Leading coordinate captures frequency | R²_1 > 0.3 | 0.304 | ✅ Holds |
| C2 | Frequency in spectral head | top-10 R² > 0.5 | 0.413 | ❌ Reversed |
| C3 | Trailing coordinates capture little | R²_100+ < 0.01 | 0.000095 | ✅ Holds |
| C4 | R² decays with coordinate rank | monotone decay | 0.304 → 0.157 → 0.068 → ... | ✅ Holds |

**The key surprise.** Even with all 576 coordinates, R² only reaches 0.628 — frequency is not fully linearly readable from spectral space. The spectral head (top-10) captures 41%; the remaining 22 points of R² are spread thinly across 566 coordinates.

**Verdict in one line.** The leading spectral coordinate captures 30.4% of token frequency variance; R² decays monotonically to essentially zero by coordinate 100; but even all 576 coordinates together only explain 62.8%.

---

## Unit 3 — Truncation and output fidelity (complete)

**Design.** For each of 8 matrices, truncate W at k = 1, 10, k90, 0.5*k90. Measure theoretical Frobenius error, actual output drift (KL divergence), and cosine similarity.

**Measured findings (SmolLM2-135M, not assumed):**

| # | Claim | Predicted | Measured | Verdict |
|---|---|---|---|---|
| C1 | Eckart-Young holds numerically | error <= 1e-6 | actual == theory in every row | Holds |
| C2 | Flat matrices tolerate more truncation | FFN rel_err < attention | opposite at fixed k | Reversed |
| C3 | Drift correlates with Frobenius error | high error => high KL | up worst rel_err but moderate KL | Reversed |
| C4 | k90 truncation preserves output | KL < 0.1 | all KL < 0.05 | Holds |

**The key insight.** Frobenius error does not predict output drift. At k=1, up loses 99.7% of its Frobenius energy but the output only drifts 0.1 nats. v loses 98.4% but drifts only 0.005 nats. The output is insensitive to directions that matter in Frobenius norm.

**Verdict in one line.** The Eckart-Young theorem holds exactly, and truncation at k90 preserves output (KL < 0.05) — but Frobenius error does NOT predict output drift, so pruning decisions need output-level measurement.

---

## Unit 4 — Scale-out: spectra across 135M to 1.7B (complete)

**Design.** Extract same 8 matrices from SmolLM2 135M, 360M, 1.7B. Compare k90/n, top1, alpha across sizes.

**Measured findings (SmolLM2 family, not assumed):**

| # | Claim | Predicted | Measured | Verdict |
|---|---|---|---|---|
| C1 | k90/n flat across sizes | span ≤ 0.05 | emb 0.137, v 0.284 | ⚠️ Partial |
| C2 | top1 constant per matrix | span ≤ 0.05 | emb 0.413, most drift | ❌ Reversed |
| C3 | W_E concentration persists | top1_E highest | 0.506 → 0.093 | ❌ Reversed |
| C4 | α is size-dependent | α changes | k/v span ~0.54 | ⚠️ Partial |

**The single biggest finding.** W_E's top1 drops from 0.506 (135M) to 0.093 (1.7B). The common direction that dominates the embedding at 135M is nearly gone at 1.7B. Spectral structure does NOT travel cleanly across scale.

**Verdict in one line.** Bigger models are *less* concentrated, not more — W_E's top1 collapses from 0.506 to 0.093 — so every spectral default is model-specific.

---

## Unit 5 — Weight probing via spectral decomposition (complete)

**Design.** Project tokens into spectral space (Z = U). Train linear probes from top-k coordinates to predict log-frequency, char length, byte length, is_alpha.

**Measured findings (SmolLM2-135M, not assumed):**

| # | Claim | Predicted | Measured | Verdict |
|---|---|---|---|---|
| C1 | Frequency readable from head | R² > 0.3 at k=10 | 0.413 | ✅ Holds |
| C2 | Length needs more coordinates | R²_length < R²_freq | 0.230 vs 0.440 at k=50 | ✅ Holds |
| C3 | All converge at k→576 | R² → 0.6+ | all reach 0.63+ | ✅ Holds |
| C4 | is_alphabetic poorly readable | R² < 0.1 | 0.937 | ❌ Reversed |

**The big surprise.** is_alpha is the MOST readable at k=576 (0.937) — it just needs the spectral tail. Frequency is head-encoded; is_alpha is tail-encoded.

**Verdict in one line.** Frequency is head-encoded (R²=0.30 at k=1), char_len rises slowly from near zero, and is_alpha is the "late bloomer" — near zero at k=1, reaches 0.94 at k=576.

---

## Unit 6 — Deployment: spectral pruning (complete)

**Design.** Replace W with W_k at k90 and 0.5·k90. Measure output drift (KL) and latency (ms/call).

**Measured findings (SmolLM2-135M, not assumed):**

| # | Claim | Predicted | Measured | Verdict |
|---|---|---|---|---|
| C1 | k90 gives KL < 0.05 | KL < 0.05 | all KL < 0.05 | ✅ Holds |
| C2 | 0.5·k90 gives speedup | ms/token reduced | minimal speedup | ❌ Reversed |
| C3 | FFN tolerates pruning | FFN drift < attention | comparable | ⚠️ Partial |

**The honest conclusion.** Pruning at k90 is *safe* (low drift) but *not useful* (no speedup) on CPU. The bottleneck is memory bandwidth, not multiply count.

**Recommended config.** Prune gate/up/down at k90; skip W_E and W_Q.

**Verdict in one line.** Safe but not useful on CPU for these matrix sizes.

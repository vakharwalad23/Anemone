# Math

The formulas behind Anemone's speed, energy and quality numbers. Each worked example uses measurements from the lab prototype on an M3 Pro, macOS 27.0.1, AC power ([Benchmarks](benchmarks.md)).

## Decode time is set by weight reads

Generating one token reads every weight once:

```math
t_{\text{token}} \approx \frac{W}{B}
```

- $W$: bytes of weights read per call.
- $B$: bandwidth the ANE reaches when streaming weights.

Example: the four Qwen3-8B chunks hold $W = 3.84$ GB. A timing study of random-weight layers measured $B \approx 57$ GB/s, which predicts 67 ms. Measured: 65.8 ms per token, 15.1 tokens/s.

Extra tokens in the same call reuse the weights already streamed, so they are nearly free:

| verify width | 8 | 16 | 32 | 64 |
|---|---|---|---|---|
| ms per call (Swift) | 63.9 | 65.3 | 72.0 | 82.4 |

Going from 8 to 16 tokens adds 0.18 ms per token, against 65.8 ms for a token decoded on its own.

## Speculative decoding speed

Each cycle drafts guesses, verifies them in one call, and yields the accepted guesses plus one token from the 8B:

```math
\text{tokens/s} = \frac{\bar{a} + 1}{t_d + t_v + t_h}
```

- $\bar{a}$: mean number of guesses accepted per cycle.
- $t_d$: draft time. $t_v$: verify time. $t_h$: host time (CPU loop, KV alignment).

Example, the math prompt: $\bar{a} = 5.07$, $t_d = 21.1$ ms, $t_v = 67.5$ ms gives 68.5 tokens/s without host time. Measured: 65.6 tokens/s, so $t_h \approx 4$ ms per cycle.

Speculation beats plain decoding (time $t_p$ per token) when

```math
\bar{a} > \frac{t_d + t_v}{t_p} - 1
```

- On the ANE: $(21.1 + 67.5) / 65.8 - 1 = 0.35$. Every prompt in the lab clears it, so drafting wins even on short translations.
- On the GPU with MLX: plain decode is 32.5 ms per token and a cycle 110 to 116 ms, so drafting needs $\bar{a} > 2.4$ to $2.6$. Only the math and code prompts clear it.

This is why the drafter helps the ANE far more than the GPU: the ANE's slow single token makes the bar low, and its spare compute makes the verify nearly free.

## Energy per token

```math
E_{\text{token}} = \frac{1}{N} \int_0^T P(t)\, dt, \qquad E_{\text{above idle}} = \frac{1}{N} \int_0^T \big(P(t) - P_{\text{idle}}\big)\, dt
```

- $P(t)$: whole-system power from the SMC, sampled over a window of length $T$ that covers all 6 prompts plus a 2 s tail.
- $N$: tokens generated in the window. $P_{\text{idle}}$: power with nothing running (1.5 to 2.8 W on the test machine).

Example: 19.8 W at 39.9 tokens/s gives $19.8 / 39.9 = 0.50$ J as a lower bound; the tail in the window brings the measured value to 0.53 J. MLX plain: 1.10 J.

## Quality against the bf16 model

Both models read the same text (teacher forcing). At each position $t$, $p_t$ is the bf16 model's next-token distribution and $q_t$ the 4-bit model's:

```math
\mathrm{KL} = \frac{1}{n} \sum_{t=1}^{n} \sum_{v} p_t(v) \log \frac{p_t(v)}{q_t(v)}, \qquad \text{top-1} = \frac{1}{n} \sum_{t=1}^{n} \mathbf{1}\big[\arg\max p_t = \arg\max q_t\big]
```

- $n$: positions scored. Docs: 4 files x 512 tokens. Chat: 6 prompts with their answers.
- $v$: tokens of the 151936-entry vocabulary.

Example: Anemone's 4-bit weights on the ANE score KL 0.047 on docs; the Core AI default scores 0.341.

## Lookup-table weights

Core AI stores each 4-bit weight as an index into a table of $2^b$ fp16 values shared by a group of $g$ output rows:

```math
\text{bits per weight} = b + \frac{16 \cdot 2^b}{g \cdot n_{\text{in}}}
```

- $b = 4$ index bits, $g = 32$ rows per table, $n_{\text{in}}$ the layer's input width.

Example: $n_{\text{in}} = 4096$ gives $4 + 256 / 131072 = 4.002$ bits per weight.

### Choosing the 16 values

Weighted k-means over the group's weights, where each input column counts by how much signal it carries:

```math
\min_{C} \sum_{i \in G} \sum_{j} h_j \big(w_{ij} - c(w_{ij})\big)^2, \qquad h_j = \sum_{s} x_{sj}^2
```

- $G$: the 32 rows of the group. $C$: its 16 values. $c(w)$: the value nearest to $w$.
- $x_{sj}$: input $j$ of the layer for calibration token $s$.

### GPTQ onto a fixed table

Snapping each weight to its nearest value ignores how errors add up in the layer's output. GPTQ instead minimizes the output error $\lVert (W - Q) X^\top \rVert_F^2$ column by column. Anemone keeps the k-means tables fixed and only changes which entry each weight snaps to:

```math
H = X^\top X + \lambda\, \overline{H_{jj}}\, I, \qquad H^{-1} = U^\top U \;\; (U \text{ upper triangular})
```

Then, for columns $j$ in descending order of $H_{jj}$:

```math
q_{:,j} = \mathrm{snap}\big(w_{:,j}\big), \qquad e_{:,j} = \frac{w_{:,j} - q_{:,j}}{U_{jj}}, \qquad w_{:,k} \leftarrow w_{:,k} - e_{:,j}\, U_{jk} \quad (k > j)
```

- $X$: calibration inputs to the layer, one row per token. $\lambda = 0.01$.
- $\mathrm{snap}$ picks, for each row, the nearest of that row group's 16 values.
- Columns are processed in blocks of 128 with the update to later blocks applied once per block; the result is the same.

Every group still holds at most 16 values, so the weights export through Core AI unchanged.

Example: simulated on Qwen3-8B with the same 12.9k calibration tokens, docs KL goes from 0.136 with weighted k-means alone to 0.043 with GPTQ. On the ANE the GPTQ weights measure 0.044.

## Token trees

The free slots of a verify call hold second-choice branches. With chain confidences $c_0, \dots, c_6$, the value of node $m$ ($m = 0, 1, \dots$) on the branch that leaves the chain at step $j$ is

```math
v_{j,m} = r_j\, (1 - c_j) \prod_{i=j+1}^{j+m} c_i, \qquad r_j = \prod_{i<j} c_i
```

- $r_j$: chance the chain survives to step $j$. $1 - c_j$: chance it misses there. The product: chance the branch survives $m$ more steps, estimated with the chain's confidences.

The budget goes to the highest-value nodes first.

Example: confidences 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3 and a budget of 6 nodes give branches of length 2 at steps 1 and 2 and of length 1 at steps 3 and 4.

## KV cache traffic at long context

```math
\text{KV bytes per chunk} = 2 \cdot L \cdot n_{kv} \cdot d \cdot c \cdot 2
```

- 2 for keys and values, $L = 9$ layers per chunk, $n_{kv} = 8$ KV heads, $d = 128$ head size, $c$ context length, 2 bytes per fp16 value.

Example: $c = 8192$ gives 302 MB, about 5.1 ms at 59 GB/s. Measured: a chunk costs 21.72 ms at 4096 and 46.25 ms at 8192. The extra 151 MB of KV explains 2.6 ms of the 24.5 ms jump, so the cliff at 8192 is not bandwidth. It was the transpose of the attention scores in Apple's SDPA; with row-wise attention a chunk costs 30.96 ms at 8192 ([Benchmarks](benchmarks.md#long-context-one-chunk-width-16)).

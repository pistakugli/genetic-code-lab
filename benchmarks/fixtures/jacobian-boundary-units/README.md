# Which units the Jacobian lane recovers, and how they were chosen

These lists decide what "64 of 64 units of layer 15" means, and until this file existed the
answer was unrecorded — the fixtures were bare lists of numbers carried over from an earlier
session. A result on a curated subset is not a result on a block of the layer, so the scope
belongs next to the numbers. Audit finding F23.

## Selection

Uniform random draw over the layer's 8192 units, without replacement.

The upper lists lived only in `/tmp` until a ladder run failed on all three rungs above 256
with an empty unit string - the script read a fixture that was never committed. They are here
now, which is the point of the directory.

`64.txt` and `128.txt` were drawn independently and share no unit: their intersection is
empty. `256.txt` extends `128.txt` and `512` extends `256`, so the upper rungs are nested
supersets rather than fresh samples. That is a deliberate design for measuring what **density**
does — the same units are re-recovered in a more crowded block — but it is not what "an
eightfold increase in block size" reads like, so it is stated here and in the roadmap. Audit
finding F22.

| file | units | relationship |
| --- | --- | --- |
| `64.txt` | 64 | independent draw |
| `128.txt` | 128 | independent draw, disjoint from 64 |
| `256.txt` | 256 | superset of `128.txt` |
| `512.txt` | 512 | superset of `256.txt` |
| `1024.txt` | 1024 | superset of `512.txt` |
| `2048.txt` | 2048 | superset of `1024.txt` |
| `4096.txt` | 4096 | superset of `2048.txt`, the first block with more units than the layer has output channels |

## Are they representative

Index uniformity says nothing about difficulty: what decides how hard a unit is to recover is
its output norm `‖d_i‖` and its gate norm, not where it sits in the tensor. Measured against
all 8192 units of layer 15, where a uniform sample would average the 50th percentile on both:

| file | mean percentile of `‖g‖` | mean percentile of `‖d‖` |
| --- | --- | --- |
| `64.txt` | 51.2 | 50.1 |
| `128.txt` | 50.8 | 47.4 |
| `256.txt` | 50.2 | 49.1 |

Layer 15 medians for reference: `‖g‖` 1.0407, `‖d‖` 0.9762.

So these are ordinary blocks, neither the easy tail nor the hard one, and the lane's headline
number can be read as a claim about a typical block of this layer. Regenerate this table with
the snippet in `docs/nn-decompiler-roadmap.md` if the fixtures ever change.

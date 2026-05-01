# `qwen3-1.7b-drgrpo` vs `qwen3-1.7b-run4` — paired analysis (n=500)

Compared: `qwen3-1.7b-drgrpo` vs `qwen3-1.7b-run4`

## Headline

| | Correct | Accuracy |
|---|---|---|
| qwen3-1.7b-run4 | 144/500 | **28.8%** |
| qwen3-1.7b-drgrpo | 163/500 | **32.6%** |
| **Delta** | | **+3.8 pp** |

McNemar exact two-sided **p = 0.0558** (discordant 89; smaller side 35)

## Transition matrix

|   | base ✓ | base ✗ |
|---|---|---|
| **ft ✓** | 109 | 54  *(improved)* |
| **ft ✗** | 35  *(regressed)* | 302 |

## Grader-path counts

| Path | base | ft | delta |
|---|---|---|---|
| exact | 71 | 78 | +7 |
| normalized | 6 | 2 | -4 |
| symbolic | 12 | 13 | +1 |
| judge | 55 | 70 | +15 |
| miss | 356 | 337 | -19 |

## Miss decomposition (ft model)

- Total `miss`: **337** (67.4%)
- ...of which **saturated at ≥16384 output tokens**: 151 (45% of misses)
- ...of which **saturated AND no `\boxed{}`** (convergence failure): 148 (44% of misses)
- ...of which **wrong-but-committed** (output < ceiling): 186 (55% of misses)

For comparison, base had 356 misses, 198 saturated (56%), 190 saturated-and-no-boxed.

## Where the fine-tune helped most (top 5 competitions, n≥2)

| Competition | n | base | ft | delta |
|---|---|---|---|---|
| China Mathematical Competition | 10 | 4 | 8 | +4 |
| Baltic Way shortlist | 2 | 0 | 2 | +2 |
| China Girls' Mathematical Olympiad | 4 | 0 | 2 | +2 |
| SAUDI ARABIAN MATHEMATICAL COMPETITIONS | 11 | 2 | 4 | +2 |
| Mongolian Mathematical Olympiad | 12 | 3 | 5 | +2 |

## Where the fine-tune regressed most (bottom 5 competitions, n≥2)

| Competition | n | base | ft | delta |
|---|---|---|---|---|
| South African Mathematics Olympiad | 8 | 7 | 5 | -2 |
| Saudi Arabian IMO Booklet | 6 | 1 | 0 | -1 |
| 58th Ukrainian National Mathematical Olympiad | 6 | 3 | 2 | -1 |
| Croatia_2018 | 5 | 2 | 1 | -1 |
| China Western Mathematical Olympiad | 4 | 1 | 0 | -1 |

# `qwen3-1.7b-drgrpo` vs `qwen3-1.7b-base` — paired analysis (n=500)

Compared: `qwen3-1.7b-drgrpo` vs `qwen3-1.7b-base`

## Headline

| | Correct | Accuracy |
|---|---|---|
| qwen3-1.7b-base | 184/500 | **36.8%** |
| qwen3-1.7b-drgrpo | 163/500 | **32.6%** |
| **Delta** | | **-4.2 pp** |

McNemar exact two-sided **p = 0.0238** (discordant 79; smaller side 29)

## Transition matrix

|   | base ✓ | base ✗ |
|---|---|---|
| **ft ✓** | 134 | 29  *(improved)* |
| **ft ✗** | 50  *(regressed)* | 287 |

## Grader-path counts

| Path | base | ft | delta |
|---|---|---|---|
| exact | 89 | 78 | -11 |
| normalized | 2 | 2 | +0 |
| symbolic | 15 | 13 | -2 |
| judge | 78 | 70 | -8 |
| miss | 316 | 337 | +21 |

## Miss decomposition (ft model)

- Total `miss`: **337** (67.4%)
- ...of which **saturated at ≥16384 output tokens**: 151 (45% of misses)
- ...of which **saturated AND no `\boxed{}`** (convergence failure): 148 (44% of misses)
- ...of which **wrong-but-committed** (output < ceiling): 186 (55% of misses)

For comparison, base had 316 misses, 157 saturated (50%), 145 saturated-and-no-boxed.

## Where the fine-tune helped most (top 5 competitions, n≥2)

| Competition | n | base | ft | delta |
|---|---|---|---|---|
| Baltic Way shortlist | 2 | 0 | 2 | +2 |
| Belarusian Mathematical Olympiad | 5 | 2 | 4 | +2 |
| Mongolian Mathematical Olympiad | 12 | 3 | 5 | +2 |
| APMO | 2 | 0 | 1 | +1 |
| Argentine National Olympiad 2015 | 2 | 0 | 1 | +1 |

## Where the fine-tune regressed most (bottom 5 competitions, n≥2)

| Competition | n | base | ft | delta |
|---|---|---|---|---|
| 58th Ukrainian National Mathematical Olympiad | 6 | 5 | 2 | -3 |
| Japan Mathematical Olympiad | 8 | 4 | 2 | -2 |
| Croatian Mathematical Society Competitions | 6 | 6 | 4 | -2 |
| Estonian Math Competitions | 6 | 3 | 1 | -2 |
| Saudi Arabian IMO Booklet | 6 | 2 | 0 | -2 |

# OpenAI safety-filter flagged problems

Collected from `logs/*.log` during the Day-2 full-eval run. These are MathNet problems that OpenAI's reasoning API rejected with `invalid_prompt` (a 400, not a transient 429/5xx). Our retry logic correctly does not retry these.

Purpose: (a) denominator accounting for the methodology caveat (`accuracy = n_correct / n_scored` rather than / N), and (b) blog-post side observation about what kinds of olympiad math trip the safety filter.

## Per-model filter rate

| Model | Processed so far | Flagged | Rate |
|---|---|---|---|
| `gemini-3-pro` | 300 | 0 | 0.00% |
| `gpt-5.4` | 500 | 5 | 1.00% |
| `gpt-5.4-mini` | 500 | 2 | 0.40% |
| `grade_all` | 0 | 0 | — |
| `opus-4-7` | 100 | 0 | 0.00% |
| `sonnet-4-6` | 500 | 0 | 0.00% |


## gpt-5.4 — 5 flagged

### `0df4`  *(#11 of 500)*

- **Country**: Saudi Arabia   **Competition**: Saudi Arabian IMO Booklet   **Language**: English   **Type**: proof and answer
- **Topics**: Algebra > Prealgebra / Basic Algebra > Simple Equations
- **Gold answer**: `(a, b, c) = (1/2, 1/3, 1/6) and (a, b, c) being any permutation of (1, 1, -1).`

**OpenAI error**: Invalid prompt: your prompt was flagged as potentially violating our usage policy. Please try again with a different prompt: https://platform.openai.com/docs/guides/reasoning#advice-on-prompting

**Problem**:

```
Find all triples $(a, b, c)$ of real numbers satisfying
$$
a + b + c = 1 \quad \text{and} \quad 3(a + bc) = 4(b + ca) = 5(c + ab).
$$
```

---

### `01xa`  *(#71 of 500)*

- **Country**: Belarus   **Competition**: 69th Belarusian Mathematical Olympiad   **Language**: English   **Type**: proof and answer
- **Topics**: Discrete Mathematics > Combinatorics > Pigeonhole principle, Discrete Mathematics > Combinatorics > Games / greedy algorithms
- **Gold answer**: `S ≤ 152`

**OpenAI error**: Invalid prompt: your prompt was flagged as potentially violating our usage policy. Please try again with a different prompt: https://platform.openai.com/docs/guides/reasoning#advice-on-prompting

**Problem**:

```
The sum of several (not necessarily different) positive integers not exceeding $10$ is equal to $S$.
Find all possible values of $S$ such that these numbers can always be partitioned into two groups with the sums of the numbers in each group not exceeding $80$.
```

---

### `09uk`  *(#83 of 500)*

- **Country**: Netherlands   **Competition**: First Round, January 2019   **Language**: English   **Type**: MCQ
- **Topics**: Discrete Mathematics > Combinatorics > Invariants / monovariants, Discrete Mathematics > Algorithms
- **Gold answer**: `B`

**OpenAI error**: Invalid prompt: your prompt was flagged as potentially violating our usage policy. Please try again with a different prompt: https://platform.openai.com/docs/guides/reasoning#advice-on-prompting

**Problem**:

```
There are $13$ distinct multiples of $7$ that consist of two digits. You want to create a longest possible chain consisting of these multiples, where two multiples can only be adjacent if the last digit of the left multiple equals the first digit of the right multiple. You can use each multiple at most once. For example, $21$ – $14$ – $49$ is an admissible chain of length $3$. What is the maximum length of an admissible chain?
A) $6$  B) $7$  C) $8$  D) $9$  E) $10$
```

---

### `0584`  *(#198 of 500)*

- **Country**: Estonia   **Competition**: Estonian Math Competitions   **Language**: English   **Type**: proof and answer
- **Topics**: Discrete Mathematics > Combinatorics > Games / greedy algorithms, Discrete Mathematics > Combinatorics > Invariants / monovariants, Number Theory > Divisibility / Factorization > Least common multiples (lcm)
- **Gold answer**: `a: first player; b: first player`

**OpenAI error**: Invalid prompt: your prompt was flagged as potentially violating our usage policy. Please try again with a different prompt: https://platform.openai.com/docs/guides/reasoning#advice-on-prompting

**Problem**:

```
Natural numbers $1$ through $n$ are written on a blackboard. On each move, one erases from the blackboard $2$ or more numbers whose sum is divisible by any of the chosen numbers and writes their sum on the blackboard. Two players make moves by turns and the player who cannot move loses the game. Which player can win the game against any play by the opponent, if:

a. $n = 6$;

b. $n = 11$?
```

---

### `0gs0`  *(#453 of 500)*

- **Country**: Turkey   **Competition**: Team Selection Test for IMO 2019   **Language**: English   **Type**: proof and answer
- **Topics**: Algebra > Algebraic Expressions > Sequences and Series > Recurrence relations, Number Theory > Divisibility / Factorization > Prime numbers, Number Theory > Modular Arithmetic
- **Gold answer**: `3, 5, 19`

**OpenAI error**: Invalid prompt: your prompt was flagged as potentially violating our usage policy. Please try again with a different prompt: https://platform.openai.com/docs/guides/reasoning#advice-on-prompting

**Problem**:

```
Let $(a_n)_{n=1}^\infty$ be a sequence of integers with $a_1 = 1$, $a_2 = 2$ and
$$
a_{n+2} = a_{n+1}^2 + (n+2)a_{n+1} - a_n^2 - n a_n
$$
for all $n \ge 1$.

a) Show that there exist infinitely many prime numbers dividing at least one term of this sequence.

b) Find three different prime numbers not dividing any term of this sequence.
```

---


## gpt-5.4-mini — 2 flagged

### `04bw`  *(#133 of 500)*

- **Country**: Croatia   **Competition**: Mathematica competitions in Croatia   **Language**: English   **Type**: proof and answer
- **Topics**: Algebra > Prealgebra / Basic Algebra > Integers, Algebra > Intermediate Algebra > Quadratic functions, Algebra > Equations and Inequalities > Linear and quadratic inequalities
- **Gold answer**: `1, 4, 9`

**OpenAI error**: Invalid prompt: your prompt was flagged as potentially violating our usage policy. Please try again with a different prompt: https://platform.openai.com/docs/guides/reasoning#advice-on-prompting

**Problem**:

```
Determine all positive integers smaller than $1000$ that are equal to the sum of squares of their digits.
```

---

### `03fi`  *(#135 of 500)*

- **Country**: Bulgaria   **Competition**: Bulgarian Spring Tournament   **Language**: English   **Type**: proof and answer
- **Topics**: Algebra > Equations and Inequalities > QM-AM-GM-HM / Power Mean, Algebra > Algebraic Expressions > Polynomials > Symmetric functions, Number Theory > Diophantine Equations > Infinite descent / root flipping, Number Theory > Algebraic Number Theory > Unique factorization
- **Gold answer**: `m = 1, M = 9/8; No, the minimum cannot be attained by any triple of nonnegative rational numbers.`

**OpenAI error**: Invalid prompt: your prompt was flagged as potentially violating our usage policy. Please try again with a different prompt: https://platform.openai.com/docs/guides/reasoning#advice-on-prompting

**Problem**:

```
The nonnegative real numbers $x, y, z$ are such that $(x + y)(y+z)(z+x) = 1$. We denote by $m$ and $M$ respectively the smallest and largest possible values of the expression $A = (xy + yz + zx)(x + y + z)$.

a) Find $m$ and $M$.

b) Is there a triple of nonnegative rational numbers $(x, y, z)$ satisfying the given equality for which $A = m$?
```

---

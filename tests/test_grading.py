"""Unit tests for mathnet_eval.grading.

Cover the layers we rely on without hitting any network / API.
LLM-judge path is opt-in so it never runs in these tests.
"""

from __future__ import annotations

from mathnet_eval.grading import (
    extract_answer,
    grade,
    normalize,
    normalize_for_exact,
    symbolic_equal,
)


class TestExtractAnswer:
    def test_final_answer_label(self):
        assert extract_answer("derivation...\nFinal answer: 42") == "42"

    def test_boxed(self):
        assert extract_answer(r"Thus the answer is \boxed{-1/4}.") == "-1/4"

    def test_the_answer_is(self):
        assert extract_answer("Therefore the answer is 7.") == "7"

    def test_prefers_last_match(self):
        # A reasoning step sometimes says "the answer is X" before the real final answer.
        txt = "The answer is probably 3. But checking again... Final answer: 5"
        assert extract_answer(txt) == "5"

    def test_no_marker(self):
        assert extract_answer("just some prose with no clear answer") is None


class TestNormalize:
    def test_strips_dollar_delimiters(self):
        assert normalize("$-1/4$") == "-1/4"

    def test_dfrac_becomes_frac(self):
        assert normalize(r"-\dfrac{1}{4}") == r"-\frac{1}{4}"

    def test_left_right(self):
        assert normalize(r"\left( x \right)") == "( x )"


class TestNormalizeForExact:
    def test_case_fold(self):
        assert normalize_for_exact("All integers k") == normalize_for_exact("all integers k")

    def test_latex_geq_equals_unicode(self):
        # The Day-1 judge-review case that should have been a cheap-layer catch.
        assert normalize_for_exact(r"All integers $k \geq 2$.") == normalize_for_exact("all integers k ≥ 2")

    def test_latex_leq_and_neq(self):
        assert normalize_for_exact(r"$x \leq 5$") == normalize_for_exact("x ≤ 5")
        assert normalize_for_exact(r"$x \neq 0$") == normalize_for_exact("x ≠ 0")

    def test_varname_prefix_stripped(self):
        # The Day-1 case `$A = -1$` vs `-1`.
        assert normalize_for_exact("$A = -1$") == normalize_for_exact("-1")
        assert normalize_for_exact("a_n = 2^n") == normalize_for_exact("2^n")

    def test_trailing_punctuation_stripped(self):
        assert normalize_for_exact("42.") == normalize_for_exact("42")
        assert normalize_for_exact("42;") == normalize_for_exact("42")

    def test_cdot_and_times(self):
        assert normalize_for_exact(r"2 \cdot 3") == normalize_for_exact("2 · 3")
        assert normalize_for_exact(r"2 \times 3") == normalize_for_exact("2 × 3")


class TestSymbolicEqual:
    def test_numeric_fraction(self):
        assert symbolic_equal("1/4", "0.25")

    def test_latex_frac_vs_plain(self):
        assert symbolic_equal(r"-\dfrac{1}{4}", "-1/4")

    def test_simple_inequality(self):
        assert not symbolic_equal("2/3", "3/2")


class TestGrade:
    def test_exact_match_wins_layer_1(self):
        g = grade("Final answer: 42", "42")
        assert g.correct and g.method == "exact"

    def test_normalize_catches_latex_delims(self):
        g = grade("Final answer: $42$", "42")
        assert g.correct and g.method == "normalized"

    def test_normalize_catches_varname_prefix(self):
        # Day-1 case 0db7 — this should now be layer 2, not layer 4.
        g = grade("Final answer: $A = -1$", "-1")
        assert g.correct and g.method == "normalized"

    def test_normalize_catches_latex_geq_and_case(self):
        # Day-1 case 04sm.
        g = grade(r"Final answer: All integers $k \geq 2$.", "all integers k ≥ 2")
        assert g.correct and g.method == "normalized"

    def test_sympy_catches_latex_frac(self):
        g = grade(r"Final answer: $-\dfrac{1}{4}$", "-1/4")
        assert g.correct and g.method == "symbolic"

    def test_miss_on_no_marker(self):
        g = grade("I have no idea.", "7")
        assert not g.correct and g.method == "miss"

    def test_miss_on_wrong_answer(self):
        g = grade("Final answer: 11", "7")
        assert not g.correct

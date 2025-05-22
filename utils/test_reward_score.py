from reward_score import calculate_score

def verify(score, expected, tol=1e-3):
    assert abs(score - expected) < tol, f"{score} ≠ {expected}"

def test_exact_match():
    p ="Target = 98; Numbers = [44, 19, 35]"
    r ="(44 + 35) + 19"
    verify(calculate_score(p,r), 1.0)

def test_off_by_one():
    p ="Target = 99; Numbers = [44, 19, 35]"
    r ="(44 + 35) + 19"         # diff =1 → 1-1/98 ≈ .9898
    score = calculate_score(p,r)
    assert 0.95 < score

def test_invalid_number():
    p ="Target = 98; Numbers = [44, 19, 35]"
    r ="(44 + 35) + 20"
    verify(calculate_score(p,r), 0.0)

def test_invalid_expr():
    p ="Target = 98; Numbers = [44, 19, 35]"
    r ="foo bar baz"
    verify(calculate_score(p,r), 0.0)

def test_partial_use_allowed():
    p ="Target = 98; Numbers = [44, 19, 35]"
    r ="44 + 35"        # result 79  → 1 - 19/98 ≈ .806
    score = calculate_score(p,r)
    assert 0.7 < score < 0.9

def test_duplicate_use():
    p ="Target = 98; Numbers = [44, 19, 35]"
    r ="44 + 44 + 10"
    verify(calculate_score(p,r), 0.0)

def test_bad_prompt():
    p ="This prompt is broken"
    r ="(44+35)+19"
    verify(calculate_score(p,r), 0.0)

def run_all():
    for fn in list(globals().values()):
        if callable(fn) and fn.__name__.startswith("test_"):
            fn()
    print("✅ all countdown reward tests passed")

if __name__ == "__main__":
    run_all()

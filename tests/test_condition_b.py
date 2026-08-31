import sentinel_condition_b as condition_b


def test_retry_returns_fallback_after_all_failures(monkeypatch):
    monkeypatch.setattr(condition_b.time, "sleep", lambda _: None)

    assert condition_b.retry(lambda: (_ for _ in ()).throw(RuntimeError()), 0.5) == 0.5


def test_interpret_results_preserves_condition_b_thresholds():
    _, partial_delta, partial_verdict = condition_b.interpret_results([0.82])
    _, sufficient_delta, sufficient_verdict = condition_b.interpret_results([0.81])

    assert partial_delta < -0.02
    assert partial_verdict.startswith("Partial dataset effect")
    assert sufficient_delta < -0.05
    assert sufficient_verdict.startswith("Dataset is sufficient driver")
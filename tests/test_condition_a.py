import sentinel_condition_a as condition_a


def test_retry_returns_value_after_transient_failure(monkeypatch):
    attempts = 0

    def operation():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary failure")
        return "success"

    monkeypatch.setattr(condition_a.time, "sleep", lambda _: None)

    assert condition_a.retry(operation, "fallback") == "success"
    assert attempts == 2


def test_interpret_results_preserves_condition_a_thresholds():
    delta, verdict = condition_a.interpret_results([0.8667])

    assert delta == 0.0
    assert verdict.startswith("No meaningful degradation")
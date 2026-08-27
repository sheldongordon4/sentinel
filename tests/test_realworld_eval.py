from types import SimpleNamespace

import pytest

import sentinel_realworld_eval as legacy_eval
import sentinel_realworld_eval_v2 as current_eval


class FailingOpenAIClient:
    class Chat:
        class Completions:
            @staticmethod
            def create(**kwargs):
                raise RuntimeError("temporary API failure")

        completions = Completions()

    chat = Chat()


class InvalidGeminiClient:
    class Models:
        @staticmethod
        def generate_content(**kwargs):
            return SimpleNamespace(text='{"score": NaN}')

    models = Models()


def test_legacy_judge_retries_and_returns_neutral_score(monkeypatch):
    monkeypatch.setattr(legacy_eval, "client", FailingOpenAIClient())
    monkeypatch.setattr(legacy_eval.time, "sleep", lambda _: None)

    assert legacy_eval.judge_response("question", "response", [], retries=2) == 0.5


def test_v2_judge_rejects_nonfinite_scores_and_returns_neutral(monkeypatch):
    monkeypatch.setattr(current_eval, "gemini", InvalidGeminiClient())
    monkeypatch.setattr(current_eval.time, "sleep", lambda _: None)

    assert current_eval.judge_response("question", "response", []) == 0.5


@pytest.mark.parametrize(
    "module, message",
    [
        (legacy_eval, "OPENAI_API_KEY"),
    ],
)
def test_client_initialization_validates_credentials(monkeypatch, module, message):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match=message):
        module.initialize_client()

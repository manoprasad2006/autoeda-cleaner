from types import SimpleNamespace
from modules.ai.client import GeminiClient
from modules.ai.chatbot import ChatSession, ask_chatbot


class FakeModels:
    def __init__(self, text=None, error=None):
        self.text = text
        self.error = error
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return SimpleNamespace(text=self.text)


def test_ai_success_without_network():
    models = FakeModels("  Useful summary  ")
    client = GeminiClient("", "test", client=SimpleNamespace(models=models))
    response = client.generate("test")
    assert response.success and response.text == "Useful summary"


def test_ai_error_does_not_expose_exception_details():
    models = FakeModels(error=RuntimeError("private key or server detail"))
    client = GeminiClient("", "test", client=SimpleNamespace(models=models))
    response = client.generate("test")
    assert not response.success
    assert "private" not in response.error
    assert models.calls == 1


def test_empty_ai_responses_have_bounded_retries():
    models = FakeModels(" ")
    client = GeminiClient("", "test", client=SimpleNamespace(models=models))
    assert not client.generate("test").success
    assert models.calls == 3


def test_chat_question_limit_prevents_request():
    models = FakeModels("ok")
    client = GeminiClient("", "test", client=SimpleNamespace(models=models))
    session = ChatSession("Dataset facts")
    assert "2,000" in ask_chatbot(client, session, "x" * 2001)
    assert models.calls == 0
    assert not session.messages

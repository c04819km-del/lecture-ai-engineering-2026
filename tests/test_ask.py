from types import SimpleNamespace

from app import ask as ask_module


def test_ask_prints_token_counts_and_response_time_on_one_line(monkeypatch, capsys):
    class FakeChat:
        def send_message(self, question):
            assert question == "質問"
            return SimpleNamespace(
                text="回答",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=12,
                    candidates_token_count=7,
                ),
            )

    times = iter([1.0, 2.234])
    monkeypatch.setattr(ask_module.time, "perf_counter", lambda: next(times))

    result = ask_module.ask("質問", FakeChat())

    assert result == "回答"
    assert capsys.readouterr().out == (
        "[トークン数] 入力: 12, 出力: 7 | [応答時間] 1.23秒\n"
    )


def test_main_sends_multiple_questions_to_the_same_chat(monkeypatch, capsys):
    class FakeChat:
        def __init__(self):
            self.questions = []

        def send_message(self, question):
            self.questions.append(question)
            return SimpleNamespace(
                text=f"回答: {question}",
                usage_metadata=None,
            )

    class FakeChats:
        def __init__(self, chat):
            self.chat = chat
            self.models = []

        def create(self, *, model):
            self.models.append(model)
            return self.chat

    chat = FakeChat()
    chats = FakeChats(chat)
    def make_client(*, api_key):
        assert api_key == "test-key"
        return SimpleNamespace(chats=chats)

    monkeypatch.setattr(ask_module.genai, "Client", make_client)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr("sys.argv", ["app.ask", "最初の質問"])
    follow_up_questions = iter(["続けて質問", ""])

    def get_next_question(prompt):
        assert "次の質問" in prompt
        return next(follow_up_questions)

    monkeypatch.setattr(
        "builtins.input",
        get_next_question,
    )

    ask_module.main()

    assert chats.models == [ask_module.MODEL]
    assert chat.questions == ["最初の質問", "続けて質問"]
    output = capsys.readouterr().out
    assert "回答: 最初の質問" in output
    assert "回答: 続けて質問" in output

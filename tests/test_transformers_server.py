from scripts.transformers_openai_server import render_messages


class FakeTokenizer:
    def __init__(self):
        self.template_kwargs = None
        self.text = None

    def apply_chat_template(self, messages, **kwargs):
        self.template_kwargs = kwargs
        return "PROMPT"

    def __call__(self, text, **kwargs):
        self.text = text
        return {"input_ids": "ids"}


def test_normal_request_adds_generation_prompt():
    tok = FakeTokenizer()
    result = render_messages(tok, [{"role": "user", "content": "hi"}])
    assert result == {"input_ids": "ids"}
    assert tok.template_kwargs == {"tokenize": False, "add_generation_prompt": True}
    assert tok.text == "PROMPT"


def test_prefill_continues_final_assistant_message():
    tok = FakeTokenizer()
    render_messages(
        tok,
        [{"role": "user", "content": "hi"},
         {"role": "assistant", "content": "Alice:"}],
        continue_final_message=True,
    )
    assert tok.template_kwargs == {
        "tokenize": False,
        "continue_final_message": True,
    }

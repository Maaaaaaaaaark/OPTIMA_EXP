"""Chat-template adaptation tests.

google/gemma-2-2b-it's official jinja template raises "System role not
supported" for a leading system message and "Conversation roles must
alternate user/assistant/user/assistant/..." for any other sequence. The
pipeline adapts message lists instead (merge the system prompt into the
first user message) without touching the prompt text or the agents' private
information.
"""
import pytest

from message.message import adapt_messages_for_chat_template

# Verbatim copy of google/gemma-2-2b-it's chat_template (transformers 4.46
# era). Used only to prove that the adapted message lists actually render.
GEMMA_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% if messages[0]['role'] == 'system' %}"
    "{{ raise_exception('System role not supported') }}"
    "{% endif %}"
    "{% for message in messages %}"
    "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
    "{% endif %}"
    "{% if message['role'] == 'user' %}"
    "{{ '<start_of_turn>user\\n' + message['content'] | trim + '<end_of_turn>\\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<start_of_turn>model\\n' + message['content'] | trim + '<end_of_turn>\\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<start_of_turn>model\\n' }}"
    "{% endif %}"
)


def make_gemma_tokenizer():
    """A tiny word-level tokenizer carrying the official Gemma template."""
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    vocab = {
        "<bos>": 0, "<eos>": 1, "<pad>": 2, "<unk>": 3,
        "<start_of_turn>": 4, "<end_of_turn>": 5,
        "user": 6, "model": 7, "\n": 8,
    }
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    fast.chat_template = GEMMA_CHAT_TEMPLATE
    return fast


# ------------------------------------------------------------------ unit tests
def test_noop_without_flag():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u1"},
    ]
    assert adapt_messages_for_chat_template(msgs, False) == msgs
    assert adapt_messages_for_chat_template(msgs, False) is not msgs  # new list


def test_leading_system_becomes_user():
    msgs = [{"role": "system", "content": "SYS"}]
    assert adapt_messages_for_chat_template(msgs, True) == [{"role": "user", "content": "SYS"}]


def test_alice_pattern_system_then_assistant():
    # Alice's memory: [system, own turn (assistant), partner turn (user)]
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "B1"},
        {"role": "assistant", "content": "A2"},
    ]
    out = adapt_messages_for_chat_template(msgs, True)
    assert out == [
        {"role": "user", "content": "SYS"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "B1"},
        {"role": "assistant", "content": "A2"},
    ]


def test_bob_pattern_system_then_user_merges():
    # Bob's memory: [system, partner turn (user), own turn (assistant)]
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "A1"},
        {"role": "assistant", "content": "B1"},
    ]
    out = adapt_messages_for_chat_template(msgs, True)
    assert out == [
        {"role": "user", "content": "SYS\n\nA1"},
        {"role": "assistant", "content": "B1"},
    ]


def test_content_is_preserved_verbatim():
    # the prompt text itself must never be modified (requirement 4)
    sys = "You are Bob, a special agent.\nInformation:\nctx_b\n"
    turn = "Alice: found <A>X</A>"
    out = adapt_messages_for_chat_template(
        [{"role": "system", "content": sys}, {"role": "user", "content": turn}], True
    )
    assert out[0]["content"] == sys + "\n\n" + turn


def test_private_information_is_not_mixed():
    # each adapted list contains exactly its own agent's system prompt
    alice = adapt_messages_for_chat_template(
        [
            {"role": "system", "content": "SYS_ALICE_CTX"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "B1"},
        ],
        True,
    )
    bob = adapt_messages_for_chat_template(
        [
            {"role": "system", "content": "SYS_BOB_CTX"},
            {"role": "user", "content": "A1"},
            {"role": "assistant", "content": "B1"},
        ],
        True,
    )
    assert "SYS_BOB_CTX" not in alice[0]["content"]
    assert "SYS_ALICE_CTX" not in bob[0]["content"]
    assert "SYS_ALICE_CTX" in alice[0]["content"]
    assert "SYS_BOB_CTX" in bob[0]["content"]


# --------------------------------------------------- real-template integration
def test_gemma_template_rejects_system_role():
    tok = make_gemma_tokenizer()
    with pytest.raises(Exception):
        tok.apply_chat_template(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}],
            tokenize=False,
        )


def test_gemma_template_requires_alternation():
    tok = make_gemma_tokenizer()
    with pytest.raises(Exception):
        tok.apply_chat_template(
            [{"role": "user", "content": "u1"}, {"role": "user", "content": "u2"}],
            tokenize=False,
        )


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "system", "content": "SYS"}],
        [{"role": "system", "content": "SYS"}, {"role": "assistant", "content": "A1"}],
        [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "A1"},
            {"role": "assistant", "content": "B1"},
        ],
        [
            {"role": "system", "content": "SYS"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "B1"},
            {"role": "assistant", "content": "A2"},
        ],
    ],
)
def test_adapted_messages_render_with_gemma_template(messages):
    tok = make_gemma_tokenizer()
    adapted = adapt_messages_for_chat_template(messages, True)
    text = tok.apply_chat_template(adapted, tokenize=False)
    assert "<start_of_turn>user\n" in text
    assert "SYS" in text
    # generation prompt: the model continues after a model header
    assert text.endswith("<start_of_turn>model\n")

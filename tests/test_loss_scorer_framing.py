"""Loss-scorer framing tests.

``frame_utterance_for_loss`` is the one place the Gemma adaptation touches
the reward pipeline: Gemma's template forbids assistant-only conversations,
so with ``merge_system_into_user`` the utterance is framed as (empty user
turn +) assistant turn, rendering the same ``<start_of_turn>model`` header.
The Qwen path (flag off) must render byte-identical strings to before, so
the R_loss formula stays unchanged.
"""
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("rouge")   # reward.reward (imported by reward.scorer)
pytest.importorskip("openai")
pytest.importorskip("tokenizers")

from reward.scorer import LossScorer, frame_utterance_for_loss

from tests.test_chat_adapter import make_gemma_tokenizer

# Qwen2.5-style chatml template (the default-system-prompt variant), enough
# to pin the Qwen regression: a lone assistant message renders a plain
# "<|im_start|>assistant\n" header.
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{% if loop.first and messages[0]['role'] != 'system' %}"
    "{{ '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}"
    "{% endif %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def make_chatml_tokenizer():
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    vocab = {"<unk>": 0, "<|im_start|>": 1, "<|im_end|>": 2, "assistant": 3, "\n": 4}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token=None,
        eos_token=None,
        pad_token="<|endoftext|>",
        unk_token="<unk>",
    )
    fast.chat_template = CHATML_TEMPLATE
    return fast


def test_gemma_framing_with_merge_renders_model_header():
    tok = make_gemma_tokenizer()
    text = frame_utterance_for_loss(tok, "Alice: hi", merge_system_into_user=True)
    assert "<start_of_turn>user\n" in text  # empty user turn satisfies alternation
    assert "<start_of_turn>model\nAlice: hi" in text


def test_gemma_framing_without_merge_raises():
    # documents WHY the flag exists: Gemma's template rejects assistant-only
    # conversations, so the old framing cannot be used as-is
    tok = make_gemma_tokenizer()
    with pytest.raises(Exception):
        frame_utterance_for_loss(tok, "Alice: hi", merge_system_into_user=False)


def test_qwen_framing_is_unchanged():
    tok = make_chatml_tokenizer()
    text = frame_utterance_for_loss(tok, "Alice: hi", merge_system_into_user=False)
    # identical to the old pipeline's per-sentence assistant framing
    assert "<|im_start|>assistant\nAlice: hi<|im_end|>" in text


class FakeTokenizer:
    """Mimics a tokenizer with no pad token (Gemma)."""

    def __init__(self, pad_token):
        self.pad_token = pad_token
        self.eos_token = "<eos>"
        self.pad_token_id = None


class FakeModel:
    def to(self, device):
        return self

    def eval(self):
        return self

    def parameters(self):
        return []


def test_loss_scorer_falls_back_to_eos_pad(monkeypatch):
    monkeypatch.setattr(
        "transformers.AutoTokenizer",
        type("FakeAutoTokenizer", (), {"from_pretrained": lambda path: FakeTokenizer(None)}),
    )
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM",
        type("FakeAutoModelForCausalLM", (), {"from_pretrained": lambda *a, **k: FakeModel()}),
    )
    scorer = LossScorer("fake/path", device="cpu", batch_size=1, merge_system_into_user=True)
    assert scorer.tokenizer.pad_token == "<eos>"


def test_loss_scorer_keeps_existing_pad(monkeypatch):
    monkeypatch.setattr(
        "transformers.AutoTokenizer",
        type(
            "FakeAutoTokenizer",
            (),
            {"from_pretrained": lambda path: FakeTokenizer("<pad>")},
        ),
    )
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM",
        type("FakeAutoModelForCausalLM", (), {"from_pretrained": lambda *a, **k: FakeModel()}),
    )
    scorer = LossScorer("fake/path", device="cpu", batch_size=1, merge_system_into_user=False)
    assert scorer.tokenizer.pad_token == "<pad>"  # Qwen tokenizers already have one

"""Alice/Bob prompt construction: the two agents must get different system
prompts and different private contexts built from the same template."""
from string import Template

from utils.prompt_template import prompt


def build(name, partner, question, information):
    return Template(prompt).safe_substitute(
        {
            "name": name,
            "partner": partner,
            "question": question,
            "information": information,
        }
    )


def test_alice_bob_system_prompts_differ():
    alice = build("Alice", "Bob", "Q?", "ctx_a_1\nctx_a_2")
    bob = build("Bob", "Alice", "Q?", "ctx_b_1")
    assert alice != bob
    assert "Alice" in alice and "Bob" in alice  # own name + partner name
    assert "Bob" in bob and "Alice" in bob


def test_private_context_isolation_in_prompts():
    alice = build("Alice", "Bob", "Q?", "PRIVATE_ALICE")
    bob = build("Bob", "Alice", "Q?", "PRIVATE_BOB")
    assert "PRIVATE_ALICE" in alice and "PRIVATE_BOB" not in alice
    assert "PRIVATE_BOB" in bob and "PRIVATE_ALICE" not in bob


def test_base_prompt_keeps_author_protocol():
    # author-faithful protocol markers survive in the rendered prompt
    text = build("Alice", "Bob", "Q?", "ctx")
    assert "<A>{answer}</A>" in text
    assert 'You must begin your response with "${name}:".' in text
    assert "continuous communication with your partner" in text


def test_agreement_rule_is_documented():
    # the conversation ends only when both agents output the same <A> answer
    assert "The conversation ends only when all agents output the answer" in prompt

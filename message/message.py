from pydantic import BaseModel
from typing import Optional, List, Dict


class llmMessage(BaseModel):
    """
    Represents a message in an LLM (large language model) conversation.

    Attributes:
        role (str): The role of the message sender. Defaults to "user".
        content (str): The content of the message. Defaults to an empty string.
    """

    role: str = "user"
    content: str = ""

    def to_dict(self):
        return self.content


class Turn(llmMessage):
    """
    A single utterance produced by one agent during generation.

    Compatible with llmMessage (role/content/to_dict), so legacy
    conversation() code keeps working unchanged.
    """

    speaker: str = ""                # "Alice" / "Bob"
    token_count: int = 0
    parsed_answer: Optional[str] = None   # content inside <A>...</A>, else None
    finish_reason: str = ""          # e.g. "stop" / "length" / "error"


def adapt_messages_for_chat_template(
    messages: List[Dict], merge_system_into_user: bool = False
) -> List[Dict]:
    """Adapt a message list for chat templates that reject the "system" role
    and require strict user/assistant alternation starting with user (Gemma).

    With ``merge_system_into_user=True``:
    - a leading system message becomes a user message (content unchanged);
    - consecutive same-role messages are coalesced, contents joined with
      ``"\\n\\n"``. So ``[system, user, assistant, ...]`` becomes
      ``[user(system + "\\n\\n" + user), assistant, ...]`` while
      ``[system, assistant, user, ...]`` becomes
      ``[user(system), assistant, user, ...]``.

    Prompt text is never modified or dropped; each agent still sees only its
    own system prompt and the partner's turns. Returns a new list of plain
    dicts (input objects are not mutated).
    """
    out: List[Dict] = []
    for message in messages:
        role = message.get("role", "user") if isinstance(message, dict) else getattr(message, "role", "user")
        content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
        if merge_system_into_user and not out and role == "system":
            role = "user"
        if out and out[-1]["role"] == role:
            out[-1]["content"] = f"{out[-1]['content']}\n\n{content}"
        else:
            out.append({"role": role, "content": content})
    return out

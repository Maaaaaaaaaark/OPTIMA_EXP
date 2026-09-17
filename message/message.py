from pydantic import BaseModel
from typing import Optional


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

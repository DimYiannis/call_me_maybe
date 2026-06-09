"""Load and index the model vocabulary for token-level constraint lookups."""
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_sdk import Small_LLM_Model


class Vocabulary:
    """
    Wrapper around the model vocab file.

    Provides fast token-id lookups and prefix-based filtering needed
    by the JSON state machine during constrained decoding.
    """

    def __init__(self, model: "Small_LLM_Model") -> None:
        """
        Load vocabulary from the model's vocab.json file.

        Args:
            model: Loaded Small_LLM_Model instance.
        """
        vocab_path = model.get_path_to_vocab_file()
        with open(vocab_path, "r", encoding="utf-8") as f:
            token_to_id: dict[str, int] = json.load(f)

        self._token_to_id: dict[str, int] = token_to_id
        self._id_to_token: dict[int, str] = {
            v: k for k, v in token_to_id.items()
        }

    def size(self) -> int:
        """Return number of tokens in the vocabulary."""
        return len(self._token_to_id)

    def items(self) -> list[tuple[str, int]]:
        """
        Return all (token_string, token_id) pairs.

        Used by the constraint layer to scan the full vocabulary when
        deciding which tokens are valid at a generation step.

        Returns:
            List of (token, id) tuples.
        """
        return list(self._token_to_id.items())

    def get_token(self, token_id: int) -> str:
        """
        Return the string for a token id.

        Args:
            token_id: Integer token id.

        Returns:
            Token string.

        Raises:
            KeyError: If token_id not in vocabulary.
        """
        return self._id_to_token[token_id]

    def get_id(self, token: str) -> int:
        """
        Return the id for a token string.

        Args:
            token: Token string.

        Returns:
            Integer token id.

        Raises:
            KeyError: If token not in vocabulary.
        """
        return self._token_to_id[token]

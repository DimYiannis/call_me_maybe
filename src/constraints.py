"""
Keep generation schema-compliant DURING decoding.

A JSON state machine that, at every generation step, decides which
vocabulary tokens are valid next given the JSON parsed so far and the
function schema. Invalid tokens are masked by the decoder.

The machine produces COMPACT JSON (no whitespace) of the form:

    {"name":"<fn>","parameters":{"<arg>":<value>,...}}

Compact output means each vocabulary key equals its literal characters
for every token we accept, so no byte-level decoding is needed.

Approach
--------
The expected output is a sequence of "instructions":
  - ('lit', s)            emit the fixed string s exactly
  - ('name',)             emit a function name (model's choice of a
                          known name) plus its closing quote
  - ('value', kind, term) emit a typed value, terminated when the
                          following literal's first char `term` appears

A character-level acceptor advances through these instructions. A token
(possibly several characters) is valid iff every one of its characters
can be consumed from the current state.
"""
from .models import FunctionDefinition
from .vocabulary import Vocabulary


# instruction tuple type alias kept loose on purpose
Instruction = tuple


class Constraint:
    """
    JSON-schema state machine for one prompt's generation.

    Tracks the current parse position and exposes the set of valid next
    token ids. A fresh Constraint is used per prompt.
    """

    def __init__(
        self,
        functions: list[FunctionDefinition],
        vocab: Vocabulary,
    ) -> None:
        """
        Build the per-function output programs.

        Args:
            functions: Known function definitions (the schema).
            vocab: Loaded model vocabulary.
        """
        self._vocab = vocab
        self._names: list[str] = [f.name for f in functions]
        self._name_set: set[str] = set(self._names)

        self._programs: dict[str, list[Instruction]] = {
            "prefix": [("lit", '{"name":"'), ("name",)],
        }
        for func in functions:
            self._programs[func.name] = self._build_suffix(func)

        self._alphabet: list[str] = [chr(c) for c in range(0x21, 0x7F)]

        # mutable parse state
        self._program_id: str = "prefix"
        self._index: int = 0
        self._lit_pos: int = 0
        self._name_buf: str = ""
        self._val_state: str = ""
        self._val_index: int = -1
        self._done: bool = False

    # -- program construction ------------------------------------------

    @staticmethod
    def _kind(type_str: str) -> str:
        """Map a schema type string to an internal value kind."""
        t = type_str.lower()
        if t in ("number", "float"):
            return "number"
        if t in ("integer", "int"):
            return "integer"
        if t in ("boolean", "bool"):
            return "boolean"
        return "string"

    def _build_suffix(
        self,
        func: FunctionDefinition,
    ) -> list[Instruction]:
        """
        Build the instruction list emitted after the function name.

        Args:
            func: The selected function definition.

        Returns:
            Instruction list for everything after the name's quote.
        """
        params = list(func.parameters.items())
        n = len(params)
        if n == 0:
            return [("lit", ',"parameters":{}}')]

        seq: list[Instruction] = [("lit", ',"parameters":{')]
        for i, (arg, schema) in enumerate(params):
            if i > 0:
                seq.append(("lit", ","))
            seq.append(("lit", '"' + arg + '":'))
            term = "," if i < n - 1 else "}"
            seq.append(("value", self._kind(schema.type), term))
        seq.append(("lit", "}}"))
        return seq

    # -- value sub-machine ---------------------------------------------

    @staticmethod
    def _is_str_char(ch: str) -> bool:
        """Return True if ch is allowed inside a string value."""
        return 0x21 <= ord(ch) <= 0x7E and ch not in ('"', "\\")

    def _value_step(self, kind: str, term: str, ch: str) -> str:
        """
        Advance the active value sub-machine by one char.

        Args:
            kind: Value kind (number/integer/string/boolean).
            term: Char that terminates the value (start of next literal).
            ch: Candidate character.

        Returns:
            'consumed', 'exit' (value done, ch belongs to next
            instruction), or 'reject'.
        """
        st = self._val_state
        if kind == "number":
            return self._number_step(st, term, ch)
        if kind == "integer":
            return self._integer_step(st, term, ch)
        if kind == "string":
            return self._string_step(st, term, ch)
        return self._boolean_step(st, term, ch)

    def _number_step(self, st: str, term: str, ch: str) -> str:
        """Float grammar: -?digits.digits (forces float output)."""
        if st == "":
            if ch == "-":
                self._val_state = "sign"
                return "consumed"
            if ch.isdigit():
                self._val_state = "int"
                return "consumed"
            return "reject"
        if st == "sign":
            if ch.isdigit():
                self._val_state = "int"
                return "consumed"
            return "reject"
        if st == "int":
            if ch.isdigit():
                return "consumed"
            if ch == ".":
                self._val_state = "dot"
                return "consumed"
            return "reject"
        if st == "dot":
            if ch.isdigit():
                self._val_state = "frac"
                return "consumed"
            return "reject"
        # st == "frac"
        if ch.isdigit():
            return "consumed"
        if ch == term:
            return "exit"
        return "reject"

    def _integer_step(self, st: str, term: str, ch: str) -> str:
        """Integer grammar: -?digits."""
        if st == "":
            if ch == "-":
                self._val_state = "sign"
                return "consumed"
            if ch.isdigit():
                self._val_state = "int"
                return "consumed"
            return "reject"
        if st == "sign":
            if ch.isdigit():
                self._val_state = "int"
                return "consumed"
            return "reject"
        # st == "int"
        if ch.isdigit():
            return "consumed"
        if ch == term:
            return "exit"
        return "reject"

    def _string_step(self, st: str, term: str, ch: str) -> str:
        """String grammar: "chars" (no spaces, no escapes)."""
        if st == "":
            if ch == '"':
                self._val_state = "open"
                return "consumed"
            return "reject"
        if st == "open":
            if ch == '"':
                self._val_state = "closed"
                return "consumed"
            if self._is_str_char(ch):
                return "consumed"
            return "reject"
        # st == "closed"
        if ch == term:
            return "exit"
        return "reject"

    def _boolean_step(self, st: str, term: str, ch: str) -> str:
        """Boolean grammar: true | false."""
        options = ("true", "false")
        if st in options and ch == term:
            return "exit"
        cand = st + ch
        if any(o.startswith(cand) for o in options):
            self._val_state = cand
            return "consumed"
        return "reject"

    # -- core stepping -------------------------------------------------

    def _step(self, ch: str) -> bool:
        """
        Advance the machine by one character (mutating).

        Loops across instruction boundaries when a value exits onto the
        terminator char so that char is re-fed to the next instruction.

        Args:
            ch: Candidate character.

        Returns:
            True if the character was accepted, else False.
        """
        while True:
            prog = self._programs[self._program_id]
            if self._index >= len(prog):
                return False
            instr = prog[self._index]
            kind = instr[0]

            if kind == "lit":
                s = instr[1]
                if ch == s[self._lit_pos]:
                    self._lit_pos += 1
                    if self._lit_pos >= len(s):
                        self._lit_pos = 0
                        self._index += 1
                        if self._index >= len(prog):
                            self._done = True
                    return True
                return False

            if kind == "name":
                if ch == '"':
                    if self._name_buf in self._name_set:
                        self._program_id = self._name_buf
                        self._index = 0
                        self._lit_pos = 0
                        self._val_index = -1
                        self._name_buf = ""
                        return True
                    return False
                cand = self._name_buf + ch
                if any(n.startswith(cand) for n in self._names):
                    self._name_buf = cand
                    return True
                return False

            if kind == "value":
                if self._val_index != self._index:
                    self._val_index = self._index
                    self._val_state = ""
                res = self._value_step(instr[1], instr[2], ch)
                if res == "consumed":
                    return True
                if res == "exit":
                    self._index += 1
                    if self._index >= len(prog):
                        self._done = True
                    continue
                return False

            return False

    # -- snapshot / restore for non-mutating queries -------------------

    def _snapshot(self) -> tuple:
        """Capture the mutable state for later restore."""
        return (
            self._program_id,
            self._index,
            self._lit_pos,
            self._name_buf,
            self._val_state,
            self._val_index,
            self._done,
        )

    def _restore(self, snap: tuple) -> None:
        """Restore state captured by _snapshot."""
        (
            self._program_id,
            self._index,
            self._lit_pos,
            self._name_buf,
            self._val_state,
            self._val_index,
            self._done,
        ) = snap

    def _valid_first_chars(self) -> set[str]:
        """Return the set of characters accepted from the current state."""
        snap = self._snapshot()
        result: set[str] = set()
        for ch in self._alphabet:
            if self._step(ch):
                result.add(ch)
            self._restore(snap)
        return result

    def _can_consume(self, token: str) -> bool:
        """Return True if every char of token is acceptable in sequence."""
        snap = self._snapshot()
        ok = True
        for ch in token:
            if not self._step(ch):
                ok = False
                break
        self._restore(snap)
        return ok

    # -- public API ----------------------------------------------------

    def is_complete(self) -> bool:
        """Return True once a full valid JSON object has been emitted."""
        return self._done

    def valid_next_ids(self) -> list[int]:
        """
        Return all token ids that keep the output schema-valid.

        Returns:
            List of valid token ids; empty when generation is complete.
        """
        if self._done:
            return []
        first = self._valid_first_chars()
        if not first:
            return []
        ids: list[int] = []
        for token, tid in self._vocab.items():
            if not token or token[0] not in first:
                continue
            if self._can_consume(token):
                ids.append(tid)
        return ids

    def accept(self, token_id: int) -> None:
        """
        Commit a chosen token, advancing the real parse state.

        Args:
            token_id: The token id selected by the decoder. Assumed to
                be one returned by valid_next_ids.
        """
        for ch in self._vocab.get_token(token_id):
            self._step(ch)

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal


SlotStrength = Literal["hard", "soft"]


@dataclass(frozen=True)
class SlotValue:
    value: object
    source_turn: int
    source_text: str
    confidence: float = 1.0
    strength: SlotStrength = "hard"


@dataclass
class SessionState:
    session_id: str
    user_profile: dict = field(default_factory=dict)
    dialog_history: list[dict[str, str]] = field(default_factory=list)
    slots: dict[str, SlotValue] = field(default_factory=dict)
    negated_slots: dict[str, set[str]] = field(default_factory=dict)
    asked_attributes: list[str] = field(default_factory=list)
    declined_attributes: set[str] = field(default_factory=set)
    last_asked_attribute: str | None = None
    current_intent: str | None = None
    subject_terms: list[str] = field(default_factory=list)
    shown_asins: set[str] = field(default_factory=set)
    retrieval_history: list[dict[str, object]] = field(default_factory=list)

    @classmethod
    def create(cls, session_id: str, user_profile: dict) -> "SessionState":
        return cls(session_id=session_id, user_profile=deepcopy(user_profile))

    @property
    def accumulated_slots(self) -> dict[str, object]:
        """Compatibility view used by tests and retrieval code."""
        return {key: slot.value for key, slot in self.slots.items()}

    @property
    def negated_terms(self) -> set[str]:
        """Flattened compatibility view for diagnostics and older tests."""
        return {value for values in self.negated_slots.values() for value in values}

    def set_slot(
        self,
        key: str,
        value: object,
        turn: int,
        source_text: str,
        *,
        strength: SlotStrength = "hard",
    ) -> None:
        self.slots[key] = SlotValue(value, turn, source_text, strength=strength)
        normalized = str(value).lower()
        if key in self.negated_slots:
            remaining = {
                negated for negated in self.negated_slots[key]
                if negated.lower() != normalized
            }
            if remaining:
                self.negated_slots[key] = remaining
            else:
                del self.negated_slots[key]

    def negate_slot(self, key: str, value: str) -> None:
        normalized = value.lower()
        self.negated_slots.setdefault(key, set()).add(value)
        active = self.slots.get(key)
        if active is not None and str(active.value).lower() == normalized:
            del self.slots[key]

    def clear_soft_preferences(self) -> None:
        self.slots = {
            key: slot for key, slot in self.slots.items() if slot.strength == "hard"
        }

"""
Style presets and refinement prompts for Blackwell-Flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import StyleType


@dataclass(frozen=True)
class StylePreset:
    """A style preset for LLM refinement."""

    name: str
    description: str
    system_prompt: str
    example_output: str | None = None


STYLE_PRESETS: dict[str, StylePreset] = {
    "professional": StylePreset(
        name="Professional",
        description="Formal, clear language suitable for business communication",
        system_prompt="""You are a professional editor. Clean up the following transcript for clarity, grammar, and flow.

Rules:
- Remove filler words (um, uh, like, you know, basically, actually, sort of, kind of)
- Fix grammatical errors
- Maintain formal, professional tone
- Preserve technical terminology accurately
- Keep sentences concise and clear
- Maintain the speaker's original intent
- Output ONLY the corrected text, no explanations or commentary""",
    ),

    "casual": StylePreset(
        name="Casual",
        description="Relaxed, conversational tone for informal communication",
        system_prompt="""You are a friendly editor. Clean up this transcript while keeping it casual and natural.

Rules:
- Remove excessive filler words but keep some for natural flow
- Fix obvious grammatical mistakes
- Keep contractions and informal language
- Preserve the speaker's personality and voice
- Output ONLY the corrected text, no explanations""",
    ),

    "creative": StylePreset(
        name="Creative",
        description="Expressive, varied vocabulary for creative writing",
        system_prompt="""You are a creative writing editor. Enhance this transcript for expressive, engaging prose.

Rules:
- Remove filler words completely
- Improve word choice for variety and impact
- Add appropriate punctuation for rhythm
- Enhance descriptive elements if present
- Maintain the core message and intent
- Output ONLY the enhanced text, no explanations""",
    ),

    "bullet_points": StylePreset(
        name="Bullet Points",
        description="Structured list format for notes and documentation",
        system_prompt="""You are a technical documentation editor. Convert this transcript into organized bullet points.

Rules:
- Extract key points and organize hierarchically
- Use clear, concise language
- Remove all filler words and redundancy
- Group related items together
- Use proper markdown bullet formatting (- for items)
- Output ONLY the bullet point list, no explanations""",
    ),
}


def get_style_preset(style: StyleType) -> StylePreset:
    """Get the style preset for the given style name."""
    return STYLE_PRESETS[style]


def get_refinement_prompt(style: StyleType, transcript: str) -> str:
    """Generate the full refinement prompt for the LLM."""
    preset = get_style_preset(style)
    return f"{preset.system_prompt}\n\nTranscript:\n{transcript}\n\nCorrected text:"


def cycle_style(current: StyleType) -> StyleType:
    """Cycle to the next style in the list."""
    styles: list[StyleType] = ["professional", "casual", "creative", "bullet_points"]
    current_idx = styles.index(current)
    next_idx = (current_idx + 1) % len(styles)
    return styles[next_idx]
